import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
os.environ["WANDB_SILENT"] = "true"
import wandb
from wandb.integration.sb3 import WandbCallback

class CartPoleSwingUpEnv(gym.Env):
    def __init__(self, render_mode="none"):
        super(CartPoleSwingUpEnv, self).__init__()
        
        self.model = mujoco.MjModel.from_xml_path("m.625/urdf/mjmodel.xml") # cpu
        self.data = mujoco.MjData(self.model) 

        self.render_mode = render_mode
        self.viewer = None
        self.current_step = 0 
        self.prev_action = 0.0 # init action tracking
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.array([0.5, 5.0, 1.0, 1.0, 20.0], dtype=np.float32), 
            high=np.array([0.5, 5.0, 1.0, 1.0, 20.0], dtype=np.float32), 
            dtype=np.float32
        )
        self.max_steps = 1000 # episode step limit
        self.current_step = 0 
        self.force_scale = 60.0 
        self.actual_force = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # init state
        self.data.qpos[0] = 0.0  
        self.data.qpos[1] = self.np_random.uniform(-0.1, 0.1) 
        self.data.qvel[:] = 0.0

        # body/joint id lookup
        body_pole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pole")
        jnt_cart_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cart_prismatic")
        jnt_pole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole_continuous")

        # mass randomization
        self.model.body_mass[body_pole_id] = 0.174 * self.np_random.uniform(0.95, 1.05)
        
        # prismatic joint
        nom_cart_damping  = 0.2   
        nom_cart_friction = 0.2   
        nom_cart_armature = 0.3   
        
        # revolute joint
        nom_pole_damping  = 0.01  
        nom_pole_friction = 0.01  

        # dr: random +/- 20% 
        self.model.dof_damping[jnt_cart_id]      = nom_cart_damping  * self.np_random.uniform(0.8, 1.2)
        self.model.dof_frictionloss[jnt_cart_id] = nom_cart_friction * self.np_random.uniform(0.8, 1.2)
        self.model.dof_armature[jnt_cart_id]     = nom_cart_armature * self.np_random.uniform(0.8, 1.2) 
        self.model.dof_damping[jnt_pole_id]      = nom_pole_damping  * self.np_random.uniform(0.8, 1.2)
        self.model.dof_frictionloss[jnt_pole_id] = nom_pole_friction * self.np_random.uniform(0.8, 1.2)

        # force scale randomization 
        self.force_scale = 60.0 * self.np_random.uniform(0.85, 1.15)

        # reset logic memory
        self.actual_force = 0.0 
        self.current_step = 0
        self.prev_action = 0.0  

        # forward kinematics update
        mujoco.mj_forward(self.model, self.data)
   
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), {}

    def step(self, action):
        action_value = float(action[0])

        # apply force scale & ema filter
        target_force = action_value * self.force_scale
        alpha = 0.15 
        self.actual_force = (1.0 - alpha) * self.actual_force + alpha * target_force 
        self.data.ctrl[0] = float(self.actual_force)

        # sync physics freq
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        cart_x, cart_vel, cos_th, sin_th, pole_vel = obs
        self.current_step += 1

        # priority
        reward_theta = 1.0 - cos_th
        # prevent sudden action value changes
        action_diff = action_value - self.prev_action
        self.prev_action = action_value  # update from action history
        
        # discourage motor action
        penalty_action = 0.005 * (action_value**2) 

        # 1. smooth reward shaping (gaussian)
        error_to_top = 2.0 - reward_theta # distance to top (0 when at top, 2 when at bottom)
        smooth_bonus = 2.0 * np.exp(-5.0 * (error_to_top**2)) #gaussian bell, 0 at bottom, 2 at top, smooth in between
        norm_bonus = smooth_bonus / 2.0 # dynamic penalties: scaled [0, 1]
        
        # 2. cart's vel constraints 
        penalty_vx = (0.01 + 0.05 * norm_bonus) * (cart_vel**2)
        # 3.  cart's pos constraints 
        penalty_x = (0.5 + 2.0 * norm_bonus) * (cart_x**2) 
        
        # 4. motor constraints for sudden changes 
        penalty_action_rate = (0.005 + 0.05 * norm_bonus) * (action_diff**2)
        
        # 5. pole vel constraints: preventing overshoot
        penalty_vth = (0.001 + 0.1 * norm_bonus) * (pole_vel**2)

        # soft boundary penalty
        penalty_boundary = 0.0
        if abs(cart_x) > 0.25:
            penalty_boundary = 10.0 * (abs(cart_x) - 0.25)

        reward = float(reward_theta - penalty_x - penalty_action - penalty_action_rate - penalty_boundary - penalty_vx - penalty_vth + smooth_bonus)

        # termination
        terminated = bool(abs(cart_x) > 0.44)
        if terminated:
            reward -= 5.0  # harsh penalty for dropping
        truncated = bool(self.current_step >= self.max_steps)

        # env wrapper info
        info = {}
        if truncated:
            info["TimeLimit.truncated"] = True # required for evaluation scripts

        if self.render_mode == "human":
            self._render_frame()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # sensor noise injection
        x = self.data.qpos[0] + self.np_random.normal(0, 0.0005)
        theta = self.data.qpos[1] + self.np_random.normal(0, 0.0005)
        v_x = self.data.qvel[0] + self.np_random.normal(0, 0.005)
        v_th = self.data.qvel[1] + self.np_random.normal(0, 0.01)
        return np.array([x, v_x, np.cos(theta), np.sin(theta), v_th], dtype=np.float32)

    def _render_frame(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

if __name__ == "__main__":
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize
    import datetime
    
    TOTAL_TIMESTEPS = 4096000 
    current_seed = 42 # fix seed for fair reward tuning comparison

    # dynamic run naming
    version_id = datetime.datetime.now().strftime("%H%M%S")
    run_name = f"tune_{version_id}_seed{current_seed}"

    print(f"\n{run_name} started!\n")
    
    # 1. wandb
    run = wandb.init(
        project="pendulum-ppo", 
        name=run_name,
        config={
            "total_timesteps": TOTAL_TIMESTEPS,
            "learning_rate": "linear_schedule(3e-4)",
            "architecture": "128x128",
            "seed": current_seed,
            "n_stack": 8,
            "learning_rate": "linear_schedule(0.00013279)", 
            "batch_size": 128,
            "ent_coef": 0.001223,
            "gamma": 0.9893,
            "n_epochs": 10,
            "num_envs": 8,
            "n_steps_per_env": 512,
            "total_rollout_buffer": 4096,
            "note": "Optuna Rank 1 (CPU Parallel 8 Envs)"
        },
        sync_tensorboard=True,  
        monitor_gym=False,       
        save_code=True,         
        reinit=True                 
    )
    
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")

    # 2. env setup
    num_envs = 8 
    
    def make_env(env_seed):
        # factory function to create independent env instances
        def _init():
            env = CartPoleSwingUpEnv(render_mode="none")
            env.reset(seed=env_seed)
            return Monitor(env)
        return _init

    # create a list of functions, each with a unique seed to ensure diverse data collection
    env_fns = [make_env(current_seed + i) for i in range(num_envs)]
    
    # launch parallel subprocesses
    vec_env = SubprocVecEnv(env_fns)

    vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    vec_env = VecFrameStack(vec_norm, n_stack=8) 

    custom_arch = dict(activation_fn=nn.Tanh, net_arch=dict(pi=[128, 128], vf=[128, 128])) 

    # 3. hyperparameters setup
    model = PPO("MlpPolicy", vec_env, verbose=0,
                n_steps=512,
                batch_size=128,
                learning_rate=linear_schedule(0.00013279), # Best Initial LR
                ent_coef=0.001223,                         # Best Entropy
                n_epochs=10,                               # Best n_epochs
                gamma=0.9893, 
                target_kl=0.015, 
                seed=current_seed,  
                policy_kwargs=custom_arch, 
                tensorboard_log="./tensorboard/tensorboard_logs/",
                device="cpu")
    
    # 4. training loop 
    model.learn(total_timesteps=TOTAL_TIMESTEPS, 
                tb_log_name=run_name, 
                callback=WandbCallback())
    
    # 5. save model
    os.makedirs("models/env", exist_ok=True)
    os.makedirs("vec/env", exist_ok=True)
    
    model.save(f"models/env/{run_name}")
    vec_norm.save(f"vec/env/vec_normalize_{run_name}.pkl")
    
    vec_env.close()
    run.finish() 

    print(f"Done: {run_name}!")