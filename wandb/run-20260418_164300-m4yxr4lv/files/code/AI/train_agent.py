import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

class CartPoleSwingUpEnv(gym.Env):
    def __init__(self, render_mode="none"):
        super(CartPoleSwingUpEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path("625/urdf/mjmodel.xml")
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        self.current_step = 0 
        self.prev_action = 0.0 # Init tracking for action smoothing
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.array([5.0, 50.0, 1.0, 1.0, 50.0], dtype=np.float32), 
            high=np.array([5.0, 50.0, 1.0, 1.0, 50.0], dtype=np.float32), 
            dtype=np.float32
        )
        self.max_steps = 1000 
        self.current_step = 0 
        self.force_scale = 60.0 
        self.actual_force = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        #initial state
        self.data.qpos[0] = 0.0  
        self.data.qpos[1] = self.np_random.uniform(-0.1, 0.1) 
        self.data.qvel[:] = 0.0

        # ID lookup
        body_pole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pole")
        jnt_cart_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cart_prismatic")
        jnt_pole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole_continuous")

        # mass randomization (+/- 15%)
        self.model.body_mass[body_pole_id] = 0.174 * self.np_random.uniform(0.95, 1.05)
        # friction, damping, armature randomization for cart
        self.model.dof_damping[jnt_cart_id] = self.np_random.uniform(0.2, 0.4)
        self.model.dof_frictionloss[jnt_cart_id] = self.np_random.uniform(0.15, 0.3)
        self.model.dof_armature[jnt_cart_id] = self.np_random.uniform(0.15, 0.3) 
        # friction, damping randomization for pole
        self.model.dof_damping[jnt_pole_id] = self.np_random.uniform(0.005, 0.02)
        self.model.dof_frictionloss[jnt_pole_id] = self.np_random.uniform(0.005, 0.02)

        # force scale randomization 
        self.force_scale = 60.0 * self.np_random.uniform(0.85, 1.15)

        # reset logic memory
        self.actual_force = 0.0 # reset force filter memory
        self.current_step = 0
        self.prev_action = 0.0  # reset action history

        # end of randomness
        mujoco.mj_forward(self.model, self.data)
   
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), {}

    def step(self, action):
        # Support scalar or array-like actions; clip to action_space
        action_value = float(np.clip(np.atleast_1d(action)[0], self.action_space.low[0], self.action_space.high[0]))
        #motor deadband
        #if abs(action_value) < 0.08:
        #     action_value = 0.0 

        # apply randomized force scale
        target_force = action_value * self.force_scale
        #(0.1-0.3)
        alpha = 0.15 

        self.actual_force = (1.0 - alpha)*self.actual_force + alpha*target_force
        self.data.ctrl[0] = float(self.actual_force)

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        cart_x, cart_vel, cos_th, sin_th, pole_vel = obs
        self.current_step += 1

        reward_theta = 1.0 - cos_th
        # Penalties: softer scaling, prevent gradient spike
        penalty_x = 1.0 * (cart_x**2)
        penalty_action = 0.005 * (action_value**2)
        # Action rate penalty: limit high-frequency changes (chattering)
        action_diff = action_value - self.prev_action
        penalty_action_rate = 0.05 * (action_diff**2)
        self.prev_action = action_value  # Update for next step
        # Velocity penalty: prevent reward hacking (swing-up by dropping)
        penalty_vx = 0.01 * (cart_vel**2)
        penalty_vth = 0.001 * (pole_vel**2)
        penalty_boundary = 0.0
        # Soft boundary penalty (> 0.3m)
        if abs(cart_x) > 0.3:
            penalty_boundary = 10.0 * (abs(cart_x) - 0.3)  # soft penalty

        reward = float(reward_theta- penalty_x - penalty_action- penalty_action_rate- penalty_boundary- penalty_vx- penalty_vth)

        terminated = bool(abs(cart_x) > 0.44)
        if terminated:
            reward -= 5.0  # Hard penalty: reduced from 100 to avoid reward cliff
        truncated = bool(self.current_step >= self.max_steps)

        # Provide terminal observation in info (used by VecNormalize)
        info = {}
        if terminated or truncated:
            info["terminal_observation"] = obs.copy()
            if truncated:
                info["TimeLimit.truncated"] = True

        if self.render_mode == "human":
            self._render_frame()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # sensor noise injection
        x = self.data.qpos[0] + self.np_random.normal(0, 0.001)
        theta = self.data.qpos[1] + self.np_random.normal(0, 0.002)
        v_x = self.data.qvel[0] + self.np_random.normal(0, 0.01)
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

# Linear schedule: progress from 1.0 to 0.0 (LR schedule)
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func
if __name__ == "__main__":
    # 1. WandB
    run = wandb.init(
        project="CartPole-Mujoco-Lab",
        config={
            "total_timesteps": 500000,
            "learning_rate": 3e-4,
            "architecture": "128x128",
        },
        sync_tensorboard=True,  # data from Tensorboard 
        monitor_gym=False,       # Video
        save_code=True,         # Code version storing
    )

    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
    # Create a fresh Monitor-wrapped env per worker to avoid reusing the same instance
    vec_env = DummyVecEnv([lambda: Monitor(CartPoleSwingUpEnv(render_mode="none"))])
    # Normalize observations/rewards and keep reference to save stats later
    vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    # Frame stack after normalization (history of 8 frames)
    vec_env = VecFrameStack(vec_norm, n_stack=8)
    custom_arch = dict(activation_fn=nn.Tanh, net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO("MlpPolicy", vec_env, verbose=0,
                learning_rate=linear_schedule(3e-4), # LR decay
                target_kl=0.015, # KL constraint: early stop policy update (target_kl)
                policy_kwargs=custom_arch, 
                tensorboard_log="./tensorboard/tensorboard_logs/",
                device="cpu")
    print("Training begins...")
    model.learn(total_timesteps=500000, tb_log_name="ppo_force_real", callback=WandbCallback())
    model.save("models/ppo_force_real")
    # Save VecNormalize statistics
    vec_norm.save("vec/vec_normalize.pkl")
    run.finish() #for wandb
    print("Done!")