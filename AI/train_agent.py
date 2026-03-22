import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import torch.nn as nn
from stable_baselines3 import PPO

class CartPoleSwingUpEnv(gym.Env):
    def __init__(self, render_mode="none"):
        super(CartPoleSwingUpEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path("625/urdf/mjmodel.xml")
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.array([5.0, 50.0, 1.0, 1.0, 50.0], dtype=np.float32), 
            high=np.array([5.0, 50.0, 1.0, 1.0, 50.0], dtype=np.float32), 
            dtype=np.float32
        )
        self.max_steps = 1000 
        self.current_step = 0 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = 0.0  
        self.data.qpos[1] = self.np_random.uniform(-0.1, 0.1) 
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        
        # flag: track if agent successfully swung up
        self.has_reached_top = False 
        
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), {}
    
    def step(self, action):
        self.data.ctrl[0] = action[0]
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        cart_x, cart_vel, cos_th, sin_th, pole_vel = obs
        self.current_step += 1
        
        if not hasattr(self, "previous_action"):
            self.previous_action = action[0]

        # base theta reward
        reward_theta = 1.0 - cos_th
        
        # spatial constraint: free swing-up, soft boundary penalty
        penalty_x = 0.0 
        penalty_boundary = 0.0
        if cart_x > 0.35 or cart_x < -0.35:
            penalty_boundary = 50.0 * (abs(cart_x) - 0.35)

        # jerk calc
        action_delta = abs(action[0] - self.previous_action)

        balance_bonus = 0.0
        penalty_action = 0.0
        penalty_smoothness = 0.0
        penalty_cart_vel = 0.0
        penalty_pole_vel = 0.0
        drop_penalty = 0.0 

        # state-machine: milestone check
        if cos_th < -0.9:
            self.has_reached_top = True

        # state-machine: split logic top vs bottom
        if cos_th < -0.7: 
            # top phase: massive reward, strict stabilization
            balance_bonus = 30.0 
            penalty_action = 0.5 * (action[0]**2)
            penalty_cart_vel = 1.0 * (cart_vel**2)
            penalty_pole_vel = 1.0 * (pole_vel**2)
            penalty_smoothness = 1.0 * action_delta
        else:
            # bottom phase: absolute freedom, massive drop penalty
            penalty_action = 0.0
            if getattr(self, "has_reached_top", False):
                drop_penalty = 50.0

        reward = float(reward_theta + balance_bonus - penalty_x - penalty_boundary - penalty_action - penalty_smoothness - penalty_cart_vel - penalty_pole_vel - drop_penalty)
        self.previous_action = action[0]

        # termination: physical limit
        terminated = bool(cart_x < -0.45 or cart_x > 0.45) 
        if terminated:
            reward -= 100.0
            
        truncated = bool(self.current_step >= self.max_steps)
        if self.render_mode == "human":
            self._render_frame()
            
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        x = self.data.qpos[0]
        theta = self.data.qpos[1]
        v_x = self.data.qvel[0]
        v_th = self.data.qvel[1]
        return np.array([x, v_x, np.cos(theta), np.sin(theta), v_th], dtype=np.float32)

    def _render_frame(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

if __name__ == "__main__":
    env = CartPoleSwingUpEnv(render_mode="none")
    custom_arch = dict(activation_fn=nn.Tanh, net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=1e-4,
                gamma=0.999,
                clip_range=0.1,
                policy_kwargs=custom_arch, 
                tensorboard_log="./tensorboard/tensorboard_logs/",
                device="cpu")
    print("Training begins...")
    model.learn(total_timesteps=2000000, tb_log_name="Fix_Reward_Ray1Met_30N")
    model.save("models/ppo_cartpole_swingup_hardware")
    print("Done and saved!")