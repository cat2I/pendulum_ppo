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
        # Action space for stepper motor velocity control
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
        # Rewards and penalties
        #1. Phan thuong tinh tien (luon >= 0)
        reward_theta = 1.0-cos_th 
        balance_bonus = 0.0
        penalty_pole_vel = 0.0
        penalty_cart_vel = 0.0
        #2. Pheu doc & Phanh gat ( kich hoat khi nghieng <25 do)
        if cos_th < -0.9: 
            # Diem dao dong tu 20 den 30 tuy do thang dung
            balance_bonus = 20.0 + 100.0*(-cos_th - 0.9)
            penalty_pole_vel = 1.0 * (pole_vel**2)
            penalty_cart_vel = 0.5 * (cart_vel**2)
        #3. Phat hanh vi co ban (Noi long de AI de tho)
        penalty_x = 0.5 * (cart_x**2)
        #penalty_action = 0.001 * (action[0]**2)
        if cos_th < -0.85: # Smart Action Penalty (Phạt thông minh) 
            # 1. Khi con lắc đang ở trên đỉnh (Vùng thăng bằng)
            penalty_action = 0.005 * (action[0]**2) #Phạt NẶNG hành động để ép AI phải rà động cơ thật mượt, tránh giật cục
        else:
            # 2. Khi con lắc ở nửa dưới (Vùng cần lấy đà)
            penalty_action = 0.0001 * (action[0]**2) #Phạt NHẸ để cho phép AI bung tối đa sức mạnh NEMA 23 hất con lắc lên
        penalty_boundary = 0.0
        # Soft boundary penalty (> 0.3m)
        if cart_x > 0.3 or cart_x < -0.3:
            penalty_boundary = 100.0 * (abs(cart_x) - 0.3)
        reward = float(reward_theta + balance_bonus - penalty_pole_vel - penalty_cart_vel- penalty_x - penalty_action - penalty_boundary)
        # Hard termination (0.45m limit)
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
                policy_kwargs=custom_arch, 
                tensorboard_log="./tensorboard/tensorboard_logs/",
                device="cpu")
    print("Training begins...")
    model.learn(total_timesteps=2000000, tb_log_name="Fix_Reward_Ray1Met_30N")
    model.save("models/ppo_cartpole_swingup_hardware")
    print("Done and saved!")