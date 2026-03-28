import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

class CartPoleSwingUpEnv(gym.Env):
    def __init__(self, render_mode="none"):
        super(CartPoleSwingUpEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path("625/urdf/mjmodel.xml")
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        self.prev_action = 0.0 # Init tracking for action smoothing
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
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
        self.prev_action = 0.0 # Clear history on new episode
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), {}

    def step(self, action):
        target_vel = action[0] * 1.0
        self.data.ctrl[0] = target_vel
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)    
        obs = self._get_obs()
        cart_x, cart_vel, cos_th, sin_th, pole_vel = obs
        self.current_step += 1

        reward_theta = 1.0 - cos_th
        action_diff = action[0] - self.prev_action

        if cos_th < -0.8: 
            balance_bonus = 10.0 + 20.0 * (-cos_th - 0.8)
            penalty_pole_vel = 2.0 * (pole_vel**2)
            penalty_action = 0.0 
            penalty_action_rate = 0.05 * (action_diff**2)
            penalty_x = 0.5 * (cart_x**2)
            swing_up_bonus = 0.0 
        else:
            balance_bonus = 0.0
            penalty_pole_vel = 0.0
            penalty_action = 0.0
            penalty_action_rate = 0.0  

            if abs(cart_x) < 0.3: 
                penalty_x = 0.0
            else:
                penalty_x = 50.0 * (abs(cart_x) - 0.3)**2

            swing_up_bonus = 0.5 * abs(cart_vel)

        self.prev_action = action[0] 

        reward = float(reward_theta + swing_up_bonus + balance_bonus - penalty_pole_vel - penalty_x - penalty_action - penalty_action_rate)

        terminated = bool(abs(cart_x) > 0.44) 
        if terminated:
            reward -= 20.0

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

from typing import Callable

# Linear schedule: progress from 1.0 to 0.0 (LR schedule)
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func
if __name__ == "__main__":
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    env = CartPoleSwingUpEnv(render_mode="none")
    env = Monitor(env)
    # 2. Bọc môi trường để tương thích với SB3 vectorization
    vec_env = DummyVecEnv([lambda: env])
    # 3. Kích hoạt Frame Stacking (Lưu lịch sử 8 frames)
    # Lưu ý: Lúc này Observation Space của bạn sẽ tự động tăng lên gấp 8 lần (5 x 8 = 40 chiều)
    vec_env = VecFrameStack(vec_env, n_stack=8)
    custom_arch = dict(activation_fn=nn.Tanh, net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO("MlpPolicy", vec_env, verbose=1,
                learning_rate=linear_schedule(3e-4), # LR decay
                target_kl=0.015, # KL constraint: early stop policy update (target_kl)
                policy_kwargs=custom_arch, 
                tensorboard_log="./tensorboard/tensorboard_logs/",
                device="cpu")
    print("Training begins...")
    model.learn(total_timesteps=500000, tb_log_name="ppo_stepper_vel")
    model.save("models/ppo_stepper_vel")
    print("Done!")