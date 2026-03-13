import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import torch.nn as nn
from stable_baselines3 import PPO

class CartPoleSwingUpEnv(gym.Env):
    
    def __init__(self, render_mode="human"):
        super(CartPoleSwingUpEnv, self).__init__()
        
        # 1. Nạp mô hình vật lý
        self.model = mujoco.MjModel.from_xml_path("cartpole_mujoco/urdf/cartpole_native.xml")
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.viewer = None

        # 2. Không gian hành động: Lực đẩy 15N (Đủ để múa lấy đà trong không gian hẹp)
        self.action_space = spaces.Box(low=-15.0, high=15.0, shape=(1,), dtype=np.float32)
        
        # 3. Không gian trạng thái
        high = np.array([5.0, 50.0, 1.0, 1.0, 50.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # 4. Giới hạn thời gian (Bom hẹn giờ) để ép đồ thị Rollout hiện lên
        self.max_steps = 1000 
        self.current_step = 0 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # ĐẶT TRẠNG THÁI BAN ĐẦU: Rủ xuống dưới (Góc 0 theo hệ SolidWorks)
        self.data.qpos[0] = 0.0  
        self.data.qpos[1] = self.np_random.uniform(-0.1, 0.1) 
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Reset lại đồng hồ mỗi khi bắt đầu ván mới
        self.current_step = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        # Truyền lực vào động cơ
        self.data.ctrl[0] = action[0]
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        cart_x, cart_vel, cos_th, sin_th, pole_vel = obs
        
        # Cập nhật đồng hồ đếm bước
        self.current_step += 1

        # ==========================================
        # HÀM CHẤM ĐIỂM DÀNH RIÊNG CHO RAY DÀI 1 MÉT
        # ==========================================
        # Hướng lên được +1 điểm, rủ xuống bị -1 điểm
        reward_theta = -cos_th 
        
        # Phạt nặng dần khi chạy xa trung tâm
        penalty_x = 1.0 * (cart_x**2)
        
        # Phạt nhẹ nếu dùng lực quá lố (Ép múa mượt)
        penalty_action = 0.001 * (action[0]**2)

        # BIỂN BÁO: Bắt đầu hãm phanh khi xe vượt qua mốc 0.3m (cách vách 0.15m)
        penalty_boundary = 0.0
        if cart_x > 0.3 or cart_x < -0.3:
            # Hệ số 100.0 ép AI quay đầu ngay lập tức vì không gian còn lại rất ngắn
            penalty_boundary = 100.0 * (abs(cart_x) - 0.3)
        
        # Cộng trừ tổng điểm
        reward = float(reward_theta - penalty_x - penalty_action - penalty_boundary)
        # ==========================================
        
        # KẾT THÚC (TERMINATED) - Ray 1m -> Nửa ray là 0.5m -> Tâm xe đụng 0.45m là vỡ ray!
        terminated = bool(cart_x < -0.45 or cart_x > 0.45) 

        # ÁN TỬ HÌNH: Trừ 100 điểm nếu dám đâm vách!
        if terminated:
            reward -= 100.0

        # HẾT GIỜ (TRUNCATED) - Ép kết thúc ván nếu sống đủ 1000 bước
        truncated = bool(self.current_step >= self.max_steps)
        
        if self.render_mode == "human":
            self._render_frame()
            
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """Đọc cảm biến từ MuJoCo"""
        x = self.data.qpos[0]
        theta = self.data.qpos[1]
        v_x = self.data.qvel[0]
        v_th = self.data.qvel[1]
        return np.array([x, v_x, np.cos(theta), np.sin(theta), v_th], dtype=np.float32)

    def _render_frame(self):
        """Chạy cửa sổ 3D ngầm"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

# AI activation
if __name__ == "__main__":
    # Đang để "none" cho chạy train thần tốc. Đổi thành "human" nếu muốn xem nó múa!
    env = CartPoleSwingUpEnv(render_mode="none")

    # 128*128 neural network
    custom_arch = dict(activation_fn=nn.Tanh, net_arch=dict(pi=[128, 128], vf=[128, 128]))

    # Cấu hình PPO
    model = PPO("MlpPolicy", env, verbose=1, 
                policy_kwargs=custom_arch, 
                tensorboard_log="./tensorboard/tensorboard_logs/",
                device="cpu")
    
    print("Training begins...")
    # Tên log mới: Fix_Reward_Ray1Met
    model.learn(total_timesteps=500000, tb_log_name="Fix_Reward_Ray1Met")
    
    model.save("models/ppo_cartpole_swingup")
    print("Done and saved!")