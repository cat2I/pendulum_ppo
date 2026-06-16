from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
import os
import time

# QUAN TRỌNG: Import môi trường từ train_vel.py (nơi chứa code Delta Velocity)
from train_vel import CartPoleSwingUpEnv

def main():
    env = CartPoleSwingUpEnv(render_mode="human")
    vec_env = DummyVecEnv([lambda: env])

    stats_path = "z.vel/vec_normalize_vel_42.pkl"
    model_path = "z.vel/ppo_vel_42.zip"

    # 1. Load VecNormalize (Bắt buộc load trước FrameStack)
    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = False   # Khóa cập nhật mean/var khi test
    vec_env.norm_reward = False # Không cần normalize reward lúc xem render
    
    # 2. Load FrameStack với n_stack=8 (Khớp với file train)
    vec_env = VecFrameStack(vec_env, n_stack=8) 
    
    model = PPO.load(model_path, env=vec_env)
    obs = vec_env.reset()

    try:
        while True:
            # Agent dự đoán hành động dựa trên State + History (8 frames)
            action, _states = model.predict(obs, deterministic=True)
            
            # Thực thi hành động
            obs, reward, dones, info = vec_env.step(action)
            
            # Lock physics FPS (0.01s khớp với frame_skip = 5 bước mujoco)
            time.sleep(0.01) 
            
            if dones[0]:
                is_truncated = info[0].get("TimeLimit.truncated", False)
                print("Truncated" if is_truncated else "Terminated")
                time.sleep(1) # Dừng 1 giây trước khi reset để dễ quan sát
                
    except KeyboardInterrupt:
        print("\n[*] Đã đóng chương trình Test.")
        pass
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()