from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
import os
import time

# QUAN TRỌNG: Import môi trường từ train_sb3.py (nơi chứa code Delta Velocity)
from train_sb3 import CartPoleSwingUpEnv

# mode selection flag
USE_SPECIFIC_SEED = True # Đổi thành True để test mô hình cụ thể
SEED = 42                # Seed bạn vừa train xong

def main():
    env = CartPoleSwingUpEnv(render_mode="human")
    vec_env = DummyVecEnv([lambda: env])

    # Trỏ đường dẫn tới folder 'vel' theo đúng cấu trúc thư mục của bạn
    if USE_SPECIFIC_SEED:
        stats_path = f"vec/vel/vec_normalize_vel_{SEED}.pkl"
        model_path = f"models/vel/ppo_vel_{SEED}.zip"
    else:
        # Fallback path nếu để USE_SPECIFIC_SEED = False
        stats_path = "vec/vel/vec_normalize_vel_42.pkl"
        model_path = "models/vel/ppo_vel_42.zip"

    # 1. Load VecNormalize (Bắt buộc load trước FrameStack)
    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = False   # Khóa cập nhật mean/var khi test
    vec_env.norm_reward = False # Không cần normalize reward lúc xem render
    
    # 2. Load FrameStack với n_stack=8 (Khớp với file train)
    vec_env = VecFrameStack(vec_env, n_stack=8) 

    print(f"[*] Đang load mô hình từ: {model_path}")
    print(f"[*] Đang load Normalize từ: {stats_path}")
    
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
                print("Truncated (Hết thời gian)" if is_truncated else "Terminated (Xe trượt quá giới hạn)")
                time.sleep(1) # Dừng 1 giây trước khi reset để dễ quan sát
                
    except KeyboardInterrupt:
        print("\n[*] Đã đóng chương trình Test.")
        pass
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()