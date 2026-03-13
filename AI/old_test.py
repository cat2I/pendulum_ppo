import gymnasium as gym
from stable_baselines3 import PPO
import time

# Gọi class môi trường mà bạn đã viết ở file train_agent sang đây
from train_agent import CartPoleSwingUpEnv

def main():
    # 1. Bật môi trường 3D
    env = CartPoleSwingUpEnv(render_mode="human")

    # 2. Lắp "Bộ não" (File .zip) vào xe
    print("Loading pre-trained model")
    model = PPO.load("models/ppo_cartpole_swingup.zip",env=env)

    # 3. Đưa xe về vạch xuất phát
    obs, info = env.reset()
    
    print("Start")
    try:
        # Chạy liên tục trong 2000 khung hình để xem nó giữ thăng bằng
        for i in range(2000):
            # CHÌA KHÓA: deterministic=True ép AI dùng kỹ năng xịn nhất, không múa may nữa
            action, _states = model.predict(obs, deterministic=False)
            
            # Thực thi lực đẩy
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Cập nhật màn hình 3D (Gọi hàm render_frame của bạn)
            if hasattr(env, '_render_frame'):
                env._render_frame()
                
            # Làm chậm video lại một chút để mắt người kịp nhìn
            time.sleep(0.01) 

            # Nếu xe chạy văng khỏi ray (game over), reset lại từ đầu
            if terminated or truncated:
                print("Timeout! Restarting...")
                time.sleep(1) # Nghỉ 1 giây rồi diễn lại
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        env.close()

if __name__ == "__main__":
    main()