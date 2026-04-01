from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import time
from train_agent import CartPoleSwingUpEnv

def main():
    env = CartPoleSwingUpEnv(render_mode="human")
    # 2. Bọc môi trường để tương thích với SB3 vectorization
    vec_env = DummyVecEnv([lambda: env])
    # 3. Kích hoạt Frame Stacking (Lưu lịch sử 8 frames)
    vec_env = VecFrameStack(vec_env, n_stack=8)

    print("Loading pre-trained model...")
    model = PPO.load("models/ppo_force_real.zip", env=vec_env)

    obs = vec_env.reset()
    print("Start testing... Press Ctrl+C in terminal to stop.")

    try:
        # Infinite loop: Runs until user presses Ctrl+C
        while True:
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, dones, info = vec_env.step(action)
            
            time.sleep(0.01) 

            # Auto-reset on crash or timeout, but keeps window open
            if dones[0]:
                print("Episode finished. Restarting...")
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()