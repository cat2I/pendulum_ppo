import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

# import custom env
from train_force import CartPoleSwingUpEnv

# config debug params
MODEL_NAME = "tune_030930_seed42"
FAILED_SEEDS = [31]  # insert bad seeds here
SLOW_MOTION_DELAY = 0.05    # seconds per frame (increase to slow down more)

def run_slow_motion_replay():
    print(f"\n--- INITIALIZING DEBUGGER FOR: {MODEL_NAME} ---")
    
    model_path = f"models/env/{MODEL_NAME}.zip"
    stats_path = f"vec/env/vec_normalize_{MODEL_NAME}.pkl"

    # init visual env
    raw_env_vis = CartPoleSwingUpEnv(render_mode="human")
    vec_env_vis = DummyVecEnv([lambda: raw_env_vis])
    
    # load norm stats (crucial for correct policy execution)
    vec_env_vis = VecNormalize.load(stats_path, vec_env_vis)
    vec_env_vis.training = False
    vec_env_vis.norm_reward = False
    vec_env_vis = VecFrameStack(vec_env_vis, n_stack=8)
    
    # load model
    try:
        model = PPO.load(model_path, env=vec_env_vis, device="cpu")
    except Exception as e:
        print(f"Model load error: {e}")
        return

    print("\n[READY] MuJoCo Viewer initialized.")
    
    for seed in FAILED_SEEDS:
        print("\n" + "="*40)
        print(f"TARGET SEED: {seed}")
        print("="*40)
        input(">>> Bấm ENTER để bắt đầu phát ván này <<<")
        
        # Khởi tạo môi trường MỚI bên trong vòng lặp
        raw_env_vis = CartPoleSwingUpEnv(render_mode="human")
        vec_env_vis = DummyVecEnv([lambda: raw_env_vis])
        vec_env_vis = VecNormalize.load(stats_path, vec_env_vis)
        vec_env_vis.training = False
        vec_env_vis.norm_reward = False
        vec_env_vis = VecFrameStack(vec_env_vis, n_stack=8)
        
        # Load model lại cho môi trường mới
        model = PPO.load(model_path, env=vec_env_vis, device="cpu")
        
        vec_env_vis.seed(seed)
        obs = vec_env_vis.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = vec_env_vis.step(action)
            done = bool(dones[0])
            time.sleep(0.03) # Slow motion

        # cleanup
        raw_env_vis.close()
    print("\n--- DEBUG SESSION COMPLETED ---")

if __name__ == "__main__":
    run_slow_motion_replay()