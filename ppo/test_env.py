import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

# Import môi trường từ file test của bạn
from train_force import CartPoleSwingUpEnv

NUM_EPISODES = 50

def save_eval_results(model_name, episode_num, history_theta, history_action, history_x, swing_up, overshoot, jitter, outcome, timestamp):
    result_dir = f"test_results/env/{model_name}"
    os.makedirs(result_dir, exist_ok=True)
    
    base_filename = f"{result_dir}/Ep{episode_num:02d}_{outcome}_{timestamp}"
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    fig.suptitle(f"Model: {model_name} - {outcome}\n"
                 f"Swing-up: {swing_up} steps | Overshoot: {overshoot if isinstance(overshoot, str) else f'{overshoot:.2f}'}° | Jitter: {jitter:.4f}", 
                 fontsize=12, fontweight='bold')
    
    ax1.plot(history_theta, color='blue')
    ax1.axhline(180, color='red', linestyle='--')
    ax1.axhline(-180, color='red', linestyle='--')
    ax1.set_ylabel('Theta (deg)')
    
    ax2.plot(history_action, color='orange')
    ax2.set_ylabel('Action (V)')
    
    ax3.plot(history_x, color='green')
    ax3.set_ylabel('Cart X (m)')
    ax3.set_xlabel('Steps')
    
    plt.tight_layout()
    fig.savefig(f"{base_filename}_plot.png", dpi=300)
    plt.close(fig)

def test_shaping_model(model_name):
    print(f"\nEvaluating model: {model_name} ---")
    
    model_path = f"models/env/{model_name}.zip"
    stats_path = f"vec/env/vec_normalize_{model_name}.pkl"
    
    # Tạo timestamp chung cho cả loạt test này (dùng cho cả ảnh và file txt)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"test_results/env/{model_name}"
    os.makedirs(result_dir, exist_ok=True)

    # PHA 1: Chạy ngầm (Không render) để tốc độ test nhanh nhất
    raw_env = CartPoleSwingUpEnv(render_mode="none")
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    vec_env = VecFrameStack(vec_env, n_stack=8) 

    try:
        model = PPO.load(model_path, env=vec_env, device="cpu")
    except Exception as e:
        print(f"Lỗi khi load model {model_name}: {e}")
        return
    
    trunc_count, term_count = 0, 0
    total_jitter = 0.0
    failed_seeds = [] 

    for ep_seed in range(1, NUM_EPISODES + 1):
        vec_env.seed(ep_seed) 
        obs = vec_env.reset()
        history_theta, history_action, history_x = [], [], []
        done = False
        is_truncated = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            unnorm_obs = vec_env.get_original_obs()[0]
            current_raw_obs = unnorm_obs[-5:] 
            
            history_theta.append(np.degrees(np.arctan2(current_raw_obs[3], current_raw_obs[2])))
            history_action.append(action[0][0])
            history_x.append(current_raw_obs[0])

            obs, _, dones, infos = vec_env.step(action)
            
            done = bool(dones[0])
            if done:
                is_truncated = bool(infos[0].get("TimeLimit.truncated", False))

        history_theta = np.array(history_theta)
        history_action = np.array(history_action)
        
        jitter = np.mean(np.abs(np.diff(history_action)))
        total_jitter += jitter
        
        # FIX lật dấu atan2: khoảng cách tới 180 độ
        theta_error = 180.0 - np.abs(history_theta)
        stable_idx = np.where(theta_error < 10)[0]
        swing_up = stable_idx[0] if len(stable_idx) > 0 else "N/A"
        overshoot = np.max(theta_error[swing_up:]) if isinstance(swing_up, (int, np.integer)) else "N/A"

        if is_truncated:
            trunc_count += 1
            outcome = "Truncated"
        else:
            term_count += 1
            outcome = "Terminated"
            failed_seeds.append(ep_seed) 

        save_eval_results(model_name, ep_seed, history_theta, history_action, history_x, swing_up, overshoot, jitter, outcome, timestamp)

    avg_jitter = total_jitter / NUM_EPISODES
    
    # GENERATE TEXT REPORT
    report_content = (
        f"========================================\n"
        f"Result: {model_name}\n"
        f"========================================\n"
        f"Total episodes: {NUM_EPISODES}\n"
        f"Success (Truncated): {trunc_count} episodes ({trunc_count/NUM_EPISODES*100:.1f}%)\n"
        f"Failure (Terminated): {term_count} episodes ({term_count/NUM_EPISODES*100:.1f}%)\n"
        f"Average Jitter (Lower is Better): {avg_jitter:.4f}\n"
        f"========================================\n"
    )
    
    # In ra terminal
    print("\n" + report_content)
    
    # Lưu ra file .txt
    report_path = os.path.join(result_dir, f"report_{model_name}_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[!] Saved report at: {report_path}")

    # PHA 2: Chuyên mục Replay
    if failed_seeds:
        print(f"\n[!] Found {len(failed_seeds)} failed episodes. Preparing replay...")
        print(">>> PLEASE PREPARE TO VIEW THE SIMULATION <<<")
        time.sleep(3)
        
        for bad_seed in failed_seeds:
            print(f"Replaying episode: {bad_seed}")
            
            # create isolated env for safe rendering
            raw_env_vis = CartPoleSwingUpEnv(render_mode="human")
            vec_env_vis = DummyVecEnv([lambda: raw_env_vis])
            vec_env_vis = VecNormalize.load(stats_path, vec_env_vis)
            vec_env_vis.training = False
            vec_env_vis.norm_reward = False
            vec_env_vis = VecFrameStack(vec_env_vis, n_stack=8)
            
            vec_env_vis.seed(bad_seed)
            obs = vec_env_vis.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, _ = vec_env_vis.step(action)
                done = bool(dones[0])
                time.sleep(0.01)
                
            # prevent memory leak / segfault
            raw_env_vis.close()
            
        print("Finished replaying all failed episodes!")

if __name__ == "__main__":
    test_shaping_model("tune_144553_seed42")