import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

# QUAN TRỌNG: Import môi trường từ file SB3 mà bạn đã setup cờ use_bonus_shaping
from train_sb3 import CartPoleSwingUpEnv

NUM_EPISODES = 20

def save_eval_results(model_name, episode_num, history_theta, history_action, history_x, swing_up, overshoot, jitter, outcome):
    # Tạo folder riêng cho từng model trong test_results
    result_dir = f"test_results/env/{model_name}"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{result_dir}/Ep{episode_num}_{outcome}_{timestamp}"
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Tiêu đề ghi rõ Model và thông số
    fig.suptitle(f"Model: {model_name} - {outcome}\n"
                 f"Swing-up: {swing_up} steps | Overshoot: {overshoot if isinstance(overshoot, str) else f'{overshoot:.2f}'}° | Jitter: {jitter:.4f}", 
                 fontsize=12, fontweight='bold')
    
    # Đồ thị Góc
    ax1.plot(history_theta, color='blue')
    ax1.axhline(180, color='red', linestyle='--')
    ax1.axhline(-180, color='red', linestyle='--')
    ax1.set_ylabel('Theta (deg)')
    
    # Đồ thị Lực tác động (Motor Action)
    ax2.plot(history_action, color='orange')
    ax2.set_ylabel('Action (V)')
    
    # Đồ thị Vị trí Xe
    ax3.plot(history_x, color='green')
    ax3.set_ylabel('Cart X (m)')
    ax3.set_xlabel('Steps')
    
    plt.tight_layout()
    fig.savefig(f"{base_filename}_plot.png", dpi=300)
    plt.close(fig)

def test_model(model_name, use_bonus_flag):
    print(f"\n--- Đang đánh giá model: {model_name} ---")
    
    # Đường dẫn trỏ đúng vào thư mục env/ theo ảnh của bạn
    model_path = f"models/env/ppo_{model_name}.zip"
    stats_path = f"vec/env/vec_normalize_{model_name}.pkl"

    # Setup Env
    raw_env = CartPoleSwingUpEnv(use_bonus_shaping=use_bonus_flag, render_mode="none")
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    vec_env = VecFrameStack(vec_env, n_stack=8) # Đổi thành 8 để khớp với model

    model = PPO.load(model_path, env=vec_env, device="cpu")
    
    trunc_count, term_count = 0, 0
    total_jitter = 0.0

    for ep in range(1, NUM_EPISODES + 1):
        obs = vec_env.reset()
        history_theta, history_action, history_x = [], [], []
        done = False
        is_truncated = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Với stack 8, unnorm_obs là mảng (40,), lấy [-5:] để bóc ra frame mới nhất
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
        
        theta_error = np.abs(np.abs(history_theta) - 180.0)
        stable_idx = np.where(theta_error < 10)[0]
        swing_up = stable_idx[0] if len(stable_idx) > 0 else "N/A"
        overshoot = np.max(theta_error[swing_up:]) if isinstance(swing_up, (int, np.integer)) else "N/A"

        if is_truncated:
            trunc_count += 1
            outcome = "Truncated"
        else:
            term_count += 1
            outcome = "Terminated"

        save_eval_results(model_name, ep, history_theta, history_action, history_x, swing_up, overshoot, jitter, outcome)

    avg_jitter = total_jitter / NUM_EPISODES
    print(f">> Tổng kết {model_name}: Truncated {trunc_count} | Terminated {term_count}")
    print(f">> AVERAGE JITTER (Càng thấp càng tốt): {avg_jitter:.4f}")

if __name__ == "__main__":
    # Cấu hình danh sách model cần đem ra mổ xẻ
    models_to_test = [
        {"name": "stack8_with_bonus", "use_bonus": True},
        {"name": "stack8_no_bonus", "use_bonus": False}
    ]
    
    for m in models_to_test:
        test_model(m["name"], m["use_bonus"])