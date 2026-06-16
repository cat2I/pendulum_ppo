import os
import datetime
import optuna
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from train_force import CartPoleSwingUpEnv

def objective(trial):
    # search space
    n_steps = trial.suggest_categorical("n_steps", [2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    
    # ppo constraint: n_steps % batch_size == 0
    if n_steps % batch_size != 0:
        raise optuna.exceptions.TrialPruned()

    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
    n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10])
    gamma = trial.suggest_float("gamma", 0.95, 0.999, log=True)

    env = DummyVecEnv([lambda: CartPoleSwingUpEnv(render_mode="none")])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    env = VecFrameStack(env, n_stack=8)

    model = PPO("MlpPolicy", env,
                n_steps=n_steps, batch_size=batch_size,
                learning_rate=learning_rate, ent_coef=ent_coef,
                n_epochs=n_epochs, gamma=gamma,
                vf_coef=0.5, target_kl=0.015,
                policy_kwargs=dict(activation_fn=nn.Tanh, net_arch=dict(pi=[128, 128], vf=[128, 128])),
                verbose=0, device="cuda")

    # short training for evaluation
    try:
        model.learn(total_timesteps=150000)
    except:
        return -1000

    # robust evaluation across 20 seeds
    all_rewards = []
    for s in range(200, 220):
        env.seed(s)
        mean_r, _ = evaluate_policy(model, env, n_eval_episodes=1, deterministic=True)
        all_rewards.append(mean_r)

    env.close()
    
    # objective: maximize mean reward penalizing high variance
    score = np.mean(all_rewards) - np.std(all_rewards)
    return score

if __name__ == "__main__":
    print("\n>>> STARTING HYPERPARAMETER TUNING (50 TRIALS)...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50) 
    
    # setup output dir
    out_dir = "test_results/optuna"
    os.makedirs(out_dir, exist_ok=True)

    # gen timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(out_dir, f"optuna_best_params_{timestamp}.txt")

    # format report layout
    report_content = "========================================\n"
    report_content += f"OPTUNA TUNING RESULTS | {timestamp}\n"
    report_content += "========================================\n"
    report_content += f"Total Trials: 50\n"
    report_content += f"Best Score (Mean - STD): {study.best_value:.4f}\n\n"
    
    report_content += "=== BEST HYPERPARAMETERS ===\n"
    for key, value in study.best_params.items():
        report_content += f"  {key}: {value}\n"

    # get top 3 trials for fallback options
    report_content += "\n=== TOP 3 TRIALS ===\n"
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    
    for i, trial in enumerate(completed_trials[:3]):
        report_content += f"Rank {i+1} (Trial {trial.number}) | Score: {trial.value:.4f}\n"
        for k, v in trial.params.items():
            report_content += f"    {k}: {v}\n"
        report_content += "\n"

    # write to txt
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "="*50)
    print("TUNING COMPLETED! BEST HYPERPARAMETERS:")
    print("="*50)
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nRobustness Score: {study.best_value:.2f}")
    print(f"\n[!] Report saved to: {report_path}")