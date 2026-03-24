from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from biped_env_obs3 import BipedWalkingEnv
from callbacks import RewardPlotCallback
import numpy as np
import os

# ✅ Setup monitored environment
env = Monitor(BipedWalkingEnv(render=False), filename="./logs/sac_monitor.csv")

# ✅ Define SAC model (no action_noise needed)
model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="logs/sac_goal/",
    learning_rate=0.001
)

# ✅ Callbacks
reward_callback = RewardPlotCallback()
eval_env = Monitor(BipedWalkingEnv(render=False))
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models_obs3/sac_best",
    log_path="./logs/sac_eval",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# ✅ Train with safety
try:
    model.learn(
        total_timesteps=2500000,
        callback=[eval_callback, reward_callback]
    )
except Exception as e:
    print("⚠️ Training interrupted:", e)
    model.save("models_obs2/sac_biped_crashsave")

# ✅ Final save
os.makedirs("models_obs3 ", exist_ok=True)
model.save("models_obs3/sac_biped_goal")
reward_callback.plot_rewards()

env.close()
