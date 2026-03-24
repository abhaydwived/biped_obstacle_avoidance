import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class RewardPlotCallback(BaseCallback):
    def __init__(self, log_interval=100, verbose=1):
        super().__init__(verbose)
        self.rewards = []
        self.log_interval = log_interval

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_reward = info['episode']['r']
                self.rewards.append(ep_reward)
        return True


    def plot_rewards(self, window_size=50):
        import numpy as np
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards, label="Episode Reward")

        if len(self.rewards) >= window_size:
            moving_avg = np.convolve(self.rewards, np.ones(window_size) / window_size, mode='valid')
            plt.plot(range(window_size - 1, len(self.rewards)), moving_avg, color='red', linestyle='--',
                    label=f"Moving Avg (window={window_size})")

        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Training Reward Progress")
        plt.legend()
        plt.grid(True)
        plt.show()

