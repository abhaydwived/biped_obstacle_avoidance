from biped_env_obs3 import BipedWalkingEnv
import time


# Simple environment test
def test_env(num_episodes=50, max_steps=1000):
    env = BipedWalkingEnv(render=True)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        
        step = 0

        while not (terminated or truncated or step >= max_steps):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            time.sleep(1 / 60)  # optional slowdown for visualization

        print(f"Episode {ep+1} finished after {step} steps. Terminated={terminated}, Truncated={truncated}")

    env.close()

if __name__ == '__main__':
    test_env()
