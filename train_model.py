import os
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from asv_env import ASVNavEnv
import torch  # Ensure PyTorch is imported


class SaveGifCallback(BaseCallback):
    """
    Callback for saving GIFs of the environment.
    """

    def __init__(self, env, save_freq=50, gif_dir="./gifs", verbose=0):
        super(SaveGifCallback, self).__init__(verbose)
        self.env = env
        self.save_freq = save_freq
        self.gif_dir = gif_dir
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("done", False):
            self.episode_count += 1
            if self.episode_count % self.save_freq == 0:
                print(f"Saving GIF for episode {self.episode_count}")
                frames = []
                obs_img, _ = self.env.reset()

                done = False
                trun = False
                # Run one episode to capture frames
                while not done and not trun:
                    action, _ = self.model.predict(obs_img, deterministic=True)
                    frame = self.env.render(return_img=True)
                    frames.append(frame)
                    obs_img, _, done, trun, _ = self.env.step(action)

                # Save frames as a GIF
                gif_path = f"{self.gif_dir}/episode_{self.episode_count}.gif"
                imageio.mimsave(gif_path, frames, fps=30)
                print(f"Saved GIF for episode {self.episode_count} at {gif_path}")

        return True


class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        Called at each step to log rewards and other metrics.
        """
        # Access the unwrapped environment
        env = (
            self.locals["env"].envs[0].unwrapped
        )  # Access the underlying unwrapped environment

        # Log individual reward components
        self.logger.record("reward/distance_reward", env.distance_reward)
        self.logger.record("reward/heading_reward", env.heading_reward)
        self.logger.record("reward/target_bonus", env.target_bonus)
        self.logger.record("reward/truncation_penalty", env.truncation_penalty)
        self.logger.record("reward/thruster_used", env.thruster_used)
        self.logger.record("reward/thruster_penalty", env.thruster_penalty)

        return True


def main():
    """
    Main function to train the model.
    """
    training_steps = 250000
    env = ASVNavEnv(training_steps=training_steps, headless=True)

    print("Checking the environment ....")
    check_env(env, warn=True)
    print("Environment is ALL GOOD!")

    log_dir = "./logs/"

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create the PPO model with the appropriate device
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, device=device)

    # to add checkpoint and eval callbacks

    gif_dir = "./gifs"
    os.makedirs(gif_dir, exist_ok=True)
    save_gif_callback = SaveGifCallback(env, save_freq=50, gif_dir=gif_dir)
    reward_logging_callback = RewardLoggingCallback()

    model.learn(
        total_timesteps=training_steps,
        callback=[save_gif_callback, reward_logging_callback],
    )
    model.save("ppo_asv_navigation-ppo2")


if __name__ == "__main__":
    main()
