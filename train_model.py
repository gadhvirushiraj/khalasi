import os
import numpy as np
import imageio
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from asv_env import ASVNavEnv
import torch  # Ensure PyTorch is imported


class CheckpointCallback(BaseCallback):
    """
    Callback for saving model checkpoints at regular intervals.
    """

    def __init__(self, save_freq: int, save_dir: str = "./checkpoints", verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure directory exists

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:  # Check if step is multiple of save_freq
            checkpoint_path = os.path.join(
                self.save_dir, f"model_step_{self.n_calls}.zip"
            )
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"Saved checkpoint: {checkpoint_path}")
        return True


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
                    action, _ = self.model.predict(obs_img)
                    frame = self.env.render(return_img=True)
                    frames.append(frame)
                    obs_img, _, done, trun, _ = self.env.step(action)

                # Save frames as a GIF
                gif_path = f"{self.gif_dir}/episode_{self.episode_count}.gif"
                imageio.mimsave(gif_path, frames, fps=30)
                print(f"Saved GIF for episode {self.episode_count} at {gif_path}")

        return True


class RewardLoggingCallback(BaseCallback):
    """
    Callback of Logging the Rewards.
    """

    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        env = self.locals["env"].envs[0].unwrapped

        # Log individual reward components
        self.logger.record("reward/distance_reward", env.distance_reward)
        self.logger.record("reward/heading_reward", env.heading_reward)
        self.logger.record("reward/target_bonus", env.target_bonus)
        self.logger.record("reward/truncation_penalty", env.truncation_penalty)
        self.logger.record("reward/thruster_used", env.thruster_used)
        self.logger.record("reward/thruster_penalty", env.thruster_penalty)

        # Log goal-reached event
        if getattr(env, "goal_reached", False):
            self.logger.record("event/goal_reached", 1)
        else:
            self.logger.record("event/goal_reached", 0)

        return True


def main():
    """
    Main function to train the model with CNN-LSTM policy.
    """
    training_steps = 250000
    env = ASVNavEnv(training_steps=training_steps, headless=True)

    print(
        " ---------------------- Checking the environment ---------------------- ",
        end="\n",
    )
    check_env(env, warn=True)
    print(
        " ---------------------- Environment is ALL GOOD! ---------------------- ",
        end="\n",
    )

    log_dir = "./logs/"

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Use the CnnLstmPolicy with RecurrentPPO
    # model = RecurrentPPO(
    #     "CnnLstmPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log=log_dir,
    #     device=device,
    #     learning_rate=1e-4,
    #     n_steps=1024,  # Number of steps per rollout (adjust for sequence length)
    #     batch_size=64,  # Adjust based on memory
    #     ent_coef=0.01,  # Entropy coefficient for exploration
    # )

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, device=device)

    # Create directories for outputs
    gif_dir = "./gifs"
    checkpoint_dir = "./checkpoints"
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize callbacks
    save_gif_callback = SaveGifCallback(env, save_freq=50, gif_dir=gif_dir)
    reward_logging_callback = RewardLoggingCallback()
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_dir=checkpoint_dir)

    # Start training
    model.learn(
        total_timesteps=training_steps,
        callback=[save_gif_callback, reward_logging_callback, checkpoint_callback],
    )
    model.save("ppo_asv_navigation")


if __name__ == "__main__":
    main()
