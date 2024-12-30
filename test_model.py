import os
from stable_baselines3 import PPO
from asv_env import ASVNavEnv


def test_saved_model(model_path, env, num_episodes=5, render=True):
    """
    Test a saved PPO model in the environment.

    :param model_path: Path to the saved PPO model (.zip file).
    :param env: The environment to test in.
    :param num_episodes: Number of episodes to run.
    :param render: Whether to render the environment during testing.
    """
    # Check if the model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0

        print(f"Starting Episode {episode + 1}")

        while not done and not truncated:
            # Get action from the model
            action, _ = model.predict(obs)

            # Take a step in the environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if render:
                env.render()

        print(
            f"Episode {episode + 1} completed: Total Reward: {total_reward}, Steps: {step_count}"
        )

    print("Testing complete!")


if __name__ == "__main__":
    # Define the path to the saved model
    saved_model_path = "./checkpoints-work-forward-only/model_step_150000.zip"

    # Create the test environment (make sure it matches training env)
    test_env = ASVNavEnv(training_steps=0, headless=False)

    # Test the saved model
    test_saved_model(saved_model_path, test_env, num_episodes=5, render=True)
