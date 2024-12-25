"""Basic testing of the ASVNavEnv environment."""

import pygame
from asv_env import ASVNavEnv
from simple_flow_feild import SimpleFlowField


def main():
    """
    Test the ASVNavEnv environment by taking random actions and printing the them.
    if agent is "test" then the agent will have no thruster and will move based on flow.
    """

    agent = "custom"
    env = ASVNavEnv(agent=agent, headless=False) 
    #include flow_field_type=SimpleFlowField to use the simple flow field

    running = True
    nssteps = 0
    clock = pygame.time.Clock()
    obs_img, _ = env.reset()

    while running:

        nssteps += 1
        # Sample a random action from the action space
        if agent == "test":
            action = [0.5, 0.5]
        else:
            action = env.action_space.sample()
        print("Action:", action)

        # Take a step in the environment
        obs_img, _, done, truncated, _ = env.step(action)
        running = not done and not truncated

        thruster1, thruster2 = action
        env.render(obs_image=obs_img, thruster1=thruster1, thruster2=thruster2)
        clock.tick(30)

    pygame.display.quit()
    print(f"Number of steps: {nssteps}")


if __name__ == "__main__":
    main()
