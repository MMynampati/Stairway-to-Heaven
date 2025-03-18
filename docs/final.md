---
layout: default
title: Final Report
---

## Video


## Project Summary
The goal of our project is to train an agent that can navigate rooms to progress through as many floors as possible in a procedurally generated obstacle tower. Each floor will become increasingly difficult as the agent goes up the tower. Each tower is randomized with every floor having a different floor layout so the agent will have to learn to generalize to new obstacles and changes in the environment, such as locked doors, dead ends, and puzzles to progress. Due to the environment being random and partially observable, the agent can’t rely on brute force, pathfinding algorithms, or memorizing the floor layouts to find the exit. It doesn’t have any information about the floor layout and where the exit is located besides the images from what the player model sees as input.

The agent processes its environment by receiving 84x84 pixel image frames, which includes the time remaining and if it has a key embedded in the image, as input. It has to learn to identify the various elements related to progression in the environment, such as keys, time orbs, puzzles, and the different types of doors. The agent can choose to output a variety of actions, or combinations of actions, such as directional movements, turning the camera, and jumping. The success of the agent is measured based on the average number of floors it can reach and the average reward per episode.


## Approaches

### Observation Space:
The agent takes in an 84 x 84 RGB image for its observation and is processed with a convolutional neural network (CNN). The image includes how much time remains and the number of keys possessed.

Since the environment is partially observable, the agent doesn’t have any prior knowledge about the floor layout when it enters a floor. It doesn’t receive the positions of the exit, doors, or keys so it has to rely on its visual input to navigate each floor.

The agent has a multi-discrete action space consisting of four groups of actions. These action branches are flattened into a discrete action space with 54 actions. Each action the agent chooses represents a combination of the moves below.


### Action Space:
- Movement (No-Action, Forward, Back)
- Movement (No-Action, Right, Left)
- Camera Rotation (No-Action, Clockwise, Counter-Clockwise)
- Jump (No-Action, Jump)


Ex.: The action, [1, 0, 1, 1], causes the agent to move forward, turn the camera to the right, and jump simultaneously.

The obstacle tower is a sparse reward environment with the agent not receiving a reward most of the time for a given state.

### Rewards:
- +1 Agent completing a floor
- +0.1 Opening a door
- +0.1 Solving a puzzle
- +0.1 Picking up a key

For each episode, the agent starts off at floor zero and tries to reach an end room to move onto the next floor, with higher floors becoming more complex. The episode ends when the agent runs out of time, reaches floor 100, or touches a dangerous obstacle such as falling into a pit.

The first thing our team had to do was decide on what RL algorithm/algorithms we were going to focus on implementing for the obstacle tower agent. In order to narrow down which algorithms to consider we made a list of the different attributes of the environment.

- Large discrete action space
- Sparse rewards
- Stochastic environment

These factors combined meant that we would likely need a sample-efficient algorithm,  with the large state space and stochastic environment providing a lot of variability in terms of the data the model could train on. After some research we were able to narrow down to two algorithms that we felt could show good results.

- Proximal Policy Optimization (PPO)
- Advantage Actor Critic (A2C)

After making this list, we ran a test simulation with the simplest implementation of each algorithm in the default environment to get a sense if any seemed better initially. The results from each algorithm were all about the same, barely even getting a mean reward of 0.2.

### Approach 1:

We eventually settled on Stable Baselines3’s implementation of PPO with the default hyperparameters. PPO collects experiences and compares how much the new policy has changed from the old policy. It clips this probability ratio to keep it within a certain range. Then it takes the minimum of the clipped and unclipped values, making small policy updates and preventing instability. With the vast state and action spaces, we felt that PPO struck a good balance between exploration and stability. This and the fact that from our reading it seemed that PPO works well with ICM, which we would go on to experiment with later.

As our baseline, we trained the agent with a random tower and a different layout in each episode to see how well it would perform without any changes made to the parameters and the environment. Then, we trained another agent on a single/fixed version of the obstacle tower, where the layout doesn’t change.

The agent trained on a single tower trained slower but had a higher mean reward compared to the randomized tower agent. The random tower model struggled to receive any rewards and was stuck on floor 0, whereas the fixed tower model had the advantage of only needing to adapt to the same floor layout.

Even when placed in new towers, the fixed tower agent was able to perform better than the randomized tower agent. Even though the layout is different from the tower it was trained on, it still recognized the doors and exits it needed to go through, unlike the randomized tower agent that performed similarly to an untrained agent.

### Approach 2: Reducing Action Space to 12 Actions

When watching both agents go through the tower, both would often perform random actions instead of heading straight for the doors. This is most likely due to both agents having a hard time associating which action out of the 54 actions would lead to a reward.

To help the agent learn and explore more efficiently, we experimented with reducing the number of actions the agent can choose to lower the complexity. We reduced the actions to only moving forward, turning the camera, and jumping, bringing the action space to 12 combinations

### New Action space:
- Movement (No-Action, Forward)
- Movement (No-Action)
- Camera Rotation (No-Action, Clockwise, Counter-Clockwise)
- Jump (No-Action, Jump)

By reducing the action space to 12 actions, the agent could generalize to new environments and perform better than the baseline randomized tower agent and the fixed tower agent. Since it had fewer options to choose from, there was a higher chance of choosing the correct action. The new agent had an easier time learning which actions brought it closer to the exit instead of constantly exploring random actions.

This approach also led to a faster training time than the fixed tower agent with similar performance, being able to consistently make it to floor one and sometimes floor 2.

However, reducing the number of actions caused the agent to often get stuck on walls and doorways since it could no longer move backward from obstacles.

### Approach 3: Reducing Action Space to 8

Reducing the action space to 12 actions had a large impact on the agent’s ability to learn so we reduced it further to 8 actions to see if there would be further improvements. 

### New Action space:
- No actions
- Forward
- Backward
- Turn camera left
- Turn camera right
- Forward + camera left
- Forward + camera right
- Jump + Forward

In our previous approach, the action space included multiple combinations involving the jump action. Although there wasn’t as much random jumping compared to an untrained agent, the agent still jumped when it was unnecessary. We limited the jump action to only one combination where the agent jumps and moves forward at the same time. We also added back the backward movement so that it wouldn’t get stuck as often, but it still sometimes struggled when going through doorways.


Although reducing to 8 actions didn’t provide significant improvement to the agent, it was able to achieve a similar amount of reward faster than the action space of 12 so we kept this approach.

### Approach 4: Hyperparameter Tuning

Next, we focused on optimizing the hyperparameters of PPO to improve performance.
Hyperparameters:
- n_steps = 512
- n_epochs = 8
- ent_coef = 0.001

We decreased the n_steps from 2048 to 512 and n_epochs from 10 to 8 for faster updates to the policy since the environment is always changing and for faster training. We also changed the entropy coefficient from 0 to 0.001 to encourage exploring new actions so that it doesn’t converge to using the same actions.

These changes didn’t affect the training speed as much, but it collected more rewards than the agent with the default hyperparameters. This is most likely because of the introduction of entropy causing the agent to explore different actions more so it found better actions than the previous agents.

### Approach 5: Frame skipping and Framestacking

We added a frame skipping parameter of 2 and a frame stacking of 4 on top of the previous approaches. For frame skipping, the agent only takes action every two frames and will return every 2nd frame. This means the agent will repeat an action for two frames and decreases the total training time because the agent doesn’t have to make a decision at every step. This would also help prevent the agent from getting stuck on walls since it could repeat turning away or moving back instead of choosing an action that would get it stuck again.

Then, we combined/stacked the last 4 frames that weren’t skipped into a single observation. This gives the agent more information about the environment, such as movement and previous changes.

- Old Observation Space: (3, 84, 84)
- New Observation Space: (12, 84, 84)

<div style="text-align:center"><img src="PPO_fixed_environment.png" width="450" height="290"/></div>


## Evaluation


## References
**PPO Algorithm**
- [Stable Baselines3 PPO Algorithm](https://stable-baselines3.readthedocs.io/en/v1.7.0/modules/ppo.html)
- [Original Proximal Policy Optimization Algorithms Paper](https://arxiv.org/abs/1707.06347)

**Obstacle Tower**
- [Obstacle Tower Environment](https://github.com/Unity-Technologies/obstacle-tower-env)
- [Obstacle Tower Research Paper](https://arxiv.org/abs/1902.01378)
- [Obstacle Tower Evaluation Code](https://github.com/Unity-Technologies/obstacle-tower-env/blob/master/examples/evaluation.py)

**ML-Agents Curiosity**
- [Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/pdf/1808.04355)
- [Unity Curiosity](https://unity.com/blog/engine-platform/solving-sparse-reward-tasks-with-curiosity)
- [Ml-Agents Configuration Parameters](https://unity-technologies.github.io/ml-agents/Training-Configuration-File/)
- [Training ML-Agents](https://unity-technologies.github.io/ml-agents/Training-ML-Agents/)

**Frame Skipping**
- [Frame Skipping Explaination](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)

## AI Tool Usage

Used ChatGPT to help troubleshoot installing and setting up the environment because of compatibility errors
