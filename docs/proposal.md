---
layout: default
title: Proposal
---


## Summary of the Project
The goal of our project is to train an AI agent to climb as many floors as possible in an obstacle tower by solving puzzles to find the exit to move to the next floor. The agent will have to recognize symbols and objects, such as keys to solve puzzles and locked doors. It will receive image frames by observing its environment to identify symbols and objects. The agent will output a movement action. It can choose to move in the four directions, turn left and right, or jump to navigate through the levels.

## AI/ML Algorithms
We will train the agent using PPO with a convolutional neural network for object detection.

## Evaluation Plan
The agent will be evaluated based on the number of floors completed, the time taken to exit a floor, and the number of puzzles solved. The number of floors completed will be the main metric used to judge the agent's success since the goal is to climb as high as possible through the tower. Its performance will be compared to an agent that performs actions randomly as the baseline. A random agent is estimated to only complete about one to three floors, and our agent should improve this by beating more floors with a faster time than the random agent.

The sanity cases for our approach would be to see if the agent can solve the first couple of floors with simple puzzles. One sanity case would be if it can find the exit of a floor with no puzzles. Another would be if it could find a key to unlock a door to the exit before going to harder puzzles. Our moonshot case is if the agent can make it past 10 floors, where it would have to learn to generalize and solve complex puzzles.


## Meet the Instructor
January 28th - 12:30 PM

## AI Tool Usage
