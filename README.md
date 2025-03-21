# Text Flappy Bird – Monte Carlo and Sarsa(λ) Agents

This project was completed as part of the 3MD3220 Reinforcement Learning course at CentraleSupélec.

## Overview

The goal of this assignment was to implement and compare two reinforcement learning algorithms—Monte Carlo and Sarsa(λ)—on the Text Flappy Bird environment. Two versions of the environment were explored:

- **State-based** (numerical observations: x- and y-distance to the pipe gap)
- **Screen-based** (pixel grid of the environment)

Agents were evaluated based on learning efficiency, performance stability, and generalization to unseen environment configurations.

## Contents

- `main_state_based.py`: Training and evaluation code for the state-based environment.
- `main_screen_based.py`: Training and evaluation code for the screen-based environment.
- `report.pdf`: The final report detailing methodology, results, and analysis.
- `figures/`: Folder containing all plots and visualizations.
- `README.md`: Project summary and structure.

## Results Summary

- **Sarsa(λ)** generally outperformed **Monte Carlo** in both environments in terms of learning speed and final performance.
- However, Sarsa(λ) showed higher variance and poorer generalization when tested in modified environments.

## Repository

Environment source:  
[Text Flappy Bird Gym Repository](https://gitlab-research.centralesupelec.fr/stergios.christodoulidis/text-flappy-bird-gym)

## Author

Ludovic Nic
