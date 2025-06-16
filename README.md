# 🐍 Snake Game with Q-Learning AI

This project implements a classic **Snake Game** enhanced with **Q-Learning**, a reinforcement learning algorithm. The agent learns how to play Snake by exploring the environment, avoiding collisions, and maximizing the reward by eating food over multiple training episodes.

---

## 🎮 Game Overview

- Grid-based snake game built using **Pygame**
- Snake receives rewards based on actions:
  - +10 for eating food 🍎
  - -0.1 for each step (to discourage idling)
  - -10 for collisions 💥
  - -5 if too many steps pass without eating (timeout)
- AI agent learns using **Q-learning** with epsilon-greedy strategy

---

## 🧠 Q-Learning Setup

- **State Representation**:  
  `(food_dx_bin, food_dy_bin, current_direction, danger_forward, danger_left, danger_right)`

- **Actions**:
  - 0 = Turn Left
  - 1 = Go Straight
  - 2 = Turn Right

- **Hyperparameters**:
  - Learning rate: `0.1`
  - Discount factor: `0.95`
  - Epsilon decay: `0.995`
  - Min epsilon: `0.01`
  - Episodes: `5000`

---
