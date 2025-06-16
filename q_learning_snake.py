import pygame
import random
import numpy as np
from collections import deque

# Game Constants
GRID_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
WIDTH, HEIGHT = GRID_WIDTH * GRID_SIZE, GRID_HEIGHT * GRID_SIZE

# Q-Learning Parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 5000

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()


class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.direction = 1  # 0=up, 1=right, 2=down, 3=left
        self.head = [GRID_WIDTH // 2, GRID_HEIGHT // 2]
        self.snake = deque([self.head.copy()])
        self.food = self.place_food()
        self.score = 0
        self.steps_since_food = 0
        self.done = False
        return self.get_state()

    def place_food(self):
        while True:
            pos = [random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)]
            if pos not in self.snake:
                return pos

    def get_state(self):
        dx = self.food[0] - self.head[0]
        dy = self.food[1] - self.head[1]

        # Danger detection
        dangers = [0, 0, 0]  # Forward, left, right
        directions = [self.direction, (self.direction - 1) % 4, (self.direction + 1) % 4]

        for i, d in enumerate(directions):
            next_pos = self.head.copy()
            if d == 0:
                next_pos[1] -= 1
            elif d == 1:
                next_pos[0] += 1
            elif d == 2:
                next_pos[1] += 1
            elif d == 3:
                next_pos[0] -= 1

            if (next_pos[0] < 0 or next_pos[0] >= GRID_WIDTH or
                    next_pos[1] < 0 or next_pos[1] >= GRID_HEIGHT or
                    next_pos in self.snake):
                dangers[i] = 1

        return np.array([dx, dy, self.direction, *dangers], dtype=np.int32)

    def step(self, action):
        self.direction = (self.direction + action - 1) % 4
        reward = 0
        self.steps_since_food += 1

        # Move snake
        new_head = self.head.copy()
        if self.direction == 0:
            new_head[1] -= 1
        elif self.direction == 1:
            new_head[0] += 1
        elif self.direction == 2:
            new_head[1] += 1
        elif self.direction == 3:
            new_head[0] -= 1

        # Collision check
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
                new_head[1] < 0 or new_head[1] >= GRID_HEIGHT or
                tuple(new_head) in map(tuple, self.snake)):
            self.done = True
            reward = -10
            return self.get_state(), reward, self.done

        self.snake.append(new_head.copy())
        self.head = new_head

        # Food check
        if tuple(new_head) == tuple(self.food):
            self.score += 1
            self.food = self.place_food()
            reward = 10
            self.steps_since_food = 0
        else:
            self.snake.popleft()
            reward = -0.1

        # Timeout
        if self.steps_since_food > 100:
            self.done = True
            reward = -5

        return self.get_state(), reward, self.done

    def render(self):
        screen.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(screen, (0, 255, 0),
                             (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1))
        pygame.draw.rect(screen, (255, 0, 0),
                         (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1))
        pygame.display.flip()
        clock.tick(10)  # Control rendering speed here


# Discretization and Q-Table setup
def discretize(state):
    dx, dy, direction, *dangers = state
    dx_bin = np.digitize(dx, [-5, 0, 5])
    dy_bin = np.digitize(dy, [-5, 0, 5])
    return (dx_bin, dy_bin, direction, *dangers)


q_table = {}


def get_q(state):
    state_key = discretize(state)
    if state_key not in q_table:
        q_table[state_key] = np.zeros(3)
    return q_table[state_key]


# Training loop with visualization control
high_score = 0
render_every = 100  # Render every N episodes
for episode in range(EPISODES):
    game = SnakeGame()
    state = game.reset()
    total_reward = 0
    done = False
    render = (episode % render_every == 0)  # Render periodically

    while not done:
        if render:
            game.render()
            # Handle window close events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # Epsilon-greedy action
        if random.random() < EPSILON:
            action = random.randint(0, 2)
        else:
            action = np.argmax(get_q(state))

        next_state, reward, done = game.step(action)
        total_reward += reward

        # Q-learning update
        old_q = get_q(state)[action]
        max_future_q = np.max(get_q(next_state))
        new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - old_q)
        get_q(state)[action] = new_q

        state = next_state

    # Decay epsilon
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    # Update high score
    if game.score > high_score:
        high_score = game.score

    print(f"Episode {episode + 1} | Score: {game.score} | High: {high_score} | ε: {EPSILON:.3f}")

pygame.quit()