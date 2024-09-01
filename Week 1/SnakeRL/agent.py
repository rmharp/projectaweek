import torch
import random
import numpy as np
from collections import deque, namedtuple
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# we can store up to 100,000 items in the memory
MAX_MEMORY = 100_000
BATCH_SIZE = 1024
LR = 0.0005
Experience = namedtuple('Experience', ('priority', 'state', 'action', 'reward', 'next_state', 'done'))

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.99 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() to remove the oldest memory
        self.model = Linear_QNet(11, [256, 128], 3) # 11 inputs, 512 hidden layers, 3 outputs with dropout
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # trainer object
    
    def get_state(self, game):
        head = game.snake[0]
        # danger
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # directions
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            
            # danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            
            # danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            
            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down 
        ]
        
        return np.array(state, dtype=int)
    
    def get_priority(self, reward, state, next_state):
        # Simple priority based on the absolute reward, with a small constant added to avoid zero priority
        return abs(reward) + 0.01
    
    def remember(self, state, action, reward, next_state, done):
        priority = self.get_priority(reward, state, next_state)
        experience = Experience(priority, state, action, reward, next_state, done)
        self.memory.append(experience) # popleft if MAX_MEMORY is reached
    
    def sample_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # Sample based on priority
            priorities = np.array([exp.priority for exp in self.memory])
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(len(self.memory), BATCH_SIZE, p=probabilities)
            mini_sample = [self.memory[i] for i in indices]
        else:
            mini_sample = self.memory
        
        return zip(*mini_sample)
    
    def train_long_memory(self):
        # Sample the memory and unpack the experiences
        sampled_experiences = self.sample_memory()
            
        # Unpack ignoring the priority
        _, states, actions, rewards, next_states, dones = sampled_experiences
        
        # Train the model
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    last_distance_to_food = None  # Initialize the last distance to food
    
    while True:
        # get the old state
        state_old = agent.get_state(game)
        
        # get the move
        final_move = agent.get_action(state_old)
        
        # perform the move and get new state
        reward, done, score = game.play_step(final_move)
        
        # Get the new position of the snake's head
        snake_head_new = game.snake[0]
        
        # Encourage moving toward food for first 50 games
        if agent.n_games < 120:
            current_distance_to_food = abs(game.food.x - snake_head_new.x) + abs(game.food.y - snake_head_new.y)
            if last_distance_to_food is not None:
                if current_distance_to_food < last_distance_to_food:
                    reward += 0.3  # Reward for moving closer to the food
                else:
                    reward -= 0.3  # Penalize for moving away from the food
            last_distance_to_food = current_distance_to_food
        
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # train long memory (replay memory so it trains again on all the previous games)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()