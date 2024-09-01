import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        # If hidden_sizes is an integer, convert it to a list
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        
        # Creating a list of layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Registering layers as a ModuleList
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        # Apply ReLU to all layers except the last one
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            
        # 1: predicted Q values with current state
        pred = self.model(state)
        
        # 2: Q_new = r + gamma * max(next_predicted Q value) -> only do this if not done
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            # Convert the one-hot encoded action to the action index
            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new
            
        # 3: loss = (Q_new - Q)^2
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()