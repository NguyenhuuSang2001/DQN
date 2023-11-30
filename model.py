from lib import *

class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc_input = nn.Linear(self.input_dim, 128)
        self.fc_hidden = nn.Linear(128, 256)
        self.fc_output = nn.Linear(256, self.output_dim)

    def forward(self, state):
        x = self.fc_input(state)
        x = F.relu(x)

        x = self.fc_hidden(x)
        x = F.relu(x)
        
        y = self.fc_output(x)
        y = F.relu(y)

        return y