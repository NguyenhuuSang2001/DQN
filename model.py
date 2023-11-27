from lib import *

class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_layer=1):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer = hidden_layer
        
        self.fc_input = nn.Linear(self.input_dim, 128)
        self.fc_hidden = [nn.Linear(128, 128) for _ in range(self.hidden_layer)]
        self.fc_output = nn.Linear(128, self.output_dim)

    def forward(self, state):
        x = self.fc_input(state)
        x = F.relu(x)

        for fc in self.fc_hidden:
            x = fc(x)
            x = F.relu(x)
        
        y = self.fc_output(x)
        y = F.relu(y)

        return y