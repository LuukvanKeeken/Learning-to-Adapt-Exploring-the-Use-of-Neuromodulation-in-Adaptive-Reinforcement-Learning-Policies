import torch.nn as nn
import random
import torch
import numpy as np



class NeuromodulatedAgent(nn.Module):
    def __init__(self, policy_net, neuromod_net, policy_has_hidden_state = False, neuromod_has_hidden_state = False):
        super(NeuromodulatedAgent, self).__init__()

        self.policy_net = policy_net
        self.neuromod_net = neuromod_net

        self.policy_has_hidden_state = policy_has_hidden_state
        self.neuromod_has_hidden_state = neuromod_has_hidden_state

    
    def forward(self, policy_input, neuromod_input, policy_hidden_state = None, neuromod_hidden_state = None):
        if self.neuromod_has_hidden_state:
            neuromod_output, neuromod_hidden_state = self.neuromod_net(neuromod_input, neuromod_hidden_state)
        else:
            neuromod_output = self.neuromod_net(neuromod_input)

        if self.policy_has_hidden_state:
            policy_output, value, policy_hidden_state = self.policy_net(policy_input, policy_hidden_state, neuromod_output)
        else:
            policy_output, value = self.policy_net(policy_input, neuromod_output)

        if self.policy_has_hidden_state:
            if self.neuromod_has_hidden_state:
                return policy_output, value, policy_hidden_state, neuromod_hidden_state
            else:
                return policy_output, value, policy_hidden_state
        else:
            if self.neuromod_has_hidden_state:
                return policy_output, value, neuromod_hidden_state
            else:
                return policy_output, value