
from copy import deepcopy

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import random

import numpy as np



class BP_RNetwork(nn.Module):
    
    def __init__(self, isize, hsize, num_actions, seed, external_neuromodulation = False, continuous_actions = False): 
        super(BP_RNetwork, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.continuous_actions = continuous_actions
        self.external_neuromodulation = external_neuromodulation

        self.hsize, self.isize  = hsize, isize 

        self.i2h = torch.nn.Linear(isize, hsize)    # Weights from input to recurrent layer
        self.w =  torch.nn.Parameter(.001 * torch.rand(hsize, hsize))   # Baseline (non-plastic) component of the plastic recurrent layer
        
        self.alpha =  torch.nn.Parameter(.001 * torch.rand(hsize, hsize))   # Plasticity coefficients of the plastic recurrent layer; one alpha coefficient per recurrent connection

        if not external_neuromodulation:
            self.h2mod = torch.nn.Linear(hsize, 1)      # Weights from the recurrent layer to the (single) neurodulator output
            self.modfanout = torch.nn.Linear(1, hsize)  # The modulator output is passed through a different 'weight' for each neuron (it 'fans out' over neurons)
        
        self.h2o = torch.nn.Linear(hsize, num_actions)    # From recurrent to outputs (action probabilities)
        if continuous_actions:
            self.h2std = torch.nn.Linear(hsize, num_actions)
        
        self.h2v = torch.nn.Linear(hsize, 1)            # From recurrent to value-prediction (used for A2C)


        
    def forward(self, inputs, hidden, neuromod_signal = None): # hidden is a tuple containing the h-state (i.e. the recurrent hidden state) and the hebbian trace 
        if hidden is None:
            if len(inputs.shape) == 2:
                hidden = (self.initialZeroState(inputs.size(0)), self.initialZeroHebb(inputs.size(0)))
            else:
                hidden = (self.initialZeroState(inputs.size(1)), self.initialZeroHebb(inputs.size(1)))
        
        # hidden[0] is the h-state; hidden[1] is the Hebbian trace
        hebb = hidden[1]
    

        # Each *column* of w, alpha and hebb contains the inputs weights to a single neuron
        hactiv = torch.tanh( self.i2h(inputs) + hidden[0].unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(1)  )  # Update the h-state
        activout = self.h2o(hactiv)  # Pure linear, raw scores - to be softmaxed later, outside the function
        if self.continuous_actions:
            std = F.softplus(self.h2std(hactiv)) + 1e-5
            activout = (activout, std)
        valueout = self.h2v(hactiv)

        # Now computing the Hebbian updates...
        part1 = hidden[0]  # Batched outer product of previous hidden state with new hidden state
        part2 = hactiv  # Batched outer product of new hidden state with itself
        deltahebb = torch.bmm(part1.unsqueeze(2), part2.unsqueeze(1))  # Batched outer product of previous hidden state with new hidden state
        # deltahebb = torch.bmm(hidden[0].unsqueeze(2), hactiv)  # Batched outer product of previous hidden state with new hidden state
        
        if not self.external_neuromodulation:
            neuromod_eta = torch.tanh(self.h2mod(hactiv)).unsqueeze(2)  # Shape: BatchSize x 1 x 1
            
            # The neuromodulated eta is passed through a vector of fanout weights, one per neuron.
            # Each *column* in w, hebb and alpha constitutes the inputs to a single cell.
            # For w and alpha, columns are 2nd dimension (i.e. dim 1); for hebb, it's dimension 2 (dimension 0 is batch)
            # The output of the following line has shape BatchSize x 1 x NHidden, i.e. 1 line and NHidden columns for each 
            # batch element. When multiplying by hebb (BatchSize x NHidden x NHidden), broadcasting will provide a different
            # value for each cell but the same value for all inputs of a cell, as required by fanout concept.
            neuromod_eta = self.modfanout(neuromod_eta) 
        else:
            assert neuromod_signal is not None
            neuromod_eta = neuromod_signal.unsqueeze(2)  # Shape: BatchSize x 1 x 1
        
        # Updating Hebbian traces, with a hard clip (other choices are possible)
        self.clipval = 2.0
        hebb = torch.clamp(hebb + neuromod_eta * deltahebb, min=-self.clipval, max=self.clipval)

        hidden = (hactiv, hebb)
        return activout, valueout, hidden




    def initialZeroState(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize), requires_grad=False )

    # In plastic networks, we must also initialize the Hebbian state:
    def initialZeroHebb(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize, self.hsize) , requires_grad=False)
    
    def loadWeights(self, weights):
        weights = deepcopy(weights)
        self.i2h.weight = torch.nn.Parameter(weights["i2h.weight"])
        self.i2h.bias = torch.nn.Parameter(weights["i2h.bias"])
        self.w = torch.nn.Parameter(weights["w"])
        self.alpha = torch.nn.Parameter(weights["alpha"])
        self.h2mod.weight = torch.nn.Parameter(weights["h2mod.weight"])
        self.h2mod.bias = torch.nn.Parameter(weights["h2mod.bias"])
        self.modfanout.weight = torch.nn.Parameter(weights["modfanout.weight"])
        self.modfanout.bias = torch.nn.Parameter(weights["modfanout.bias"])
        self.h2o.weight = torch.nn.Parameter(weights["h2o.weight"])
        self.h2o.bias = torch.nn.Parameter(weights["h2o.bias"])
        self.h2v.weight = torch.nn.Parameter(weights["h2v.weight"])
        self.h2v.bias = torch.nn.Parameter(weights["h2v.bias"])




class Standard_RNetwork(nn.Module):
    
    def __init__(self, isize, hsize, num_actions, seed, continuous_actions = False): 
        super(Standard_RNetwork, self).__init__()

        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.continuous_actions = continuous_actions

        self.hsize, self.isize  = hsize, isize 

        self.i2h = torch.nn.Linear(isize, hsize)    # Weights from input to recurrent layer
        self.w =  torch.nn.Parameter(.001 * torch.rand(hsize, hsize))   # Baseline (non-plastic) component of the plastic recurrent layer
        
        self.h2o = torch.nn.Linear(hsize, num_actions)    # From recurrent to outputs (action probabilities)
        if continuous_actions:
            self.h2std = torch.nn.Linear(hsize, num_actions)    
        
        self.h2v = torch.nn.Linear(hsize, 1)            # From recurrent to value-prediction (used for A2C)


        
    def forward(self, inputs, hidden): # hidden is a tuple containing the h-state (i.e. the recurrent hidden state) and the hebbian trace 
        if hidden is None:
            if len(inputs.shape) == 2:
                hidden = (self.initialZeroState(inputs.size(0)), self.initialZeroHebb(inputs.size(0)))
            else:
                hidden = (self.initialZeroState(inputs.size(1)), self.initialZeroHebb(inputs.size(1)))
        
        
        
        # hidden[0] is the h-state; hidden[1] is the Hebbian trace
        hebb = hidden[1]


        # Each *column* of w, alpha and hebb contains the inputs weights to a single neuron
        hactiv = torch.tanh( self.i2h(inputs) + torch.matmul(hidden[0], self.w))  # Update the h-state
        activout = self.h2o(hactiv)  # Pure linear, raw scores - to be softmaxed later, outside the function
        if self.continuous_actions:
            std = F.softplus(self.h2std(hactiv)) + 1e-5
            activout = (activout, std)
        valueout = self.h2v(hactiv)

    
        hidden = (hactiv, hebb)
        return activout, valueout, hidden




    def initialZeroState(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize), requires_grad=False )

    # In plastic networks, we must also initialize the Hebbian state:
    def initialZeroHebb(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize, self.hsize) , requires_grad=False)
    
    def loadWeights(self, weights):
        weights = deepcopy(weights)
        self.i2h.weight = torch.nn.Parameter(weights["i2h.weight"])
        self.i2h.bias = torch.nn.Parameter(weights["i2h.bias"])
        self.w = torch.nn.Parameter(weights["w"])
        self.h2o.weight = torch.nn.Parameter(weights["h2o.weight"])
        self.h2o.bias = torch.nn.Parameter(weights["h2o.bias"])
        self.h2v.weight = torch.nn.Parameter(weights["h2v.weight"])
        self.h2v.bias = torch.nn.Parameter(weights["h2v.bias"])



class Standard_FFNetwork(nn.Module):
    def __init__(self, input_size, first_hidden_size, second_hidden_size, num_actions, seed): 
        super(Standard_FFNetwork, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.input_size, self.first_layer_size, self.second_layer_size  = input_size, first_hidden_size, second_hidden_size

        self.i2h = torch.nn.Linear(input_size, first_hidden_size)    # Weights from input to recurrent layer
        self.w =  torch.nn.Linear(first_hidden_size, second_hidden_size)   # Baseline (non-plastic) component of the plastic recurrent layer
        
        self.h2o = torch.nn.Linear(second_hidden_size, num_actions)    # From recurrent to outputs (action probabilities)
        self.h2v = torch.nn.Linear(second_hidden_size, 1)            # From recurrent to value-prediction (used for A2C)

    def forward(self, inputs, hidden): # hidden is a tuple containing the h-state (i.e. the recurrent hidden state) and the hebbian trace 
        
        activation = self.w(self.i2h(inputs))

        activout = self.h2o(activation)
        valueout = self.h2v(activation)

        
    
        return activout, valueout, hidden




    def initialZeroState(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.first_layer_size), requires_grad=False )

    # In plastic networks, we must also initialize the Hebbian state:
    def initialZeroHebb(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.first_layer_size, self.second_layer_size) , requires_grad=False)
    
    def loadWeights(self, weights):
        weights = deepcopy(weights)
        self.i2h.weight = torch.nn.Parameter(weights["i2h.weight"])
        self.i2h.bias = torch.nn.Parameter(weights["i2h.bias"])
        self.w.weight = torch.nn.Parameter(weights["w.weight"])
        self.w.bias = torch.nn.Parameter(weights["w.bias"])
        self.h2o.weight = torch.nn.Parameter(weights["h2o.weight"])
        self.h2o.bias = torch.nn.Parameter(weights["h2o.bias"])
        self.h2v.weight = torch.nn.Parameter(weights["h2v.weight"])
        self.h2v.bias = torch.nn.Parameter(weights["h2v.bias"])