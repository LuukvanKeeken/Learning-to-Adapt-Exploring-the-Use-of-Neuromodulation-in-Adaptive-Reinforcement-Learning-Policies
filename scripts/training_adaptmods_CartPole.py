from collections import OrderedDict
from copy import deepcopy
from datetime import date
import random
import gym
import numpy as np
import torch
from learning_to_adapt.LTC_A2C import LTC_Network, CfC_Network
from learning_to_adapt.Adaptation_Module import StandardRNN
from learning_to_adapt.backpropamine_A2C import BP_RNetwork
from learning_to_adapt.modifiable_async_vector_env import ModifiableAsyncVectorEnv
import os
import argparse
import time
torch.autograd.set_detect_anomaly(True)



def get_privileged_info(randomized_env_params):
    params_values = [[d['length'], d['masspole'], d['force_mag']] for d in randomized_env_params]
    params_tensor = torch.tensor(params_values, dtype=torch.float32)

    return params_tensor

def get_privileged_info2(env):
    pole_length = env.unwrapped.length
    masspole = env.unwrapped.masspole
    force_mag = env.unwrapped.force_mag

    privileged_info = [pole_length, masspole, force_mag]
    return torch.tensor(privileged_info, dtype=torch.float32)


def get_random_env_paramvals(env, randomization_params, batch_size = 1):
    pole_length = env.unwrapped.length
    masspole = env.unwrapped.masspole
    force_mag = env.unwrapped.force_mag

    orig_params = [pole_length, masspole, force_mag]
    param_names = ['length', 'masspole', 'force_mag']
    new_params = [{name: None for name in param_names} for _ in range(batch_size)]
    
    for i in range(len(orig_params)):
        if isinstance(randomization_params[i], float):
            low = orig_params[i] - orig_params[i] * randomization_params[i]
            high = orig_params[i] + orig_params[i] * randomization_params[i]
        elif isinstance(randomization_params[i], tuple):
            low = orig_params[i]*randomization_params[i][0]
            high = orig_params[i]*randomization_params[i][1]
            
        sampled_values = np.random.uniform(low, high, batch_size)

        for j in range(batch_size):
            new_params[j][param_names[i]] = sampled_values[j]

    return new_params



def validate_adaptation_module_old(agent_net, encoder, adaptation_module, evaluation_seeds, env_name, num_validation_eps, max_steps):
    with torch.no_grad():
        pole_length_mods = [0.55, 10.5]
        pole_mass_mods = [3.0]
        force_mag_mods = [0.6, 3.5]
        
        
        validation_rewards = []
        validation_losses = []
        settings = []

        for episode in range(num_validation_eps):
            env = gym.make(env_name)
            policy_hidden_state = None
            adaptation_module_hidden_state = None
            adaptation_module_outputs = []
            encoder_outputs = []

            
            pole_length_mod = np.random.choice(pole_length_mods)
            pole_mass_mod = np.random.choice(pole_mass_mods)
            force_mag_mod = np.random.choice(force_mag_mods)
            settings.append((pole_length_mod, pole_mass_mod, force_mag_mod))
            # env.seed(int(evaluation_seeds[episode]))

            env.unwrapped.length *= pole_length_mod
            env.unwrapped.masspole *= pole_mass_mod
            env.unwrapped.force_mag *= force_mag_mod

            state = env.reset()
            total_reward = 0
            done = False

            # Randomly sample an action and state to 
            # feed as the first input to the adaptation
            # module.
            prev_action = env.action_space.sample()
            prev_action = torch.tensor(prev_action).view(1, -1)
            prev_state = env.observation_space.sample()
            prev_state = torch.from_numpy(prev_state)
            prev_state = prev_state.unsqueeze(0).to(device)

            for step in range(max_steps):

                adaptation_module_input = torch.cat((prev_state, prev_action), 1).to(torch.float32).to(device)
                adaptation_module_output, adaptation_module_hidden_state = adaptation_module(adaptation_module_input, adaptation_module_hidden_state)

                privileged_info = get_privileged_info2(env).unsqueeze(0).to(device)
                encoder_output = encoder(privileged_info)

                # Transform the state to the correct format and save it
                # to be used as previous state in the next time step.
                state = torch.from_numpy(state)
                state = state.unsqueeze(0).to(device)
                prev_state = state

                # In the first step, we don't save the output of the adaptation module
                # and the encoder, since they are based on a randomly sampled state and action.
                if not step == 0:
                    adaptation_module_outputs.append(adaptation_module_output)
                    encoder_outputs.append(encoder_output)
                
                # Feed the state and adaptation module input into the agent network
                policy_output, value, policy_hidden_state = agent_net(state.float(), policy_hidden_state, adaptation_module_output)

                # Get distribution over the action space and select
                # the action with the highest probability.
                policy_dist = torch.softmax(policy_output, dim = 1)
                action = torch.argmax(policy_dist).item()
                prev_action = action
                prev_action = torch.tensor(prev_action).view(1, -1)

                # Take a step in the environment
                state, r, done, _ = env.step(action)
                total_reward += r

                if done or step == max_steps - 1:
                    validation_rewards.append(total_reward)
                    assert len(adaptation_module_outputs) == len(encoder_outputs)
                    loss_function = torch.nn.MSELoss()
                    loss_val = loss_function(torch.stack(adaptation_module_outputs), torch.stack(encoder_outputs))
                    validation_losses.append(loss_val.item())
                    break

        return np.mean(validation_losses), np.mean(validation_rewards), np.std(validation_rewards)



def validate_adaptation_module(agent_net, encoder, adaptation_module, evaluation_seeds, env_name, num_validation_eps, max_steps):
    with torch.no_grad():
        validation_ranges = [[(0.55, 0.775), (5.75, 10.5)], [(2.0, 3.0)], [(0.6, 0.8), (2.25, 3.5)]]
        
        
        validation_rewards = []
        validation_losses = []
        current_np_seed = np.random.get_state()
        current_r_seed = random.getstate()
        for episode in range(num_validation_eps):
            env = gym.make(env_name)
            policy_hidden_state = None
            adaptation_module_hidden_state = None
            adaptation_module_outputs = []
            encoder_outputs = []

            
            np.random.seed((evaluation_seeds[episode] + seed)%(2**32))
            random.seed((evaluation_seeds[episode] + seed)%(2**32))
            pole_length_range = random.choice(validation_ranges[0])
            pole_length_mod = np.random.uniform(pole_length_range[0], pole_length_range[1])
            pole_mass_mod = np.random.uniform(validation_ranges[1][0][0], validation_ranges[1][0][1])
            force_mag_range = random.choice(validation_ranges[2])
            force_mag_mod = np.random.uniform(force_mag_range[0], force_mag_range[1])
            
            env.seed(int(evaluation_seeds[episode]))

            env.unwrapped.length *= pole_length_mod
            env.unwrapped.masspole *= pole_mass_mod
            env.unwrapped.force_mag *= force_mag_mod

            state = env.reset()
            total_reward = 0
            done = False

            # Randomly sample an action and state to 
            # feed as the first input to the adaptation
            # module.
            prev_action = env.action_space.sample()
            prev_action = torch.tensor(prev_action).view(1, -1)
            prev_state = env.observation_space.sample()
            prev_state = torch.from_numpy(prev_state)
            prev_state = prev_state.unsqueeze(0).to(device)

            for step in range(max_steps):

                adaptation_module_input = torch.cat((prev_state, prev_action), 1).to(torch.float32).to(device)
                adaptation_module_output, adaptation_module_hidden_state = adaptation_module(adaptation_module_input, adaptation_module_hidden_state)

                privileged_info = get_privileged_info2(env).unsqueeze(0).to(device)
                encoder_output = encoder(privileged_info)

                # Transform the state to the correct format and save it
                # to be used as previous state in the next time step.
                state = torch.from_numpy(state)
                state = state.unsqueeze(0).to(device)
                prev_state = state

                # In the first step, we don't save the output of the adaptation module
                # and the encoder, since they are based on a randomly sampled state and action.
                if not step == 0:
                    adaptation_module_outputs.append(adaptation_module_output)
                    encoder_outputs.append(encoder_output)
                
                # Feed the state and adaptation module input into the agent network
                policy_output, value, policy_hidden_state = agent_net(state.float(), policy_hidden_state, adaptation_module_output)

                # Get distribution over the action space and select
                # the action with the highest probability.
                policy_dist = torch.softmax(policy_output, dim = 1)
                action = torch.argmax(policy_dist).item()
                prev_action = action
                prev_action = torch.tensor(prev_action).view(1, -1)

                # Take a step in the environment
                state, r, done, _ = env.step(action)
                total_reward += r

                if done or step == max_steps - 1:
                    validation_rewards.append(total_reward)
                    assert len(adaptation_module_outputs) == len(encoder_outputs)
                    loss_function = torch.nn.MSELoss()
                    loss_val = loss_function(torch.stack(adaptation_module_outputs), torch.stack(encoder_outputs))
                    validation_losses.append(loss_val.item())
                    break
        
        np.random.set_state(current_np_seed)
        random.setstate(current_r_seed)
        return np.mean(validation_losses), np.mean(validation_rewards), np.std(validation_rewards)



def train_adaptation_module(env, num_parallel_envs, batch_size, num_training_episodes, max_steps, agent_net, num_outputs, evaluation_seeds, i_run, neuron_type, encoder, adaptation_module, optimizer, selection_method = "100 episode average", gamma = 0.99, max_reward = 200, env_name = "CartPole-v0", num_validation_eps = 10, validate_every = 10, randomization_params = None, randomize_every = 5):
    loss_function = torch.nn.MSELoss()

    training_losses = []
    training_total_rewards = []
    validation_losses = []
    validation_total_rewards = []

    best_validation_reward = -np.inf
    best_validation_reward_after = -1
    best_validation_loss = np.inf
    best_validation_loss_after = -1

    total_episodes_trained = 0

    vec_env = ModifiableAsyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_parallel_envs)])

        
    while total_episodes_trained < num_training_episodes:
        encoder_outputs_batch = []
        adaptation_module_outputs_batch = []
        total_training_rewards_batch = []

        running_encoder_outputs = [[] for _ in range(num_parallel_envs)]
        running_adaptation_module_outputs = [[] for _ in range(num_parallel_envs)]

        running_step_numbers = [0 for _ in range(num_parallel_envs)]
        running_total_training_rewards = [0 for _ in range(num_parallel_envs)]

        # Initlialize the hidden states for the policy network and the adaptation module
        policy_hidden_states = None
        adaptation_module_hidden_states = None

        # Initialize the environments' parameters
        randomized_env_params = get_random_env_paramvals(env, randomization_params, num_parallel_envs)
        vec_env.set_env_params(randomized_env_params)

        dones = [False for _ in range(num_parallel_envs)]

        prev_actions = vec_env.action_space.sample()
        prev_actions = torch.tensor(prev_actions).detach().to(device)
        prev_states = vec_env.observation_space.sample()
        if neuron_type == "BP":
            prev_states = torch.from_numpy(prev_states).detach().to(device)
        else:
            prev_states = torch.from_numpy(prev_states).unsqueeze(0).detach().to(device)


        states = vec_env.reset()
        start = time.time()
        while len(encoder_outputs_batch) < batch_size:
            
            # Feed the settings of the parallel environments into the encoder
            vec_encoder_outputs = encoder(get_privileged_info(randomized_env_params))

            # Feed the previous states and actions into the adaptation module
            if neuron_type == "BP":
                adaptation_module_inputs = torch.cat((prev_states, prev_actions.unsqueeze(-1)), -1).to(torch.float32).to(device)
            else:
                adaptation_module_inputs = torch.cat((prev_states, prev_actions.unsqueeze(0).unsqueeze(-1)), -1).to(torch.float32).to(device)
            vec_adaptation_module_outputs, adaptation_module_hidden_states = adaptation_module(adaptation_module_inputs, adaptation_module_hidden_states)
            
            if neuron_type == "BP":
                states = torch.from_numpy(states).detach().to(device)
            else:
                states = torch.from_numpy(states).unsqueeze(0).detach().to(device)
            prev_states = states


            # In the first step, we don't save the output of the adaptation module
            # and the encoder, since they are based on a randomly sampled state and action.
            for i, (encoder_output, adapt_mod_output) in enumerate(zip(vec_encoder_outputs, vec_adaptation_module_outputs.squeeze(0))):
                if running_step_numbers[i] > 0:
                    running_encoder_outputs[i].append(encoder_output)
                    running_adaptation_module_outputs[i].append(adapt_mod_output)


            # No gradient calculations required when the neuromodulation signals
            # have already been calculated.
            with torch.no_grad():
                policy_outputs, values, policy_hidden_states = agent_net(states.float(), policy_hidden_states, vec_adaptation_module_outputs.squeeze(0))
                
                if neuron_type == "BP":
                    policy_dists = torch.softmax(policy_outputs, dim = 1)
                    actions = torch.argmax(policy_dists, dim = 1).squeeze().tolist()
                else:
                    policy_dists = torch.softmax(policy_outputs, dim = 2)
                    actions = torch.argmax(policy_dists, dim = 2).squeeze().tolist()
                prev_actions = torch.tensor(actions).to(device)
                
                states, rewards, dones, _ = vec_env.step(actions)


            for i, (state, reward, done) in enumerate(zip(states, rewards, dones)):
                running_step_numbers[i] += 1
                running_total_training_rewards[i] += reward

                if done:
                    if neuron_type == "BP":
                        policy_hidden_states[0][i] = torch.zeros_like(policy_hidden_states[0][i])
                        policy_hidden_states[1][i] = torch.zeros_like(policy_hidden_states[1][i])
                    else:
                        policy_hidden_states[i] = torch.zeros_like(policy_hidden_states[i])
                    adaptation_module_hidden_states[0][i] = torch.zeros_like(adaptation_module_hidden_states[0][i])

                    if len(running_encoder_outputs[i]) > 0:
                        encoder_outputs_batch.append(torch.stack(running_encoder_outputs[i]))
                        adaptation_module_outputs_batch.append(torch.stack(running_adaptation_module_outputs[i]))

                        running_encoder_outputs[i] = []
                        running_adaptation_module_outputs[i] = []


                    running_step_numbers[i] = 0
                    total_training_rewards_batch.append(running_total_training_rewards[i])
                    running_total_training_rewards[i] = 0
                
        
        

        total_episodes_trained += batch_size

        losses = []
        for adaptation_module_output, encoder_output in zip(adaptation_module_outputs_batch, encoder_outputs_batch):
            loss_val = loss_function(adaptation_module_output, encoder_output)
            losses.append(loss_val)

        average_loss = sum(losses) / len(losses)
        
        optimizer.zero_grad()
        average_loss.backward()
        optimizer.step()
        training_losses.append(average_loss.item())
        
        average_training_reward = np.mean(total_training_rewards_batch)
        stddev_training_reward = np.std(total_training_rewards_batch)
        training_total_rewards.append(average_training_reward)


        print(f"Episode {total_episodes_trained}/{num_training_episodes}, average batch training loss: {average_loss.item()}, average batch training reward: {average_training_reward} +/- {stddev_training_reward:.2f}")
        

        mean_valid_loss, mean_valid_reward, std_valid_reward = validate_adaptation_module(agent_net, encoder, adaptation_module, evaluation_seeds, env_name, num_validation_eps, max_steps)
        validation_losses.append(mean_valid_loss)
        validation_total_rewards.append(mean_valid_reward)
        print(f"average validation loss: {mean_valid_loss}, average validation reward: {mean_valid_reward} +/- {std_valid_reward}")

        if mean_valid_reward >= best_validation_reward:
            best_validation_reward = mean_valid_reward
            best_validation_reward_after = total_episodes_trained
            torch.save(adaptation_module.state_dict(), f"{results_dir}/best_adaptation_module_reward_{neuron_type}_A2C_{i_run}.pt")
        if mean_valid_loss <= best_validation_loss:
            best_validation_loss = mean_valid_loss
            best_validation_loss_after = total_episodes_trained
            torch.save(adaptation_module.state_dict(), f"{results_dir}/best_adaptation_module_loss_{neuron_type}_A2C_{i_run}.pt")
            

        
    

    print(f"Best validation reward: {best_validation_reward} after {best_validation_reward_after} episodes")
    print(f"Best validation loss: {best_validation_loss} after {best_validation_loss_after} episodes")
    
    return training_losses, training_total_rewards, validation_losses, validation_total_rewards, best_validation_reward, best_validation_reward_after, best_validation_loss, best_validation_loss_after



parser = argparse.ArgumentParser(description='Train adaptation module for neuromodulated CfC')
parser.add_argument('--neuron_type', type=str, default='CfC', help='Type of neuron to train')
parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
parser.add_argument('--state_dims', type=int, default=4, help='Number of state dimensions')
parser.add_argument('--action_dims', type=int, default=1, help='Number of action dimensions')
parser.add_argument('--num_neurons_policy', type=int, default=64, help='Number of neurons in the policy network')
parser.add_argument('--num_neurons_adaptation', type=int, default=64, help='Number of neurons in the adaptation module')
parser.add_argument('--num_actions', type=int, default=2, help='Number of actions')
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--mode', type=str, default='only_neuromodulated', help='Mode of the CfC network')
parser.add_argument('--wiring', type=str, default='None', help='Wiring of the CfC network')
parser.add_argument('--neuromod_network_dims', type=int, nargs='+', default = [3, 256, 128], help='Dimensions of the neuromodulation network, without output layer')
parser.add_argument('--num_training_eps', type=int, default=20000, help="Number of episodes to train the adaptation module")
parser.add_argument('--env_name', type=str, default="CartPole-v0", help="Gym RL environment name")
parser.add_argument('--lr_adapt_mod', type=float, default=0.0005, help="Learning rate of the adaptation module")
parser.add_argument('--wd_adapt_mod', type=float, default=0.0, help="Weight decay of the adaptation module")
parser.add_argument('--training_range', type=str, default='training_range', help='Range from which training data is sampled')
parser.add_argument('--randomize_every', type=int, default=1, help='Number of episodes between randomization of environment parameters')
parser.add_argument('--validate_every', type=int, default=10, help='Number of training episodes between validations')
parser.add_argument('--num_validation_eps', type=int, default=20, help='Number of episodes to validate the adaptation module')
parser.add_argument('--adapt_mod_type', type=str, default='StandardRNN', help='Type of adaptation module to use')
parser.add_argument('--result_id', type=int, default=-1, help='ID of the result')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training the adaptation module')
parser.add_argument('--num_parallel_envs', type=int, default=10, help='Number of parallel environments to train the adaptation module')
parser.add_argument('--encoder_hidden_activation', type=str, default='tanh', help='Activation function for the encoder hidden layers')
parser.add_argument('--encoder_output_activation', type=str, default='tanh', help='Activation function for the encoder output layer')


args = parser.parse_args()
neuron_type = args.neuron_type
device = args.device
state_dims = args.state_dims
action_dims = args.action_dims
num_neurons_policy = args.num_neurons_policy
num_neurons_adaptation = args.num_neurons_adaptation
num_actions = args.num_actions
seed = args.seed
mode = args.mode
if args.wiring == 'None':
    wiring = None
neuromod_network_dims = args.neuromod_network_dims
neuromod_network_dims.append(num_neurons_policy)
num_training_eps = args.num_training_eps
env_name = args.env_name
lr_adapt_mod = args.lr_adapt_mod
wd_adapt_mod = args.wd_adapt_mod
training_range = args.training_range
randomize_every = args.randomize_every
validate_every = args.validate_every
num_validation_eps = args.num_validation_eps
adapt_mod_type = args.adapt_mod_type
result_id = args.result_id
batch_size = args.batch_size
num_parallel_envs = args.num_parallel_envs
if neuron_type == "BP":
    top_dir = "BP_A2C"
else:
    top_dir = "LTC_A2C"
if args.encoder_hidden_activation == 'relu':
    encoder_hidden_activation = torch.nn.ReLU()
elif args.encoder_hidden_activation == 'tanh':
    encoder_hidden_activation = torch.nn.Tanh()
else:
    raise NotImplementedError

if args.encoder_output_activation == 'relu':
    encoder_output_activation = torch.nn.ReLU()
elif args.encoder_output_activation == 'tanh':
    encoder_output_activation = torch.nn.Tanh()
else:
    raise NotImplementedError


if training_range == 'training_range':
    randomization_params = [(0.775, 5.75), (1.0, 2.0), (0.8, 2.25)]
else:
    raise NotImplementedError

if env_name == 'CartPole-v0':
    env = gym.make('CartPole-v0')
else:
    raise NotImplementedError
evaluation_seeds = np.load('learning_to_adapt/rstdp_cartpole/seeds/evaluation_seeds.npy')

phase_1_dir = "CfC_a2c_result_62055_202458_learningrate_0.001_numneurons_64_encoutact_tanh_neuromod_network_dims_3_256_128_64"

if result_id == -1:
    dirs = os.listdir(f'learning_to_adapt/results/{top_dir}/')
    if not any('adaptation_module' in d for d in dirs):
        result_id = 1
    else:
        results = [d for d in dirs if 'adaptation_module' in d]
        result_id = len(results) + 1

d = date.today()


results_dir = f"learning_to_adapt/results/{top_dir}/adaptation_module_{adapt_mod_type}_result_{result_id}_{str(d.year) + str(d.month) + str(d.day)}_BP_a2c_result_2169_202448_numneuronsadaptmod_{num_neurons_adaptation}_lradaptmod_{lr_adapt_mod}_wdadaptmod_{wd_adapt_mod}"
os.mkdir(results_dir)


weights_0 = torch.load(f'learning_to_adapt/results/{top_dir}/{phase_1_dir}/checkpoint_{neuron_type}_A2C_0.pt', map_location=torch.device(device))
weights_1 = torch.load(f'learning_to_adapt/results/{top_dir}/{phase_1_dir}/checkpoint_{neuron_type}_A2C_1.pt', map_location=torch.device(device))
weights_2 = torch.load(f'learning_to_adapt/results/{top_dir}/{phase_1_dir}/checkpoint_{neuron_type}_A2C_2.pt', map_location=torch.device(device))
weights_3 = torch.load(f'learning_to_adapt/results/{top_dir}/{phase_1_dir}/checkpoint_{neuron_type}_A2C_3.pt', map_location=torch.device(device))
weights_4 = torch.load(f'learning_to_adapt/results/{top_dir}/{phase_1_dir}/checkpoint_{neuron_type}_A2C_4.pt', map_location=torch.device(device))
weights_5 = torch.load(f'learning_to_adapt/results/{top_dir}/{phase_1_dir}/checkpoint_{neuron_type}_A2C_5.pt', map_location=torch.device(device))
weights_6 = torch.load(f'learning_to_adapt/results/{top_dir}/{phase_1_dir}/checkpoint_{neuron_type}_A2C_6.pt', map_location=torch.device(device))
weights_7 = torch.load(f'learning_to_adapt/results/{top_dir}/{phase_1_dir}/checkpoint_{neuron_type}_A2C_7.pt', map_location=torch.device(device))
weights_8 = torch.load(f'learning_to_adapt/results/{top_dir}/{phase_1_dir}/checkpoint_{neuron_type}_A2C_8.pt', map_location=torch.device(device))
weights_9 = torch.load(f'learning_to_adapt/results/{top_dir}/{phase_1_dir}/checkpoint_{neuron_type}_A2C_9.pt', map_location=torch.device(device))
weights = [weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9]


all_training_losses = []
all_training_total_rewards = []
all_validation_losses = []
all_validation_total_rewards = []
best_validation_rewards = []
best_validation_losses = []
for i, w in enumerate(weights):
    print(f"Training adaptation module for model {i+1}")

    if neuron_type == "CfC":
        policy_net = CfC_Network(state_dims, num_neurons_policy, num_actions, seed, mode = mode, wiring = wiring)
        
        
        layer_list = []
        for dim in range(len(neuromod_network_dims) - 1):
            layer_list.append(torch.nn.Linear(neuromod_network_dims[dim], neuromod_network_dims[dim + 1]))
            if dim < len(neuromod_network_dims)-2:
                layer_list.append(encoder_hidden_activation)
            else:
                layer_list.append(encoder_output_activation)
        encoder = torch.nn.Sequential(*layer_list)

        w['policy_net.cfc_model.rnn_cell.tau_system'] = torch.reshape(w['policy_net.cfc_model.rnn_cell.tau_system'], (num_neurons_policy,))
        w_policy = OrderedDict((k.split('.', 1)[-1], v) for k, v in w.items() if 'neuromod' not in k)
        w_encoder = OrderedDict((k.split('.', 1)[-1], v) for k, v in w.items() if 'neuromod' in k)
    elif neuron_type == "LTC":
        raise NotImplementedError
    elif neuron_type == "BP":
        policy_net = BP_RNetwork(4, num_neurons_policy, 2, seed, external_neuromodulation = True).to(device)

        layer_list = []
        for dim in range(len(neuromod_network_dims) - 1):
            layer_list.append(torch.nn.Linear(neuromod_network_dims[dim], neuromod_network_dims[dim + 1]))
            if dim < len(neuromod_network_dims)-2:
                layer_list.append(encoder_hidden_activation)
            else:
                layer_list.append(encoder_output_activation)
        encoder = torch.nn.Sequential(*layer_list)

        w_policy = OrderedDict((k.split('.', 1)[-1], v) for k, v in w.items() if 'neuromod' not in k)
        w_encoder = OrderedDict((k.split('.', 1)[-1], v) for k, v in w.items() if 'neuromod' in k)
    
    policy_net.load_state_dict(w_policy)
    encoder.load_state_dict(w_encoder)

    
    for name, param in encoder.named_parameters():
        param.requires_grad = False

    if adapt_mod_type == 'StandardRNN':
        adaptation_module = StandardRNN(state_dims + action_dims, num_neurons_adaptation, num_neurons_policy, seed = seed)
    else:
        raise NotImplementedError
    

    optimizer = torch.optim.Adam(adaptation_module.parameters(), lr = lr_adapt_mod, weight_decay = wd_adapt_mod)

    training_losses, training_total_rewards, validation_losses, validation_total_rewards, best_validation_reward, best_validation_reward_after, best_validation_loss, best_validation_loss_after = train_adaptation_module(env, num_parallel_envs, batch_size, num_training_eps, 200, policy_net, num_actions, evaluation_seeds, i, neuron_type, encoder, adaptation_module, optimizer, randomization_params=randomization_params, randomize_every=randomize_every, validate_every=validate_every, num_validation_eps=num_validation_eps)
    all_training_losses.append(training_losses)
    all_training_total_rewards.append(training_total_rewards)
    all_validation_losses.append(validation_losses)
    all_validation_total_rewards.append(validation_total_rewards)
    best_validation_rewards.append((best_validation_reward, best_validation_reward_after))
    best_validation_losses.append((best_validation_loss, best_validation_loss_after))
    
np.save(f"{results_dir}/all_training_losses.npy", all_training_losses)
np.save(f"{results_dir}/all_training_total_rewards.npy", all_training_total_rewards)
np.save(f"{results_dir}/all_validation_losses.npy", all_validation_losses)
np.save(f"{results_dir}/all_validation_total_rewards.npy", all_validation_total_rewards)


with open(f"{results_dir}/best_validation_reward_after.txt", "w") as f:
    for i, reward_results in enumerate(best_validation_rewards):
        f.write(f"{i}: {reward_results[0]} after {reward_results[1]}\n")

    f.write(f"Average training episodes: {np.mean([x[1] for x in best_validation_rewards])}\n")
    f.write(f"Mean average reward: {np.mean([x[0] for x in best_validation_rewards])} +/- {np.std([x[0] for x in best_validation_rewards])}")

with open(f"{results_dir}/best_validation_loss_after.txt", "w") as f:
    for i, loss_results in enumerate(best_validation_losses):
        f.write(f"{i}: {loss_results[0]} after {loss_results[1]}\n")

    f.write(f"Average training episodes: {np.mean([x[1] for x in best_validation_losses])}\n")
    f.write(f"Mean average loss: {np.mean([x[0] for x in best_validation_losses])} +/- {np.std([x[0] for x in best_validation_losses])}")
