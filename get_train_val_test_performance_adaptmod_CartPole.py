from collections import OrderedDict
from datetime import date
import random
import gym
import numpy as np
import torch
from Master_Thesis_Code.Adaptation_Module import StandardRNN
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
from Master_Thesis_Code.Neuromodulated_Agent import NeuromodulatedAgent
from Master_Thesis_Code.backpropamine_A2C import BP_RNetwork
from neuromodulated_ncps.ncps.wirings import AutoNCP

def evaluate_agent_all_params(policy_net, adaptation_module, env_name, num_episodes, evaluation_seeds, pole_length_modifier, pole_mass_modifier, force_mag_modifier):
    
    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.length *= pole_length_modifier
    env.unwrapped.masspole *= pole_mass_modifier
    env.unwrapped.force_mag *= force_mag_modifier
        
    for i_episode in range(num_episodes):
        policy_hidden_state = None
        adaptation_module_hidden_state = None
        
        env.seed(int(evaluation_seeds[i_episode]))
        
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

        while not done:
            adaptation_module_input = torch.cat((prev_state, prev_action), 1).to(torch.float32).to(device)
            adaptation_module_output, adaptation_module_hidden_state = adaptation_module(adaptation_module_input, adaptation_module_hidden_state)
            
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device) #This as well?
            prev_state = state


            policy_output, _, policy_hidden_state = policy_net(state.float(), policy_hidden_state, adaptation_module_output)
            
            # Get distribution over the action space and select
            # the action with the highest probability.
            policy_dist = torch.softmax(policy_output, dim = 1)
            action = torch.argmax(policy_dist).item()
            prev_action = action
            prev_action = torch.tensor(prev_action).view(1, -1)
            

            # Take a step in the environment
            state, r, done, _ = env.step(action)
            total_reward += r

        eval_rewards.append(total_reward)

    return eval_rewards



def get_privileged_info(env):
    pole_length = env.unwrapped.length
    masspole = env.unwrapped.masspole
    force_mag = env.unwrapped.force_mag

    privileged_info = [pole_length, masspole, force_mag]
    return torch.tensor(privileged_info, dtype=torch.float32)




training_ranges = [(0.775, 5.75), (1.0, 2.0), (0.8, 2.25)]
validation_ranges = [[(0.55, 0.775), (5.75, 10.5)], [(2.0, 3.0)], [(0.6, 0.8), (2.25, 3.5)]]
testing_ranges = [[(0.1, 0.55), (10.5, 20.0)], [(5.0, 13.0)], [(0.2, 0.6), (3.5, 6.0)]]



device = "cpu"
neuron_type = "BP"
if neuron_type == "BP":
    top_dir = "BP_A2C"
else:
    top_dir = "LTC_A2C"
mode = "neuromodulated"
adapt_mod_type = "StandardRNN"
num_neurons_policy = 48
num_neurons_adaptmod = 64
state_dims = 4
action_dims = 1
num_actions = 2

num_models = 10
seed = 5
env_name = "CartPole-v0"
n_evaluations = 1000

wiring = None
batch_dir_am = ""
batch_dir_p = ""





evaluation_seeds = np.load('learning_to_adapt/rstdp_cartpole/seeds/evaluation_seeds.npy')
arrays = [evaluation_seeds]

# Generate the new arrays and add them to the list
for i in range(1, 10):
    new_array = evaluation_seeds + i
    arrays.append(new_array)

# Concatenate all arrays together
evaluation_seeds = np.concatenate(arrays)

policy_dir = "BP_a2c_result_43092_202453_learningrate_0.0001_numneurons_48_encoutact_tanh_neuromod_network_dims_3_128_80_48"
adapt_mod_dir = "adaptation_module_StandardRNN_result_66002_202459_BP_a2c_result_2169_202448_numneuronsadaptmod_64_lradaptmod_0.001_wdadaptmod_0.01"


policy_weights_0 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_p}/{policy_dir}/checkpoint_{neuron_type}_A2C_0.pt', map_location=torch.device(device))
policy_weights_1 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_p}/{policy_dir}/checkpoint_{neuron_type}_A2C_1.pt', map_location=torch.device(device))
policy_weights_2 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_p}/{policy_dir}/checkpoint_{neuron_type}_A2C_2.pt', map_location=torch.device(device))
policy_weights_3 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_p}/{policy_dir}/checkpoint_{neuron_type}_A2C_3.pt', map_location=torch.device(device))
policy_weights_4 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_p}/{policy_dir}/checkpoint_{neuron_type}_A2C_4.pt', map_location=torch.device(device))
policy_weights_5 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_p}/{policy_dir}/checkpoint_{neuron_type}_A2C_5.pt', map_location=torch.device(device))
policy_weights_6 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_p}/{policy_dir}/checkpoint_{neuron_type}_A2C_6.pt', map_location=torch.device(device))
policy_weights_7 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_p}/{policy_dir}/checkpoint_{neuron_type}_A2C_7.pt', map_location=torch.device(device))
policy_weights_8 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_p}/{policy_dir}/checkpoint_{neuron_type}_A2C_8.pt', map_location=torch.device(device))
policy_weights_9 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_p}/{policy_dir}/checkpoint_{neuron_type}_A2C_9.pt', map_location=torch.device(device))
policy_weights = [policy_weights_0, policy_weights_1, policy_weights_2, policy_weights_3, policy_weights_4, policy_weights_5, policy_weights_6, policy_weights_7, policy_weights_8, policy_weights_9]


am_weights_0 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_am}/{adapt_mod_dir}/best_adaptation_module_loss_{neuron_type}_A2C_0.pt', map_location=torch.device(device))
am_weights_1 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_am}/{adapt_mod_dir}/best_adaptation_module_loss_{neuron_type}_A2C_1.pt', map_location=torch.device(device))
am_weights_2 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_am}/{adapt_mod_dir}/best_adaptation_module_loss_{neuron_type}_A2C_2.pt', map_location=torch.device(device))
am_weights_3 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_am}/{adapt_mod_dir}/best_adaptation_module_loss_{neuron_type}_A2C_3.pt', map_location=torch.device(device))
am_weights_4 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_am}/{adapt_mod_dir}/best_adaptation_module_loss_{neuron_type}_A2C_4.pt', map_location=torch.device(device))
am_weights_5 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_am}/{adapt_mod_dir}/best_adaptation_module_loss_{neuron_type}_A2C_5.pt', map_location=torch.device(device))
am_weights_6 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_am}/{adapt_mod_dir}/best_adaptation_module_loss_{neuron_type}_A2C_6.pt', map_location=torch.device(device))
am_weights_7 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_am}/{adapt_mod_dir}/best_adaptation_module_loss_{neuron_type}_A2C_7.pt', map_location=torch.device(device))
am_weights_8 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_am}/{adapt_mod_dir}/best_adaptation_module_loss_{neuron_type}_A2C_8.pt', map_location=torch.device(device))
am_weights_9 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir_am}/{adapt_mod_dir}/best_adaptation_module_loss_{neuron_type}_A2C_9.pt', map_location=torch.device(device))
am_weights = [am_weights_0, am_weights_1, am_weights_2, am_weights_3, am_weights_4, am_weights_5, am_weights_6, am_weights_7, am_weights_8, am_weights_9]



eraser = '\b \b'

with torch.no_grad():
    training_rewards = []
    validation_rewards = []
    testing_rewards = []
    for i, (pw, amw) in enumerate(zip(policy_weights, am_weights)):
        print('Run {:02d} ...'.format(i), end='')
        if neuron_type == "LTC":
            # agent_net = LTC_Network(4, num_neurons_policy, 2, seed, wiring = wiring).to(device)
            raise NotImplementedError
        elif neuron_type == "CfC":
            if adapt_mod_type == 'StandardRNN':
                adaptation_module = StandardRNN(state_dims + action_dims, num_neurons_adaptmod, num_neurons_policy, seed = seed)
            else:
                raise NotImplementedError

            policy_net = CfC_Network(state_dims, num_neurons_policy, num_actions, seed, mode = mode, wiring = wiring).to(device)
            w_policy = OrderedDict((k.split('.', 1)[-1], v) for k, v in pw.items() if 'neuromod' not in k)
            w_policy['cfc_model.rnn_cell.tau_system'] = torch.reshape(w_policy['cfc_model.rnn_cell.tau_system'], (num_neurons_policy,))
            policy_net.load_state_dict(w_policy)

            adaptation_module.load_state_dict(amw)
        elif neuron_type == "BP":
            if adapt_mod_type == 'StandardRNN':
                adaptation_module = StandardRNN(state_dims + action_dims, num_neurons_adaptmod, num_neurons_policy, seed = seed)
            else:
                raise NotImplementedError
            
            policy_net = BP_RNetwork(4, num_neurons_policy, 2, seed, external_neuromodulation = True).to(device)
            w_policy = OrderedDict((k.split('.', 1)[-1], v) for k, v in pw.items() if 'neuromod' not in k)
            policy_net.load_state_dict(w_policy)

            adaptation_module.load_state_dict(amw)

        

        
        # Training rewards ---------------------------
        rewards_sum = 0
        for i in range(n_evaluations):
            np.random.seed(evaluation_seeds[i])
            
            pole_length_mod = np.random.uniform(training_ranges[0][0], training_ranges[0][1])
            pole_mass_mod = np.random.uniform(training_ranges[1][0], training_ranges[1][1])
            force_mag_mod = np.random.uniform(training_ranges[2][0], training_ranges[2][1])

            rewards_sum += np.mean(evaluate_agent_all_params(policy_net, adaptation_module, env_name, 1, evaluation_seeds[i:], pole_length_mod, pole_mass_mod, force_mag_mod))
        
        rewards_sum /= n_evaluations
        training_rewards.append(rewards_sum)
        print(eraser*3 + '-> Avg training reward: {:7.2f}'.format(rewards_sum))


        # Validation rewards ---------------------------
        rewards_sum = 0
        for i in range(n_evaluations):
            np.random.seed(evaluation_seeds[i])
            random.seed(evaluation_seeds[i])
            
            pole_length_range = random.choice(validation_ranges[0])
            pole_length_mod = np.random.uniform(pole_length_range[0], pole_length_range[1])
            pole_mass_mod = np.random.uniform(validation_ranges[1][0][0], validation_ranges[1][0][1])
            force_mag_range = random.choice(validation_ranges[2])
            force_mag_mod = np.random.uniform(force_mag_range[0], force_mag_range[1])

            rewards_sum += np.mean(evaluate_agent_all_params(policy_net, adaptation_module, env_name, 1, evaluation_seeds[i:], pole_length_mod, pole_mass_mod, force_mag_mod))

        rewards_sum /= n_evaluations
        validation_rewards.append(rewards_sum)
        print(eraser*3 + '-> Avg validation reward: {:7.2f}'.format(rewards_sum))


        # Testing rewards ---------------------------
        rewards_sum = 0
        for i in range(n_evaluations):
            np.random.seed(evaluation_seeds[i])
            random.seed(evaluation_seeds[i])
            
            pole_length_range = random.choice(testing_ranges[0])
            pole_length_mod = np.random.uniform(pole_length_range[0], pole_length_range[1])
            pole_mass_mod = np.random.uniform(testing_ranges[1][0][0], testing_ranges[1][0][1])
            force_mag_range = random.choice(testing_ranges[2])
            force_mag_mod = np.random.uniform(force_mag_range[0], force_mag_range[1])

            rewards_sum += np.mean(evaluate_agent_all_params(policy_net, adaptation_module, env_name, 1, evaluation_seeds[i:], pole_length_mod, pole_mass_mod, force_mag_mod))

        rewards_sum /= n_evaluations
        testing_rewards.append(rewards_sum)
        print(eraser*3 + '-> Avg testing reward: {:7.2f}'.format(rewards_sum))


    with open(f"learning_to_adapt/results/{top_dir}/{batch_dir_am}/{adapt_mod_dir}/train_val_test.txt", "w") as f:
        f.write(f"Mean avg training reward: {np.mean(training_rewards)} +/- {np.std(training_rewards)}\n")
        print(training_rewards)
        print(f"Mean avg training reward: {np.mean(training_rewards)} +/- {np.std(training_rewards)}")
        f.write(f"Mean avg validation reward: {np.mean(validation_rewards)} +/- {np.std(validation_rewards)}\n")
        print(validation_rewards)
        print(f"Mean avg validation reward: {np.mean(validation_rewards)} +/- {np.std(validation_rewards)}")
        f.write(f"Mean avg testing reward: {np.mean(testing_rewards)} +/- {np.std(testing_rewards)}\n")
        print(testing_rewards)
        print(f"Mean avg testing reward: {np.mean(testing_rewards)} +/- {np.std(testing_rewards)}")




            
            

            


