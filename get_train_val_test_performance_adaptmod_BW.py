from collections import OrderedDict
from datetime import date
import random
import gym
import numpy as np
import torch
from learning_to_adapt.Adaptation_Module import StandardRNN
from learning_to_adapt.LTC_A2C import LTC_Network, CfC_Network
from learning_to_adapt.Neuromodulated_Agent import NeuromodulatedAgent
from learning_to_adapt.backpropamine_A2C import BP_RNetwork
from neuromodulated_ncps.ncps.wirings import AutoNCP



def evaluate_BW(policy_net, adaptation_module, env_name, num_episodes, evaluation_seeds, env_parameter_settings = None):
    with torch.no_grad():
        eval_rewards = []
        env = gym.make(env_name)
        
        for param, value in env_parameter_settings.items():
            setattr(env.unwrapped, param, value)

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
                
                
                # Transform the state to the correct format and save it
                # to be used as previous state in the next time step.
                state = torch.from_numpy(state)
                state = state.unsqueeze(0).to(device)
                prev_state = state


                # Feed the state and adaptation module input into the agent network
                policy_output, value, policy_hidden_state = policy_net(state.float(), policy_hidden_state, adaptation_module_output)

                means, std_devs = policy_output
                
                # Get greedy action
                action = means
                

                state, r, done, _ = env.step(action[0].cpu().numpy())

                total_reward += r
            eval_rewards.append(total_reward)

        return eval_rewards




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
            # random_neuromod_output = torch.randn(num_neurons_policy).to(device)
            
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




gym.envs.registration.register(
    id='AdjustableBipedalWalker-v3',
    entry_point='learning_to_adapt.AdjustableBipedalWalker:AdjustableBipedalWalker',
    max_episode_steps=1600,
    reward_threshold=300,
)



training_ranges = [(0.925, 1.125), (0.9875, 1.025), (0.925, 1.125), (0.9875, 1.025), (0.875, 1.125), (0.925, 1.025), (0.95, 1.075), (0.875, 1.125), (0.875, 1.125), (0.975, 1.05), (0.925, 1.01875)]
validation_ranges = [[(0.85, 0.925), (1.125, 1.25)], [(0.975, 0.9875), (1.025, 1.05)], [(0.85, 0.925), (1.125, 1.25)], [(0.975, 0.9875), (1.025, 1.05)], [(0.75, 0.875), (1.125, 1.25)], [(0.85, 0.925), (1.025, 1.05)], [(0.9, 0.95), (1.075, 1.15)], [(0.75, 0.875), (1.125, 1.25)], [(0.75, 0.875), (1.125, 1.25)], [(0.95, 0.975), (1.05, 1.1)], [(0.85, 0.925), (1.01875, 1.0375)]]
testing_ranges = [[(0.7, 0.85), (1.25, 1.5)], [(0.95, 0.975), (1.05, 1.1)], [(0.7, 0.85), (1.25, 1.5)], [(0.95, 0.975), (1.05, 1.1)], [(0.5, 0.75), (1.25, 1.5)], [(0.7, 0.85), (1.05, 1.1)], [(0.8, 0.9), (1.15, 1.3)], [(0.5, 0.75), (1.25, 1.5)], [(0.5, 0.75), (1.25, 1.5)], [(0.9, 0.95), (1.1, 1.2)], [(0.7, 0.85), (1.0375, 1.075)]]

default_values = [8.0, 34.0, 8.0, 34.0, 2.5, 4.0, 6.0, 1.0, 1.0, 5.0, 160.0]
env_params = ['left_leg_w_unscaled', 'left_leg_h_unscaled', 'right_leg_w_unscaled', 'right_leg_h_unscaled', 'terrain_friction', 'speed_hip', 'speed_knee', 'left_leg_density', 'right_leg_density', 'hull_density', 'lidar_range_unscaled']


device = "cpu"
neuron_type = "BP"
if neuron_type == "BP":
    top_dir = "BP_A2C"
else:
    top_dir = "LTC_A2C"
mode = "neuromodulated"
adapt_mod_type = "StandardRNN"
num_neurons_policy = 96
num_neurons_adaptmod = 96
state_dims = 24
action_dims = 4
num_actions = 2

num_models = 10
seed = 5
env_name = "AdjustableBipedalWalker-v3"
n_evaluations = 1000

wiring = None




evaluation_seeds = np.load('learning_to_adapt/rstdp_cartpole/seeds/evaluation_seeds.npy')
arrays = [evaluation_seeds]

# Generate the new arrays and add them to the list
for i in range(1, 10):
    new_array = evaluation_seeds + i
    arrays.append(new_array)

# Concatenate all arrays together
evaluation_seeds = np.concatenate(arrays)



policy_weights_0 = torch.load(f'learning_to_adapt/results/{top_dir}/BP_a2c_result_4400000_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_neuromod_network_dims_11_256_128_96/checkpoint_BP_A2C_0.pt')
policy_weights_1 = torch.load(f'learning_to_adapt/results/{top_dir}/BP_a2c_result_4400001_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_neuromod_network_dims_11_256_128_96/checkpoint_BP_A2C_0.pt')
policy_weights_2 = torch.load(f'learning_to_adapt/results/{top_dir}/Continued_BP_a2c_result_4400002_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_neuromod_network_dims_11_256_128_96/checkpoint_BP_A2C_0.pt')
policy_weights_3 = torch.load(f'learning_to_adapt/results/{top_dir}/BP_a2c_result_4400003_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_neuromod_network_dims_11_256_128_96/checkpoint_BP_A2C_0.pt')
policy_weights_4 = torch.load(f'learning_to_adapt/results/{top_dir}/BP_a2c_result_4400004_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_neuromod_network_dims_11_256_128_96/checkpoint_BP_A2C_0.pt')
policy_weights_5 = torch.load(f'learning_to_adapt/results/{top_dir}/BP_a2c_result_4400005_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_neuromod_network_dims_11_256_128_96/checkpoint_BP_A2C_0.pt')
policy_weights_6 = torch.load(f'learning_to_adapt/results/{top_dir}/BP_a2c_result_4400006_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_neuromod_network_dims_11_256_128_96/checkpoint_BP_A2C_0.pt')
policy_weights_7 = torch.load(f'learning_to_adapt/results/{top_dir}/BP_a2c_result_4400007_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_neuromod_network_dims_11_256_128_96/checkpoint_BP_A2C_0.pt')
policy_weights_8 = torch.load(f'learning_to_adapt/results/{top_dir}/BP_a2c_result_4400008_2024718_learningrate_0.0001_numneurons_96_encoutact_tanh_neuromod_network_dims_11_256_128_96/checkpoint_BP_A2C_0.pt')
policy_weights_9 = torch.load(f'learning_to_adapt/results/{top_dir}/BP_a2c_result_3300005_2024628_learningrate_0.0001_numneurons_96_encoutact_tanh_neuromod_network_dims_11_256_128_96/checkpoint_BP_A2C_0.pt')
policy_weights = [policy_weights_0, policy_weights_1, policy_weights_2, policy_weights_3, policy_weights_4, policy_weights_5, policy_weights_6, policy_weights_7, policy_weights_8, policy_weights_9]


am_weights_0 = torch.load(f'learning_to_adapt/results/{top_dir}/adaptation_module/training_results/adaptation_module_StandardRNN_result_6000000_202485_BP_a2c_result_2169_202448_numneuronsadaptmod_96_lradaptmod_0.01_wdadaptmod_0.0/best_adaptation_module_loss_BP_A2C_0.pt')
am_weights_1 = torch.load(f'learning_to_adapt/results/{top_dir}/adaptation_module/training_results/adaptation_module_StandardRNN_result_6000001_202485_BP_a2c_result_2169_202448_numneuronsadaptmod_96_lradaptmod_0.01_wdadaptmod_0.0/best_adaptation_module_loss_BP_A2C_0.pt')
am_weights_2 = torch.load(f'learning_to_adapt/results/{top_dir}/adaptation_module/training_results/adaptation_module_StandardRNN_result_8000000_2024811_BP_a2c_result_2169_202448_numneuronsadaptmod_96_lradaptmod_0.01_wdadaptmod_0.0/best_adaptation_module_loss_BP_A2C_0.pt')
am_weights_3 = torch.load(f'learning_to_adapt/results/{top_dir}/adaptation_module/training_results/adaptation_module_StandardRNN_result_6000003_202485_BP_a2c_result_2169_202448_numneuronsadaptmod_96_lradaptmod_0.01_wdadaptmod_0.0/best_adaptation_module_loss_BP_A2C_0.pt')
am_weights_4 = torch.load(f'learning_to_adapt/results/{top_dir}/adaptation_module/training_results/adaptation_module_StandardRNN_result_6000004_202485_BP_a2c_result_2169_202448_numneuronsadaptmod_96_lradaptmod_0.01_wdadaptmod_0.0/best_adaptation_module_loss_BP_A2C_0.pt')
am_weights_5 = torch.load(f'learning_to_adapt/results/{top_dir}/adaptation_module/training_results/adaptation_module_StandardRNN_result_6000005_202485_BP_a2c_result_2169_202448_numneuronsadaptmod_96_lradaptmod_0.01_wdadaptmod_0.0/best_adaptation_module_loss_BP_A2C_0.pt')
am_weights_6 = torch.load(f'learning_to_adapt/results/{top_dir}/adaptation_module/training_results/adaptation_module_StandardRNN_result_6000006_202485_BP_a2c_result_2169_202448_numneuronsadaptmod_96_lradaptmod_0.01_wdadaptmod_0.0/best_adaptation_module_loss_BP_A2C_0.pt')
am_weights_7 = torch.load(f'learning_to_adapt/results/{top_dir}/adaptation_module/training_results/adaptation_module_StandardRNN_result_6000007_202485_BP_a2c_result_2169_202448_numneuronsadaptmod_96_lradaptmod_0.01_wdadaptmod_0.0/best_adaptation_module_loss_BP_A2C_0.pt')
am_weights_8 = torch.load(f'learning_to_adapt/results/{top_dir}/adaptation_module/training_results/adaptation_module_StandardRNN_result_6000008_202485_BP_a2c_result_2169_202448_numneuronsadaptmod_96_lradaptmod_0.01_wdadaptmod_0.0/best_adaptation_module_loss_BP_A2C_0.pt')
am_weights_9 = torch.load(f'learning_to_adapt/results/{top_dir}/adaptation_module/training_results/adaptation_module_StandardRNN_result_5000000_2024721_BP_a2c_result_2169_202448_numneuronsadaptmod_96_lradaptmod_0.01_wdadaptmod_0.0/best_adaptation_module_loss_BP_A2C_0.pt')
am_weights = [am_weights_0, am_weights_1, am_weights_2, am_weights_3, am_weights_4, am_weights_5, am_weights_6, am_weights_7, am_weights_8, am_weights_9]



eraser = '\b \b'

with torch.no_grad():
    training_rewards = []
    validation_rewards = []
    testing_rewards = []
    for i, (pw, amw) in enumerate(zip(policy_weights, am_weights)):
        print('Run {:02d} ...'.format(i), end='')
        if neuron_type == "LTC":
            raise NotImplementedError
        elif neuron_type == "CfC":
            if adapt_mod_type == 'StandardRNN':
                adaptation_module = StandardRNN(state_dims + action_dims, num_neurons_adaptmod, num_neurons_policy, seed = seed)
            else:
                raise NotImplementedError

            policy_net = CfC_Network(state_dims, num_neurons_policy, action_dims, seed, mode = mode, wiring = wiring, continuous_actions=True).to(device)
            w_policy = OrderedDict((k.split('.', 1)[-1], v) for k, v in pw.items() if 'neuromod' not in k)
            w_policy['cfc_model.rnn_cell.tau_system'] = torch.reshape(w_policy['cfc_model.rnn_cell.tau_system'], (num_neurons_policy,))
            policy_net.load_state_dict(w_policy)

            adaptation_module.load_state_dict(amw)
        elif neuron_type == "BP":
            if adapt_mod_type == 'StandardRNN':
                adaptation_module = StandardRNN(state_dims + action_dims, num_neurons_adaptmod, num_neurons_policy, seed = seed)
            else:
                raise NotImplementedError
            
            policy_net = BP_RNetwork(state_dims, num_neurons_policy, action_dims, seed, external_neuromodulation = True, continuous_actions=True).to(device)
            w_policy = OrderedDict((k.split('.', 1)[-1], v) for k, v in pw.items() if 'neuromod' not in k)
            policy_net.load_state_dict(w_policy)

            adaptation_module.load_state_dict(amw)

        

        
        # Training rewards ---------------------------
        rewards_sum = 0
        all_rewards = []
        for i in range(n_evaluations):
            np.random.seed(evaluation_seeds[i])
            random.seed(evaluation_seeds[i])
            random_env_param_settings = {}
            for param, fract_range, default_val in zip(env_params, training_ranges, default_values):
                random_env_param_settings[param] = np.random.uniform(fract_range[0], fract_range[1]) * default_val

            r = np.mean(evaluate_BW(policy_net, adaptation_module, env_name, 1, evaluation_seeds[i:], random_env_param_settings))
            rewards_sum += r
            all_rewards.append(r)

        rewards_sum /= n_evaluations
        training_rewards.append(rewards_sum)
        print(eraser*3 + '-> Avg training reward: {:7.2f}'.format(rewards_sum))
        print(np.std(all_rewards))
        all_rewards_train = all_rewards

        # Validation rewards ---------------------------
        rewards_sum = 0
        all_rewards = []
        for i in range(n_evaluations):
            np.random.seed(evaluation_seeds[i])
            random.seed(evaluation_seeds[i])
            random_env_param_settings = {}
            for param, ranges, default_val in zip(env_params, validation_ranges, default_values):
                val_range = random.choice(ranges)
                random_env_param_settings[param] = np.random.uniform(val_range[0], val_range[1]) * default_val

            r = np.mean(evaluate_BW(policy_net, adaptation_module, env_name, 1, evaluation_seeds[i:], random_env_param_settings))
            rewards_sum += r
            all_rewards.append(r)

        rewards_sum /= n_evaluations
        validation_rewards.append(rewards_sum)
        print(eraser*3 + '-> Avg validation reward: {:7.2f}'.format(rewards_sum))
        print(np.std(all_rewards))
        all_rewards_val = all_rewards


        # Testing rewards ---------------------------
        rewards_sum = 0
        all_rewards = []
        for i in range(n_evaluations):
            np.random.seed(evaluation_seeds[i])
            random.seed(evaluation_seeds[i])
            random_env_param_settings = {}
            for param, ranges, default_val in zip(env_params, testing_ranges, default_values):
                test_range = random.choice(ranges)
                random_env_param_settings[param] = np.random.uniform(test_range[0], test_range[1]) * default_val

            r = np.mean(evaluate_BW(policy_net, adaptation_module, env_name, 1, evaluation_seeds[i:], random_env_param_settings))
            rewards_sum += r
            all_rewards.append(r)

        rewards_sum /= n_evaluations
        testing_rewards.append(rewards_sum)
        print(eraser*3 + '-> Avg testing reward: {:7.2f}'.format(rewards_sum))
        print(np.std(all_rewards))
        all_rewards_test = all_rewards

    with open(f"learning_to_adapt/results/{top_dir}/train_val_test.txt", "w") as f:
        f.write(f"All training rewards: {training_rewards}\n")
        print(f"All training rewards: {training_rewards}")
        print(f"Alll training rewards: {all_rewards_train}")
        f.write(f"Mean avg training reward: {np.mean(training_rewards)} +/- {np.std(training_rewards)}\n")
        print(f"Mean avg training reward: {np.mean(training_rewards)} +/- {np.std(training_rewards)}")
        f.write(f"All validation rewards: {validation_rewards}\n")
        print(f"All validation rewards: {validation_rewards}")
        print(f"All validation rewards: {all_rewards_val}")
        f.write(f"Mean avg validation reward: {np.mean(validation_rewards)} +/- {np.std(validation_rewards)}\n")
        print(f"Mean avg validation reward: {np.mean(validation_rewards)} +/- {np.std(validation_rewards)}")
        f.write(f"All testing rewards: {testing_rewards}\n")
        print(f"All testing rewards: {testing_rewards}")
        print(f"All testing rewards: {all_rewards_test}")
        f.write(f"Mean avg testing reward: {np.mean(testing_rewards)} +/- {np.std(testing_rewards)}\n")
        print(f"Mean avg testing reward: {np.mean(testing_rewards)} +/- {np.std(testing_rewards)}")




            
            

            


