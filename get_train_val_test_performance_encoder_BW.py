from datetime import date
import random
import gym
import numpy as np
import torch
from learning_to_adapt.LTC_A2C import LTC_Network, CfC_Network
from learning_to_adapt.Neuromodulated_Agent import NeuromodulatedAgent
from learning_to_adapt.backpropamine_A2C import BP_RNetwork
from neuromodulated_ncps.ncps.wirings import AutoNCP



def evaluate_BW(agent_net, env_name, num_episodes, evaluation_seeds, env_parameter_settings = None):
    with torch.no_grad():
        eval_rewards = []
        env = gym.make(env_name)

        for param, value in env_parameter_settings.items():
            setattr(env.unwrapped, param, value)
            
        for i_episode in range(num_episodes):
            policy_hidden_states = None
            
            env.seed(int(evaluation_seeds[i_episode]))
            
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                state = torch.from_numpy(state)
                state = state.unsqueeze(0).to(device)
                privileged_info = get_privileged_info(env_parameter_settings).unsqueeze(0).to(device)
                policy_outputs, values, policy_hidden_states = agent_net(state.float(), privileged_info, policy_hidden_states)

                
                means, std_devs = policy_outputs
                
                # Get greedy action
                action = means
                

                state, r, done, _ = env.step(action[0].cpu().numpy())

                total_reward += r
            eval_rewards.append(total_reward)

        return eval_rewards



def get_privileged_info(env_parameter_settings):
    params_values = [val for val in env_parameter_settings.values()]
    params_tensor = torch.tensor(params_values, dtype=torch.float32)

    return params_tensor





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
    model_signifier = "BP"
else:
    top_dir = "LTC_A2C"
    model_signifier = "CfC"
mode = "neuromodulated"
num_neurons_policy = 96
neuromod_network_dims = [11, 256, 128, num_neurons_policy]
batch_dir = ""
encoder_func = "tanh"
num_models = 1
seed = 5
env_name = "AdjustableBipedalWalker-v3"
n_evaluations = 1000

wiring = None

if encoder_func == "tanh":
    encoder_hidden_activation = torch.nn.Tanh()
    encoder_output_activation = torch.nn.Tanh()
elif encoder_func == "relu":
    encoder_hidden_activation = torch.nn.ReLU()
    encoder_output_activation = torch.nn.ReLU()



evaluation_seeds = np.load('learning_to_adapt/rstdp_cartpole/seeds/evaluation_seeds.npy')
arrays = [evaluation_seeds]

# Generate the new arrays and add them to the list
for i in range(1, 10):
    new_array = evaluation_seeds + i
    arrays.append(new_array)

# Concatenate all arrays together
evaluation_seeds = np.concatenate(arrays)

result_dir = ""


policy_weights_0 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_0.pt', map_location=torch.device(device))
policy_weights_1 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_0.pt', map_location=torch.device(device))
policy_weights_2 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_0.pt', map_location=torch.device(device))
policy_weights_3 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_0.pt', map_location=torch.device(device))
policy_weights_4 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_0.pt', map_location=torch.device(device))
policy_weights_5 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_0.pt', map_location=torch.device(device))
policy_weights_6 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_0.pt', map_location=torch.device(device))
policy_weights_7 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_0.pt', map_location=torch.device(device))
policy_weights_8 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_0.pt', map_location=torch.device(device))
policy_weights_9 = torch.load(f'Master_Thesis_Code/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_0.pt', map_location=torch.device(device))
policy_weights = [policy_weights_0, policy_weights_1, policy_weights_2, policy_weights_3, policy_weights_4, policy_weights_5, policy_weights_6, policy_weights_7, policy_weights_8, policy_weights_9]

eraser = '\b \b'


training_rewards = []
validation_rewards = []
testing_rewards = []
with torch.no_grad():
    for i, w in enumerate(policy_weights):
        print('Run {:02d} ...'.format(i), end='')
        if neuron_type == "LTC":
            agent_net = LTC_Network(24, num_neurons_policy, 4, seed, wiring = wiring).to(device)
        elif neuron_type == "CfC":
            layer_list = []
            for dim in range(len(neuromod_network_dims) - 1):
                layer_list.append(torch.nn.Linear(neuromod_network_dims[dim], neuromod_network_dims[dim + 1]))
                if dim < len(neuromod_network_dims)-2:
                    layer_list.append(encoder_hidden_activation)
                else:
                    layer_list.append(encoder_output_activation)
            encoder = torch.nn.Sequential(*layer_list)

            policy_net = CfC_Network(24, num_neurons_policy, 4, seed, mode = mode, wiring = wiring, continuous_actions=True).to(device)

            agent_net = NeuromodulatedAgent(policy_net, encoder, policy_has_hidden_state=True).to(device)
            w['policy_net.cfc_model.rnn_cell.tau_system'] = torch.reshape(w['policy_net.cfc_model.rnn_cell.tau_system'], (num_neurons_policy,))
        elif neuron_type == "BP":
            layer_list = []
            for dim in range(len(neuromod_network_dims) - 1):
                layer_list.append(torch.nn.Linear(neuromod_network_dims[dim], neuromod_network_dims[dim + 1]))
                if dim < len(neuromod_network_dims)-2:
                    layer_list.append(encoder_hidden_activation)
                else:
                    layer_list.append(encoder_output_activation)
            encoder = torch.nn.Sequential(*layer_list)

            policy_net = BP_RNetwork(24, num_neurons_policy, 4, seed, external_neuromodulation = True, continuous_actions=True).to(device)

            agent_net = NeuromodulatedAgent(policy_net, encoder, policy_has_hidden_state=True).to(device)

        agent_net.load_state_dict(w)

        
        # Training rewards ---------------------------
        rewards_sum = 0
        all_rewards = []
        for i in range(n_evaluations):
            np.random.seed(evaluation_seeds[i])
            random.seed(evaluation_seeds[i])
            random_env_param_settings = {}
            for param, fract_range, default_val in zip(env_params, training_ranges, default_values):
                random_env_param_settings[param] = np.random.uniform(fract_range[0], fract_range[1]) * default_val

            r = np.mean(evaluate_BW(agent_net, env_name, 1, evaluation_seeds[i:], random_env_param_settings))
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

            r = np.mean(evaluate_BW(agent_net, env_name, 1, evaluation_seeds[i:], random_env_param_settings))
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

            r = np.mean(evaluate_BW(agent_net, env_name, 1, evaluation_seeds[i:], random_env_param_settings))
            rewards_sum += r
            all_rewards.append(r)

        rewards_sum /= n_evaluations
        testing_rewards.append(rewards_sum)
        print(eraser*3 + '-> Avg testing reward: {:7.2f}'.format(rewards_sum))
        print(np.std(all_rewards))
        all_rewards_test = all_rewards







    with open(f"learning_to_adapt/results/{top_dir}/{batch_dir}/train_val_test.txt", "w") as f:
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



        
        

        


