from datetime import date
import random
import gym
import numpy as np
import torch
from Master_Thesis_Code.LTC_A2C import LTC_Network, CfC_Network
from Master_Thesis_Code.Neuromodulated_Agent import NeuromodulatedAgent
from Master_Thesis_Code.backpropamine_A2C import BP_RNetwork, Standard_RNetwork, Standard_FFNetwork
from neuromodulated_ncps.ncps.wirings import AutoNCP
from Master_Thesis_Code.BP_A2C.BP_A2C_agent import evaluate_BP_agent_all_params



def evaluate_agent_all_params(agent_net, env_name, num_episodes, evaluation_seeds, pole_length_modifier, pole_mass_modifier, force_mag_modifier):
    eval_rewards = []
    env = gym.make(env_name)
    env.unwrapped.length *= pole_length_modifier
    env.unwrapped.masspole *= pole_mass_modifier
    env.unwrapped.force_mag *= force_mag_modifier
        
    for i_episode in range(num_episodes):
        hidden_state = None
        
        env.seed(int(evaluation_seeds[i_episode]))
        
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).to(device) #This as well?
            policy_output, value, hidden_state = agent_net(state.float(), hidden_state)
            
            policy_dist = torch.softmax(policy_output, dim = 1)
            
            action = torch.argmax(policy_dist)
            

            state, r, done, _ = env.step(action.item())

            total_reward += r
        eval_rewards.append(total_reward)

    return eval_rewards
    

training_ranges = [(0.775, 5.75), (1.0, 2.0), (0.8, 2.25)]
validation_ranges = [[(0.55, 0.775), (5.75, 10.5)], [(2.0, 3.0)], [(0.6, 0.8), (2.25, 3.5)]]
testing_ranges = [[(0.1, 0.55), (10.5, 20.0)], [(5.0, 13.0)], [(0.2, 0.6), (3.5, 6.0)]]



device = "cpu"
neuron_type = "BP"
if neuron_type == "BP" or neuron_type == "StandardRNN" or neuron_type == "StandardMLP":
    top_dir = "BP_A2C"
    if neuron_type == "StandardRNN":
        model_signifier = "Standard_RNN"
    elif neuron_type == "BP":
        model_signifier = "BP_RNN"
    elif neuron_type == "StandardMLP":
        model_signifier = "Standard_MLP"
else:
    top_dir = "LTC_A2C"
    if neuron_type == "CfC":
        model_signifier = "CfC"
    elif neuron_type == "LTC":
        model_signifier = "LTC"
mode = "pure"
num_neurons_policy = 32
batch_dir = ""

num_models = 10
seed = 5
env_name = "CartPole-v0"
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


result_dir = "BP_RNN_a2c_result_128_2024828_plm_0.33_pmm_9.0_fmm_4.75_entropycoef_0.0001_valuepredcoef_0.01_learningrate_0.001_numtrainepisodes_20000_selectionmethod_true_range_eval_all_params_trainingmethod_quarter_range_numneurons_32"



policy_weights_0 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_0.pt', map_location=torch.device(device))
policy_weights_1 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_1.pt', map_location=torch.device(device))
policy_weights_2 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_2.pt', map_location=torch.device(device))
policy_weights_3 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_3.pt', map_location=torch.device(device))
policy_weights_4 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_4.pt', map_location=torch.device(device))
policy_weights_5 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_5.pt', map_location=torch.device(device))
policy_weights_6 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_6.pt', map_location=torch.device(device))
policy_weights_7 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_7.pt', map_location=torch.device(device))
policy_weights_8 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_8.pt', map_location=torch.device(device))
policy_weights_9 = torch.load(f'learning_to_adapt/results/{top_dir}/{batch_dir}/{result_dir}/checkpoint_{model_signifier}_A2C_9.pt', map_location=torch.device(device))
policy_weights = [policy_weights_0, policy_weights_1, policy_weights_2, policy_weights_3, policy_weights_4, policy_weights_5, policy_weights_6, policy_weights_7, policy_weights_8, policy_weights_9]

eraser = '\b \b'


training_rewards = []
validation_rewards = []
testing_rewards = []
with torch.no_grad():
    for i, w in enumerate(policy_weights):
        print('Run {:02d} ...'.format(i), end='')
        if neuron_type == "LTC":
            agent_net = LTC_Network(4, num_neurons_policy, 2, seed, wiring = wiring).to(device)
        elif neuron_type == "CfC":
            agent_net = CfC_Network(4, num_neurons_policy, 2, seed, mode = mode, wiring = wiring).to(device)
            w['cfc_model.rnn_cell.tau_system'] = torch.reshape(w['cfc_model.rnn_cell.tau_system'], (num_neurons_policy,))
        elif neuron_type == "BP":
            agent_net = BP_RNetwork(4, num_neurons_policy, 2, seed).to(device)
        elif neuron_type == "StandardRNN":
            agent_net = Standard_RNetwork(4, num_neurons_policy, 2, seed).to(device)
        elif neuron_type == "StandardMLP":
            agent_net = Standard_FFNetwork(4, num_neurons_policy, num_neurons_policy, 2, seed).to(device)

        agent_net.load_state_dict(w)

        
        # Training rewards ---------------------------
        rewards_sum = 0
        for j in range(n_evaluations):
            np.random.seed(evaluation_seeds[j])
            
            pole_length_mod = np.random.uniform(training_ranges[0][0], training_ranges[0][1])
            pole_mass_mod = np.random.uniform(training_ranges[1][0], training_ranges[1][1])
            force_mag_mod = np.random.uniform(training_ranges[2][0], training_ranges[2][1])

            if neuron_type == "BP" or neuron_type == "StandardRNN" or neuron_type == "StandardMLP":
                rewards_sum += np.mean(evaluate_BP_agent_all_params(agent_net, env_name, 1, evaluation_seeds[j:], pole_length_mod, pole_mass_mod, force_mag_mod))
            else:
                rewards_sum += np.mean(evaluate_agent_all_params(agent_net, env_name, 1, evaluation_seeds[j:], pole_length_mod, pole_mass_mod, force_mag_mod))
        
        rewards_sum /= n_evaluations
        training_rewards.append(rewards_sum)
        print(eraser*3 + '-> Avg training reward: {:7.2f}'.format(rewards_sum))


        # Validation rewards ---------------------------
        rewards_sum = 0
        for j in range(n_evaluations):
            np.random.seed(evaluation_seeds[j])
            random.seed(evaluation_seeds[j])
            
            pole_length_range = random.choice(validation_ranges[0])
            pole_length_mod = np.random.uniform(pole_length_range[0], pole_length_range[1])
            pole_mass_mod = np.random.uniform(validation_ranges[1][0][0], validation_ranges[1][0][1])
            force_mag_range = random.choice(validation_ranges[2])
            force_mag_mod = np.random.uniform(force_mag_range[0], force_mag_range[1])
            
            if neuron_type == "BP" or neuron_type == "StandardRNN" or neuron_type == "StandardMLP":
                rewards_sum += np.mean(evaluate_BP_agent_all_params(agent_net, env_name, 1, evaluation_seeds[j:], pole_length_mod, pole_mass_mod, force_mag_mod))
            else:
                rewards_sum += np.mean(evaluate_agent_all_params(agent_net, env_name, 1, evaluation_seeds[j:], pole_length_mod, pole_mass_mod, force_mag_mod))

        rewards_sum /= n_evaluations
        validation_rewards.append(rewards_sum)
        print(eraser*3 + '-> Avg validation reward: {:7.2f}'.format(rewards_sum))


        # Testing rewards ---------------------------
        rewards_sum = 0
        for j in range(n_evaluations):
            np.random.seed(evaluation_seeds[j])
            random.seed(evaluation_seeds[j])
            
            pole_length_range = random.choice(testing_ranges[0])
            pole_length_mod = np.random.uniform(pole_length_range[0], pole_length_range[1])
            pole_mass_mod = np.random.uniform(testing_ranges[1][0][0], testing_ranges[1][0][1])
            force_mag_range = random.choice(testing_ranges[2])
            force_mag_mod = np.random.uniform(force_mag_range[0], force_mag_range[1])

            if neuron_type == "BP" or neuron_type == "StandardRNN" or neuron_type == "StandardMLP":
                rewards_sum += np.mean(evaluate_BP_agent_all_params(agent_net, env_name, 1, evaluation_seeds[j:], pole_length_mod, pole_mass_mod, force_mag_mod))
            else:
                rewards_sum += np.mean(evaluate_agent_all_params(agent_net, env_name, 1, evaluation_seeds[j:], pole_length_mod, pole_mass_mod, force_mag_mod))

        rewards_sum /= n_evaluations
        testing_rewards.append(rewards_sum)
        print(eraser*3 + '-> Avg testing reward: {:7.2f}'.format(rewards_sum))

    

    with open(f"learning_to_adapt/results/{top_dir}/{batch_dir}/{result_dir}/train_val_test.txt", "w") as f:
        f.write(f"All training rewards: {training_rewards}\n")
        print(f"All training rewards: {training_rewards}")
        f.write(f"Mean avg training reward: {np.mean(training_rewards)} +/- {np.std(training_rewards)}\n")
        print(f"Mean avg training reward: {np.mean(training_rewards)} +/- {np.std(training_rewards)}")
        f.write(f"All validation rewards: {validation_rewards}\n")
        print(f"All validation rewards: {validation_rewards}")
        f.write(f"Mean avg validation reward: {np.mean(validation_rewards)} +/- {np.std(validation_rewards)}\n")
        print(f"Mean avg validation reward: {np.mean(validation_rewards)} +/- {np.std(validation_rewards)}")
        f.write(f"All testing rewards: {testing_rewards}\n")
        print(f"All testing rewards: {testing_rewards}")
        f.write(f"Mean avg testing reward: {np.mean(testing_rewards)} +/- {np.std(testing_rewards)}\n")
        print(f"Mean avg testing reward: {np.mean(testing_rewards)} +/- {np.std(testing_rewards)}")



        
        

        


