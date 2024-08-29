import argparse
import time
import gym
import torch
import numpy as np
import random
import os
from datetime import date
from learning_to_adapt.backpropamine_A2C import BP_RNetwork, Standard_RNetwork, Standard_FFNetwork
from learning_to_adapt.BP_A2C.BP_A2C_agent import A2C_Agent


from Master_Thesis_Code.modifiable_async_vector_env import ModifiableAsyncVectorEnv


parser = argparse.ArgumentParser(description='Train an A2C agent on the BipedalWalker environment')
parser.add_argument('--num_neurons', type=int, default=96, help='Number of neurons in the hidden layer')
parser.add_argument('--network_type', type=str, default='Standard_RNN', help='Type of network to use')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the agent')
parser.add_argument('--num_models', type=int, default=1, help='Number of models to train')
parser.add_argument('--selection_method', type=str, default='range', help='Method to use for selecting the best model')
parser.add_argument('--training_method', type=str, default = "training_range", help='Method to train the agent')
parser.add_argument('--result_id', type=int, default=-1, help='ID to use for the results directory')
parser.add_argument('--env_name', type=str, default='AdjustableBipedalWalker-v3', help='Name of the environment to use')
parser.add_argument('--input_dims', type=int, default=24, help='Number of input dimensions to the network')
parser.add_argument('--output_dims', type=int, default=4, help='Number of output dimensions to the network')
parser.add_argument('--continuous_actions', type=bool, default=True, help='Whether the environment has continuous actions')
parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient for the agent')
parser.add_argument('--value_pred_coef', type=float, default=0.01, help='Value prediction coefficient for the agent')
parser.add_argument('--num_training_episodes', type=int, default=1000000, help='Number of training episodes to run')
parser.add_argument('--num_evaluation_episodes', type=int, default=50, help='Number of evaluation episodes to run')
parser.add_argument('--training_episodes_per_section', type=int, default=1000, help='Number of training episodes to run per section')
parser.add_argument('--evaluate_every', type=int, default=50, help='How often to evaluate the agent')
parser.add_argument('--batch_size', type=int, default=5, help='Batch size to use for training')
parser.add_argument('--num_parallel_envs', type=int, default=5, help='Number of parallel environments to use')
parser.add_argument('--additional_seed', type=int, default=0, help='Additional number to add to the seed')
parser.add_argument('--gammaR', type=float, default=0.99, help='Discount factor for the agent')
parser.add_argument('--max_grad_norm', type=float, default=10, help='Maximum gradient norm for the agent')


gym.envs.registration.register(
    id='AdjustableBipedalWalker-v3',
    entry_point='learning_to_adapt.AdjustableBipedalWalker:AdjustableBipedalWalker',
    max_episode_steps=1600,
    reward_threshold=300,
)


args = parser.parse_args()
learning_rate = args.learning_rate
num_neurons = args.num_neurons
network_type = args.network_type
num_models = args.num_models
selection_method = args.selection_method
training_method = args.training_method
result_id = args.result_id
env_name = args.env_name
input_dims = args.input_dims
output_dims = args.output_dims
continuous_actions = args.continuous_actions
entropy_coef = args.entropy_coef
value_pred_coef = args.value_pred_coef
num_training_episodes = args.num_training_episodes
evaluate_every = args.evaluate_every
training_eps_per_section = args.training_episodes_per_section
num_evaluation_episodes = args.num_evaluation_episodes
batch_size = args.batch_size
num_parallel_envs = args.num_parallel_envs
additional_seed = args.additional_seed
gammaR = args.gammaR
max_grad_norm = args.max_grad_norm

device = torch.device("cpu")

assert evaluate_every % batch_size == 0



max_steps = 1600




evaluation_seeds = np.load('learning_to_adapt/rstdp_cartpole/seeds/evaluation_seeds.npy')
training_seeds = np.load('learning_to_adapt/rstdp_cartpole/seeds/training_seeds.npy')
max_reward = 300


if training_method == "training_range":
    randomization_params = [(0.925, 1.125), (0.9875, 1.025), (0.925, 1.125), (0.9875, 1.025), (0.875, 1.125), (0.925, 1.025), (0.95, 1.075), (0.875, 1.125), (0.875, 1.125), (0.975, 1.05), (0.925, 1.01875)]
else:
    randomization_params = None

if result_id == -1:
    # Create Results Directory
    dirs = os.listdir('learning_to_adapt/results/BP_A2C/')
    if not any('a2c_result' in d for d in dirs):
        result_id = 1
    else:
        results = [d for d in dirs if 'a2c_result' in d]
        result_id = len(results) + 1

# Get today's date and add it to the results directory
d = date.today()
result_dir = f'learning_to_adapt/results/BP_A2C/{network_type}_a2c_result_' + str(result_id) + "_{}_entropycoef_{}_valuepredcoef_{}_\
learningrate_{}_numtrainepisodes_{}_selectionmethod_{}_trainingmethod_{}_numneurons_{}".format(
    str(d.year) + str(d.month) + str(d.day), entropy_coef, value_pred_coef,
    learning_rate, num_training_episodes, selection_method, training_method, num_neurons)


os.mkdir(result_dir)
print('Created Directory {} to store the results in'.format(result_dir))




all_training_losses = []
all_training_total_rewards = []
all_validation_losses = []
all_validation_total_rewards = []
best_average_after_all = []
best_average_all = []
start_time = time.time()
vec_env = ModifiableAsyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_parallel_envs)])
for i_run in range(num_models):
    print("Run # {}".format(i_run))
    seed = int(training_seeds[i_run]+additional_seed)
    
    torch.manual_seed(seed)
    random.seed(seed)

    if network_type == 'BP_RNN':
        agent_net = BP_RNetwork(input_dims, num_neurons, output_dims, seed, continuous_actions=continuous_actions).to(device)
    elif network_type == 'Standard_RNN':
        agent_net = Standard_RNetwork(input_dims, num_neurons, output_dims, seed, continuous_actions=continuous_actions)
    elif network_type == 'Standard_MLP':
        agent_net = Standard_FFNetwork(input_dims, num_neurons, num_neurons, output_dims, seed)
    else:
        raise NotImplementedError("Network type not recognized")
    
    optimizer = torch.optim.Adam(agent_net.parameters(), lr = learning_rate)
    agent = A2C_Agent(env_name, seed, agent_net, entropy_coef, value_pred_coef, gammaR,
                      max_grad_norm, max_steps, batch_size, training_eps_per_section, optimizer, 
                      i_run, result_dir, selection_method, num_evaluation_episodes, evaluation_seeds, max_reward, evaluate_every, network_type, continuous_actions)

    for section in range(0, int(num_training_episodes/training_eps_per_section)):
        print(f"Section {section+1} out of {int(num_training_episodes/training_eps_per_section)} sections")

        if section == 0:
            smoothed_scores, scores, best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses = agent.train_agent_continuous_vectorized_v3(vec_env, training_eps_per_section, section, randomization_params = randomization_params, num_parallel_envs=num_parallel_envs)
        else:
            smoothed_scores, scores, best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses = agent.train_agent_continuous_vectorized_v3(vec_env, training_eps_per_section, section, randomization_params = randomization_params, best_average=best_average_all[i_run], best_average_after=best_average_after_all[i_run], num_parallel_envs=num_parallel_envs)
        
        
        
        
        if section == 0:
            best_average_after_all.append(best_average_after)
            best_average_all.append(best_average)
            all_training_losses.append([tensor.cpu().numpy() for tensor in training_losses])
            all_training_total_rewards.append(training_total_rewards)
            all_validation_losses.append([tensor.cpu().numpy() for tensor in validation_losses])
            all_validation_total_rewards.append(validation_total_rewards)
        else:
            if best_average > best_average_all[i_run]:
                best_average_after_all[i_run] = best_average_after
                best_average_all[i_run] = best_average
            all_training_losses[i_run] = np.concatenate((all_training_losses[i_run], [tensor.cpu().numpy() for tensor in training_losses]))
            all_training_total_rewards[i_run] = np.concatenate((all_training_total_rewards[i_run], training_total_rewards))
            all_validation_losses[i_run] = np.concatenate((all_validation_losses[i_run], [tensor.cpu().numpy() for tensor in validation_losses]))
            all_validation_total_rewards[i_run] = np.concatenate((all_validation_total_rewards[i_run], validation_total_rewards))

        np.save(f"{result_dir}/all_training_losses_{i_run}.npy", all_training_losses[i_run])
        np.save(f"{result_dir}/all_training_total_rewards_{i_run}.npy", all_training_total_rewards[i_run])
        np.save(f"{result_dir}/all_validation_losses_{i_run}.npy", all_validation_losses[i_run])
        np.save(f"{result_dir}/all_validation_total_rewards_{i_run}.npy", all_validation_total_rewards[i_run])



        with open(f"{result_dir}/best_average_after.txt", 'w') as f:
            for i, best_episode in enumerate(best_average_after_all):
                if best_average_all[i] == max_reward:
                    f.write(f"{i}: {best_average_all[i]} after {best_episode} (total trained: {best_episode})\n")
                else:
                    if i == i_run:
                        f.write(f"{i}: {best_average_all[i]} after {best_episode} (total trained: {(section+1)*training_eps_per_section})\n")
                    else:
                        f.write(f"{i}: {best_average_all[i]} after {best_episode} (total trained: {num_training_episodes})\n")

            f.write(f"Average training episodes: {np.mean(best_average_after_all)}, std dev: {np.std(best_average_after_all)}\n")
            f.write(f"Mean average performance: {np.mean(best_average_all)}, std dev: {np.std(best_average_all)}")

        if best_average_all[i_run] >= max_reward:
            break


    print(f"Best average after {best_average_after_all[i_run]} episodes: {best_average_all[i_run]}")
print(f"Training took {time.time() - start_time} seconds")