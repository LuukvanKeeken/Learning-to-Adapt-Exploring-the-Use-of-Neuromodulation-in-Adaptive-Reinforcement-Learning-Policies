import argparse
from datetime import date
import os
import random
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import deque
import torch
from learning_to_adapt.LTC_A2C import LTC_Network, CfC_Network
from learning_to_adapt.modifiable_async_vector_env import ModifiableAsyncVectorEnv
from neuromodulated_ncps.ncps.wirings import AutoNCP
from torch.distributions import Categorical
from neuromodulated_ncps.ncps.wirings import Wiring


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_BW(agent_net, env_name, num_episodes, evaluation_seeds, env_parameter_settings = None):
    with torch.no_grad():
        eval_rewards = []
        env = gym.make(env_name)

        for param, value in env_parameter_settings.items():
            setattr(env.unwrapped, param, value)
            
        for i_episode in range(num_episodes):
            hidden_state = None
            
            env.seed(int(evaluation_seeds[i_episode]))
            
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                state = torch.from_numpy(state)
                state = state.unsqueeze(0).to(device) #This as well?
                policy_output, value, hidden_sate = agent_net.forward(state.float(), hidden_state)
                
                means, std_devs = policy_output
                
                # Get greedy action
                action = means
                

                state, r, done, _ = env.step(action[0].cpu().numpy())

                total_reward += r
            eval_rewards.append(total_reward)

        return eval_rewards



def get_random_env_paramvals_BW(randomization_params, batch_size = 1):

    default_values = [8.0, 34.0, 8.0, 34.0, 2.5, 4.0, 6.0, 1.0, 1.0, 5.0, 160.0]
    param_names = ['left_leg_w_unscaled', 'left_leg_h_unscaled', 'right_leg_w_unscaled', 'right_leg_h_unscaled', 'terrain_friction', 'speed_hip', 'speed_knee', 'left_leg_density', 'right_leg_density', 'hull_density', 'lidar_range_unscaled']
    new_params = [{name: None for name in param_names} for _ in range(batch_size)]
    
    for i in range(len(default_values)):
        if isinstance(randomization_params[i], float):
            low = default_values[i] - default_values[i] * randomization_params[i]
            high = default_values[i] + default_values[i] * randomization_params[i]
        elif isinstance(randomization_params[i], tuple):
            low = default_values[i]*randomization_params[i][0]
            high = default_values[i]*randomization_params[i][1]
            
        sampled_values = np.random.uniform(low, high, batch_size)

        for j in range(batch_size):
            new_params[j][param_names[i]] = sampled_values[j]

    return new_params


def train_agent_batched(vec_env, training_eps_per_section, section, num_parallel_envs, batch_size, max_grad_norm, gamma, best_average = -np.inf, best_average_after = np.inf):
    
    
    training_total_rewards = []
    training_losses = []
    validation_total_rewards = []
    validation_losses = []

    eps_trained = 1 + section*training_eps_per_section
    end_of_section = (section+1)*training_eps_per_section

    while eps_trained <= end_of_section:
        log_probs_batch = []
        values_batch = []
        rewards_batch = []
        entropies_batch = []

        running_log_probs = [[] for _ in range(num_parallel_envs)]
        running_values = [[] for _ in range(num_parallel_envs)]
        running_rewards = [[] for _ in range(num_parallel_envs)]
        running_entropies = [[] for _ in range(num_parallel_envs)]

        hidden_states = None

        if randomization_params:
            randomized_env_params = get_random_env_paramvals_BW(randomization_params, num_parallel_envs)
            vec_env.set_env_params(randomized_env_params)

        states = vec_env.reset()
        while len(log_probs_batch) < batch_size:
            states = torch.from_numpy(states).to(device)
            policy_outputs, values, hidden_states = agent_net(states, hidden_states)
            mus, sigmas = policy_outputs[0], policy_outputs[1]
            sigmas = torch.diag_embed(sigmas)
            dists = torch.distributions.MultivariateNormal(mus, sigmas)
            actions = dists.sample()
            log_probs = dists.log_prob(actions)
            entropies = dists.entropy()
            states,rewards, dones, _ = vec_env.step(actions.cpu().numpy())

            for i, (state, reward, done, log_prob, value, entropy) in enumerate(zip(states, rewards, dones, log_probs, values, entropies)):
                running_log_probs[i].append(log_prob.unsqueeze(0))
                running_values[i].append(value)
                running_rewards[i].append(reward)
                running_entropies[i].append(entropy.unsqueeze(0))

                if done:
                    training_total_rewards.append(np.sum(running_rewards[i]))
                    log_probs_batch.append(running_log_probs[i])
                    values_batch.append(running_values[i])
                    rewards_batch.append(running_rewards[i])
                    entropies_batch.append(running_entropies[i])

                    running_log_probs[i] = []
                    running_values[i] = []
                    running_rewards[i] = []
                    running_entropies[i] = []

                    hidden_states_new = hidden_states.clone()
                    hidden_states_new[i] = torch.zeros_like(hidden_states[i]).to(device)
                    hidden_states = hidden_states_new

                    randomized_env_params = get_random_env_paramvals_BW(randomization_params)
                    vec_env.set_env_params(randomized_env_params[0], i)

                    if len(log_probs_batch) == batch_size:
                        break

        eps_trained += batch_size

        summed_loss = 0
        for rewards_history, log_probs_history, values_history, entropies_history in zip(rewards_batch, log_probs_batch, values_batch, entropies_batch):
            returns = []
            R = 0
            for r in rewards_history[::-1]:
                R = r + gamma * R
                returns.insert(0, R)
            
            log_probs_history = torch.cat(log_probs_history).to(device)
            values_history = torch.cat(values_history).squeeze().to(device)
            entropies_history = torch.cat(entropies_history).to(device)
            returns = torch.FloatTensor(returns).to(device)

            advantage_history = returns - values_history
            actor_loss = -(log_probs_history * advantage_history.detach()).mean()
            critic_loss = advantage_history.pow(2).mean()
            entropy_loss = entropies_history.mean()
            total_loss = actor_loss + value_pred_coef * critic_loss - entropy_coef * entropy_loss

            summed_loss += total_loss
            training_losses.append(total_loss.detach())

        average_total_loss = summed_loss / batch_size
        optimizer.zero_grad()
        average_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_net.parameters(), max_grad_norm)
        optimizer.step()


        if ((selection_method == "original") and ((eps_trained-1) % evaluate_every == 0)):
            evaluation_performance = np.mean(evaluate_BW(agent_net, env_name, num_evaluation_episodes, evaluation_seeds))
            print(f"Episode {eps_trained-1}\tAverage evaluation: {evaluation_performance}")
            validation_total_rewards.append(evaluation_performance)
            if evaluation_performance > best_average:
                best_average = evaluation_performance
                best_average_after = eps_trained-1
                torch.save(agent_net.state_dict(),
                        result_dir + '/checkpoint_BP_A2C_{}.pt'.format(i_run))
                
            if best_average == max_reward:
                print(f'Best {selection_method}: ', best_average, ' reached at episode ',
                best_average_after, '. Model saved in folder best.')
                return best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses
        elif ((selection_method == "range") and ((eps_trained - 1) % evaluate_every == 0)):
            validation_ranges = [[(0.85, 0.925), (1.125, 1.25)], [(0.975, 0.9875), (1.025, 1.05)], [(0.85, 0.925), (1.125, 1.25)], [(0.975, 0.9875), (1.025, 1.05)], [(0.75, 0.875), (1.125, 1.25)], [(0.85, 0.925), (1.025, 1.05)], [(0.9, 0.95), (1.075, 1.15)], [(0.75, 0.875), (1.125, 1.25)], [(0.75, 0.875), (1.125, 1.25)], [(0.95, 0.975), (1.05, 1.1)], [(0.85, 0.925), (1.01875, 1.0375)]]
            default_values = [8.0, 34.0, 8.0, 34.0, 2.5, 4.0, 6.0, 1.0, 1.0, 5.0, 160.0]
            env_params = ['left_leg_w_unscaled', 'left_leg_h_unscaled', 'right_leg_w_unscaled', 'right_leg_h_unscaled', 'terrain_friction', 'speed_hip', 'speed_knee', 'left_leg_density', 'right_leg_density', 'hull_density', 'lidar_range_unscaled']
            
            eps_per_setting = 1
            evaluation_performance = 0
            current_np_seed = np.random.get_state()
            current_r_seed = random.getstate()
            for i in range(num_evaluation_episodes):
                np.random.seed((evaluation_seeds[i+eps_per_setting-1] + seed)%(2**32))
                random.seed((evaluation_seeds[i+eps_per_setting-1] + seed)%(2**32))
                
                random_env_param_settings = {}
                for param, ranges, default_val in zip(env_params, validation_ranges, default_values):
                    val_range = random.choice(ranges)
                    random_env_param_settings[param] = np.random.uniform(val_range[0], val_range[1])*default_val
                
                evaluation_performance += np.mean(evaluate_BW(agent_net, env_name, eps_per_setting, evaluation_seeds[i+eps_per_setting:], env_parameter_settings=random_env_param_settings))

            evaluation_performance /= num_evaluation_episodes
            print(f"Episode {eps_trained-1}\tAverage evaluation: {evaluation_performance}")
            validation_total_rewards.append(evaluation_performance)
            if evaluation_performance > best_average:
                best_average = evaluation_performance
                best_average_after = eps_trained-1
                torch.save(agent_net.state_dict(),
                        result_dir + '/checkpoint_BP_A2C_{}.pt'.format(i_run))
                
            if best_average == max_reward:
                print(f'Best {selection_method}: ', best_average, ' reached at episode ',
                best_average_after, '. Model saved in folder best.')
                return best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses
            
            np.random.set_state(current_np_seed)
            random.setstate(current_r_seed)

    print(f'Current best {selection_method}: ', best_average, ' reached at episode ', best_average_after, '.')

    return best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses



parser = argparse.ArgumentParser(description='Train an A2C agent on the CartPole environment')
parser.add_argument('--num_neurons', type=int, default=96, help='Number of neurons in the hidden layer')
parser.add_argument('--network_type', type=str, default='CfC', help='Type of neuron, either "LTC" or "CfC"')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='Learning rate for the agent')
parser.add_argument('--training_method', type=str, default = "training_range", help='Method to train the agent')
parser.add_argument('--selection_method', type=str, default = "range", help='Method to select the best model')
parser.add_argument('--seed', type=int, default=5, help='Seed for the random number generator')
parser.add_argument('--result_id', type=int, default=-1, help='ID for the result directory')
parser.add_argument('--num_models', type=int, default=10, help='Number of models to train')
parser.add_argument('--entropy_coef', type=float, default=0.001, help='Entropy coefficient for the agent')
parser.add_argument('--value_pred_coef', type=float, default=0.5, help='Value prediction coefficient for the agent')
parser.add_argument('--num_training_episodes', type=int, default=1000000, help='Number of training episodes to run')
parser.add_argument('--training_episodes_per_section', type=int, default=1000, help='Number of training episodes to run per section')
parser.add_argument('--max_reward', type=int, default=200, help='Maximum number of steps to run in the environment')
parser.add_argument('--num_evaluation_episodes', type=int, default=50, help='Number of evaluation episodes to run')
parser.add_argument('--evaluate_every', type=int, default=50, help='How often to evaluate the agent')
parser.add_argument('--env_name', type=str, default='AdjustableBipedalWalker-v3', help='Name of the environment to use')
parser.add_argument('--batch_size', type=int, default=5, help='Batch size to use for training')
parser.add_argument('--num_parallel_envs', type=int, default=5, help='Number of parallel environments to use')
parser.add_argument('--input_dims', type=int, default=24, help='Number of input dimensions to the network')
parser.add_argument('--output_dims', type=int, default=4, help='Number of output dimensions to the network')
parser.add_argument('--continuous_actions', type=bool, default=True, help='Whether the environment has continuous actions')
parser.add_argument('--additional_seed', type=int, default=0, help='Additional number to add to the seed')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for the rewards')
parser.add_argument('--max_grad_norm', type=float, default=10, help='Maximum gradient norm for the agent')
parser.add_argument('--mode', type=str, default='pure', help='Mode for the CfC network')
parser.add_argument('--tau_sys_extraction', type=bool, default=True, help='Whether to extract the tau system')
parser.add_argument('--sparsity_level', type=float, default=0.5, help='Sparsity level for the network')
parser.add_argument('--wiring', type=Wiring, default=None, help='Whether to use a wiring for the network')


gym.envs.registration.register(
    id='AdjustableBipedalWalker-v3',
    entry_point='learning_to_adapt.AdjustableBipedalWalker:AdjustableBipedalWalker',
    max_episode_steps=1600,
    reward_threshold=300,
)


args = parser.parse_args()

result_id = args.result_id
num_neurons = args.num_neurons
network_type = args.network_type
learning_rate = args.learning_rate
training_method = args.training_method
selection_method = args.selection_method
num_models = args.num_models
entropy_coef = args.entropy_coef
value_pred_coef = args.value_pred_coef
training_eps_per_section = args.training_episodes_per_section
num_training_episodes = args.num_training_episodes
max_reward = args.max_reward
num_evaluation_episodes = args.num_evaluation_episodes
evaluate_every = args.evaluate_every
env_name = args.env_name
batch_size = args.batch_size
num_parallel_envs = args.num_parallel_envs
input_dims = args.input_dims
output_dims = args.output_dims
continuous_actions = args.continuous_actions
additional_seed = args.additional_seed
gamma = args.gamma
max_grad_norm = args.max_grad_norm
mode = args.mode
tau_sys_extraction = args.tau_sys_extraction
sparsity_level = args.sparsity_level
wiring = args.wiring
if num_training_episodes % training_eps_per_section != 0:
    raise ValueError("Number of training episodes must be divisible by training episodes per section")


print(f"Num neurons: {num_neurons}, learning rate: {learning_rate}, neuron type: {network_type}")
device = "cpu"


if training_method == "training_range":
    randomization_params = [(0.925, 1.125), (0.9875, 1.025), (0.925, 1.125), (0.9875, 1.025), (0.875, 1.125), (0.925, 1.025), (0.95, 1.075), (0.875, 1.125), (0.875, 1.125), (0.975, 1.05), (0.925, 1.01875)]
else:
    randomization_params = None

if result_id == -1:
    dirs = os.listdir('learning_to_adapt/results/LTC_A2C/')
    if not any('a2c_result' in d for d in dirs):
        result_id = 1
    else:
        results = [d for d in dirs if 'a2c_result' in d]
        result_id = len(results) + 1


d = date.today()
result_dir = f'learning_to_adapt/results/LTC_A2C/{network_type}_a2c_result_' + str(result_id) + f'_{str(d.year)+str(d.month)+str(d.day)}_learningrate_{learning_rate}_selectiomethod_{selection_method}_trainingmethod_{training_method}_numneurons_{num_neurons}'
if network_type == "CfC":
    result_dir += "_mode_" + mode
if wiring:
    result_dir += "_wiring_" + "AutoNCP" + f"_sparsity_{sparsity_level}"
os.mkdir(result_dir)
print('Created Directory {} to store the results in'.format(result_dir))





evaluation_seeds = np.load('learning_to_adapt/rstdp_cartpole/seeds/evaluation_seeds.npy')
training_seeds = np.load('learning_to_adapt/rstdp_cartpole/seeds/training_seeds.npy')


all_training_losses = []
all_training_total_rewards = []
all_validation_losses = []
all_validation_total_rewards = []
best_average_after_all = []
best_average_all = []
vec_env = ModifiableAsyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_parallel_envs)])
for i_run in range(num_models):
    print(f"Run # {i_run}")
    seed = int(training_seeds[i_run]+additional_seed)

    torch.manual_seed(seed)
    random.seed(seed)
    
    if network_type == "LTC":
        agent_net = LTC_Network(input_dims, num_neurons, output_dims, seed, wiring = wiring).to(device)
    elif network_type == "CfC":
        agent_net = CfC_Network(input_dims, num_neurons, output_dims, seed, mode = mode, wiring = wiring, continuous_actions=continuous_actions).to(device)

    optimizer = torch.optim.Adam(agent_net.parameters(), lr=learning_rate)


    for section in range(0, int(num_training_episodes/training_eps_per_section)):
        print(f"Section {section+1} out of {int(num_training_episodes/training_eps_per_section)} sections")
        
        # Make sure that the training takes into account the actual best average
        # when deciding to save the model, and not just the best performance in
        # the current section.
        if section == 0:
            best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses = train_agent_batched(vec_env, training_eps_per_section, section, num_parallel_envs, batch_size, max_grad_norm, gamma)
        else:
            best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses = train_agent_batched(vec_env, training_eps_per_section, section, num_parallel_envs, batch_size, max_grad_norm, gamma)

        if section == 0:
            best_average_after_all.append(best_average_after)
            best_average_all.append(best_average)
            all_training_losses.append(training_losses)
            all_training_total_rewards.append(training_total_rewards)
            all_validation_losses.append(validation_losses)
            all_validation_total_rewards.append(validation_total_rewards)
        else:
            if best_average > best_average_all[i_run]:
                best_average_after_all[i_run] = best_average_after
                best_average_all[i_run] = best_average
            all_training_losses[i_run] = np.concatenate((all_training_losses[i_run], training_losses))
            all_training_total_rewards[i_run] = np.concatenate((all_training_total_rewards[i_run], training_total_rewards))
            all_validation_losses[i_run] = np.concatenate((all_validation_losses[i_run], validation_losses))
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

        if best_average_all[i_run] == max_reward:
            break

    print(f"Best average after {best_average_after_all[i_run]} episodes: {best_average_all[i_run]}")



