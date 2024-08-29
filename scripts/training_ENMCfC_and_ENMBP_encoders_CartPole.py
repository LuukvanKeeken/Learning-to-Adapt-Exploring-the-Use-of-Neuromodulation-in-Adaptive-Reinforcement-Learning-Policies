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
from learning_to_adapt.Neuromodulated_Agent import NeuromodulatedAgent
from learning_to_adapt.backpropamine_A2C import BP_RNetwork
from neuromodulated_ncps.ncps.wirings import AutoNCP
from torch.distributions import Categorical
import argparse




device = "cpu"


def evaluate_agent_pole_length(agent_net, env_name, num_episodes, evaluation_seeds, pole_length_modifier):
    with torch.no_grad():
        eval_rewards = []
        env = gym.make(env_name)
        env.unwrapped.length *= pole_length_modifier
            
        for i_episode in range(num_episodes):
            policy_hidden_state = None
            
            env.seed(int(evaluation_seeds[i_episode]))
            
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                state = torch.from_numpy(state)
                state = state.unsqueeze(0).to(device) #This as well?
                privileged_info = get_privileged_info(env).unsqueeze(0).to(device)
                policy_output, value, policy_hidden_state = agent_net(state.float(), privileged_info, policy_hidden_state)
                
                policy_dist = torch.softmax(policy_output, dim = 1)
                
                action = torch.argmax(policy_dist)
                

                state, r, done, _ = env.step(action.item())

                total_reward += r
            eval_rewards.append(total_reward)

        return eval_rewards



def evaluate_agent_all_params(agent_net, env_name, num_episodes, evaluation_seeds, pole_length_modifier, pole_mass_modifier, force_mag_modifier):
    with torch.no_grad():
        eval_rewards = []
        env = gym.make(env_name)
        env.unwrapped.length *= pole_length_modifier
        env.unwrapped.masspole *= pole_mass_modifier
        env.unwrapped.force_mag *= force_mag_modifier
            
        for i_episode in range(num_episodes):
            policy_hidden_state = None
            
            env.seed(int(evaluation_seeds[i_episode]))
            
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                state = torch.from_numpy(state)
                state = state.unsqueeze(0).to(device) #This as well?
                privileged_info = get_privileged_info(env).unsqueeze(0).to(device)
                policy_output, value, policy_hidden_state = agent_net(state.float(), privileged_info, policy_hidden_state)
                
                policy_dist = torch.softmax(policy_output, dim = 1)
                
                action = torch.argmax(policy_dist)
                

                state, r, done, _ = env.step(action.item())

                total_reward += r
            eval_rewards.append(total_reward)

        return eval_rewards



def get_privileged_info(env):
    pole_length = env.unwrapped.length
    masspole = env.unwrapped.masspole
    force_mag = env.unwrapped.force_mag

    privileged_info = [pole_length, masspole, force_mag]
    return torch.tensor(privileged_info, dtype=torch.float32)

def randomize_env_params(env, randomization_params, schedule_factor = None):
    pole_length = env.unwrapped.length
    masspole = env.unwrapped.masspole
    force_mag = env.unwrapped.force_mag

    params = [pole_length, masspole, force_mag]

    
    for i in range(len(params)):
        if isinstance(randomization_params[i], float):
            if schedule_factor:
                raise NotImplementedError
            low = params[i] - params[i] * randomization_params[i]
            high = params[i] + params[i] * randomization_params[i]
        elif isinstance(randomization_params[i], tuple):
            # The schedule factor determines what percentage of the total training
            # range of parameter modifiers can be sampled from.
            if schedule_factor:
                low_factor = randomization_params[i][0] * schedule_factor + (1 - schedule_factor)
                high_factor = randomization_params[i][1] * schedule_factor + (1 - schedule_factor)
                low = params[i]*low_factor
                high = params[i]*high_factor
            else:
                low = params[i]*randomization_params[i][0]
                high = params[i]*randomization_params[i][1]
        elif isinstance(randomization_params[i], list):
            param_range = random.choice(randomization_params[i])
            low = param_range[0]
            high = param_range[1]
            
        params[i] = np.random.uniform(low, high)

    env.unwrapped.length = params[0]
    env.unwrapped.masspole = params[1]
    env.unwrapped.force_mag = params[2]

    return env
    

def train_agent(env, num_training_episodes, max_steps, agent_net, num_outputs, 
                evaluation_seeds, seed, i_run, neuron_type, section, training_eps_per_section,
                selection_method = "100 episode average", 
                gamma = 0.99, max_reward = 200, env_name = "CartPole-v0", num_evaluation_episodes = 10, 
                evaluate_every = 10, randomization_params = None, randomize_every = 5, 
                schedule_start = None, schedule_end = None, schedule_type = None, value_pred_coef = 0.5,
                entropy_coef = 0.01, best_average = -np.inf, best_average_after = np.inf):
    
    
    scores = []
    smoothed_scores = []


    training_total_rewards = []
    training_losses = []
    validation_total_rewards = []
    validation_losses = []

    for episode in range(1 + section*training_eps_per_section, (section+1)*training_eps_per_section + 1):
        
        if randomization_params and episode % randomize_every == 0:
            env = gym.make(env_name)
            
            if schedule_type == 'linear':
                schedule_factor = (episode/num_training_episodes) * (schedule_end - schedule_start) + schedule_start
                env = randomize_env_params(env, randomization_params, schedule_factor=schedule_factor)
            else:
                env = randomize_env_params(env, randomization_params)

        policy_hidden_state = None

        score = 0

        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        for steps in range(max_steps):
            state = torch.FloatTensor(state).unsqueeze(0)
            privileged_info = get_privileged_info(env).unsqueeze(0).to(device)
            policy_logits, value, policy_hidden_state = agent_net(state.float(), privileged_info, policy_hidden_state)
            dist = Categorical(logits=policy_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
            next_state, reward, done, _ = env.step(action.item())

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            score += reward
            state = next_state
            
            
            

            if done or steps == max_steps - 1:
                training_total_rewards.append(score)

                
                if ((selection_method == "evaluation") and (episode % evaluate_every == 0)):
                    evaluation_performance = np.mean(evaluate_agent_pole_length(agent_net, env_name, num_evaluation_episodes, evaluation_seeds, 1.0))
                    print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                    if evaluation_performance >= best_average:
                        best_average = evaluation_performance
                        best_average_after = episode
                        torch.save(agent_net.state_dict(),
                                       result_dir + f'/checkpoint_{neuron_type}_A2C_{i_run}.pt')

                    if best_average == max_reward:
                        print(f'Best {selection_method}: ', best_average, ' reached at episode ',
                        best_average_after, f'. Model saved in folder {result_dir}')
                        return smoothed_scores, scores, best_average, best_average_after

                
                elif ((selection_method == "validation_range") and (episode % evaluate_every == 0)):
                    validation_ranges = [[(0.55, 0.775), (5.75, 10.5)], [(2.0, 3.0)], [(0.6, 0.8), (2.25, 3.5)]]

                    eps_per_setting = 1
                    evaluation_performance = 0
                    current_np_seed = np.random.get_state()
                    current_r_seed = random.getstate()
                    for i in range(num_evaluation_episodes):
                        np.random.seed((evaluation_seeds[i+eps_per_setting-1] + seed)%(2**32))
                        random.seed((evaluation_seeds[i+eps_per_setting-1] + seed)%(2**32))
                        pole_length_range = random.choice(validation_ranges[0])
                        pole_length_mod = np.random.uniform(pole_length_range[0], pole_length_range[1])
                        pole_mass_mod = np.random.uniform(validation_ranges[1][0][0], validation_ranges[1][0][1])
                        force_mag_range = random.choice(validation_ranges[2])
                        force_mag_mod = np.random.uniform(force_mag_range[0], force_mag_range[1])
                        evaluation_performance += np.mean(evaluate_agent_all_params(agent_net, env_name, eps_per_setting, evaluation_seeds[i+eps_per_setting:], pole_length_mod, pole_mass_mod, force_mag_mod))

                    evaluation_performance /= num_evaluation_episodes
                    print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")
                    validation_total_rewards
                    if evaluation_performance >= best_average:
                        best_average = evaluation_performance
                        best_average_after = episode
                        torch.save(agent_net.state_dict(),
                                       result_dir + f'/checkpoint_{neuron_type}_A2C_{i_run}.pt')
                        
                    if best_average == max_reward:
                        print(f'Best {selection_method}: ', best_average, ' reached at episode ',
                        best_average_after, f'. Model saved in folder {result_dir}')
                        return smoothed_scores, scores, best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses

                    np.random.set_state(current_np_seed)
                    random.setstate(current_r_seed)
                

                elif ((selection_method == "testing_range") and (episode % evaluate_every == 0)):
                    validation_ranges = [[(0.1, 0.55), (10.5, 20.0)], [(5.0, 13.0)], [(0.2, 0.6), (3.5, 6.0)]]

                    eps_per_setting = 1
                    evaluation_performance = 0
                    total_eval_eps = 10
                    current_np_seed = np.random.get_state()
                    current_r_seed = random.getstate()
                    for i in range(total_eval_eps):
                        np.random.seed((evaluation_seeds[i+eps_per_setting-1] + seed)%(2**32))
                        random.seed((evaluation_seeds[i+eps_per_setting-1] + seed)%(2**32))
                        pole_length_range = random.choice(validation_ranges[0])
                        pole_length_mod = np.random.uniform(pole_length_range[0], pole_length_range[1])
                        pole_mass_mod = np.random.uniform(validation_ranges[1][0][0], validation_ranges[1][0][1])
                        force_mag_range = random.choice(validation_ranges[2])
                        force_mag_mod = np.random.uniform(force_mag_range[0], force_mag_range[1])
                        evaluation_performance += np.mean(evaluate_agent_all_params(agent_net, env_name, eps_per_setting, evaluation_seeds[i+eps_per_setting:], pole_length_mod, pole_mass_mod, force_mag_mod))

                    evaluation_performance /= total_eval_eps
                    print(f"Episode {episode}\tAverage evaluation: {evaluation_performance}")

                    if evaluation_performance >= best_average:
                        best_average = evaluation_performance
                        best_average_after = episode
                        torch.save(agent_net.state_dict(),
                                       result_dir + f'/checkpoint_{neuron_type}_A2C_{i_run}.pt')
                        
                    if best_average == max_reward:
                        print(f'Best {selection_method}: ', best_average, ' reached at episode ',
                        best_average_after, f'. Model saved in folder {result_dir}')
                        return smoothed_scores, scores, best_average, best_average_after

                    np.random.set_state(current_np_seed)
                    random.setstate(current_r_seed)

                        
                break

        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        
        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze()
        returns = torch.FloatTensor(returns)
        
        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        entropy_loss = entropy.mean()  
        total_loss = actor_loss + value_pred_coef * critic_loss - entropy_coef * entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        training_losses.append(total_loss.detach())
    
    print(f'Current best {selection_method}: ', best_average, ' reached at episode ',
              best_average_after)
    
    return smoothed_scores, scores, best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses




parser = argparse.ArgumentParser(description='Train an A2C agent on the CartPole environment')
parser.add_argument('--num_neurons', type=int, default=48, help='Number of neurons in the hidden layer')
parser.add_argument('--randomization_factor', type=float, default=0.5, help='Factor to randomize the environment parameters')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the agent')
parser.add_argument('--training_method', type=str, default = "training_range", help='Method to train the agent')
parser.add_argument('--neuromod_network_dims', type=int, nargs='+', default = [3, 192, 96], help='Dimensions of the neuromodulation network, without output layer')
parser.add_argument('--selection_method', type=str, default = "validation_range", help='Method to select the best model')
parser.add_argument('--num_models', type=int, default=10, help='Number of models to train')
parser.add_argument('--num_training_episodes', type=int, default=25000, help='Number of episodes to train the agent')
parser.add_argument('--training_episodes_per_section', type=int, default=1000, help='Number of training episodes to run per section')
parser.add_argument('--encoder_output_activation', type=str, default="relu", help="Activation function of the encoder's output layer")
parser.add_argument('--encoder_hidden_activation', type=str, default="relu", help="Activation function of the encoder's hidden layers")
parser.add_argument('--result_id', type=int, default=-1, help='ID of the result folder')
parser.add_argument('--mode', type=str, default="only_neuromodulated", help="The mode of the CfC network.")
parser.add_argument('--schedule_start', type=float, default=0.00001, help="The starting value of the schedule factor")
parser.add_argument('--schedule_end', type=float, default=1.0, help="The end value of the schedule factor")
parser.add_argument('--schedule_type', type=str, default='None', help="The type of schedule to use for the schedule factor")
parser.add_argument('--neuron_type', type=str, default='CfC', help="The type of neuron to use")
parser.add_argument('--value_pred_coef', type=float, default=0.5, help="The coefficient for the value prediction loss")
parser.add_argument('--entropy_coef', type=float, default=0.01, help="The coefficient for the entropy loss")
parser.add_argument('--num_evaluation_episodes', type=int, default=20, help='Number of evaluation episodes to run')
parser.add_argument('--evaluate_every', type=int, default=20, help='How often to evaluate the agent')
parser.add_argument('--max_reward', type=int, default=200, help='Maximum number of steps to run in the environment')
parser.add_argument('-gamma', type=float, default=0.99, help='Discount factor for the rewards')
parser.add_argument('--sparsity_level', type=float, default=0.5, help='Sparsity level of the network')
parser.add_argument('--tau_sys_extraction', type=bool, default=True, help='Whether to extract the tau_sys from the LTC network')
args = parser.parse_args()


num_neurons = args.num_neurons
factor = args.randomization_factor
learning_rate = args.learning_rate
training_method = args.training_method
selection_method = args.selection_method
neuromod_network_dims = args.neuromod_network_dims
neuromod_network_dims.append(num_neurons)
num_models = args.num_models
num_training_episodes = args.num_training_episodes
result_id = args.result_id
mode = args.mode
schedule_start = args.schedule_start
schedule_end = args.schedule_end
schedule_type = args.schedule_type
neuron_type = args.neuron_type
value_pred_coef = args.value_pred_coef
entropy_coef = args.entropy_coef
num_evaluation_episodes = args.num_evaluation_episodes
evaluate_every = args.evaluate_every
training_eps_per_section = args.training_episodes_per_section
max_reward = args.max_reward
gamma = args.gamma
sparsity_level = args.sparsity_level
tau_sys_extraction = args.tau_sys_extraction

if args.encoder_output_activation == "identity":
    encoder_output_activation = torch.nn.Identity()
elif args.encoder_output_activation == "relu":
    encoder_output_activation = torch.nn.ReLU()
elif args.encoder_output_activation == "tanh":
    encoder_output_activation = torch.nn.Tanh()
else:
    raise NotImplementedError

if args.encoder_hidden_activation == "relu":
    encoder_hidden_activation = torch.nn.ReLU()
elif args.encoder_hidden_activation == "tanh":
    encoder_hidden_activation = torch.nn.Tanh()
else:
    raise NotImplementedError

if num_training_episodes % training_eps_per_section != 0:
    raise ValueError("Number of training episodes must be divisible by training episodes per section")


print(f"Num neurons: {num_neurons}, learning rate: {learning_rate}, rand factor: {factor}")
device = "cpu"



if training_method == "training_range":
    randomization_params = [(0.775, 5.75), (1.0, 2.0), (0.8, 2.25)]
elif training_method == "original":
    randomization_params = None
elif training_method == "testing_range":
    randomization_params = [[(0.1, 0.55), (10.5, 20.0)], [(5.0, 13.0)], [(0.2, 0.6), (3.5, 6.0)]]


wiring = None

if neuron_type == "BP":
    top_dir = "BP_A2C"
elif neuron_type == "LTC" or neuron_type == "CfC":
    top_dir = "LTC_A2C"

if result_id == -1:
    dirs = os.listdir(f'learning_to_adapt/results/{top_dir}/')
    if not any('a2c_result' in d for d in dirs):
        result_id = 1
    else:
        results = [d for d in dirs if 'a2c_result' in d]
        result_id = len(results) + 1


d = date.today()
result_dir = f'learning_to_adapt/results/{top_dir}/{neuron_type}_a2c_result_' + str(result_id) + f'_{str(d.year)+str(d.month)+str(d.day)}_learningrate_{learning_rate}_numneurons_{num_neurons}_encoutact_{args.encoder_output_activation}'

if mode == "neuromodulated" or mode == "only_neuromodulated":
    result_dir += "_neuromod_network_dims_" + "_".join(map(str, neuromod_network_dims))
if wiring:
    result_dir += "_wiring_" + "AutoNCP" + f"_sparsity_{sparsity_level}"

os.mkdir(result_dir)
print('Created Directory {} to store the results in'.format(result_dir))




env = gym.make('CartPole-v0')
evaluation_seeds = np.load('learning_to_adapt/rstdp_cartpole/seeds/evaluation_seeds.npy')
training_seeds = np.load('learning_to_adapt/rstdp_cartpole/seeds/training_seeds.npy')


all_training_losses = []
all_training_total_rewards = []
all_validation_losses = []
all_validation_total_rewards = []
best_average_after_all = []
best_average_all = []
for i_run in range(num_models):
    print(f"Run # {i_run}")
    seed = int(training_seeds[i_run])

    torch.manual_seed(seed)
    random.seed(seed)
    
    if neuron_type == "LTC":
        raise NotImplementedError
        agent_net = LTC_Network(4, num_neurons, 2, seed, wiring = wiring).to(device)
    elif neuron_type == "CfC":
        layer_list = []
        for dim in range(len(neuromod_network_dims) - 1):
            layer_list.append(torch.nn.Linear(neuromod_network_dims[dim], neuromod_network_dims[dim + 1]))
            if dim < len(neuromod_network_dims)-2:
                layer_list.append(encoder_hidden_activation)
            else:
                layer_list.append(encoder_output_activation)
        encoder = torch.nn.Sequential(*layer_list)
        
        policy_net = CfC_Network(4, num_neurons, 2, seed, mode = mode, wiring = wiring).to(device)

        agent_net = NeuromodulatedAgent(policy_net, encoder, policy_has_hidden_state=True).to(device)
    elif neuron_type == "BP":
        layer_list = []
        for dim in range(len(neuromod_network_dims) - 1):
            layer_list.append(torch.nn.Linear(neuromod_network_dims[dim], neuromod_network_dims[dim + 1]))
            if dim < len(neuromod_network_dims)-2:
                layer_list.append(encoder_hidden_activation)
            else:
                layer_list.append(encoder_output_activation)
        encoder = torch.nn.Sequential(*layer_list)

        policy_net = BP_RNetwork(4, num_neurons, 2, seed, external_neuromodulation = True).to(device)

        agent_net = NeuromodulatedAgent(policy_net, encoder, policy_has_hidden_state=True).to(device)


    optimizer = torch.optim.Adam(agent_net.parameters(), lr=learning_rate)

    for section in range(0, int(num_training_episodes/training_eps_per_section)):
        print(f"Section {section+1} out of {int(num_training_episodes/training_eps_per_section)} sections")

        # Make sure that the training takes into account the actual best average
        # when deciding to save the model, and not just the best performance in
        # the current section.
        if section == 0:
            smoothed_scores, scores, best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses = train_agent(env, num_training_episodes, 200, agent_net, 2, evaluation_seeds, seed, i_run, neuron_type, section, training_eps_per_section, selection_method = selection_method, gamma = gamma, randomization_params=randomization_params, schedule_start = schedule_start, schedule_end=schedule_end, schedule_type=schedule_type, value_pred_coef = value_pred_coef, entropy_coef = entropy_coef, num_evaluation_episodes=num_evaluation_episodes, evaluate_every=evaluate_every)
            best_average_after_all.append(best_average_after)
            best_average_all.append(best_average)
            all_training_losses.append(training_losses)
            all_training_total_rewards.append(training_total_rewards)
            all_validation_losses.append(validation_losses)
            all_validation_total_rewards.append(validation_total_rewards)
        
        else:
            smoothed_scores, scores, best_average, best_average_after, training_total_rewards, training_losses, validation_total_rewards, validation_losses = train_agent(env, num_training_episodes, 200, agent_net, 2, evaluation_seeds, seed, i_run, neuron_type, section, training_eps_per_section, selection_method = selection_method, gamma = gamma, randomization_params=randomization_params, schedule_start = schedule_start, schedule_end=schedule_end, schedule_type=schedule_type, value_pred_coef = value_pred_coef, entropy_coef = entropy_coef, num_evaluation_episodes=num_evaluation_episodes, evaluate_every=evaluate_every, best_average=best_average_all[i_run], best_average_after=best_average_after_all[i_run])
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




