# Learning to Adapt: Exploring the Use of Neuromodulation in Adaptive Reinforcement Learning Policies

This repository includes the code that was used in the Master thesis of Luuk van Keeken, which can be found at ... The code is in part based on the repositories for [Backpropamine](https://github.com/uber-research/backpropamine), [r-STDP](https://github.com/mahmoudakl/dsrl), and [Neural Circuit Policies](https://github.com/mlech26l/ncps).<br><br>

## Setup
1) Clone this repository in some parent directory.
2) Rename the directory to `learning_to_adapt`.
3) In the same parent directory, clone [the repository for neuromodulating CfC networks](https://github.com/LuukvanKeeken/neuromodulated_ncps). This is a fork of [the Neural Circuit Policies repository](https://github.com/mlech26l/ncps). The fork also contains some functionality for extracting the values of time-constants.
4) Install all dependencies present in `requirements.txt`.
5) Go to the parent directory.
6) Run any script in `learning_to_adapt` as follows: `python -m learning_to_adapt.scripts.example_script_name --example_arg example_arg_value`. <br><br><br>


## Overview

### Training scripts
- `training_CfC_CartPole.py`: script for training non-ENM Closed-Form Continuous-Time networks and Liquid-Time Constant networks in the CartPole environment.
- `training_RNN_and_BP_CartPole`: script for training baseline RNNs and non-ENM Backpropamine networks in the CartPole environment.
- `training_ENMCfC_and_ENMBP_encoders_CartPole.py`: script for training ENM-Backpropamine and ENM-CfC encoders in the CartPole environment.
- `training_adaptmods_CartPole.py`: script for training adaptation modules to estimate outputs of encoders, in the CartPole environment.
- `training_CfC_BW.py`: script for training non-ENM Closed-Form Continuous-Time networks and Liquid-Time Constant networks in the BipedalWalker environment.
- `training_RNN_and_BP_BW.py`: script for training baseline RNNs and non-ENM Backpropamine networks in the BipedalWalker environment.
- `training_ENMCfC_and_ENMBP_encoders_BW.py`: script for training ENM-Backpropamine and ENM-CfC encoders in the BipedalWalker environment.
- `training_adaptmods_BW.py`: script for training adaptation modules to estimate outputs of encoders, in the BipedalWalker environment.

### Evaluation scripts
- `get_train_val_test_performance_CartPole.py`: script for evaluating non-ENM methods over the training, validation, and testing ranges in CartPole.
- `get_train_val_test_performance_encoder_CartPole.py`: script for evaluating ENM encoders over the training, validation, and testing ranges in CartPole.
- `get_train_val_test_performance_adaptmod_CartPole.py`: script for evaluating ENM adaptation modules over the training, validation, and testing ranges in CartPole.
- `get_train_val_test_performance_BW.py`: script for evaluating non-ENM methods over the training, validation, and testing ranges in BipedalWalker.
- `get_train_val_test_performance_encoder_BW.py`: script for evaluating ENM encoders over the training, validation, and testing ranges in BipedalWalker.
- `get_train_val_test_performance_adaptmod_BW.py`: script for evaluating ENM adaptation modules over the training, validation, and testing ranges in BipedalWalker.

### Other files
- `Adaptation_Module.py`: class for an RNN adaptation module.
- `AdjustableBipedalWalker.py`: a modification of gym's BipedalWalker that allows for setting the values of several environment parameters.
- `backpropamine_a2c.py`: classes implementing BP networks that can be externally neuromodulated, as well as RNNs and MLPs.
- `BP_A2C/BP_A2C_agent.py`: code used in training RNN and BP models using Advantage Actor-Critc.
- `LTC_A2C.py`: classes implementing CfC and LTC networks that can be externally neuromodulated.
- `modifiable_async_vector_env.py`: adaptation of gym's AsyncVectorEnv, that allows for setting the environment parameter values of individual environments in a vectorised environment.
- `Neuromodulated_Agent.py`: class that can wrap a policy network and a network providing external neuromodulation.