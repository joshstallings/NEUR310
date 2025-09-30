


import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from netrep.metrics import LinearMetric, GaussianStochasticMetric, EnergyStochasticMetric

# My code. 
import generators

OUTPUT_FOLDER = "ranging_temperature_across_k_seeds"
EXPERIMENT_NAME = "experimentTEST"

PROMPT = "Once upon a time"
PROMPT_LENGTH = len(PROMPT.split(" "))
SEEDS = [1, 2, 3]
TEMPERATURES = [0, 0.5, 1.0, 2.0]
NUM_EXPERIMENTS = 800
NUM_TOKENS = 20

# Load in the model

# Entries are of shape [ activations, token_responses]
# where activations and token_responses are of size 800
trials = np.empty( (len(TEMPERATURES), len(SEEDS), NUM_EXPERIMENTS, PROMPT_LENGTH + NUM_TOKENS - 1, 64) )
average_trajectories = []
for i in range(len(TEMPERATURES)):
    t = TEMPERATURES[i]
    for j in range(len(SEEDS)):
        set_seed(SEEDS[j])
        model_1m = AutoModelForCausalLM.from_pretrained('/Users/joshstallings/Desktop/LipshutzLab/TinyStories-1M')
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        model_1m.eval()
        
        trajectories_to_average = []
        for k in range(NUM_EXPERIMENTS):
            activation_responses, token_responses = generators.generate(model_1m, "1m", tokenizer, PROMPT, NUM_TOKENS, 
                                                                 layer_to_record=-1)

            last_layer_trajectory = activation_responses[k][-1][0][0]
            trajectories_to_average.append(last_layer_trajectory)
            trials[i, j, k] = last_layer_trajectory


        # Convert the list into a numpy array and take the mean over 64 neurons
        np_array = np.array(trajectories_to_average)

# THIS IS TO HELP ME REMEMBER THE LAYOUT OF THIS MATRIX
# print(len(activation_responses)) # Number of experiments
# print(len(activation_responses[0])) # Number of tokens
# print(len(activation_responses[0][0])) # == 2, something about the output
# print(activation_responses[0][19][0].shape) # [1, <token>, 64]
# print(activation_responses[0][0][1].shape) # [1, 16, 6, 6]

if(not os.path.isdir(OUTPUT_FOLDER)):
    os.mkdir(f"{OUTPUT_FOLDER}")

np.save(f"{OUTPUT_FOLDER}{os.sep}{EXPERIMENT_NAME}_per_trial_tensor.npy", trials)
