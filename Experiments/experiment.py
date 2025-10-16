


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

OUTPUT_FOLDER = "Experiments/data"
EXPERIMENT_NAME = f"2_1M_experiment800N_EXTREME_TEMPS_PROMPTB"

PROMPT = "A long time ago"
PROMPT_LENGTH = 4
SEEDS = [100, 400, 900]
TEMPERATURES = [2, 5, 10]
NUM_EXPERIMENTS = 800
NUM_TOKENS = 20



PARAMS = {
    "PROMPT": PROMPT,
    "PROMPT_LENGTH": PROMPT_LENGTH,
    "SEEDS": SEEDS,
    "TEMPERATURES": TEMPERATURES,
    "NUM_EXPERIMENTS": NUM_EXPERIMENTS,
    "NUM_TOKENS": NUM_TOKENS
}

# Load in the model
# Entries are of shape [ activations, token_responses]
# where activations and token_responses are of size 800
trials = np.empty( (len(TEMPERATURES), len(SEEDS), NUM_EXPERIMENTS, PROMPT_LENGTH + NUM_TOKENS - 1, 64) )

token_responses = []
trajectories_to_average = [] # testing
for i in range(len(TEMPERATURES)):
    t = TEMPERATURES[i]
    for j in range(len(SEEDS)):
        set_seed(SEEDS[j])
        model_1m = AutoModelForCausalLM.from_pretrained('/Users/joshstallings/Desktop/LipshutzLab/TinyStories-1M')
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        model_1m.eval()

        activation_response = []
        token_response = []
        
        for k in range(NUM_EXPERIMENTS):
            layer_to_record = -1
            model_name = "1m"
            model = model_1m


            # Generation Code. 
            # Custom hook function
            def get_activation(name):
                def hook(model, input, output):
                    # We save the output of the layer
                    saved_activations[name].append(output) # [ [x, y], .... [x, y] ]
                return hook
            
            if layer_to_record == None:
                for layer in model.transformer.h:
                    layer.register_forward_hook(get_activation(model_name))
            else:
                model.transformer.h[layer_to_record].register_forward_hook(get_activation(model_name))

            # Tokenize the input. 
            input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
            current_ids = input_ids

            saved_activations = {}
            saved_activations[model_name] = []
                
            for _ in range(NUM_TOKENS): # Generate N new tokens
                # Model forward pass. The hook will be triggered here.
                with torch.no_grad():
                    outputs = model(current_ids)
                
                # Get the logits for the next token
                # Logits are values before they are passed to the activation function.
                logits = outputs.logits
                next_token_logits = logits[:, -1, :]
                scaled_logits = next_token_logits

                if t > 0:
                    scaled_logits = next_token_logits / t


                top_k = 50  # Keep only the top 50 most likely tokens
                indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
                scaled_logits[indices_to_remove] = -float('Inf')
                
                # 3. Apply Top-P (nucleus) sampling
                top_p = 0.95  # Keep tokens that add up to 95% probability
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
                
                # Find the tokens that fall within the cumulative probability
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scaled_logits[indices_to_remove] = -float('Inf')
                next_token_id = torch.multinomial(softmax(scaled_logits, dim=-1), num_samples=1)

                # Append the new token to the sequence
                current_ids = torch.cat([current_ids, next_token_id], dim=-1)

                if(next_token_id == tokenizer.eos_token_id):
                    break
            
                activation_response.append(saved_activations[model_name]) # take the very last activation
                token_response.append(current_ids[0])             
            
            token_responses.append(token_response)
            
            # If the response was shorter than expected I zero pad the response
            last_layer_trajectory = None
            if(activation_response[k][-1][0][0].shape[0] <  PROMPT_LENGTH + NUM_TOKENS - 1):
                pad_rows = trials.shape[3] -  activation_response[k][-1][0][0].shape[0]
                last_layer_trajectory = torch.from_numpy(np.pad(activation_response[k][-1][0][0], ((0, pad_rows), (0, 0)), mode='constant', constant_values=0))
            else:
                last_layer_trajectory = activation_response[k][-1][0][0]

            trajectories_to_average.append(last_layer_trajectory)
            trials[i, j, k] = last_layer_trajectory



        # Convert the list into a numpy array and take the mean over 64 neurons

np_array = np.array(trajectories_to_average)
print(f"Total Responses shape={np_array.shape}\nTrials Shape={trials.shape}")

# THIS IS TO HELP ME REMEMBER THE LAYOUT OF THIS MATRIX
# print(len(activation_responses)) # Number of experiments
# print(len(activation_responses[0])) # Number of tokens
# print(len(activation_responses[0][0])) # == 2, something about the output
# print(activation_responses[0][19][0].shape) # [1, <token>, 64]
# print(activation_responses[0][0][1].shape) # [1, 16, 6, 6]

if(not os.path.isdir(OUTPUT_FOLDER)):
    os.mkdir(f"{OUTPUT_FOLDER}")

np.save(f"{OUTPUT_FOLDER}{os.sep}{EXPERIMENT_NAME}_per_trial_tensor.npy", trials)

with open(f"{OUTPUT_FOLDER}{os.sep}{EXPERIMENT_NAME}_text_response.txt", "w") as f:
    for i in range(len(token_responses)):
        f.write(tokenizer.decode(token_response[i])+"\n")

with open(f"{OUTPUT_FOLDER}{os.sep}{EXPERIMENT_NAME}_params.txt", "w") as f:
    for k, v in PARAMS.items():
        if(k == "PROMPT"):
            f.write(f"{k} = \"{v}\"\n")
        else:
            f.write(f"{k} = {v}\n")
    