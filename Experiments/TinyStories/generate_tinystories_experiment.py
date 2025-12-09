import os
import torch
import numpy as np
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

OUTPUT_FOLDER = "./data"
PROMPTS = ["Once upon a time", "A long time ago"]

MODELS = {"../TinyStories-33M":768}

if(not os.path.isdir(OUTPUT_FOLDER)):
    os.mkdir(f"{OUTPUT_FOLDER}")

with open(f"{OUTPUT_FOLDER}{os.sep}README.txt", "w") as f:
    f.write("This folder contains simulated data.\n")

for path, hidden_size in MODELS.items():
    model_name = path.split("-")[-1]
    for b in range(len(PROMPTS)):
        s = "A"
        if b==1:
            s = "B"

        EXPERIMENT_NAME = f"{model_name}_experiment800N_9_TEMPS_PROMPT{s}"
#         EXPERIMENT_NAME = f"TEST"

        PROMPT = PROMPTS[b]
        PROMPT_LENGTH = len(PROMPT.split())
        SEEDS = [100, 400, 900]
        TEMPERATURES = [1e-10, 0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10]
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
        # Entries are of shape [ activations, total_token_responses]
        # where activations and total_token_responses are of size 800
        trials = np.empty( (len(TEMPERATURES), len(SEEDS), NUM_EXPERIMENTS, PROMPT_LENGTH + NUM_TOKENS - 1, hidden_size) )

        total_token_responses = []
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

        trajectories_to_average = [] # testing
        for i in range(len(TEMPERATURES)):
            t = TEMPERATURES[i]
            print(f"{i+1}/{len(TEMPERATURES)}")
            for j in range(len(SEEDS)):
                set_seed(SEEDS[j])
                model_1m = AutoModelForCausalLM.from_pretrained(path)
                model_1m.eval()

                activation_response = []
                token_response = []
                for k in range(NUM_EXPERIMENTS):
                    layer_to_record = -1
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

                    total_token_responses.append(token_response[-1])

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

        print(f"Trials Shape={trials.shape}")

        # THIS IS TO HELP ME REMEMBER THE LAYOUT OF THIS MATRIX
        # print(len(activation_responses)) # Number of experiments
        # print(len(activation_responses[0])) # Number of tokens
        # print(len(activation_responses[0][0])) # == 2, something about the output
        # print(activation_responses[0][19][0].shape) # [1, <token>, 64]
        # print(activation_responses[0][0][1].shape) # [1, 16, 6, 6]

        np.save(f"{OUTPUT_FOLDER}{os.sep}{EXPERIMENT_NAME}_per_trial_tensor.npy", trials)

        with open(f"{OUTPUT_FOLDER}{os.sep}{EXPERIMENT_NAME}_text_response.txt", "w") as f:
                for i in range(len(total_token_responses)):
                    f.write(tokenizer.decode(total_token_responses[i])+"\n")

        with open(f"{OUTPUT_FOLDER}{os.sep}{EXPERIMENT_NAME}_params.txt", "w") as f:
            for k, v in PARAMS.items():
                if(k == "PROMPT"):
                    f.write(f"{k} = \"{v}\"\n")
                else:
                    f.write(f"{k} = {v}\n")

