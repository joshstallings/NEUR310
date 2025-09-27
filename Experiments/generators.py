import torch
import numpy as np
from torch.nn.functional import softmax

# My custom generate function. 
def generate(model, model_name, tokenizer, prompt, num_tokens = 20, temperature = 1,
             layer_to_record = -1, activation_responses = [], token_responses = []):

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
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    current_ids = input_ids

    saved_activations = {}
    saved_activations[model_name] = []
        
    for _ in range(num_tokens): # Generate N new tokens
        # Model forward pass. The hook will be triggered here.
        with torch.no_grad():
            outputs = model(current_ids)
        
        # Get the logits for the next token
        # Logits are values before they are passed to the activation function.
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        scaled_logits = next_token_logits

        if temperature > 0:
            scaled_logits = next_token_logits / temperature


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
    
    activation_responses.append(saved_activations[model_name]) # take the very last activation
    token_responses.append(current_ids[0]) 
            

    return activation_responses, token_responses



