import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

PROMPT = "Once upon a time there was"
NUM_EXPERIMENTS = 10

ACTIVATION_RESPONSES = []
RESPONSES = []


model = AutoModelForCausalLM.from_pretrained('/Users/joshstallings/Desktop/LipshutzLab/TinyStories-1M') 
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model.eval()

# Define the hook function
def get_activation(name):
    def hook(model, input, output):
        # We save the output of the layer
        saved_activations[name] = output
    return hook


# Get the last layer of the model
# TODO: i want to look up attention and transformers for LLMs
last_layer = model.transformer.h[-1]
# Register hook function to the model's forward pass on the last layer
last_layer.register_forward_hook(get_activation('last_layer_output'))

# Custom Generation loop. 
# TODO: is there a more sophisticated way to write my own generation loops?
# TODO: is this even a correct generation loop? 
for i in range(NUM_EXPERIMENTS):
    # Tokenize the input. 
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    current_ids = input_ids
    saved_activations = {}
    for _ in range(10): # Generate 10 new tokens
        # Model forward pass. The hook will be triggered here.
        with torch.no_grad():
            outputs = model(current_ids)
        
        # Get the logits for the next token
        # Logits are values before they are passed to the activation function.
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        
        # Select the next token with greedy decoding
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Append the new token to the sequence
        current_ids = torch.cat([current_ids, next_token_id], dim=-1)
        
        # Access the saved activations for the current step
        # You can process saved_activations['last_layer_output'] here

    ACTIVATION_RESPONSES.append(saved_activations['last_layer_output'])
    RESPONSES.append(current_ids[0])


with open("activation_values.txt", "w") as f:
    for response in ACTIVATION_RESPONSES:
        f.write(str(response)+"\n")

with open("text_responses.txt", "w") as f:
    for response in RESPONSES:
        f.write(str(tokenizer.decode(response))+"\n")

# Print the final generated sequence
print("Done!")