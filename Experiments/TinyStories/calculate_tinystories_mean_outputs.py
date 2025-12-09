
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM


# Dict of model path to list of data files
MODEL_TO_DATA = {
    "../TinyStories-1M": ["generated_data_text_responses/1M_experiment800N_9_TEMPS_PROMPTA_per_trial_tensor.npy",
                          "generated_data_text_responses/1M_experiment800N_9_TEMPS_PROMPTB_per_trial_tensor.npy"],
    "../TinyStories-3M": ["generated_data_text_responses/3M_experiment800N_9_TEMPS_PROMPTA_per_trial_tensor.npy",
                          "generated_data_text_responses/3M_experiment800N_9_TEMPS_PROMPTB_per_trial_tensor.npy"],
    "../tinystories-custom-8M": ["generated_data_text_responses/8M_experiment800N_9_TEMPS_PROMPTA_per_trial_tensor.npy",
                                "generated_data_text_responses/8M_experiment800N_9_TEMPS_PROMPTB_per_trial_tensor.npy"],
    "../tinystories-custom-21M": ["generated_data_text_responses/21M_experiment800N_9_TEMPS_PROMPTA_per_trial_tensor.npy",
                                "generated_data_text_responses/21M_experiment800N_9_TEMPS_PROMPTB_per_trial_tensor.npy"],
    "../TinyStories-33M": ["generated_data_text_responses/33M_experiment800N_9_TEMPS_PROMPTA_per_trial_tensor.npy",
                          "generated_data_text_responses/33M_experiment800N_9_TEMPS_PROMPTB_per_trial_tensor.npy"]
}

PROMPTS = ["Once upon a time", "A long time ago"]
TEMPERATURES = [1e-10, 0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10]
SEEDS = [100, 400, 900]
OUTPUT = {} # Mapping of model -> average text response. 
for MODEL, DATA_LIST in MODEL_TO_DATA.items():
    print("Using model: {MODEL}")
    OUTPUT[MODEL] = {}
    for idx, DATA in enumerate(DATA_LIST):
        if(idx >= len(PROMPTS)):
            print(f"ERROR: idx {idx} out of range for {len(PROMPTS)} prompts.")
            continue

        # Prompts to use.
        PROMPT = PROMPTS[idx]
        PROMPT_LENGTH = len(PROMPT)
        OUTPUT[MODEL][PROMPT] = []

        # Load in data and create the means. 
        DATA = [np.load(DATA)]
        AVERAGE_RESPONSES = [np.mean(d, 2) for d in DATA]

        # Instantiate the tokenizer and tokenize input prompt. 
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
        # Load model in eval mode.
        model = AutoModelForCausalLM.from_pretrained(MODEL)
        model.eval()

        # Get the average text response the temperatures. 
        mean_text_responses = []
        for temp_idx, _ in enumerate(TEMPERATURES):
            seed_idx = 2
            tensor = torch.from_numpy(AVERAGE_RESPONSES[0][temp_idx][seed_idx])
            tensor = tensor.float()
            x1 = model.transformer.ln_f(tensor)  # Apply normalization layer
            x2 = model.lm_head(x1)  # Send through language model head

            # Compute its text output. 
            to_decode = torch.argmax(x2, dim=-1)
            to_decode[:len(input_ids[0])] = input_ids
            decoded_text = tokenizer.decode(to_decode.tolist())
            mean_text_responses.append(decoded_text)
        
        OUTPUT[MODEL][PROMPT].append(mean_text_responses)

print("Exporting..")
with open("mean_text_responses.json", "w") as f:
    json.dump(OUTPUT, f, indent=4)

print("Finished!")