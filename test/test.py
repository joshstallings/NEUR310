from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained('/Users/joshstallings/Desktop/LipshutzLab/TinyStories-1M') 

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

prompt = "Once upon a time there was"

input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length = 1000, num_beams=1)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
