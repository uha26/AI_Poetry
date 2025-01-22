from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'  # You can use 'gpt2-medium' or 'gpt2-large' for a larger model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the pad_token_id to the eos_token_id to avoid padding issues
tokenizer.pad_token = tokenizer.eos_token

def generate_poem(theme):
    # Encode the input theme
    input_ids = tokenizer.encode(theme, return_tensors='pt')

    # Manually create an attention mask (set to 1 for all tokens)
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

    # Generate text using the model with sampling enabled
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,  # Explicitly pass the attention mask
        max_length=150,
        num_return_sequences=1,
        do_sample=True,  # Enable sampling
        temperature=0.6,  # Control the randomness of the generation
        top_p=0.9,  # Top-p sampling
        no_repeat_ngram_size=2  # Avoid repeating n-grams
    )

    # Decode the generated text
    poem = tokenizer.decode(output[0], skip_special_tokens=True)
    return poem

# Get theme from the user
theme = input("Enter the theme for the poem: ")

# Generate and print the poem based on the user input
poem = generate_poem(theme)
print("\nGenerated Poem:\n")
print(poem)
