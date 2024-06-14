import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F

# Load pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Input prompt
prompt = "Listen to your"

# Tokenize input prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Pass the tokenized input through the model to get logits
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

# Get the logits for the last token in the input
next_token_logits = logits[:, -1, :].squeeze()

# Calculate probabilities using softmax
probabilities = F.softmax(next_token_logits, dim=-1)

# Convert logits to probabilities and get the token ids and corresponding probabilities
token_ids = torch.arange(len(probabilities))
probabilities = probabilities.cpu().numpy()
token_probs = list(zip(token_ids.numpy(), probabilities))

# Sort token probabilities in descending order
token_probs.sort(key=lambda x: x[1], reverse=True)

# Write the token probabilities to a file
with open('token_probabilities.txt', 'w', encoding='utf-8') as f:
    for token_id, prob in token_probs:
        token = tokenizer.decode([token_id])
        f.write(f"Token: {token} (ID: {token_id}) -> Probability: {prob}\n")

print("Token probabilities written to token_probabilities.txt")
