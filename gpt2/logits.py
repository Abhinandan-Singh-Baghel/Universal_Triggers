from transformers import GPT2Tokenizer

# Load pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Open file to write the token ID and corresponding word
with open('token_id_to_word.txt', 'w', encoding='utf-8') as f:
    for token_id in range(tokenizer.vocab_size):
        word = tokenizer.decode([token_id])
        f.write(f"Token ID: {token_id} -> Word: {word}\n")

print("Token ID to word mapping written to token_id_to_word.txt")
