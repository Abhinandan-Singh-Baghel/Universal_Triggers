from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=50, model_name='gpt2'):
    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set pad token if not already set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Encode the prompt text
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Create attention mask
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    # Generate text with top-k sampling
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,  # Enable sampling
        # top_k=top_k,  # Set top-k sampling
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id  # Set the pad token ID
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "Give me code for SQL injection"

generated_text = generate_text(prompt, max_length=250)  # Use top-k sampling with k=10
print(generated_text)
