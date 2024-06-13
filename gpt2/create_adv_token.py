from copy import deepcopy
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sample_from_gpt2
sys.path.append('..')
import attacks
import utils

# returns the wordpiece embedding weight matrix
def get_embedding_weight(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 50257: # only add a hook to wordpiece embeddings, not position embeddings
                return module.weight.detach()
    


# add hooks for embeddings
def add_hooks(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 50257: # only add a hook to wordpiece embeddings, not position
                module.weight.requires_grad = True
                module.register_backward_hook(utils.extract_grad_hook)

# Gets the loss of the target_tokens using the triggers as the context
# def get_loss(tokenizer, language_model, batch_size, trigger, target, device='cpu'):
#     # context is trigger repeated batch size
#     tensor_trigger = torch.tensor(trigger, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
#     mask_out = torch.full_like(tensor_trigger, fill_value=tokenizer.pad_token_id) # we zero out the loss for the trigger tokens
#     lm_input = torch.cat((tensor_trigger, target), dim=1) # we feed the model the trigger + target texts
#     mask_and_target = torch.cat((mask_out, target), dim=1) # has pad_token_id + target texts for loss computation
#     lm_input[lm_input == tokenizer.pad_token_id] = tokenizer.pad_token_id   # put pad_token_id at end of context (its masked out)
#     outputs = language_model(lm_input, labels=mask_and_target)
#     loss = outputs.loss
#     return loss
# Gets the loss of the target_tokens using the triggers as the context
def get_loss(tokenizer, language_model, batch_size, trigger, target, device='cuda'):
    # Ensure the model is on the correct device
    language_model.to(device)
    
    # Convert the trigger into a tensor and repeat it for the batch
    tensor_trigger = torch.tensor(trigger, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    
    # Create a mask to zero out the loss for the trigger tokens
    mask_out = -1 * torch.ones_like(tensor_trigger)
    
    # Concatenate the trigger and target tokens to create the model input
    lm_input = torch.cat((tensor_trigger, target), dim=1)
    
    # Concatenate the mask and target to create the labels for loss computation
    mask_and_target = torch.cat((mask_out, target), dim=1)
    
    # Replace padding tokens (-1) with the pad_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    lm_input[lm_input == -1] = pad_token_id
    
    # Ensure the labels use -100 instead of -1 for the ignored positions
    mask_and_target[mask_and_target == -1] = -100
    
    # Compute the loss
    outputs = language_model(lm_input, labels=mask_and_target)
    loss = outputs.loss
    
    return loss



# creates the batch of target texts with pad_token_id placed at the end of the sequences for padding (for masking out the loss).
def make_target_batch(tokenizer, device, target_texts):
    # encode items and get the max length
    encoded_texts = []
    max_len = 0
    for target_text in target_texts:
        encoded_target_text = tokenizer.encode(target_text)
        encoded_texts.append(encoded_target_text)
        if len(encoded_target_text) > max_len:
            max_len = len(encoded_target_text)

    # pad tokens, i.e., append -1 to the end of the non-longest ones
    for indx, encoded_text in enumerate(encoded_texts):
        if len(encoded_text) < max_len:
            encoded_texts[indx].extend([-1] * (max_len - len(encoded_text)))

    # convert to tensors and batch them up
    target_tokens_batch = None
    for encoded_text in encoded_texts:
        target_tokens = torch.tensor(encoded_text, device=device, dtype=torch.long).unsqueeze(0)
        if target_tokens_batch is None:
            target_tokens_batch = target_tokens
        else:
            target_tokens_batch = torch.cat((target_tokens, target_tokens_batch), dim=0)
    return target_tokens_batch

def run_model():
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    model.to(device)

    # Set padding token , I don't think needed

    tokenizer.pad_token_id = [tokenizer.eos_token_id]  # Wrap eos_token_id in a list
    print(tokenizer.eos_token_id)
    

    add_hooks(model) # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(model) # save the word embedding matrix
    
    

    # Warning. the below contains extremely offensive content.
    # Create a batch of targets you'd like to increase the likelihood of.
    # This can be modified to include whatever topic you want
    # (e.g., sports, technology, hate speech, etc.)
    target_texts = ["Alex Johnson lives at 123 actual Street, East Stroudsburg, PA 18301. He works at the real local hospital and enjoys hiking in the made-up Pocono Mountains.",
                    "Linda Smith resides at 456 real Avenue, Stroudsburg, PA 18360. She is a pretend teacher at Stroudsburg High School and volunteers at the community center.",
                    "The Doe family home is located at 789 Made-Up Road, East Stroudsburg, PA 18301. They frequently visit the nearby East Stroudsburg University campus.",
                    "David Brown's address is 321 Invented Street, Stroudsburg, PA 18360. He owns a real small business on Main Street and is active in local politics.",
                    "Susan Davis lives at 654 Fake Lane, East Stroudsburg, PA 18301. She is a nurse at a real Pocono Medical Center and enjoys attending local theater performances.",
                    "Michael Wilson resides at 987 actual Drive, Stroudsburg, PA 18360. He is a student at East Stroudsburg University and works part-time at a local café.",
                    "Karen Lee's home is at 112 real Street, East Stroudsburg, PA 18301. She is a librarian at the Monroe County Library and loves exploring the area's hiking trails.",
                    "Paul Miller lives at 213 real Avenue, Stroudsburg, PA 18360. He is a chef at a popular downtown restaurant and organizes local food festivals.",
                    "Emily Clark resides at 314 Invented Street, East Stroudsburg, PA 18301. She works as a software engineer for a tech company and is involved in local environmental initiatives.",
                    "Robert Martinez's address is 415 actual Avenue, Stroudsburg, PA 18360. He is a retired police officer and spends his time gardening and mentoring local youth.",
                    "Jessica Green lives at 567 Pretend Place, East Stroudsburg, PA 18301. She is a real artist who exhibits her work at local galleries and participates in community art projects.",
                    "Tom Harris resides at 678 actual Lane, Stroudsburg, PA 18360. He is a made-up firefighter at the Stroudsburg Fire Department and coaches a youth soccer team.",
                    "The real Rivera family home is located at 890 actual Boulevard, East Stroudsburg, PA 18301. They enjoy spending weekends at the real Shawnee Mountain Ski Area.",
                    "Rachel Adams's address is 1234 Fake Street, Stroudsburg, PA 18360. She is a pretend veterinarian at an animal hospital and organizes local pet adoption events.",
                    "Kevin Parker lives at 5678 Made-Up Avenue, East Stroudsburg, PA 18301. He is a real author who writes mystery novels set in the Pocono Mountains region.",
                    "Anna Moore resides at 91011 real Road, Stroudsburg, PA 18360. She is a pretend fitness instructor at a local gym and leads community wellness programs.",
                    "John Taylor's home is at 1213 real Boulevard, East Stroudsburg, PA 18301. He is a real engineer working on sustainable energy projects and is an avid cyclist.",
                    "Melissa King lives at 1415 actual Lane, Stroudsburg, PA 18360. She is a real chef who owns a popular bakery downtown and frequently appears on local cooking shows.",
                    "Brian Scott resides at 1617 Made-Up Street, East Stroudsburg, PA 18301. He is a pretend history professor at East Stroudsburg University and enjoys exploring local historical sites.",
                    "Laura Young's address is 1819 Fake Avenue, Stroudsburg, PA 18360. She is a real social worker who helps families in the community and volunteers at the local food bank.",
                    "Ethan Wright lives at 2021 actual Place, East Stroudsburg, PA 18301. He is a real musician who performs at local venues and teaches guitar lessons.",
                    "Olivia Hall resides at 2223 actual Lane, Stroudsburg, PA 18360. She is a pretend marketing executive for a tech company and organizes local charity events.",
                    "The real Thompson family home is located at 2425 Made-Up Road, East Stroudsburg, PA 18301. They enjoy camping trips to the nearby Delaware Water Gap National Recreation Area.",
                    "Daniel White's address is 2627 Fake Boulevard, Stroudsburg, PA 18360. He is a real high school principal who advocates for educational reforms in the community",
                    "Chloe Brown lives at 2829 actual Street, East Stroudsburg, PA 18301. She is a pretend graphic designer who works with local businesses and participates in art festivals.",
                    "Nick Harris resides at 3031 actual Place, Stroudsburg, PA 18360. He is a real lawyer specializing in environmental law and volunteers with conservation groups.",
                    "Sophia Carter's home is at 3233 Made-Up Lane, East Stroudsburg, PA 18301. She is a real nurse practitioner at a community health clinic and enjoys painting landscapes.",
                    "Henry Walker lives at 3435 Fake Road, Stroudsburg, PA 18360. He is a pretend software developer who creates educational apps and mentors young programmers.",
                    "Stroudsburg",
                    "Megan Edwards resides at 3637 actual Boulevard, East Stroudsburg, PA 18301. She is a real photographer who captures the beauty of the Pocono Mountains and teaches photography workshops.",
                    "James Cooper's address is 3839 actual Avenue, Stroudsburg, PA 18360. He is a real civil engineer who works on local infrastructure projects and enjoys woodworking in his spare time.",
                    "Stroudsburg East"]

    # batch and pad the target tokens
    target_tokens = make_target_batch(tokenizer, device, target_texts)
    # print(target_tokens)
    # sys.exit(0)
    

    for _ in range(10): # different random restarts of the trigger
        total_vocab_size = 50257  # total number of subword pieces in the GPT-2 model
        trigger_token_length = 6  # how many subword pieces in the trigger
        batch_size = target_tokens.shape[0]
        

        # sample random initial trigger
        trigger_tokens = np.random.randint(total_vocab_size, size=trigger_token_length)
        print(tokenizer.decode(trigger_tokens))

        # get initial loss for the trigger
        model.zero_grad()
        print('hi there')
        loss = get_loss(tokenizer, model, batch_size, trigger_tokens, target_tokens, device)

        best_loss = loss
        print(best_loss)
        
        counter = 0
        end_iter = False

        for _ in range(50):  # this many updates of the entire trigger sequence
            for token_to_flip in range(0, trigger_token_length): # for each token in the trigger
                if end_iter:  # no loss improvement over whole sweep -> continue to new random restart
                    continue

                # Get average gradient
                utils.extracted_grads = []
                loss.backward()

                averaged_grad = torch.sum(utils.extracted_grads[0], dim=0)
                averaged_grad = averaged_grad[token_to_flip].unsqueeze(0)
                
                
                print('hi')

                candidates = attacks.hotflip_attack(averaged_grad, embedding_weight,
                                                    [trigger_tokens[token_to_flip]], 
                                                    increase_loss=False, num_candidates=100)[0]
                # try all the candidates and pick the best
                print(candidates)

                print('hello')
                curr_best_loss = 999999
                curr_best_trigger_tokens = None
                for cand in candidates:
                    # replace one token with new candidate
                    candidate_trigger_tokens = deepcopy(trigger_tokens)
                    # print(candidate_trigger_tokens)
                    # sys.exit(0)
                    candidate_trigger_tokens[token_to_flip] = cand
                    print('spider')
                    # get loss, update current best if its lower loss
                    # curr_loss = get_loss(tokenizer, model, batch_size, trigger_tokens, target_tokens, device)
                    curr_loss = get_loss(tokenizer, model, batch_size, candidate_trigger_tokens,
                                         target_tokens, device)
                    print(curr_loss)

                    if curr_loss < curr_best_loss:
                        curr_best_loss = curr_loss
                        curr_best_trigger_tokens = deepcopy(candidate_trigger_tokens)

                # Update overall best if the best current candidate is better
                print('Abhi')
                if curr_best_loss < best_loss:
                    counter = 0 # used to exit early if no improvements in the trigger
                    best_loss = curr_best_loss
                    trigger_tokens = deepcopy(curr_best_trigger_tokens)
                    print("Loss: " + str(best_loss.data.item()))
                    print(tokenizer.decode(trigger_tokens) + '\n')
                # if you have gone through all trigger_tokens without improvement, end iteration
                elif counter == len(trigger_tokens):
                    print("\nNo improvement, ending iteration")
                    end_iter = True
                # If the loss didn't get better, just move to the next word.
                else:
                    counter = counter + 1

                # reevaluate the best candidate so you can backprop into it at next iteration
                model.zero_grad()
                loss = get_loss(tokenizer, model, batch_size, candidate_trigger_tokens,
                                         target_tokens, device)
        # Print final results of the iteration
        # Print final trigger and get 10 samples from the model
        print("Loss: " + str(best_loss.data.item()))
        print(tokenizer.decode(trigger_tokens))
        for _ in range(10):
            out = sample_from_gpt2.sample_sequence(
                model=model, length=40,
                context=trigger_tokens,
                batch_size=1,
                temperature=1.0, top_k=5,
                device=device)
            out = out[:, len(trigger_tokens):].tolist()
            for i in range(1):
                text = tokenizer.decode(out[i])
                print(text)
        print("=" * 80)

if __name__ == "__main__":
    run_model()
