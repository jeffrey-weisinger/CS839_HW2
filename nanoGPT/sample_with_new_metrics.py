"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import numpy as np
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with open("./generated_output/generated_output.txt", "w") as f:
    f.write("")
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            with open("./generated_output/generated_output.txt", "a") as f:
                f.write(decode(y[0].tolist()))
                f.write('\n')
k = 5
train_data = "shakespeare_char"
training_lines = []
#getting all individual input lines in a list.
with open(f"./data/{train_data}/input.txt", "r", encoding="utf-8", errors="ignore") as f:
    training_data = f.read()
    training_data_arr = training_data.split("\n")
    for line in training_data_arr:
        if line != "":
            training_lines.append(line)

generated_lines = []
#getting all individual output/generated lines in a list.
with open("./generated_output/generated_output.txt", "r", encoding="utf-8", errors="ignore") as f:
    generated_data = f.read()
    generated_data_arr = generated_data.split("\n")
    for line in generated_data_arr:
        if line != "":
            generated_lines.append(line)

generated_lines = generated_lines[:11]

#embed, then encode
train_embed = []
for line in training_lines:
    encoded_line = encode(line)
    #use wte to embed
    embedded_line = model.transformer.wte(torch.tensor(encoded_line))
    #average over each embedding for easy comparison in kNN.
    embedded_line_average = embedded_line.mean(dim = 0)
    train_embed.append(embedded_line_average)

generated_embed = []
generated_encode = []
for line in generated_lines:
    encoded_line = encode(line)
    #full encoded line, used for perplexity.
    generated_encode.append(encoded_line)
    #use wte to embed
    embedded_line = model.transformer.wte(torch.tensor(encoded_line))
    #average over each embedding for easy comparison in kNN.
    embedded_line_average = embedded_line.mean(dim = 0)
    generated_embed.append(embedded_line_average)

#Specific, kNN Approach
distances = []
for generated_elem in generated_embed[:10]:
    current_distances = []
    for train_elem in train_embed:
        l2_distance = torch.sqrt(torch.sum((generated_elem - train_elem)**2))
        l2_distance = l2_distance.item()
        current_distances.append(np.abs(l2_distance))
    current_distances = sorted(current_distances)
    distances += current_distances[:k]

average_distance = np.mean(distances)

#General, Perplexity Approach:
perplexities = []
for encode in generated_encode:
    #make long string of all tokens for the encoding.
    encoded_tokens = []
    for token in encode:
        encoded_tokens.append(token)
    all_tokens = torch.tensor(encoded_tokens)
    #make singular row within a 2D tensor matrix.
    all_tokens = all_tokens.view(1, -1)
    #shorten token count if necessary.s
    if len(all_tokens[0]) > model.config.block_size:
        all_tokens = all_tokens[:, :model.config.block_size]
    #get losses where index is n-1 tokens, and targets is n-1 tokens shifted one to the right.
    _, loss = model(idx=all_tokens[:, :-1], targets=all_tokens[:, 1:])
    #get perplexity directly from loss
    perplexity = torch.exp(loss).item() 
    perplexities.append(perplexity)
average_perplexity = np.mean(perplexities)

print(f"Average {k}-NN Distance Across 10 lines: {average_distance}")
print(f"Average Perplexity Across 10 lines: {average_perplexity}")