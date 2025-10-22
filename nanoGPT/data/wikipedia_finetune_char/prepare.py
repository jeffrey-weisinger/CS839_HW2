import os
import pickle
import requests
import numpy as np
import re

n = 45100000

valid_lines = []
with open("./../../wikipedia_dataset.txt", "r", encoding="utf-8") as f:
    data = f.read()
    data = data[:n]
    valid_lines = re.split(r'[\n]', data)

with open("./../../wikipedia_prepared_dataset.txt", "w", encoding="utf-8") as f:
    for valid_line in valid_lines:
        if valid_line != "":
            f.write(valid_line)
            f.write("\n")

with open("./../../wikipedia_prepared_dataset.txt", 'r', encoding="utf-8") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
meta = pickle.load(open("../shakespeare_char/meta.pkl", "rb"))
stoi = meta["stoi"]
itos = meta["itos"]
def encode(s):
    encoded_list = []
    for c in s:
        if c in stoi.keys():
            encoded_list.append(stoi[c]) #if the keys exist, return them
        else:
            encoded_list.append(0) #otherwise, return an array of just 0.
    return encoded_list

def decode(l):
    decoded_string = ""
    for i in l: #for each i, we can add it to return string.
        if i in itos.keys(): 
            decoded_string += itos[i] #append to return string if it exists
        #if it doesn't exist, we won't append anything. in the worst case, we just return an empty string.
    return decoded_string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
