# Import necessary libraries
import torch
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stream = open("config.yaml")
config = load(stream, Loader)
stream.close()

# Take in the text file
input_loc = config['dataset']
file = open(input_loc, "r")
text = file.read()
file.close()

# All the unique characters that the text has
chars = sorted(list(set(text)))
vocab_size = len(chars)


# Create mapping from character to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

sample_text = 'Hello World'
assert decode(encode(sample_text)) == sample_text
# Google uses sentencepiece - Subword tokenizer
# Open AI has tiktoken

# Encoding the entire dataset
data = torch.tensor(encode(text), dtype = torch.long)

# Splitting up the data
n = int(config['split-ratio']*len(data))
train_data = data[:n]
val_data = data[n:]

# Implementing the batch operation and data loader

torch.manual_seed(1337)
batch_size = config['batch-size']
block_size = config['block-size']

def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

# Do the estimate of the loss function for general data
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(config['eval-iters'])
        for k in range(config['eval-iters']):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Implementing the model
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = self(idx)
            # Focus only on the last time stamp
            logits = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B,T+1)
        return idx


model = BigramLanguageModel().to(device)

# Optimization object 
optimizer = torch.optim.AdamW(model.parameters(), lr = config['learning-rate'])

# Training Loop
for iter in range(config['max-iters']):
    if iter % config['eval-iters'] == 0:
        losses = estimate_loss()
        print(f'Step {iter}: train loss {losses["train"]:.4f} val loss {losses["val"]:.4f}')

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Evaluating the model    
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(model.generate(context,max_new_tokens=500)[0].tolist()))