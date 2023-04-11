import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    #pytorch requires one dimensional tensor to be represented by a tuple with a trailing comma - (batch_size,):
    ix = torch.randint(len(data) - block_size, (batch_size,)) 
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    #a self attention head

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, Head_size)
        q = self.query(x) # (B, T, Head_size)
        v = self.value(x) #(B, T, Head_size)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 #(B, T, Head_size)  @ (B, Head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #set all future values to -inf , (B, T, T) 
        wei = F.softmax(wei, dim=1) # (B, T, T)
        wei = self.dropout(wei)
        out  = wei@v # apply the value matrix (B, T, T) -> (B, T, Head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #create a list of heads
        self.proj = nn.Linear(head_size * num_heads, n_embd) #head_size * num_heads dimension to become n_embd dimension
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #concatenate these heads. (B, T, Head_size) -> (B, T, Head_size * num_heads)
        out = self.dropout(self.proj(out)) #linear transformation + (B, T, Head_size) -> (B, T, n_embd) for correct out dimensions
        return out

class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    #one decoder block

    def __init__(self, n_embd, n_head):
        #n_head -> amt of heads
        super().__init__()
        head_size = n_embd // n_head 
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
         #define LayerNorm functions. 2 because they won't just reduce mean to 0 and variance to 1, they will be trainable
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        #adds & layernorms per transformer architecture. Here layernorm is applied before the sublayer as is more common nowadays
        x = x + self.sa(self.ln1(x)) #after attention
        x = x + self.ffwd(self.ln2(x)) #after forward layer
        return x

    
# super simple bigram model
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #positional encoding table, for each value in the block. does not necessarily have to have same dimensionality, n_embd
        #as the token embedding table, but does here
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # * because nn.Sequential takes separate arguments, not an iterable as input
        self.ln_f = nn.LayerNorm(n_embd)
        #linear transformation layer, also changes n_embd features to vocab_size features, which we need since our
        #token embedding table is n_embd features tall, but logits must be vocab_size features tall
        self.lm_head = nn.Linear(n_embd, vocab_size) 

        #in the github for the project he mentions an apply init, but it is not covered in the video so I will omit it

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T, C=n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C=n_embd)
        x = tok_emb + pos_emb # (B, T, C=n_embd)
        x = self.blocks(x) #run decoder blocks (B, T, C=n_embd)
        x = self.ln_f(x) # (B, T, C=n_embd)
        logits = self.lm_head(x) # (B, T, C=vocab_size)

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
            #crop idx to the last set of block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond) #targets = None, so no loss is calculated
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C), and works because no loss is calcualted, so logits dimensions are (B, T, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # Evaluate loss every eval_interval, as well as the last iteration
    if iter % eval_interval == 0 or iter == max_iters -1 :
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
open('output.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
#print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
