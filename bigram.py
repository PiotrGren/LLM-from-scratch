import torch
import torch.nn as nn
from torch.nn import functional as F

#Hyperparameters
batch_size = 32
block_size = 8
max_iters = 30000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' #If you have GPU this code will enable to run model on GPU using CUDA instead of just CPU (it makes training a lot more faster)
eval_iters = 200

torch.manual_seed(1337)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


#Training and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


#Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i + block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) #if we have GPU we will use CUDA so we got to make sure we load the data to device
    return x, y


@torch.no_grad() #Telling the pytorch that this function is not intend to do backpropagation
def estimate_loss():
    out = {}
    model.eval() #setting the model to the evaluation phase
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() #setting the model back to training phase
    return out


#Simple BIGRAM model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) #Each token directly reads off the logits for the next token from a lookup table

    def forward(self, idx, targets=None): #make targets oprtional cause of generating function
        #idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) #(B, T, C) - B(batch_size), T(time - block_size), C(channel - vocab_size)

        if targets is None:
            loss = None
        else:
            #We need to reshape our logits because it is (B, T, C) but torch cross entropy loss function expects C to be the 2nd parameter - [(B, C, T)]
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #We are changing our tensor to 2-dimenasional tensor where B and T are stretched out to one dimension and in this way C is the 2nd dimension just as loss function expects
            #We have to do the same thing to targets
            targets = targets.view(B*T)

            #loss function
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    #Generation function takes idx (one block/sequenc) that is (B, T) and generate and concat it to be (B, T + max_new_tokens)
    #For exmaple we give it idx with 8 characters (8 time steps) and want to generate 3 more time steps
    #It will generate 3 new characters based on probability and distribution and our new idx will be (B, T+3)
    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #Get the predictions
            logits, loss = self(idx)

            #Focus only on the last time step
            logits = logits[:, -1, :] #Become (B, C)

            #Apply softmax to get probabilitics
            probs = F.softmax(logits, dim = -1) #(B, C)

            #Sample from distribiution
            idx_next = torch.multinomial(probs, num_samples = 1) #(B, 1)

            #Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx
    

model = BigramLanguageModel(vocab_size)
m = model.to(device) #same as data loading

#Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

for iter in range(max_iters):
    #Every once in a while evalute the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step [{iter}]\nTrain loss: {losses['train']:.4f}, Validation loss: {losses['val']:.4f}\n")

    #Sample a batch od data
    xb, yb = get_batch('train')

    #Evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()


#Generate from trained model
context = torch.zeros((1, 1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))