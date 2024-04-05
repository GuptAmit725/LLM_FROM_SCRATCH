import torch
from torch.nn import functional as F
import torch.nn as nn
from tqdm.notebook import tqdm

data = ''
with open("C:/Users/amiti/Desktop/CustomLLM/data/MLBOOK.txt", "r", encoding="utf8") as f:
    data = f.read()
f.close()
data = data.replace('\n',' ')

words = list(set(data.split()))
vocab_size = len(words)

#Create a mapping for words to integers.

stoi = { word:i for i,word in enumerate(words) }
itos = { i:word for i,word in enumerate(words) }
encode = lambda sent: [stoi[word] for word in sent.split()]
decode = lambda l: ' '.join(itos[i] for i in l)

print(encode('INTRODUCTION TO  MACHINE LEARNING'))
print(decode(encode('INTRODUCTION TO  MACHINE LEARNING')))

#Preparing data tensor
data_tensor = torch.tensor(encode(data))
#Splitting data into train and validation data
n = int(0.9*len(data))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

torch.manual_seed(596)
batch_size = 8
context_length = 32

x = train_data[:context_length].tolist()
y = train_data[1:context_length+1].tolist()

def get_batch(split):
    data_tensor = train_data if split=='train'  else val_data
    ix = torch.randint(len(data_tensor)-context_length, (batch_size,))
    x = torch.stack([data_tensor[i:i+context_length] for i in ix])
    y = torch.stack([data_tensor[i+1:i+context_length+1] for i in ix])

    return x,y

xb, yb = get_batch('train')

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss=None
        else:
            B, T, C = logits.shape
            #print(B, T , C) # B=batch_size, T=context_lebgth, C=vocab_size
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # print(logits.shape, targets.shape)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits, loss = self(idx) #B, T, C
            # print(logits.shape)
            #Pluck the last token embedding from each batch 
            logits = logits[:, -1, :] #B,C
            #Get the softmax score for each token logits in the batch.
            probs = F.softmax(logits, dim=-1) # B,C
            #Next token prediction
            idx_next = torch.multinomial(probs, num_samples=1) #B,1
            # print(idx.shape, idx_next.shape)
            idx = torch.cat((idx, idx_next), dim=1) # B, T+1
        return idx
            

# Let's train the weights 

batch_size = 32
epochs = 1000
lr = 1e-3
m = BigramModel(vocab_size)
optimizer = torch.optim.Adam(m.parameters(), lr)

for epoch in tqdm(range(epochs)):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(f'loss: {loss}')

inp = encode('What is Machine Learning ?')
#print(m.generate(torch.tensor([inp], dtype=torch.long), 10).tolist())
output = decode(m.generate(torch.tensor([inp], dtype=torch.long), 100)[0].tolist())
print('INPUT')
print(inp)
print('OUTPUT')
print(output)