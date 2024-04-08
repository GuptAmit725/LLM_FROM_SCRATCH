#Building the very basic bigram model
import torch
from torch.nn import functional as F
import torch.nn as nn
from config import getCongig
from getData import get_batch, vocabSize

config = getCongig()
SEED = config["SET-TORCH-SEED"]
EMBEDDING_SIZE = config["EMBEDDING-SIZE"]
DEVICE = config["DEVICE"]
BLOCK_SIZE = config["TR-BLOCK-SIZE"]
ATTENTION_HEADS = config["ATTENTION-HEADS"]
MAX_NEW_TOKENS = config["MAX-NEW-TOKENS"]
VOCAB_SIZE = vocabSize()

xb, yb = get_batch('train')

torch.manual_seed(SEED)

class FeedForward(nn.Module):
    def __init__(self, embed_size=EMBEDDING_SIZE):
        super().__init__()
        self.l1 = nn.Linear(embed_size, 4*embed_size)
        self.l1.weight = torch.nn.init.normal_(self.l1.weight, mean=0.0, std=0.02)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(4*embed_size, embed_size)
        self.l2.weight = torch.nn.init.normal_(self.l2.weight, mean=0.0, std=0.02)
        self.d_out = nn.Dropout(p=0.2)

    def forward(self,x):
        out = self.d_out(self.l2(self.relu(self.l1(x))))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.embed = nn.Embedding(vocab_size, embed_size).to(DEVICE)
        self.embed.weight = torch.nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self,idx, B, T, C):
        pos_embedding = self.embed(idx).view(B*T, C)
        idx_expand = idx.view(1, B*T)
        freq = torch.pow(10000, torch.arange(self.embed_size)*(2/self.embed_size)).to(DEVICE)
        sin_idx = torch.sin(pos_embedding / freq)
        return sin_idx

class SelfAttention(nn.Module):
    def __init__(self, heads=ATTENTION_HEADS, embed_size=EMBEDDING_SIZE):
        super().__init__()
        self.head = heads
        self.embed_size = embed_size
        self.head_out_size = embed_size//heads
        self.q = nn.Linear(embed_size, embed_size//heads)
        self.q.weight = torch.nn.init.normal_(self.q.weight, mean=0.0, std=0.02)
        self.k = nn.Linear(embed_size, embed_size//heads)
        self.k.weight = torch.nn.init.normal_(self.k.weight, mean=0.0, std=0.02)
        self.v = nn.Linear(embed_size, embed_size//heads)
        self.v.weight = torch.nn.init.normal_(self.v.weight, mean=0.0, std=0.02)
        self.sm = nn.Softmax()

    def forward(self, embeddings):
        #print(self.embed_size,self.head, self.embed_size%self.head)
        assert self.embed_size%self.head == 0 , "Embedding size should be divisible by number of heads"
        B, T, C = embeddings.shape
        y_list = []
        y = torch.randn(B, T, C)
        for n_head in range(self.head):
            embeddings = embeddings.view(B*T, C)
            q = self.q(embeddings) # (B*T, embed_size//heads)
            k = self.k(embeddings) # (B*T, embed_size//heads)
            v = self.v(embeddings) # (B*T, embed_size//heads)
            qk = q @ k.T # (B*T, B*T)
            qk_scaled = qk / self.embed_size**0.5
            att = self.sm(qk_scaled)
            #print('att: ',att.shape)
            y_ = att @ v
            y_ = y_.view(B, T, self.head_out_size)
            y_list.append(y_)
        y = torch.cat(y_list, -1)
        #print(f'Multi head output shape: {y.shape}')
            
        return y

class Block(nn.Module):
    def __init__(self, embed_size=EMBEDDING_SIZE):
        super().__init__()
        self.sa = SelfAttention(heads=8).to(DEVICE)
        self.ff = FeedForward().to(DEVICE)
        self.lnorm1 = nn.LayerNorm(embed_size).to(DEVICE)
        self.lnorm2 = nn.LayerNorm(embed_size).to(DEVICE)

    def forward(self, x):
        x = x.to(DEVICE)
        x = x + self.lnorm1(self.sa(x))
        x = x + self.lnorm2(self.ff(x))
        return x
        
class gpt_model(nn.Module):
    def __init__(self, vocab_size, embed_size=EMBEDDING_SIZE, block_size=BLOCK_SIZE):
        super().__init__()
        self.embed_size = embed_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size).to(DEVICE)
        self.token_embedding_table.weight = torch.nn.init.normal_(self.token_embedding_table.weight, mean=0.0, std=0.02)
        self.pos_embed = PositionalEncoding(vocab_size, embed_size).to(DEVICE)
        self.n_block = nn.Sequential(*[Block() for i in range(block_size)])
        self.final_lnorm = nn.LayerNorm(embed_size).to(DEVICE)
        self.final_l = nn.Linear(embed_size, vocab_size).to(DEVICE)
        self.final_l.weight = torch.nn.init.normal_(self.final_l.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        idx = idx.to(DEVICE)
        embeddings = self.token_embedding_table(idx)
        B, T, C = embeddings.shape
        #embeddings = self.lnorm(embeddings)
        embeddings = embeddings.view(B*T, C)
        pos_embedding = self.pos_embed(idx, B, T, C)
        embeddings = embeddings + pos_embedding
        embeddings = embeddings.view(B, T, C)
        logits = self.final_l(self.final_lnorm(self.n_block(embeddings)))      
        if targets is None:
            loss=None
        else:
            B, T, C = logits.shape
            #print(B, T , C) # B=batch_size, T=context_lebgth, C=vocab_size
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # print(logits.shape, targets.shape)
            loss = F.cross_entropy(logits, targets)
        return logits, loss#,(self.emb_weights, self.l1_weights, self.l2_weights)

    def generate(self, idx, MAX_NEW_TOKENS):
        for i in range(MAX_NEW_TOKENS):
            idx = idx.to(DEVICE)
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
            

m = gpt_model(VOCAB_SIZE)
logits, loss = m(xb.to(DEVICE), yb.to(DEVICE)) #, (emb_weight, l1_weight, l2_weight)
print(logits.shape)
print(loss)