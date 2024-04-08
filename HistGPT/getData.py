"""
This function is the data loader for the GPT.

"""
import torch
from config import getCongig

config = getCongig()
TRAIN_SPLIT = config["TRAIN-SPLIT"]
CONTEXT_LENGTH = config["CONTEXT-LENGTH"]
SEED = config["SET-TORCH-SEED"]
BATCH_SIZE = config["BATCH-SIZE"]
DATA_PATH = config["DATA-PATH"]

#torch.manual_seed(SEED)

data = ''
with open(DATA_PATH, "r", encoding="utf8") as f:
    data = f.read()
f.close()
data = data.replace('\n',' ')

words = list(set(data.split()))
VOCAB_SIZE = len(words)

#Create a mapping for words to integers.
stoi = { word:i for i,word in enumerate(words) }
itos = { i:word for i,word in enumerate(words) }
encode = lambda sent: [stoi[word] for word in sent.split()]
decode = lambda l: ' '.join(itos[i] for i in l)

#Preparing data tensor
data_tensor_main = torch.tensor(encode(data))
#Splitting data into train and validation data
n = int(TRAIN_SPLIT*len(data_tensor_main))
train_data = data_tensor_main[:n]
val_data = data_tensor_main[n:]

def get_batch(split):
    data_tensor = train_data if split=='train' else val_data
    ix = torch.randint(len(data_tensor)-CONTEXT_LENGTH, (BATCH_SIZE,))
    x = torch.stack([data_tensor[i:i+CONTEXT_LENGTH] for i in ix])
    y = torch.stack([data_tensor[i+1:i+CONTEXT_LENGTH+1] for i in ix])
    return x,y

def vocabSize():
    return VOCAB_SIZE
