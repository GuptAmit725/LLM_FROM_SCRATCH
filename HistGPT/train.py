# Let's train the weights 
import torch
from tqdm.notebook import tqdm
from config import getCongig
from getData import get_batch, vocabSize
from generate import generateTokens
from GPTModel import *

config = getCongig()

BATCH_SIZE = config["BATCH-SIZE"]
VOCAB_SIZE = vocabSize()
EPOCHS = config["EPOCHS"]
DEVICE = config["DEVICE"]
INTERVAL= config["VERBOSE-INTERVAL"]
LR = config["LEARNING-RATE"]
OPTIMIZER = torch.optim.Adam(m.parameters(), LR)

model = gpt_model(VOCAB_SIZE)
model = model.to(DEVICE)

losses_tr, losses_val = [], []
for epoch in range(EPOCHS):
    xb, yb = get_batch('train')
    xb = xb.to(DEVICE)
    yb = yb.to(DEVICE)
    
    logits, loss = model(xb, yb)
    OPTIMIZER.zero_grad(set_to_none=True)
    loss.backward()
    OPTIMIZER.step()

    with torch.no_grad():
        xb_val, yb_val = get_batch("val")
        xb_val = xb_val.to(DEVICE)
        yb_val = yb_val.to(DEVICE)
        logits_val, loss_val = model(xb_val, yb_val)

    if epoch%INTERVAL==0:
        print(f'Epoch: {epoch} / {EPOCHS}, train loss: {loss}, validation loss: {loss_val}')
        losses_tr.append(loss)
        losses_val.append(loss_val)
        generateTokens(model)
        print("="*100)
