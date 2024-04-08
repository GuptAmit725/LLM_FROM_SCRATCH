import getData 
import torch
from config import getCongig

configs = getCongig()
INP_TEXT = configs["TEST-INPUT"]
MAX_NEW_TOKENTS = configs["MAX-NEW-TOKENS"]

def generateTokens(model):
    INPUT = getData.encode(INP_TEXT)
    OUTPUT = getData.decode(model.generate(torch.tensor([INPUT], dtype=torch.long), MAX_NEW_TOKENTS)[0].tolist())
    print('INPUT')
    print(getData.decode(INPUT))
    print('OUTPUT')
    print(OUTPUT)

    return