import torch

config_data = {
    "BATCH-SIZE": 16,
    "CONTEXT-LENGTH" : 3,
    "EMBEDDING-SIZE" : 64,
    "LEARNING-RATE" : 1e-2,
    "TRAIN-SPLIT": 0.8,
    "SET-TORCH-SEED": 313465132,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DATA-PATH": "C:/Users/amiti/Desktop/CustomLLM/LLM_FROM_SCRATCH/data/MLBOOK.txt",
    "TR-BLOCK-SIZE": 4,
    "ATTENTION-HEADS": 4,
    "EPOCHS":100000,
    "VERBOSE-INTERVAL": 1000,
    "TEST-INPUT": "Machine Learning",
    "MAX-NEW-TOKENS": 25
}


def getCongig():
    return config_data