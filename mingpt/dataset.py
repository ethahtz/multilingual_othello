import torch
from torch.utils.data import Dataset
import numpy as np

VALID_SQUARES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
                18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 37, 
                38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 
                54, 55, 56, 57, 58, 59, 60, 61, 62, 63] # 27,28,35,36 are not used


class CharDataset(Dataset):
    def __init__(self, data, config):
        if hasattr(data, "ood_perc"):
            ood_perc = data.ood_perc
            data.ood_perc = 0  # shut down the randomness
        
        chars = [-100, ] + config.vocabs

        data_size, vocab_size = len(data), len(chars)  # vocab size 61, with -100 sorted to the front

        max_len = max([len(config.to_tok_space(VALID_SQUARES, i)) for i in range(config.num_token_spaces)]) 
        print('Dataset created has %d sequences, %d unique words across %d token spaces.' % (data_size, vocab_size, config.num_token_spaces))
        
        self.retrieval_mode = config.retrieval_mode
        
        print(f"Dataset retrieval mode set to '{self.retrieval_mode}'")

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = max_len
        self.block_size = max_len - 1  # for autoregressive training
        self.vocab_size = vocab_size
        if hasattr(data, "ood_perc"):
            data.ood_perc = ood_perc  # turn on the randomness
        self.data = data
        self.nTokenSpaces = config.num_token_spaces
        self.to_tok_space = config.to_tok_space

        self.unified_output = config.unified_output
    
    def __len__(self):
        return len(self.data)
    
    def set_mode(self, mode):
        self.retrieval_mode = mode
        print(f"Dataset retrieval mode set to '{self.retrieval_mode}'")

    def __getitem__(self, idx):
        chunk = self.data[idx]

        if not self.unified_output:

            if self.retrieval_mode == "uniform": # a game sequence will be randomly mapped to a token space
                chunk = self.to_tok_space(chunk, np.random.randint(0, self.nTokenSpaces))
            else:
                chunk = self.to_tok_space(chunk, int(self.retrieval_mode))

            if len(chunk) != self.max_len:
                chunk += [-100, ] * (self.max_len - len(chunk))  # -100 can be ignored in CE

            dix = [self.stoi[s] for s in chunk]

            x = torch.tensor(dix[:-1], dtype=torch.long)
            y = torch.tensor(dix[1:], dtype=torch.long)
            return x, y

        else:

            if self.retrieval_mode == "uniform": # a game sequence will be randomly mapped to a token space except the 0th
                chunk_input, sizes = self.to_tok_space(chunk, np.random.randint(1, self.nTokenSpaces), rtnSizes=True)
            else:
                sizes = None
                chunk_input = self.to_tok_space(chunk, int(self.retrieval_mode))
            
            chunk_action = self.to_tok_space(chunk, 0) # unified output space is always token space 0

            if len(chunk_input) != self.max_len:
                chunk_input += [-100, ] * (self.max_len - len(chunk_input))

            if sizes is not None:
                chunk_action_new = [-100, ] * self.max_len

                curr_idx = 0
                for i, s in enumerate(sizes):
                    chunk_action_new[curr_idx] = chunk_action[i]
                    curr_idx += s
                
                chunk_action = chunk_action_new
            else:
                if len(chunk_action) != self.max_len:
                    chunk_action += [-100, ] * (self.max_len - len(chunk_action))
            
            dix_input = [self.stoi[s] for s in chunk_input]
            dix_action = [self.stoi[s] for s in chunk_action]
            
            x = torch.tensor(dix_input[:-1], dtype=torch.long)
            y = torch.tensor(dix_action[1:], dtype=torch.long)
            return x, y