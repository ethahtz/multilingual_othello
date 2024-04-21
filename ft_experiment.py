from mingpt.utils import set_seed
set_seed(44)
import time
import torch
from torch.utils.data import random_split, Subset
from data import get_othello
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mothello_config import MOthello_Config
import argparse

LIMITED_PREFIX = [19, 18, 17]

def get_random_subset(dataset, d_size):
    """
    Return a random subset of the dataset
    """
    indices = torch.randperm(len(dataset))[:d_size]
    return Subset(dataset, indices)

def countPrefix(seqs: list, prefix: list):
    """
    Return the number of prefix-filtered sequences and the index of the first 
    and last sequence that has the prefix. Since the synthetic sequences are 
    sorted, sequences withe the same prefix are contiguous.

    Args:
        seqs (list): List of (sorted) othello game sequences
        prefix (list): Prefix to filter the sequences
    
    Returns:
        int: Number of sequences with the prefix
        int: Index of the first sequence with the prefix
        int: Index of the last sequence with the prefix
    """
    idx, count = 0, 0
    left = None

    while idx < len(seqs) and seqs[idx][:len(prefix)] != prefix:
        idx += 1
    
    left = idx
    
    while idx < len(seqs) and seqs[idx][:len(prefix)] == prefix:
        count += 1
        idx += 1
        
    return count, left, idx


parser = argparse.ArgumentParser(description='Running Fine-tuning experiments on mOthelloGPT')

parser.add_argument('--exp_config',
                    required=True,
                    type=str,
                    help="Path to the experiment configuration file")

parser.add_argument('--max_PT_epochs',
                    default=40,
                    type=int,
                    help="Maximum number of epochs for pretraining")

parser.add_argument('--max_FT_epochs',
                    default=4,
                    type=int,
                    help="Maximum number of epochs for fine-tuning")

parser.add_argument('--batch_size_pt',
                    default=1024,
                    type=int,
                    help="Batch size for pretraining")

parser.add_argument('--batch_size_ft',
                    default=1024,
                    type=int,
                    help="Batch size for fine-tuning")

parser.add_argument('--lr',
                    default=5e-4,
                    type=float,
                    help="Learning rate")

parser.add_argument('--ft_language_index',
                    default=0,
                    type=int,
                    help="Index of the language to fine-tune on")

parser.add_argument('--ft_language_index_unified',
                    default=1,
                    type=int,
                    help="Index of the language to fine-tune on for unified output")

parser.add_argument('--ft_dataset_size',
                    default=102400,
                    type=int,
                    )

args, _ = parser.parse_known_args()


the_config = MOthello_Config(args.exp_config)

othello = get_othello(ood_num=-1, data_root=None, wthor=True)

seqs_count, left_idx, right_idx = countPrefix(othello, LIMITED_PREFIX)

# we use prefix-filtered sequences for pretraining
pretrain_dataset = CharDataset(othello[left_idx:right_idx], the_config)

# split the dataset into training and validation for early stopping
train_dataset, val_dataset = random_split(pretrain_dataset, [seqs_count-30000, 30000])

mconf = GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

t_start = time.strftime("_%Y%m%d_%H%M%S")

tconf = TrainerConfig(
    max_epochs=args.max_PT_epochs, 
    batch_size=args.batch_size_pt,
    learning_rate=args.lr,
    lr_decay=True, 
    warmup_tokens=len(train_dataset)*pretrain_dataset.block_size*5, 
    final_tokens=len(train_dataset)*pretrain_dataset.block_size*args.max_PT_epochs,
    num_workers=0, 
    ckpt_path=the_config.ckpt_dir, 
)

trainer = Trainer(model, train_dataset, val_dataset, tconf)
device = trainer.device
print(f"Pretraining - starting time: {t_start}")
trainer.train()

# load the best model so far
model.load_state_dict(torch.load(the_config.ckpt_dir))

full_dataset = CharDataset(othello, the_config)

if the_config.unified_output:
    # we fine-tune the first non-unified output language as the source language
    full_dataset.set_mode(args.ft_language_index_unified)
else:
    full_dataset.set_mode(args.ft_language_index)

# we use a random subset of the even-distribution dataset for fine-tuning
ft_dataset = get_random_subset(full_dataset, args.ft_dataset_size)

tconf = TrainerConfig(
    max_epochs=args.max_FT_epochs, 
    batch_size=args.batch_size_ft,
    learning_rate=args.lr,
    lr_decay=True, 
    warmup_tokens=len(ft_dataset)*pretrain_dataset.block_size*5, 
    final_tokens=len(ft_dataset)*pretrain_dataset.block_size*args.max_FT_epochs,
    num_workers=0, 
    ckpt_path=the_config.ckpt_dir[:-5] + "_ft.ckpt",
)

trainer = Trainer(model, ft_dataset, None, tconf, save_intermediate=True, save_interval=int((len(ft_dataset)/args.batch_size_ft)/5))
device = trainer.device

t_start = time.strftime("_%Y%m%d_%H%M%S")
print(f"Finetuning - starting time: {t_start}")
trainer.train()