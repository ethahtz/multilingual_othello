import numpy as np
import torch
from data import get_othello
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig
from mothello_config import MOthello_Config
from metrics import cross_lingual_alignment_probe_accuracy_at_layer, eval_pred
from tqdm import tqdm
from glob import glob 
from natsort import natsorted
import argparse

parser = argparse.ArgumentParser(description='Inspect fine-tuned models')

parser.add_argument('--exp_config',
                    required=True,
                    type=str,
                    help="Path to the experiment configuration file")

parser.add_argument('--n_test_seqs',
                    default=100,
                    type=int,
                    help="Number of test sequences")

args, _ = parser.parse_known_args()

othello = get_othello(ood_num=args.n_test_seqs, data_root=None, wthor=True)

the_config = MOthello_Config(args.exp_config)
exp_name = the_config.exp_name
testing_dataset = CharDataset(othello, the_config)

mconf = GPTConfig(testing_dataset.vocab_size, testing_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
pt_model = GPT(mconf)
load_res = pt_model.load_state_dict(torch.load(the_config.ckpt_dir))
device = torch.cuda.current_device()
pt_model = pt_model.to(device)

ft_models = []

ckpt_dirs = [the_config.ckpt_dir]

for ckpt_path in natsorted(glob(f"./ckpts/gpt_{exp_name}_ft_epoch*.ckpt")):
    print(f"Loading fine-tuned model from {ckpt_path}")
    mconf = GPTConfig(testing_dataset.vocab_size, testing_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    load_res = model.load_state_dict(torch.load(ckpt_path))
    ft_models.append(model)
    ckpt_dirs.append(ckpt_path)

ft_models = [model.to(device) for model in ft_models]

all_accs = [[] for _ in range(the_config.num_token_spaces)] # next legal move prediction performance on each language
all_sims = [] # representational similarity between the finetuning langauge and all other languages

# Evaluate the models performance on each language
for m in [pt_model] + ft_models:
    for token_space_idx in range(1 if the_config.unified_output else 0, the_config.num_token_spaces):
        acc = eval_pred(othello, the_config, testing_dataset, m, device, token_space_idx, the_config.unified_output)
        all_accs[token_space_idx].append(acc)

# Compute cross-lingual alignment probe accuracy between the finetuning language and all other languages
for i, m in enumerate([pt_model] + ft_models):

    accs = []    
    if the_config.unified_output:
        for token_space_idx in range(2, the_config.num_token_spaces):
            acc = cross_lingual_alignment_probe_accuracy_at_layer(exp_name, 1, token_space_idx, m, testing_dataset, device, 6, ckpt_dir=ckpt_dirs[i], tableOutput=False)
            accs.append(acc)
    else:
        for token_space_idx in range(1, the_config.num_token_spaces):
            acc = cross_lingual_alignment_probe_accuracy_at_layer(exp_name, 0, token_space_idx, m, testing_dataset, device, 6, ckpt_dir=ckpt_dirs[i], tableOutput=False)
            accs.append(acc)
    
    accs = [acc[0][1] for acc in accs]

    avg_acc = np.mean(accs)

    print(i, "-th model, representational sim: ", avg_acc)
    all_sims.append(avg_acc)

print("Next legal move prediction accuracy performance of each language throughout finetuning: ", all_accs)
print("Average cross-lingual alignment probe accuracy between FT language and all other languages throughout finetuning: ", all_sims)