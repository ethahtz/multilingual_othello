import os
# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# make deterministic
from mingpt.utils import set_seed
set_seed(44)

import time
from tqdm import tqdm
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from data import get_othello
from data.othello import OthelloBoardState
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig, GPTforProbing
from mingpt.probe_trainer import Trainer, TrainerConfig
from mothello_config import MOthello_Config
from mingpt.probe_model import BatteryProbeClassification, BatteryProbeClassificationTwoLayer

parser = argparse.ArgumentParser(description='Train probing classification network')
parser.add_argument('--layer',
                    default=6,
                    type=int,
                    help="Layer to probe")

parser.add_argument('--epo',
                    default=16,
                    type=int,
                    help="Number of epochs")

parser.add_argument('--mid_dim',
                    default=512,
                    type=int)

parser.add_argument('--token_space_idx',
                    default=0,
                    type=int,
                    help="Token space index as input for probing (for non-unified exps)")

parser.add_argument('--token_space_idx_unified',
                    default=1,
                    type=int,
                    help="Token space index as input for probing (for unified exps)")

parser.add_argument('--twolayer',
                    dest='twolayer', 
                    action='store_true',)

parser.add_argument('--random',
                    dest='random', 
                    action='store_true')

parser.add_argument('--exp',
                    default="state", 
                    type=str)

parser.add_argument('--exp_config',
                    required=True,
                    type=str,
                    help="Path to the experiment configuration file")          

parser.add_argument('--ckpt_dir',
                    default="",
                    type=str)   

args, _ = parser.parse_known_args()

config = MOthello_Config(args.exp_config)
exp_name = config.exp_name

ckpt_dir = config.ckpt_dir

if args.ckpt_dir != "":
    ckpt_dir = args.ckpt_dir

BAT_DIR = f"battery_{exp_name}"
VOCAB_SIZE = len(config.vocabs) + 1

folder_name = f"{BAT_DIR}/{args.exp}" + ckpt_dir.split("/")[-1].split(".")[0]

if args.twolayer:
    folder_name = folder_name + f"_tl{args.mid_dim}" 
if args.random:
    folder_name = folder_name + "_random"


print(f"======\nRunning experiment for {folder_name}/layer{args.layer} with {args.mid_dim} mid dimensions\n======")

# No injection of translated token space
othello = get_othello(ood_num=1000, data_root=None)

train_dataset = CharDataset(othello, config)

if config.unified_output:
    train_dataset.set_mode(args.token_space_idx_unified)
    token_space_idx_used = args.token_space_idx_unified
    assert config.ltypes[args.token_space_idx_unified] == "atomic", "Currently the probe training script only supports atomic languages, you are welcome to use the compose_helper function to train probes on the aggregated representation of other types of languages"
else:
    train_dataset.set_mode(args.token_space_idx)
    token_space_idx_used = args.token_space_idx
    assert config.ltypes[args.token_space_idx] == "atomic", "Currently the probe training script only supports atomic languages, you are welcome to use the compose_helper function to train probes on the aggregated representation of other types of languages"

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPTforProbing(mconf, probe_layer=args.layer)
if args.random:
    model.apply(model._init_weights)
else:  # trained on synthetic dataset
    load_res = model.load_state_dict(torch.load(ckpt_dir))
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = model.to(device)

loader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=1, num_workers=1)

act_container = []
property_container = []
age_container = []
for x, y in tqdm(loader, total=len(loader)):
    tbf = config.parseNuniv_repr([train_dataset.itos[_] for _ in x.tolist()[0]], parseNtokens=60)

    valid_until = tbf.index(-100) if -100 in tbf else 999
    a = OthelloBoardState()
    b = OthelloBoardState()
    properties = a.get_gt(tbf[:valid_until], "get_" + args.exp)  # [block_size, ]
    act = model(x.to(device))[0, ...].detach().cpu()  # [block_size, f]
    ages = b.get_gt(tbf[:valid_until], "get_age")

    if len(properties) != len(act):
        act = act[:len(properties)]
    act_container.extend([_[0] for _ in act.split(1, dim=0)[:valid_until]])
    property_container.extend(properties[:valid_until])
    age_container.extend(ages[:valid_until])

if args.exp == "state":
    probe_class=3

if args.twolayer:
    probe = BatteryProbeClassificationTwoLayer(device, probe_class=probe_class, num_task=64, mid_dim=args.mid_dim)
else:
    probe = BatteryProbeClassification(device, probe_class=probe_class, num_task=64)
    
class ProbingDataset(Dataset):
    def __init__(self, act, y, age):
        assert len(act) == len(y)
        assert len(act) == len(age)
        print(f"{len(act)} pairs loaded...")
        self.act = act
        self.y = y
        self.age = age
        print(np.sum(np.array(y)==0), np.sum(np.array(y)==1), np.sum(np.array(y)==2))
        
        long_age = []
        for a in age:
            long_age.extend(a)
        long_age = np.array(long_age)
        counts = [np.count_nonzero(long_age == i) for i in range(60)]
        del long_age
        print(counts)
    def __len__(self, ):
        return len(self.y)
    def __getitem__(self, idx):
        return self.act[idx], torch.tensor(self.y[idx]).to(torch.long), torch.tensor(self.age[idx]).to(torch.long)

probing_dataset = ProbingDataset(act_container, property_container, age_container)
train_size = int(0.8 * len(probing_dataset))
test_size = len(probing_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(probing_dataset, [train_size, test_size])
sampler = None
train_loader = DataLoader(train_dataset, shuffle=False, sampler=sampler, pin_memory=True, batch_size=128, num_workers=1)
test_loader = DataLoader(test_dataset, shuffle=True, pin_memory=True, batch_size=128, num_workers=1)

max_epochs = args.epo
t_start = time.strftime("_%Y%m%d_%H%M%S")
tconf = TrainerConfig(
    max_epochs=max_epochs, batch_size=1024, learning_rate=1e-3,
    betas=(.9, .999), 
    lr_decay=True, warmup_tokens=len(train_dataset)*5, 
    final_tokens=len(train_dataset)*max_epochs,
    num_workers=4, weight_decay=0., 
    ckpt_path=os.path.join("./ckpts/", folder_name, f"layer{args.layer}_toksp{token_space_idx_used}")
)
trainer = Trainer(probe, train_dataset, test_dataset, tconf)
trainer.train(prt=True)
trainer.save_traces()
trainer.save_checkpoint()
