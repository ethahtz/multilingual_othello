import time
from data import get_othello
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mothello_config import MOthello_Config
from mingpt.utils import set_seed
import argparse

parser = argparse.ArgumentParser(description='Train mOthelloGPT on Multilingual Othello')

parser.add_argument('--exp_config',
                    required=True,
                    type=str,
                    help="Path to the experiment configuration file")          

parser.add_argument('--max_epochs',
                    default=9,
                    type=int)

parser.add_argument('--batch_size',
                    default=1024,
                    type=int)

parser.add_argument('--lr',
                    default=5e-4,
                    type=float)


args, _ = parser.parse_known_args()

the_config = MOthello_Config(args.exp_config)

# make deterministic
set_seed(the_config.train_seed)
print(f"Training seed set to {the_config.train_seed}")

othello = get_othello(ood_num=-1, data_root=None, wthor=True)
train_dataset = CharDataset(othello, the_config)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

max_epochs = args.max_epochs
# initialize a trainer instance and kick off training
t_start = time.strftime("_%Y%m%d_%H%M%S")
tconf = TrainerConfig(
    max_epochs=max_epochs, 
    batch_size=args.batch_size,
    learning_rate=args.lr,
    lr_decay=True, 
    warmup_tokens=len(train_dataset)*train_dataset.block_size*5, 
    final_tokens=len(train_dataset)*train_dataset.block_size*max_epochs,
    num_workers=0, 
    ckpt_path=the_config.ckpt_dir, 
)

trainer = Trainer(model, train_dataset, None, tconf)
device = trainer.device
print(f"Starting time: {t_start}")
trainer.train()