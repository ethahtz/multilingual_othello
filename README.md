# Multilingual Othello

  

This repository contains the code for running experiments presented in [mOthello: When Do Cross-Lingual Representation Alignment and Cross-Lingual Transfer Emerge in Multilingual Models?] 

The implementation of GPT2 is based on [minGPT](https://github.com/karpathy/minGPT), and the overall codebase is mostly built on [Othello World](https://github.com/likenneth/othello_world). Many thanks to Andrej Karpathy and Kenneth Li for open-sourcing their projects!

  

## Abstract

  

> Many pretrained multilingual models exhibit cross-lingual transfer ability, which is often attributed to a learned language-neutral representation during pretraining. However, it remains unclear what factors contribute to the learning of a language-neutral representation, and whether the learned language-neutral representation suffices to facilitate cross-lingual transfer. We propose a synthetic task, Multilingual Othello (mOthello), as a testbed to delve into these two questions. We find that: (1) models trained with naive multilingual pretraining fail to learn a language-neutral representation across all input languages; (2) the introduction of "anchor tokens" (i.e., lexical items that are identical across languages) helps cross-lingual representation alignment; and (3) the learning of a language-neutral representation alone is not sufficient to facilitate cross-lingual transfer. Based on our findings, we propose a novel approach -- multilingual pretraining with unified output space -- that both induces the learning of language-neutral representation and facilitates cross-lingual transfer.

## Table of Contents

  

1. [Installation](#installation)

2. [Defining a multilingual Othello Instance](#defining-a-multilingual-othello-instance)

3. [Training mOthelloGPT](#training-mothellogpt)

4. [Probing mOthelloGPT](#probing-mothellogpt)

5. [Running Finetuning Experiments](#running-finetuning-experiments)

6. [How to Cite](#how-to-cite)

  

## Installation

  

Use these commands to set up:

```
conda env create -f environment.yml
conda activate othello
mkdir ckpts
```

  

## Defining a Multilingual Othello Instance

 

Each mOthello Instance is defined by a configuration text file in the `exp_configs` folder. We have provided some examples in that folder. For instance:

```
{ 'languagetypes': ["atomic", "atomic", "split", "compositional"],
'overlaptoks':
{57: 0, 3,
9 : 0, 3,
40: 0, 3,
32: [[0, 1], [2, 3]],
},
'trainseed': 42,
'unifiedoutput': False}
```

Each configuration needs to have 
- `language_types`: a list indicating the type of each language; 
- `overlap_toks`: a dictionary where the keys are the move indices, and the values are list of sets of language indexes that share that token (For instance: `32: [[0, 1], [2, 3]]` means that move `32` is a shared token between language `0` and language `1`, and language `2` and language `3`); 
- `train_seed`: an integer indicating the random seed used in training, and
- `unified_output`: a Boolean indicating whether the unified output space approach is to be used

 
 
 
## Training mOthelloGPT

  

We use the same synthetic dataset provided in Othello World. You can download the [synthetic dataset](https://drive.google.com/drive/folders/1pDMdMrnxMRiDnUd-CNfRNvZCi7VXFRtv?usp=sharing) and save them in `data` subfolder. 

  

Train an mOthelloGPT:
```
python train_mothello_gpt.py --exp_config CONFIG_PATH
``` 
where `CONFIG_PATH` is the path to the config file
 

Note: if your mOthello instance includes non-atomic language types, the computational requirement will be higher since the maximum length of a sequence will be longer. Consider decreasing the `--batch_size` for this type of experiments.

  

## Probing mOthelloGPT

  

Then we will use `train_probe_othello.py` to train probes.  
 
Normally, we train a two-layer probe classifier on inputs from language `LANG_IDX` using 

```
python3 trainprobeothello.py --twolayer --exp_config CONFIG_PATH --tokspidx LANGIDX
```

If the model checkpoint that you want to train a probe is different from its default checkpoint path (taht is in `CKPT_DIR` instead), you can use 

```
python3 train_probe_othello.py --twolayer --exp_config CONFIG_PATH --token_space_idx LANG_IDX --ckpt_dir CKPT_DIR
```

(This only happens during the finetuning experiments.) Note: currently we only support training probes on inputs from atomic languages. 

 
We also provided bash scripts that help you train a set of probes with convenience:
```
./train_probes_all_langs.sh CONFIG_PATH NUM_LANGS
```
Trains a probe for each language in the mOthelloGPT, and can be used for running experiments involving computing the `probeacc_layer_all_pairs` metric

---

```
./train_probes_ft_ckpts.sh CONFIG_PATH
``` 
Trains the probes with inputs from the language being finetuned on, using model checkpoints saved throughout the finetuning process, this is helpful for preparing probes before you run `ft_inspect.py`

 

## Running Metrics

 

After the mOthelloGPT and necessary probes are trained, we can compute following metrics in `metrics.py`:
```
python metrics.py --metric probeacc_layer --exp_config CONFIG_PATH --token_space_idx1 i --token_space_idx2 j
```
Computes the cross-lingual-alignment-probe accuracy from language `i` to language `j` in the mOthelloGPT trained on multilingual Othello defined by `CONFIG_PATH`.

---

 ```
 python metrics.py --metric probeacc_layer_all_pairs --exp_config CONFIG_PATH
 ```
 Computes the cross-lingual-alignment-probe accuracy between all pairs of languages in the mOthelloGPT trained on multilingual Othello defined by `CONFIG_PATH` and saves the accuracies in a `npy` file.

---

```
python metrics.py --metric cosdist --exp_config CONFIG_PATH --token_space_idx1 i --token_space_idx2 j
```
Plots the cosine distance matrix of token hidden representations of parallel sequences in language `i` and language `j` across all layers; results are saved in `figures` folder.

---


```
python metrics.py --metric evalpred --exp_config CONFIG_PATH --token_space_idx1 i
```
Evaluates the top-1 legal move prediction in language `i` of the mOthelloGPT trained on multilingual Othello defined by `CONFIG_PATH`.

## Running Finetuning Experiments

 

You can use 
```
python ft_experiment.py --exp_config CONFIG_PATH
``` 
to pretrain and finetune an mOthelloGPT in the same way as we reported in our paper. After the models are trained and checkpoints saved, you can run 
```
python ft_inspect.py --exp_config CONFIG_PATH
```
to see prediction accuracy (of each language) and cross-lingual alignment probe accuracy (averaged between the FT language and all other target languages) of the model throughout the finetuning process.

 

## How to Cite

```
@inproceedings{
hua2024mothello,
title={mOthello: When Do Cross-Lingual Representation Alignment and Cross-Lingual Transfer Emerge in Multilingual Models?},
author={Tianze Hua and Tian Yun and Ellie Pavlick},
booktitle={2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
year={2024}
}
```