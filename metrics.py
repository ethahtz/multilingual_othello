from tqdm import tqdm
import numpy as np
import torch
from data import get_othello
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig
from mingpt.probe_model import BatteryProbeClassificationTwoLayer
from mingpt.utils import sample
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from mothello_config import MOthello_Config
from data.othello import OthelloBoardState
import pandas as pd
import argparse

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    From: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # show all ticks
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")


    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.imshow(data, cmap='viridis', interpolation='nearest')
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def plotDistanceMatrix_with_config(seqIDX, tok_sp1_id, tok_sp2_id, model, test_dataset, device, exp_name, config):
    """
    Heatmap visualization of the representational similarity between tokens of parallel sequence in two languages.

    Args:
        seqIDX: The index of the sequence to visualize
        tok_sp1_id: The index of the first language
        tok_sp2_id: The index of the second language
        model: The mOthelloGPT model
        test_dataset: The testing dataset
        device: The device to run the model on
        exp_name: The experiment name
        config: The MOthello_Config object
    
    """
    
    orig_ids = [test_dataset.stoi[_] for _ in config.to_tok_space(test_dataset.data[seqIDX][:59], tok_sp1_id)]
    trans_ids = [test_dataset.stoi[_] for _ in config.to_tok_space(test_dataset.data[seqIDX][:59], tok_sp2_id)]

    orig_input  = torch.tensor([orig_ids]).to(device)
    trans_input = torch.tensor([trans_ids]).to(device)

    labels = [str(i) for i in orig_ids] + [str(i) + "t" for i in trans_ids]

    orig_hiddens = model(orig_input, hidden=True)[-1]
    trans_hiddens = model(trans_input, hidden=True)[-1]

    layer_id = 0

    for orig_hidden, trans_hidden in tqdm(zip(orig_hiddens, trans_hiddens)):

        orig_hidden = orig_hidden.squeeze().to("cpu").detach().numpy()
        trans_hidden = trans_hidden.squeeze().to("cpu").detach().numpy()

        all_hidden = np.vstack((orig_hidden, trans_hidden))

        dists = cdist(all_hidden, all_hidden, metric="cosine")

        fig, ax = plt.subplots(figsize=(30,30))

        im, cbar = heatmap(dists, labels, labels, ax=ax, cmap="viridis", cbarlabel="distance")

        plt.tight_layout()
        plt.savefig(f"figures/{exp_name}_tsp({tok_sp1_id},{tok_sp2_id})_cosDistMtx_at_l{layer_id}")
        plt.close()

        layer_id += 1


def probe_logits2boardstate(logits):
    assert logits.shape[0] == 64
    assert logits.shape[1] == 3
    probs = torch.softmax(logits, dim=-1)  # [64, 3]
    probs, preds = torch.max(probs, dim=-1)  # [64, ], [64, ]
    return (preds - 1).reshape(8,8).numpy()


def count_state_accuracy(state_gt, state_pred):
    assert state_gt.shape == state_pred.shape
    return np.sum(state_gt == state_pred)


def compose_helper(trans_hidden, seq, config, tok_sp2_id, mode="last"):
    """
    A helper function to compose the hidden states of the tokens in a language
    where a single move is represented by multiple tokens.
    """
    assert mode in ["last", "first", "mean", "sum"], "mode must be one of 'last', 'first', 'mean', 'sum'"

    compose_info = []

    for mv in seq:
        compose_info.append(len(config.to_tok_space([mv], tok_sp2_id)))

    trans_hidden_composed = []

    curr_idx = 0

    for m in compose_info:
        if mode == "last":
            trans_hidden_composed.append(trans_hidden[curr_idx + m - 1])
        elif mode == "first":
            trans_hidden_composed.append(trans_hidden[curr_idx])
        elif mode == "mean":
            trans_hidden_composed.append(trans_hidden[curr_idx : curr_idx + m].mean(0))
        else: # mode == "sum"
            trans_hidden_composed.append(trans_hidden[curr_idx : curr_idx + m].sum(0))

        curr_idx += m

    return trans_hidden_composed


def probe_accuracy_helper(tok_sp1_id, tok_sp2_id, model, test_dataset, device, probe, layerID, flip=False):
    """
    Helper function to compute the probe accuracy between two languages at a given layer of an mOthelloGPT model.
    """

    oprobe_hit, tprobe_hit, total = 0,0,0

    for seqIDX in range(len(test_dataset)):
        partial_game = [test_dataset.stoi[_] for _ in test_dataset.to_tok_space(test_dataset.data[seqIDX][:59], tok_sp1_id)]
        trans_partial_game = [test_dataset.stoi[_] for _ in test_dataset.to_tok_space(test_dataset.data[seqIDX][:59], tok_sp2_id)]

        partial_game = torch.tensor(partial_game, dtype=torch.long).to(device)
        trans_partial_game = torch.tensor(trans_partial_game, dtype=torch.long).to(device)

        _, _, hidden_acts = model(partial_game[None, :], hidden=True)
        _, _, hidden_acts_trans = model(trans_partial_game[None, :], hidden=True)

        hidden_acts_trans_composed = compose_helper(hidden_acts_trans[layerID].squeeze(), test_dataset.data[seqIDX][:59], test_dataset, tok_sp2_id)

        board_state = OthelloBoardState()

        for i in range(len(partial_game)):

            hid_orig = hidden_acts[layerID].squeeze()[i].detach()

            hid_trans = hidden_acts_trans_composed[i]

            state_orig_pred  = probe(hid_orig)[0].squeeze().detach().cpu()
            state_trans_pred = probe(hid_trans)[0].squeeze().detach().cpu()

            state_orig_pred = probe_logits2boardstate(state_orig_pred)
            state_trans_pred = probe_logits2boardstate(state_trans_pred)

            if flip:
                state_trans_pred = 0 - state_trans_pred

            board_state.update(test_dataset.data[seqIDX][i:i+1], prt=False)

            oprobe_hit += count_state_accuracy(state_orig_pred, board_state.state)
            tprobe_hit += count_state_accuracy(state_trans_pred, board_state.state)
            total += 64
        
    return (oprobe_hit, tprobe_hit, total)


def cross_lingual_alignment_probe_accuracy_at_layer(exp_name, tok_sp1_id, tok_sp2_id, model, test_dataset, device, layerID, tableOutput=True, ckpt_dir=None):
    """
    Compute the cross-lingual alignment probe accuracy between two languages at a given layer of an mOthelloGPT model.

    Args:
        exp_name: The experiment name
        tok_sp1_id: The index of the first language
        tok_sp2_id: The index of the second language
        model: The mOthelloGPT model
        test_dataset: The testing dataset
        device: The device to run the model on
        layerID: The index of the layer to evaluate the model on
        tableOutput: Whether to output the result as a table
        ckpt_dir: The path to the checkpoint file of mOthelloGPT (if different from the default one defined in the mothello_config)
    
    Returns:
        An array / pandas dataframe of cross-lingual alignment probe accuracy between the two languages
    """

    BATTERY_DIR = f"battery_{exp_name}"

    if ckpt_dir is not None:
        special_str = ckpt_dir[:-5].split("/")[-1]
    else:
        special_str = ""

    accuracy = []

    theprobe = BatteryProbeClassificationTwoLayer(device, probe_class=3, num_task=64, mid_dim=512)
    load_res = theprobe.load_state_dict(torch.load(f"./ckpts/{BATTERY_DIR}/state{special_str}_tl512/layer{layerID}_toksp{tok_sp1_id}/checkpoint.ckpt"))

    theprobe.eval()

    oprobe_hit, tprobe_hit, total = probe_accuracy_helper(tok_sp1_id, tok_sp2_id, model, test_dataset, device, theprobe, layerID)
    accuracy.append([oprobe_hit/total, tprobe_hit/total])

    if tableOutput:
        
        result_df = pd.DataFrame(accuracy).T
        
        result_df = result_df.round(decimals=3)
        result_df.columns = [f"layer{layerID}|lang{tok_sp1_id}->lang{tok_sp2_id}"]
        result_df.index = ["In-distribution Probe Accuracy", "Cross-lingual Alignment Probe Accuracy"]

        return result_df

    else:
        return accuracy

def all_pairwise_probe_accuracy_at_layer(exp_name, model, test_dataset, config, device, layer_idx, tableOutput=True, toSave=True, ckpt_dir=None):
    """
    Compute a matrix of cross-lingual alignment probe accuracy between all pairs of languages
    at a given layer of an mOthelloGPT model.

    Args:
        exp_name: The experiment name
        model: The mOthelloGPT model
        test_dataset: The testing dataset
        config: The MOthello_Config object
        device: The device to run the model on
        layer_idx: The index of the layer to evaluate the model on
        tableOutput: Whether to output the result as a table
        toSave: Whether to save the result as a npy file
        ckpt_dir: The path to the checkpoint file of mOthelloGPT (if different from the default one defined in the mothello_config)
    
    Returns:
        An np array / pandas dataframe of cross-lingual alignment probe accuracy between all pairs of languages
    """

    nTokSpace = config.num_token_spaces

    res = np.zeros((nTokSpace, nTokSpace))

    for tok_sp1_id in range(nTokSpace):
        for tok_sp2_id in range(nTokSpace):
            
            acc = cross_lingual_alignment_probe_accuracy_at_layer(exp_name, tok_sp1_id, tok_sp2_id, model, test_dataset, device, layer_idx, tableOutput=False, ckpt_dir=ckpt_dir)[0][1]
            print(f"language pair: {(tok_sp1_id, tok_sp2_id)}, acc: {acc}")

            res[tok_sp1_id, tok_sp2_id] = acc
    
    if toSave:
        np.save(f"{exp_name}_layer{layer_idx}_cross_probe_acc.npy", res)

    if tableOutput:
        result_df = pd.DataFrame(res).T
        result_df = result_df.round(decimals=3)
        result_df.columns = [f"lang{i}" for i in range(nTokSpace)]
        result_df.index = [f"lang{i}" for i in range(nTokSpace)]

        return result_df
    else:
        return res

def eval_pred(othello, the_config, test_dataset, model, device, token_space_idx, unified=False):
    """
    Evaluates the performance as accuracy in predicting the next legal moves of an mOthelloGPT
    model on the test sequences of the given language `token_space_idx`.

    Args:
        othello: The Othello object
        the_config: The MOthello_Config object
        test_dataset: The testing dataset
        model: The mOthelloGPT model
        device: The device to run the model on
        token_space_idx: The index of the language to evaluate the model on
        unified: whether the model is trained to predict tokens in a unified output space
    
    Returns:
        The accuracy of the model in predicting the next legal moves of the given language `token_space_idx`
    """
    total_nodes = 0
    success_nodes = 0

    bar = tqdm(othello.sequences)
    for whole_game in bar:
        length_of_whole_game = len(whole_game)
        for length_of_partial_game in range(1, length_of_whole_game):
            total_nodes += 1

            context = whole_game[:length_of_partial_game]

            x = torch.tensor([test_dataset.stoi[s] for s in the_config.to_tok_space(context, token_space_idx)], dtype=torch.long)[None, ...].to(device)
            
            if not unified:
                y = sample(model, x, min(3, length_of_whole_game - length_of_partial_game), temperature=1.0)[0] # (split words) sample 3 steps and see if the ground truth next (bundle of) token(s) are predicted correctly 
            else:
                y = sample(model, x, 1, temperature=1.0)[0]

            completion = [test_dataset.itos[int(i)] for i in y if i != -1]

            try:
                if not unified:
                    assert the_config.valid_in_tok_space(completion, token_space_idx, seqLen=length_of_partial_game+1)
                
                if not unified:
                    OthelloBoardState().update(the_config.parseNuniv_repr(completion, length_of_partial_game+1), prt=False)
                else:
                    OthelloBoardState().update(context + the_config.univ_repr(completion[-1:]), prt=False)
                
            except Exception:
                pass
            else:
                success_nodes += 1
        bar.set_description(f"{success_nodes/total_nodes*100:.2f}% pass rate: {success_nodes}/{total_nodes} among all searched nodes")
    print(f"LANGUAGE-{token_space_idx} -> {success_nodes/total_nodes*100:.2f}% pass rate: {success_nodes}/{total_nodes} among all searched nodes")

    return success_nodes/total_nodes * 100

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run metrics to evaluate the alignment of token spaces')

    parser.add_argument('--exp_config',
                        required=True,
                        type=str,
                        help="Path to the experiment configuration file")          

    parser.add_argument('--metric', 
                        choices=["probeacc_layer", "probeacc_layer_all_pairs", "cosdist", "evalpred"], 
                        required=True, 
                        type=str,
                        help="Type of metric to run")

    parser.add_argument('--token_space_idx1',
                        default=0,
                        type=int,
                        help="The index of the first language for the metric")

    parser.add_argument('--token_space_idx2',
                        default=1,
                        type=int,
                        help="The index of the second language for the metric")
    
    parser.add_argument('--n_test_seqs',
                        default=100,
                        type=int,
                        help="Number of test sequences to evaluate/run the metric on")

    parser.add_argument('--probe_layer',
                        default=6,
                        type=int,
                        help="Using probes trained at this layer")

    # normally, the checkpoint file is the same as the one defined in the config file
    # the discrepancy may arise when we train fine-tuned checkpoints on a base model
    parser.add_argument('--ckpt_dir',
                    default="",
                    type=str,
                    help="Path to the checkpoint file of mOthelloGPT (if different from the default one defined in the mothello_config)")

    args, _ = parser.parse_known_args()

    the_config = MOthello_Config(args.exp_config)
    exp_name = the_config.exp_name

    othello = get_othello(ood_num=args.n_test_seqs, data_root=None, wthor=True)

    test_dataset = CharDataset(othello, the_config)

    mconf = GPTConfig(test_dataset.vocab_size, test_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)

    ckpt_dir = the_config.ckpt_dir

    load_res = model.load_state_dict(torch.load(ckpt_dir))
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = model.to(device)

    if args.metric == "probeacc_layer":
        print(cross_lingual_alignment_probe_accuracy_at_layer(exp_name, args.token_space_idx1, args.token_space_idx2, model, test_dataset, device, args.probe_layer, ckpt_dir=ckpt_dir))
    elif args.metric == "probeacc_layer_all_pairs":
        print(all_pairwise_probe_accuracy_at_layer(exp_name, model, test_dataset, the_config, device, args.probe_layer, ckpt_dir=ckpt_dir))
    elif args.metric == "cosdist":
        plotDistanceMatrix_with_config(0, args.token_space_idx1, args.token_space_idx2, model, test_dataset, device, exp_name, the_config)
    elif args.metric == "evalpred":
        eval_pred(othello, the_config, test_dataset, model, device, args.token_space_idx1, unified=the_config.unified_output)
    else:
        print(f"Unknown metric: {args.metric}")