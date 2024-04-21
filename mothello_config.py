from ast import literal_eval
import numpy as np

from mingpt.dataset import CharDataset
from data import get_othello

VALID_SQUARES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
                18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 37, 
                38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 
                54, 55, 56, 57, 58, 59, 60, 61, 62, 63] # 27,28,35,36 are not used

def load_spec(dir):
    
    with open(dir, 'r') as f:
        res = literal_eval(f.read())
    
    return res

class MOthello_Config():
    def __init__(self, spec_path: str):

        spec = load_spec(spec_path)

        assert sorted(spec.keys()) == ['language_types', 'overlap_toks', 'train_seed', 'unified_output'], "Malformed specification file"

        self.exp_name = spec_path.split('/')[-1].split(".")[0]

        self.train_seed = spec["train_seed"]
        np.random.seed(self.train_seed)

        self.retrieval_mode = 'uniform'

        self.num_token_spaces = len(spec["language_types"])
        self.ltypes = spec["language_types"]

        overlap_toks = spec["overlap_toks"]

        self.ckpt_dir = f"./ckpts/gpt_{self.exp_name}.ckpt"

        self.unified_output = spec["unified_output"]

        self.tok_dict = [{} for _ in range(self.num_token_spaces)]

        # Validating the overlapping tokens in the specification and make the spec normalized
        for tok in overlap_toks:
            assert tok in VALID_SQUARES
            tok_spaces_set = set()
            
            for i in range(len(overlap_toks[tok])):
                overlap_toks[tok][i].sort() # this list indicates a tokspace group

                base_tok_space = overlap_toks[tok][i][0] # the "base" token space in that group

                for j, tokspace in enumerate(overlap_toks[tok][i]):
                    assert tokspace < self.num_token_spaces

                    if tokspace not in tok_spaces_set:
                        tok_spaces_set.add(tokspace)
                    else:
                        raise RuntimeError(f"Language {tokspace} used twice for overlapping token {tok} - collapse it manually")
                    
                    if j > 0: # not the "base" token space for that overlap token
                        self.tok_dict[tokspace][tok] = base_tok_space

        vocabs = []

        vocab_index = 0

        for tok_space_id, l_type in zip(list(range(self.num_token_spaces)), self.ltypes):

            if l_type == "compositional":

                cols = [vocab_index + i for i in range(8)]
                vocab_index += 8

                rows = [vocab_index + i for i in range(8)]
                vocab_index += 8

                vocabs += cols + rows

            else:
                cols, rows = None, None

            for sqaure_id in VALID_SQUARES:
                if sqaure_id not in self.tok_dict[tok_space_id]:

                    if l_type == "atomic":
                        vocab = vocab_index
                        vocab_index += 1
                        self.tok_dict[tok_space_id][sqaure_id] = [vocab]
                        vocabs.append(vocab)
                    elif l_type == "split":
                        nsplitwords = np.random.randint(1,4)
                        self.tok_dict[tok_space_id][sqaure_id] = []
                        for i in range(nsplitwords):
                            vocab = vocab_index
                            vocab_index += 1
                            self.tok_dict[tok_space_id][sqaure_id].append(vocab)
                            vocabs.append(vocab)
                    elif l_type == "compositional":
                        self.tok_dict[tok_space_id][sqaure_id] = [cols[sqaure_id // 8], rows[sqaure_id % 8]]
                    else:
                        raise RuntimeError(f"Unknown language type '{l_type}': only types ['atomic', 'split', 'compositional'] are supported.")
                else:
                    base_tok_space = self.tok_dict[tok_space_id][sqaure_id]
                    self.tok_dict[tok_space_id][sqaure_id] = self.tok_dict[base_tok_space][sqaure_id] # uses the same token as its base token space

        self.vocabs = vocabs

        print(f"===== MOthello configuration '{self.exp_name}' loaded =====\n {self.num_token_spaces} token spaces, {len(self.vocabs)} unique vocabs, directory: '{self.ckpt_dir}'\n======================================================================")


    def to_tok_space(self, sequence, tok_space_id, rtnSizes=False):
        """
        Given a sequence in its universal representation, translate it to
        a token space specified by the tok_space_id
        do not modify -100
        """

        assert tok_space_id < self.num_token_spaces, f"Token space ID {tok_space_id} is out of range"

        tok_lists = [self.tok_dict[tok_space_id][mov] if mov != -100 else -100 for mov in sequence]
        res = []
        sizes = []
        for l in tok_lists: 
            if isinstance(l, list): # flatten the list if necessary
                sizes.append(len(l))
                for t in l:
                    res.append(t)
            else:
                sizes.append(1)
                res.append(l)

        if rtnSizes:
            if self.ltypes[tok_space_id] == "atomic":
                return res, None
            return res, sizes
        else:
            return res

         
    def valid_in_tok_space(self, sequence, tok_space_id, seqLen=-1):
        """
        checks whether a sequence is a valid sequence in a specific token space
        """
        def findSquareID(seq, tok_space_id):
            for sqaure_id in VALID_SQUARES:
                for l in range(1, 4):
                    if self.tok_dict[tok_space_id][sqaure_id] == seq[:l]:
                        return sqaure_id, l

            return None, 0
        
        idx = 0
        res = []

        if seqLen == -1: # that means to parse the entire sequence
            while idx < len(sequence):
                sqrID, length = findSquareID(sequence[idx:], tok_space_id)
                if length == 0:
                    return False
                else:
                    res.append(sqrID)
                    idx += length
        else:
            count = 0
            while idx < len(sequence) and count < seqLen:
                sqrID, length = findSquareID(sequence[idx:], tok_space_id)
                if length == 0:
                    return False
                else:
                    res.append(sqrID)
                    idx += length
                    count += 1
        
        return True
        

    def univ_repr(self, sequence): 
        """
        Given a sequence in any token space, translate it back to its
        universal representation, which is recognizable by the Othello ground
        truth functions
        """
        return self.parseNuniv_repr(sequence)

    def parseNuniv_repr(self, seq, parseNtokens=-1):

        def findSquareID(seq):
            if seq[0] == -100:
                return -100, 1
            for sqaure_id in VALID_SQUARES:
                for i in range(self.num_token_spaces):
                    for l in range(1, 4):
                        if self.tok_dict[i][sqaure_id] == seq[:l]:
                            return sqaure_id, l

            return None, 0
        
        idx = 0
        res = []

        if parseNtokens == -1: # that means to parse the entire sequence
            while idx < len(seq):
                sqrID, length = findSquareID(seq[idx:])
                if length == 0:
                    raise RuntimeError("Fail to parse input")
                else:
                    res.append(sqrID)
                    idx += length
        else:
            count = 0
            while idx < len(seq) and count < parseNtokens:
                sqrID, length = findSquareID(seq[idx:])
                if length == 0:
                    raise RuntimeError("Fail to parse input")
                else:
                    res.append(sqrID)
                    idx += length
                    count += 1
        
        return res

if __name__ == "__main__":

    # pass
    cfg = MOthello_Config('test_config')

    for i in range(cfg.num_token_spaces):
        assert cfg.valid_in_tok_space(cfg.to_tok_space(VALID_SQUARES, i), i), f"VALIDITY TEST: Failed at token space {i}"
        assert VALID_SQUARES == cfg.univ_repr(cfg.to_tok_space(VALID_SQUARES, i)), f"UNIVERSAL_REPR TEST: Failed at token space {i}"

    othello = get_othello(ood_num=10, data_root=None, wthor=True)

    dataset = CharDataset(othello, cfg)