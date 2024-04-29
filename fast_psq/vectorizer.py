from pathlib import Path
from tqdm.auto import tqdm
import gzip
import pickle

from typing import Callable, Dict, List, Type, Union
from collections import Counter

import numpy as np
from scipy import sparse as sp

from fast_psq.utils import ensure_local_fn

def _binary_merge(d: Dict[str, Dict[str, float]], keys=None):
    if keys == None:
        keys = list(d.keys())
    l = len(keys)
    if l <= 2:
        return set([ v for k in keys for v in d[k].keys() ])
    return set(_binary_merge(d, keys[:l//2])) | set(_binary_merge(d, keys[l//2:]))

def _make_vectors(tokenizer: Callable[[str], List[str]], vocab_inv: Dict[str, int], text: List[str], doclen_normalized: bool):
    if isinstance(text, str):
        text = [text]
    
    val, col, row, doclens = [], [], [], []
    for i, t in enumerate(text):
        tokens = tokenizer(t)
        doclens.append(len(tokens))
        normalizer = doclens[-1] if doclen_normalized else 1.
    
        idx_vals = [ (vocab_inv[st], v/normalizer) for st, v in Counter(tokens).items() if st in vocab_inv ]
        if len(idx_vals) > 0:
            dcol, dval = list(zip(*idx_vals))
            val += dval
            col += dcol
            row += [i]*len(dval)

    return sp.csr_matrix((val, (row, col)), shape=(max(row)+1, len(vocab_inv) )), doclens


class PSQVectorizer:

    def __init__(self, matrix: sp.spmatrix, source_vocab: List[str], target_vocab: List[str], 
                 source_tokenizer: Callable[[str], List[str]] = None, 
                 target_tokenizer: Callable[[str], List[str]] = None, 
                 alpha: float=0.1,
                 qimp: np.ndarray=None):
        self.matrix = matrix
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.alpha = alpha
        self.qimp = qimp

        self.source_vocab_inv = { t: i for i, t in enumerate(source_vocab) }
        self.target_vocab_inv = { t: i for i, t in enumerate(target_vocab) }
    
    def _get_params(self):
        return {
            'alpha': self.alpha, 'qimp': self.qimp
        }

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, str):
            key = self.source_vocab_inv[key]
        assert isinstance(key, int)
        return self.matrix[:, key]

    def make_source_vectors(self, text: List[str], return_lengths: bool=False):
        vec, lens =  _make_vectors(self.source_tokenizer, self.source_vocab_inv, text, True)
        return (vec, lens) if return_lengths else vec

    def make_target_vectors(self, text: List[str], return_lengths: bool=False):
        vec, lens =  _make_vectors(self.target_tokenizer, self.target_vocab_inv, text, False)
        return (vec, lens) if return_lengths else vec
    
    def translate(self, source: Union[sp.spmatrix, List[str]], log_transform: bool=True, return_lengths: bool=False):
        if isinstance(source, list):
            source, lengths = self.make_source_vectors(source, return_lengths=True)
        d: sp.csc_matrix = (self.matrix @ source.T).tocsc()
        
        if log_transform:
            d.data = np.log(d.data)

        if self.qimp is not None:
            target_term_imp = self.qimp[d.indices]
            d.data = np.logaddexp(d.data + np.log(1-self.alpha), np.log(target_term_imp)) - np.log(target_term_imp)
            # d.data = np.log(d.data * (1-self.alpha) + target_term_imp) - np.log(target_term_imp)
        
        return (d, lengths) if return_lengths else d

    def decode_target(self, target_m: sp.csc_matrix):
        ret = [ {} for _ in range(target_m.shape[1]) ]
        for i in range(target_m.shape[0]):
            token = self.target_vocab[i]
            # ptrs = target_m.indptr[i: i+1]
            for j in range(target_m.indptr[i], target_m.indptr[i+1]):
                weight = float(target_m.data[j])
                ret[target_m.indices[j]][token] = weight
        
        return ret

    def load_query_term_importance(self, counts_file: str):
        importance = np.ones(len(self.target_vocab))
        total_count = 1
        for l in open(ensure_local_fn(counts_file), encoding="utf-8"):
            l = l.strip().split()
            count = int(l[0])
            token = l[1].lower()

            total_count += count
            if token in self.target_vocab_inv:
                importance[ self.target_vocab_inv[token] ] += count

        self.qimp = (importance / total_count) * self.alpha

    @classmethod
    def from_dict(cls, psq_matrix: Dict[str, Dict[str, float]], 
                  with_pbar: bool = False, **kwargs):
        source_vocab = sorted(list(psq_matrix.keys()))
        target_vocab = sorted(list(_binary_merge(psq_matrix, source_vocab)))
        source_vocab_inv = { t: i for i, t in enumerate(source_vocab) }
        target_vocab_inv = { t: i for i, t in enumerate(target_vocab) }
        
        data, row, col = [], [], []
        for source_token, possible_trans in tqdm(psq_matrix.items(), disable=not with_pbar, dynamic_ncols=True):
            for tran, v in possible_trans.items():
                row.append(target_vocab_inv[tran])
                col.append(source_vocab_inv[source_token])
                data.append(v)

        matrix = sp.csc_matrix((data, (row, col)), shape=(len(target_vocab), len(source_vocab)))
        return cls(matrix, source_vocab, target_vocab, **kwargs)
    
    @classmethod
    def load(cls, fn: Union[str, Path], **kwargs):
        loaded = pickle.load(gzip.open(fn, 'rb'))
        return cls(*loaded[:-1], **loaded[-1], **kwargs)
        
    def save(self, fn: Union[str, Path]):
        with gzip.open(fn, 'wb') as fw:
            pickle.dump((
                self.matrix, self.source_vocab, self.target_vocab, self._get_params()
            ), fw)