from typing import Iterable, Tuple

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from pathlib import Path
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import json
import gzip
import pickle
import re


def strip_newlines(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub("\s+", " ", text)
    return text


def batching(it: Iterable, batch_size: int):
    assert batch_size > 0
    ret = []
    for item in it:
        ret.append(item)
        if len(ret) >= batch_size:
            yield ret
            ret = []
    if len(ret) > 0:
        yield ret


def find_unmerged_inverted_indexes(search_in: Path, prefix: str):
    search_in = Path(search_in)
    index_files = list(search_in.glob(f"{prefix}-*.gz"))
    if len(index_files) > 0:
        return gzip.open, index_files
    return open, list(search_in.glob(f"{prefix}-*.pkl"))


def write_inverted_index(compress: bool, output_fn: Path, docnos: np.ndarray, chunk: sp.spmatrix, verbose: bool=False):
    with (open if not compress else gzip.open)(output_fn, 'wb') as fw:
        pickle.dump((docnos, chunk), fw)
        if verbose:
            tqdm.write(f"Saved index file {output_fn.name} with shape {chunk.shape} and nnz {chunk.nnz}")


def read_inverted_index(fn: Path) -> Tuple[np.ndarray, sp.spmatrix]:
    fn = Path(fn)
    return pickle.load( (gzip.open if fn.name.endswith('.gz') else open)(fn, 'rb') )


def ensure_local_fn(fn: str):
    if not Path(fn).exists():
        try:
            fn = hf_hub_download(*fn.split(":"))
        except EntryNotFoundError:
            raise FileNotFoundError(f"{fn} is not a local file nor on Huggingface model.")
    return fn


def read_raw_dictionary(fn: str):
    fn = ensure_local_fn(fn)
    return json.load((gzip.open if fn.endswith('.gz') else open)(fn, 'rt'))
