import heapq
import json
import tempfile
import argparse
from tqdm import tqdm

from pathlib import Path
import shutil
from functools import partial
from multiprocessing import Pool

from typing import Tuple, List, Dict

import numpy as np
from scipy import sparse as sp

from fast_psq.tokenizer import PSQTokenizer
from fast_psq.vectorizer import PSQVectorizer
from fast_psq.utils import (strip_newlines, batching, 
                            find_unmerged_inverted_indexes, 
                            write_inverted_index,
                            read_inverted_index, 
                            read_raw_dictionary)

import logging
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("fast-psq-index") 


def read_documents(args):
    with open(args.doc_file) as f:
        for line in f:
            temp = json.loads(line)
            title = strip_newlines(temp[args.title]) 
            text = strip_newlines(temp[args.body])
            yield temp[args.docid], title + " " + text


def prune_cdf(probs: List[Tuple[str, float]], max_cdf: float=1.0):
    probs = sorted(probs, key=lambda x: -x[1])
    idx_over_max_cdf = np.where(np.cumsum([ x[1] for x in probs ]) >= max_cdf)[0]
    
    if idx_over_max_cdf.size == 0:
        return probs
    
    probs = probs[:idx_over_max_cdf[0]]
    sum_probs = sum(p for t, p in probs)
    return [ (t, p/sum_probs) for t, p in probs ]
    

def prune_dictionary(psq_dict: Dict[str, Dict[str, float]], min_prob: float=0, max_cdf: float=1.0, max_alternative: int=-1):
    new_dict: Dict[str, Dict[str, float]] = {}
    for source_term, translations in psq_dict.items():
        translations = translations.items()

        if min_prob > 0:
            translations = list(filter(lambda x: x[1] >= min_prob, translations))

        if max_cdf < 1.0:
            translations = prune_cdf(translations, max_cdf)

        if max_alternative > 0:
            translations = heapq.nlargest(max_alternative, translations, key=lambda x: x[1])
        
        if len(translations) > 0:
            new_dict[source_term] = dict(translations)
        
    return new_dict


def prune_chunk(args, trans_chunk: sp.csc_matrix, doc_lengths: List[int]) -> sp.csc_matrix:
    trans_chunk = trans_chunk.tocsc() # just making sure
    trans_chunk.data[ trans_chunk.data < args.min_token_score ] = 0
    trans_chunk.eliminate_zeros() 

    if args.max_translated_doc_length == -1 and args.max_translated_doc_ratio == -1:
        return trans_chunk

    for i in range(trans_chunk.shape[1]):
        doc_offset_start, doc_offset_end = trans_chunk.indptr[i], trans_chunk.indptr[i+1]
        doclen = doc_offset_end - doc_offset_start

        if args.max_translated_doc_length != -1: 
            maxlength: int = int(args.max_translated_doc_length)
        else: # using ratio
            maxlength: int = int(doc_lengths[i] * args.max_translated_doc_ratio)

        if doclen <= maxlength:
            continue # no need to prune

        vals = trans_chunk.data[doc_offset_start: doc_offset_end] # returns references
        vals[ np.argpartition(vals, doclen-maxlength)[:-maxlength] ] = 0 # changes are also on .data
    
    trans_chunk.eliminate_zeros() 
    return trans_chunk


def index_chunk(args, chunk_info: Tuple[int, List[Tuple[str, str]]]):
    output_dir = Path(args.output_dir)
    tokenizer = PSQTokenizer(args.lang)
    vectorizer = PSQVectorizer.load(output_dir/'vectorizer.pkl.gz', source_tokenizer=tokenizer.tokenize)

    chunk_idx, chunk = chunk_info
    docnos = np.array([ d[0] for d in chunk ])
    text = [ d[1] for d in chunk ]
    trans_chunk: sp.csr_matrix = prune_chunk(
        args, *vectorizer.translate(text, return_lengths=True)
    )

    if not args.float64:
        trans_chunk = trans_chunk.astype(np.float32)

    if not args.do_merge: 
        fn = Path(args.output_dir) / f"{args.index_prefix}-{chunk_idx}.{'pkl' if not args.compression else 'gz'}"
        trans_chunk = trans_chunk.tocsr()
        write_inverted_index(args.compression, fn, docnos, trans_chunk, verbose=args.verbose) # write with compression
    else:
        # no need to converge to csr yet
        fn = Path(args.tmp_dir) / f"{args.index_prefix}-0.{chunk_idx}.pkl"
        write_inverted_index(False, fn, docnos, trans_chunk, verbose=args.verbose)


def index(args):
    output_dir = Path(args.output_dir)
    logger.info(f"Running indexing on doc file {args.doc_file}")

    logger.info(f"Loading psq dictionary {args.raw_dictionary}")
    psq_matrix = prune_dictionary(
        read_raw_dictionary(args.raw_dictionary), 
        args.min_translation_prob, args.max_translation_cdf, args.max_translation_alternatives
    )
    tokenizer = PSQTokenizer(args.lang)

    vectorizer = PSQVectorizer.from_dict(psq_matrix, source_tokenizer=tokenizer.tokenize, alpha=args.alpha)
    vectorizer.load_query_term_importance(args.count_file)
    vectorizer.save(output_dir / "vectorizer.pkl.gz")

    logger.info(f"Counting docs...")
    ndocs = sum(1 for _ in open(args.doc_file)) 

    # determine chunk size based on numer of workers
    chunk_size = min(args.chunk_size, int(np.ceil(ndocs/args.nworkers)))

    work_ = partial(index_chunk, args)
    with Pool(args.nworkers) as pool:
        list(tqdm(
            pool.imap_unordered(work_, enumerate( batching(read_documents(args), chunk_size) )),
            dynamic_ncols=True, total=int(np.ceil(ndocs/chunk_size))
        ))


def prune_inverted_index(args, inverted_matrix: sp.csr_matrix):
    if args.posting_list_k == -1:
        return inverted_matrix

    for i in tqdm(range(inverted_matrix.shape[0]), desc="pruning index", dynamic_ncols=True):
        start, end = inverted_matrix.indptr[i], inverted_matrix.indptr[i+1]
        posting_length = end - start
        if posting_length <= args.posting_list_k:
            continue
        vals = inverted_matrix.data[start: end]
        kth_largest_score = np.partition(vals, -args.posting_list_k)[-args.posting_list_k]
        cutoff = kth_largest_score * args.posting_list_epsilon
        vals[ vals <= cutoff ] = 0

    inverted_matrix.eliminate_zeros() 
    return inverted_matrix


def merge_pair(args, current_level: int, group_info: Tuple[int, List[Path]]):
    group_idx, group = group_info
    if len(group) < 2:
        return
    
    to_merge = [ read_inverted_index(fn) for fn in group ]

    merged_chunk = sp.hstack([ chunk.tocsc() for docno, chunk in to_merge ])
    merged_docnos = np.concatenate([ docno for docno, chunk in to_merge ])

    # merged_chunk = prun_inverted_index(merged_chunk)
    
    output_fn = Path(args.tmp_dir) / f"{args.index_prefix}-{current_level}.{group_idx}.pkl"
    write_inverted_index(False, output_fn, merged_docnos, merged_chunk)
    
    for f in group:
        f.unlink()


def merge_chunks(args):
    output_dir = Path(args.output_dir)
    tmp_dir = Path(args.tmp_dir)

    _, files_to_merge = find_unmerged_inverted_indexes(tmp_dir, args.index_prefix)
    logger.info(f"Start with {len(files_to_merge)} chunks to merge, "
                f"using {np.ceil(np.log(len(files_to_merge))/np.log(args.merge_nway)):.0f} levels of merging.")

    current_level = 1
    while len(files_to_merge) > 1:
        work_ = partial(merge_pair, args, current_level)
        with Pool(max(args.nworkers, len(files_to_merge))) as pool:
            list(tqdm(
                pool.imap_unordered(work_, enumerate( batching(files_to_merge, args.merge_nway) )),
                desc=f"Level {current_level}",
                dynamic_ncols=True, 
                total=int(np.ceil(len(files_to_merge)/args.merge_nway)),
            ))
        
        _, files_to_merge = find_unmerged_inverted_indexes(args, tmp_dir)
        current_level += 1
    
    assert len(files_to_merge) == 1, f"Weird... found {len(files_to_merge)} files but exited"

    logger.info("Final inverting csc to csr.")
    final_fn = output_dir / f"{args.index_prefix}-merged.{'pkl' if not args.compression else 'gz'}"
    all_docnos, merged_chunk = read_inverted_index(files_to_merge[0])
    merged_chunk = prune_inverted_index(args, merged_chunk.tocsr())
    write_inverted_index(args.compression, final_fn, all_docnos, merged_chunk)
    files_to_merge[0].unlink()

    logger.info(f"Saved inverted index `{final_fn}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that generates inverted index from a jsonl index file")
    parser.add_argument("--doc_file", type=str, default=None)
    parser.add_argument("--from_existing_index", type=str, default=None)

    parser.add_argument("--lang", type=str, required=True)

    parser.add_argument("--docid", type=str, default="docno")
    parser.add_argument("--title", type=str, default="title")
    parser.add_argument("--body", type=str, default="body")

    parser.add_argument("--raw_dictionary", "--raw_dict", "--psq_file", dest="raw_dictionary", type=str, required=True) 
    parser.add_argument("--count_file", type=str, default="hltcoe/psq_translation_tables:eng.cnt")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--index_prefix", type=str, default='vecpsq')

    parser.add_argument('--alpha', type=float, default=0.1)

    # pruning parameters 
    parser.add_argument('--min_translation_prob', type=float, default=0)
    parser.add_argument('--max_translation_alternatives', type=int, default=-1)
    parser.add_argument('--max_translation_cdf', type=float, default=1.0)
    parser.add_argument('--min_token_score', type=float, default=1e-9)
    parser.add_argument('--max_translated_doc_length', type=int, default=-1) 
    parser.add_argument('--max_translated_doc_ratio', type=float, default=-1) 

    # https://dl.acm.org/doi/10.1145/383952.383958
    parser.add_argument('--posting_list_k', type=int, default=-1)
    parser.add_argument('--posting_list_epsilon', type=float, default=0.1)

    parser.add_argument('--do_merge', action='store_true', default=False)
    parser.add_argument('--merge_nway', type=int, default=2)

    parser.add_argument('--chunk_size', type=int, default=5_000)
    parser.add_argument('--compression', action='store_true', default=False)
    parser.add_argument('--float64', action='store_true', default=False) 

    parser.add_argument('--nworkers', type=int, default=20)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    assert (args.doc_file is None) != (args.from_existing_index is None), \
        "Need to specify either `doc_file` or `from_existing_index`."

    assert (args.max_translated_doc_ratio == -1) or (args.max_translated_doc_length == -1), \
        "Can only specify either `max_translated_doc_ratio` or `max_translated_doc_length` not both"

    if args.posting_list_k > 0:
        assert args.do_merge, "Pruning posting lists with max doc requires running merging"

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        logger.warning(f"{output_dir} exists, will overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = tempfile.TemporaryDirectory()
    args.tmp_dir = tmp_dir.name
    logger.info(f"Use tmp dir `{args.tmp_dir}`")

    json.dump(vars(args), (output_dir / "args.json").open('w'))

    if args.from_existing_index: 
        # move the unmerge files into temp dir
        opener, files = find_unmerged_inverted_indexes(args, args.from_existing_index)
        for fn in tqdm(files, desc="Copying files", dynamic_ncols=True):
            shutil.copy(str(fn), str(Path(args.tmp_dir) / fn.name))
        shutil.copy( str(Path(args.from_existing_index) / "vectorizer.pkl.gz"), str(output_dir / "vectorizer.pkl.gz") )
    else:
        index(args)
    
    if args.do_merge:
        merge_chunks(args)
    
    tmp_dir.cleanup()