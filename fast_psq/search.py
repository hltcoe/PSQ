import argparse
from pathlib import Path
import gzip
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

import numpy as np
from scipy import sparse as sp

from fast_psq.vectorizer import PSQVectorizer
from fast_psq.tokenizer import PSQTokenizer
from fast_psq.utils import read_inverted_index

from time import time

# @profile
def search_in_shard(
        q_vec: sp.csr_matrix, # only one query
        shard: Tuple[List[str], sp.csc_matrix], 
        k: int = 1000
    ):
    doc_ids, chunk_matrix = shard

    doc_scores = q_vec.data @ chunk_matrix[ q_vec.indices ]

    if doc_ids.shape[0] > k:
        topk_idx = np.argpartition(-doc_scores, k)[:k]
        return doc_ids[topk_idx], doc_scores[topk_idx]
    return doc_ids, doc_scores


# @profile
def search_query(q_vec, qids, shards, n_retrieve) -> Dict[str, List[Tuple[str, float]]]:
    all_results = {}

    for i, qid in enumerate(tqdm(qids, desc='search', dynamic_ncols=True)):
        all_ids, all_scores = [], []
        for shard in shards:
            topk_ids, topk_scores = search_in_shard(q_vec[i], shard, k=n_retrieve)
            all_ids.append(topk_ids)
            all_scores.append(topk_scores)
        
        all_ids = np.concatenate(all_ids)
        all_scores = np.concatenate(all_scores)
        if len(all_ids) > n_retrieve:
            topk_idx = np.argpartition(-all_scores, n_retrieve)[:n_retrieve]
            all_results[qid] = list(zip(all_ids[topk_idx], all_scores[topk_idx]))
        else:
            all_results[qid] = list(zip(all_ids, all_scores))
    
    return all_results


def search_by_shards(q_vec, qids, shards, n_retrieve) -> Dict[str, List[Tuple[str, float]]]:
    all_results = { qid: [[], []] for qid in qids }
    search_time = 0

    for shard in shards:
        start_time = time()
        for i, qid in enumerate(qids):
            topk_ids, topk_scores = search_in_shard(q_vec[i], shard, k=n_retrieve)
            all_results[qid][0] = np.concatenate([all_results[qid][0], topk_ids])
            all_results[qid][1] = np.concatenate([all_results[qid][1], topk_scores])

            if len(all_results[qid][0]) > n_retrieve:
                topk_idx = np.argpartition(-all_results[qid][1], n_retrieve)[:n_retrieve]
                all_results[qid] = [all_results[qid][0][topk_idx], all_results[qid][1][topk_idx]]

        search_time += ( time() - start_time )

    
    return { qid: list(zip(ids, scores)) for qid, (ids, scores) in all_results.items() }, search_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)

    parser.add_argument("--query_lang", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    parser.add_argument("--index_prefix", type=str, default='vecpsq')
    parser.add_argument('--n_retrieve', type=int, default=1000)
    parser.add_argument('--run_name', type=str, default='fast-psq')

    parser.add_argument('--qrels', type=str, default=None)
    parser.add_argument('--metrics', nargs='+', default=['nDCG@20', 'P@10', 'MAP', 'R@100', 'R@1000'])

    parser.add_argument('--no_preload', action='store_true', default=False)

    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    index_dir = Path(args.index_dir)

    tokenizer = PSQTokenizer(args.query_lang)
    vectorizer = PSQVectorizer.load(index_dir/'vectorizer.pkl.gz', target_tokenizer=tokenizer.tokenize)

    queries = [ l.strip().split('\t') for l in open(args.query_file)]
    qids = [ q[0] for q in queries ]

    q_vec: sp.csr_matrix = vectorizer.make_target_vectors([ q[1] for q in queries ])

    # search for compressed index files first
    opener = gzip.open
    index_files = list(index_dir.glob(f"{args.index_prefix}-*.gz"))
    if len(index_files) == 0:
        opener = open
        index_files = list(index_dir.glob(f"{args.index_prefix}-*.pkl"))
    
    if args.no_preload:
        shards = (
            read_inverted_index(shard_fn)
            for shard_fn in tqdm(index_files, desc='searching shrads', dynamic_ncols=True)
        )
        all_results, query_serving_time = search_by_shards(q_vec, qids, shards, args.n_retrieve)
    else:
        shards: List[Tuple[List[str], sp.csc_matrix]] = [
            read_inverted_index(shard_fn)
            for shard_fn in tqdm(index_files, desc='loading shrads', dynamic_ncols=True)
        ]

        # align dtype
        q_vec = q_vec.astype( shards[0][1].dtype )
        
        start_time = time()
        all_results = search_query(q_vec, qids, shards, args.n_retrieve)
        query_serving_time = time() - start_time

    print(f"Searching {len(queries)} queries take {query_serving_time*1000:.4f} ms, "
            f"average latency {query_serving_time/len(queries)*1000:.4f} ms")

    with open(args.output_file, 'w') as fw:
        for qid in qids:
            if args.verbose: 
                print(f"-- qid={qid}, n_hit={len(all_results[qid])}, "
                      f"max score={max(x[1] for x in all_results[qid])}, min score={min(x[1] for x in all_results[qid])}")
            docs = sorted(all_results[qid], key=lambda x: -x[1])
            for i, (doc_id, s) in enumerate(docs):
                fw.write(f"{qid} Q0 {doc_id} {i} {s} {args.run_name}\n")

    if args.qrels is not None:
        import ir_measures as irms
        metrics = [ irms.parse_measure(m) for m in args.metrics ]
        print( irms.calc_aggregate(metrics, irms.read_trec_qrels(args.qrels), irms.read_trec_run(args.output_file)) )