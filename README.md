# Efficient Implementation of Probabilistic Structured Queries

This package is an implementation of the Probablistic Structured Queries algorithm for 
cross-langauge information retrieval. 
It leverages alignment table from statistical machine translation to translate the document 
bag-of-tokens into the query language. 

Raw translation tables are available on Huggingface Models [`hltcoe/psq_translation_tables`](https://huggingface.co/hltcoe/psq_translation_tables)

## Get started

`fast_psq` is available on PyPI.
```bash
pip install fast_psq
```

Alternatively, you can also install directly from the GitHub main branch by using the following 
command. 
```bash
pip install pip@git+https://github.com/hltcoe/PSQ
```

`fast_psq` works with `ir_datasets` and `ir_measures` quite well for accessing IR evaluation collections 
and evaluating results. You can install the two packages with the following command. 
```bash
pip install ir_datasets ir_measures
```

## Indexing

The indexing script takes a translation table (i.e., alignment matrix) and a document `jsonl` file. 
We release a number of them on Huggingface Model, which can be automatically downloaded 
in the script by placing the path in the `--psq_file` flag in the format of `{repo_id}:{flie_path}`. 
Alternatively, you can also pass in a local `.json.gz` file that contains a dictionary of dictionaries, mapping from
source tokens (string) to target tokens (string) to alignment probabilities. 
However, the default tokenizer in the script uses `mosestokenier`, which may not match the one in your own 
alignment matrix. You should either use `mosestokenier` when aligning the bitext or replace the tokenizer with yours. 

The document file should be a `jsonl` file with one document in each row. 
You can specify the field for document id, title, and body text by passing in the field name
in the file through `--docid`, `--title`, and `--body` respectively. 
Alternatively, you can also use `--doc_source` with `irds:` as prefix to use a dataset in `ir_datasets`.

The following is an example indexing command.
```bash
python -m fast_psq.index \
--doc_file irds:neuclir/1/zh/trec-2022 \
--lang zh \
--psq_file hltcoe/psq_translation_tables:zh.table.dict.gz \
--min_translation_prob 0.00010 \
--max_translation_alternatives 64 \
--max_translation_cdf 0.99 \
--docid doc_id \
--title title \
--body text \
--min_translation_prob 1e-4 \
--max_translation_alternatives 64 \
--output_dir ./indexes/neuclir-zh.f32/ \
--compression \
--nworkers 64
```

Please use `python -m fast_psq.index --help` for more information about the arguments. 

## Searching

The search script takes the index and a `tsv` query file and output a TREC style result file. 
Similarly, we support `ir_datasets` as well with `irds:` prefix in both `--query_source` and `--qrels` arguments. 

The following command is an example. 
```bash
python -m fast_psq.search \
--query_source irds:neuclir/1/zh/trec-2022 \
--query_field title \
--index_dir ./indexes/neuclir-zh.f32/ \
--qrels irds:neuclir/1/zh/trec-2022 \
--query_lang en \
--output_file ./neuclir-zh.en.title.f32.trec
```

Please use `python -m fast_psq.search --help` for more information about the arguments. 

## Citation

```bibtex
@article{psq-repro,
    title = {Efficiency-Effectiveness Tradeoff of Probabilistic Structured Queries for Cross-Language Information Retrieval},
    author = {Eugene Yang and Suraj Nair and Dawn Lawrie and James Mayfield and Douglas W. Oard and Kevin Duh},
    journal = {arXiv preprint arXiv},
    year = {2024}
}
```