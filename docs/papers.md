---
title: "fak-dev papers: local research-paper index"
description: "Fetch arXiv and Hugging Face paper metadata into a local JSONL index and search it with BM25."
---

# fak-dev papers

`fak-dev papers` pulls paper metadata from the arXiv export API and the Hugging
Face papers API into one local index, then searches it offline with BM25 (title
weighted 2x, plus abstract and authors). Every record is keyed `arxiv:<id>`
without a version, so a Hugging Face daily paper and its arXiv entry merge into
one record: longest abstract, union of categories and sources, max upvotes.

## Commands

```bash
fak-dev papers fetch  [--source arxiv|hf|all] [--query Q] [--category cs.LG] [--max 25] [--date YYYY-MM-DD]
fak-dev papers search "speculative decoding" [--live] [--max 10] [--since 30d] [--category C] [--source hf] [--json]
fak-dev papers latest [--since 7d] [--category C] [--source arxiv|hf] [--max 20] [--json]
fak-dev papers show 2211.17192 [--json]
```

- `fetch` without `--query` lists Hugging Face daily papers (latest, or `--date`);
  arXiv needs `--query` or `--category` and is skipped otherwise. It prints
  `new/updated/total` counts.
- `search --live` queries arXiv and Hugging Face for the same text, upserts the
  results, then ranks the whole index. Without `--live` it never touches the network.
- `latest` lists indexed papers newest first; `--since` takes `30d`, `2w`, `48h`,
  or a date (empty = no cutoff).
- `show` prints the full record plus abs, PDF, and Hugging Face URLs. For the
  full text as markdown use `hf papers read <id>`.

Human output is one line per paper: `date  id  [upvotes]  title`.

## Index location

Default: `<repo-root>/.fak/papers/index.jsonl` (`.fak/` is gitignored). Override
with `--index PATH` or `FAK_PAPERS_INDEX`. Writes are atomic (temp file + rename)
and the file is sorted by ID, one JSON record per line.

arXiv requests send a `fak-papers` User-Agent and keep at least 3s between
consecutive calls, per the arXiv API terms.
