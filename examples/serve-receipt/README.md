# serve-receipt facts

Facts documents for `fak serve-receipt plan` — one JSON file per model,
schema `fak.serve-receipt.facts/1` (field-by-field reference: the package doc
in `internal/servereceipt/doc.go`).

Regenerate a receipt (md to stdout, JSON with `--format json`):

    go build -o fak ./cmd/fak
    ./fak serve-receipt plan --facts examples/serve-receipt/kimi-k31-pro-nvfp4.facts.json --gpu rtx-pro-6000-blackwell-96gb --tp 8 --pp 3
    ./fak serve-receipt validate --receipt <receipt.json>

`weights_bytes` here is the provisional decimal total; reconcile against the
exact shard sum before quoting a receipt downstream (see each file's `notes`).
