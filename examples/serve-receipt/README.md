# serve-receipt facts

Facts documents for `fak serve-receipt plan` — one JSON file per model,
schema `fak.serve-receipt.facts/1` (field-by-field reference: the package doc
in `internal/servereceipt/doc.go`).

Regenerate the receipt for the bundled model (markdown to stdout):

    go run ./cmd/fak serve-receipt plan --facts examples/serve-receipt/kimi-k31-pro-nvfp4.facts.json --gpu rtx-pro-6000-blackwell-96gb --tp 8 --pp 3

A captured run is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

Regenerate with other inputs (JSON with `--format json`):

    go build -o fak ./cmd/fak
    ./fak serve-receipt plan --facts examples/serve-receipt/kimi-k31-pro-nvfp4.facts.json --gpu rtx-pro-6000-blackwell-96gb --tp 8 --pp 3
    ./fak serve-receipt validate --receipt <receipt.json>

`weights_bytes` here is the provisional decimal total; reconcile against the
exact shard sum before quoting a receipt downstream (see each file's `notes`).

## What this does not claim

- The receipt is arithmetic over the facts file. It does not claim the model
  loads, fits, or serves at the planned layout; the witness checklist in the
  output lists what still has to be measured on hardware.
- The `max_cached_tokens` figure is a planning bound from the stated KV layout
  and reserves, not a measured KV capacity.
- The facts are as observed on the date in the file. They are not a live check
  against the model hub.
