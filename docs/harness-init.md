# fak harness init

`fak harness init` creates the smallest runnable fak-based product outside the fak checkout. The generated program imports only the public `pkg/harnesskit` contract, pins an immutable Go module version, and performs one deterministic offline turn with semantic JSON events.

```text
fak harness init --dir ./my-product --module example.com/my-product
cd ./my-product
go build -o product-bin ./cmd/product
go run ./cmd/product --selfcheck
```

Ownership is explicit in `harness.lock.json`: `product/config.go` and `README.md` are user-owned and never overwritten. Generated Go/module files carry generator provenance or are listed in the lock. Re-running the command updates only recognized generated files and leaves user-owned files byte-for-byte intact.

The default pin is `github.com/anthony-chaudhary/fak@v0.43.1-0.20260814184635-613a82b762e2`, the Go proxy's immutable pseudo-version for commit `613a82b762e2` where public contract `v1alpha1` shipped. Override it explicitly with `--fak-version` when upgrading. Windows and Linux clean-room transcripts are archived under `docs/_witnesses/harness-init/`.
