// Package tokenizer is a tokenizer leaf for offline text/id conversion outside the model proof path.
//
// The in-kernel model remains token-id-in: no forward-pass oracle, R2/R14 gate, or
// model math test depends on this package. The tokenizer bounds kvmmu span precision
// and text-facing demos only; a wrong text/id boundary can shift a Segment.Len and
// evict the wrong K/V rows, but no tokenizer result certifies model numerics.
//
// Stage T-2 implements HF fast tokenizer.json ByteLevel BPE decode and encode.
// The merge-order gate is an offline corpus fixture against HF's tokenizer output;
// it is a tokenizer witness only and does not touch model numerics.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See fak/GROWTH.md for the layering contract and how a leaf bakes in.
package tokenizer
