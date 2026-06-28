// Package modelladder is the shared model-ladder/registry infrastructure used by
// the live demos (cmd/ctxdemo, cmd/demorace). It auto-detects a ladder of candidate
// models (135m → 0.5B → 1.5B → 3B) from disk, and provides a registry that lazily
// loads + quantizes each rung once and memoizes it. Both demos previously carried a
// byte-identical copy of this code; it lives here so they stay independent commands
// without forking the infrastructure.
package modelladder
