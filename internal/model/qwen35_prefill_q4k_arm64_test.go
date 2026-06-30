//go:build arm64

package model

// prefillQ4KKTol is the K/Kraw tolerance for the hybrid-Q4K prefill-vs-decode parity test on
// arm64. The QKNorm-damped K/Kraw cache stays at the strict bound; the raw V cache carries the
// documented Q8-minority reduction-order drift and reaches 2.02178955e-4 on M3 Pro.
func prefillQ4KKTol() float64 { return 1e-5 }

func prefillQ4KVTol() float64 { return 2.1e-4 }
