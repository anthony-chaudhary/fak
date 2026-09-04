package docrender

import "testing"

func BenchmarkDocRender(b *testing.B) {
	const sampleDoc = `# System Overview

This document describes the architectural guarantees and operational models.

## Architecture

- Bounded parsing ensures deterministic output.
- Fail-closed refusal guarantees no invalid Markdown constructs pass through silently.
- Standard library dependencies keep the footprint minimal.

1. High performance
2. Memory safety
3. Provenance tracking

` + "```" + `go
func HandleDoc() {
	println("processing")
}
` + "```" + `

---

### Invariants

> Invariant: doc rendering is fail-closed and deterministic.
`

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Exercise the core scanning pipeline which verifies all syntax rules
		// and construct bounds.
		refusals := Scan(sampleDoc)
		if len(refusals) != 0 {
			b.Fatalf("unexpected refusals: %v", refusals)
		}
		// Also exercise kind resolution over the input
		dec, err := ResolveKind("", "docs/overview.md", sampleDoc)
		if err != nil {
			b.Fatalf("unexpected error resolving kind: %v", err)
		}
		if dec.Kind != KindReport {
			b.Fatalf("unexpected kind: %v", dec.Kind)
		}
	}
}
