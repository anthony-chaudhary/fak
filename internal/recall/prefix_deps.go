package recall

import "github.com/anthony-chaudhary/fak/internal/cachemeta"

// prefix_deps.go — the recall-side connector for GLM52-HOSTED-CACHE-COHERENCE §A4: it
// turns a recorded session's page table into the provider-prefix coherence dependencies
// that drive the live break decision. Each page admitted under an external trust witness
// (Page.Witness) becomes a cachemeta.PrefixDependency at its token offset in the
// assembled prefix; a refuted (revoked) witness then breaks the hosted GLM prefix at its
// stale span.
//
// This is the missing input layer between the recorded prefix and the cachemeta
// coherence kernel. The live OpenAI-proxy path is then a single decoupled call:
//
//	shaped, _, _ := cachemeta.ShapeGLMTurnRevoked(turn, manifest.PrefixDeps(), vdso.Default.Revoked)
//
// The token offset is derived from cumulative page byte length (~4 bytes/token, matching
// cachemeta's relative-accounting estimate) because the page table records bytes, not
// billed tokens — honest for the linter's relative cacheable/lost accounting, never sold
// as an exact provider count.

// PrefixDeps extracts the provider-prefix coherence dependencies from this manifest's
// page table, in page order. Pages without a witness contribute their length to the
// running offset but no dependency.
func (m Manifest) PrefixDeps() []cachemeta.PrefixDependency {
	var deps []cachemeta.PrefixDependency
	var tok int64
	for _, p := range m.Pages {
		if p.Witness != "" {
			deps = append(deps, cachemeta.PrefixDependency{
				Witness:    p.Witness,
				Value:      p.Witness, // the witness string is its own recorded value for the offline value-form
				TokenStart: tok,
			})
		}
		tok += estPageTokens(p.Len)
	}
	return deps
}

// PrefixDeps on a LIVE Recorder snapshots its current page table and extracts the same
// coherence dependencies as Manifest.PrefixDeps. This is what removes the need for a
// separate "live context-assembly" deps source: mid-session the Recorder IS the live
// core image, and Manifest() is its snapshot, so the live OpenAI-proxy path uses the
// identical connector the offline/cdb path does:
//
//	shaped, _, _ := cachemeta.ShapeGLMTurnRevoked(turn, recorder.PrefixDeps(), vdso.Default.Revoked)
func (r *Recorder) PrefixDeps() []cachemeta.PrefixDependency {
	return r.Manifest().PrefixDeps()
}

// estPageTokens converts a page's byte length to a coarse token offset (~4 B/token, min
// 1 for a non-empty page), matching cachemeta's relative-accounting estimate.
func estPageTokens(n int64) int64 {
	if t := n / 4; t > 0 {
		return t
	}
	if n > 0 {
		return 1
	}
	return 0
}
