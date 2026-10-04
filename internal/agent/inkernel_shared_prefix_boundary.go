package agent

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// Shared-prefix boundary snapshots (subagent fan-out on recurrent hybrids).
//
// A recurrent hybrid (Qwen3.5/3.6 GDN) cannot truncate its recurrent state, so a
// radix match that ends mid-edge restores only the nearest ancestor that owns a
// complete PrefixSnapshot. Before this seam a request admitted at most two
// snapshots: the full prompt plus one checkpoint (the structural divergence point,
// or the last 64-token grid boundary). A parent request therefore left no snapshot
// at or before the end of its system/tools block, and the FIRST child that shared
// that block but diverged in its user turn restored zero tokens; only the second
// child benefited from the divergence checkpoint the first one materialized.
//
// The render layer knows exactly where the shared block ends: every ChatML-family
// renderer folds system text and the tool schema into ONE leading system block
// terminated by "<|im_end|>\n". preparePrompt measures that block in tokens and
// carries the offset to the prefill planner on the request context; the prefill
// then admits one more complete snapshot there. The offset is only ever a hint: a
// snapshot admitted at ANY offset is keyed by the exact token prefix ids[:offset]
// it was computed from, so a wrong or stale offset can cost memory but can never
// restore incorrect state.
//
// Memory: each admitted snapshot is an independent copy of its prefix (attention KV
// for the full-attention layers, which grows with the prefix, plus the fixed-size
// recurrent state). The boundary costs one extra snapshot per DISTINCT system/tools
// block, not per request: every later sibling or turn restores at or past the
// boundary and admits none. It stays inside the radixkv snapshot byte budget (a
// budget refusal drops this optional snapshot rather than failing the request), and
// snapshot eviction reclaims the deepest snapshots before a shared ancestor.
// ctxmmu's COW page table (COWPageTable.ForkSubagent / SessionBranch) could in
// principle share the boundary pages instead of copying them, but it pages opaque
// byte buffers that neither model.KVCache nor model.PrefixSnapshot is laid out in,
// and recurrent state is a whole-state overwrite per token, not an append-only page
// run, so wiring it here would mean re-laying out the KV cache; it is not reused.

type inKernelSharedPrefixBoundaryKey struct{}

// withInKernelSharedPrefixBoundary records the token offset where the rendered
// shared system/tools block ends. Non-positive offsets leave ctx unchanged.
func withInKernelSharedPrefixBoundary(ctx context.Context, tokens int) context.Context {
	if ctx == nil || tokens <= 0 {
		return ctx
	}
	return context.WithValue(ctx, inKernelSharedPrefixBoundaryKey{}, tokens)
}

// inKernelSharedPrefixBoundaryFromContext returns the recorded shared-block token
// offset, or 0 when the request carries none.
func inKernelSharedPrefixBoundaryFromContext(ctx context.Context) int {
	if ctx == nil {
		return 0
	}
	n, _ := ctx.Value(inKernelSharedPrefixBoundaryKey{}).(int)
	if n < 0 {
		return 0
	}
	return n
}

// inKernelSharedPrefixBoundaryTokens tokenizes the rendered shared block and returns
// its length when it is an exact token prefix of the full prompt ids. Special-token
// delimiters make the split stable for ChatML, but the prefix check keeps any
// tokenizer whose merges cross the seam from producing an off-by-some offset.
func inKernelSharedPrefixBoundaryTokens(tok *tokenizer.Tokenizer, rendered string, ids []int) int {
	if tok == nil || len(ids) == 0 {
		return 0
	}
	shared := inKernelSharedPrefixText(rendered)
	if shared == "" || len(shared) >= len(rendered) {
		return 0
	}
	prefix, err := tok.Encode(shared)
	if err != nil || len(prefix) == 0 || len(prefix) >= len(ids) {
		return 0
	}
	for i, id := range prefix {
		if ids[i] != id {
			return 0
		}
	}
	return len(prefix)
}

// inKernelPrefillCheckpoints returns the ascending, de-duplicated prompt offsets at
// which prefill admits a complete snapshot before the full prompt. Every offset is
// strictly after the restored prefix and strictly before the prompt end (the full
// prompt snapshot is admitted separately). It keeps the historical adaptive
// checkpoint and adds the shared-prefix boundary, so a request admits at most three
// snapshots; once a later request restores at or past the boundary it adds none.
func inKernelPrefillCheckpoints(matched, cacheable, sharedBoundary, promptTokens int) []int {
	var out []int
	if sharedBoundary > matched && sharedBoundary < promptTokens {
		out = append(out, sharedBoundary)
	}
	c := inKernelAdaptiveSnapshotCheckpoint(matched, cacheable, promptTokens)
	if c <= matched || c >= promptTokens {
		return out
	}
	switch {
	case len(out) == 0:
		out = append(out, c)
	case c > out[0]:
		out = append(out, c)
	case c < out[0]:
		out = []int{c, out[0]}
	}
	return out
}

// sharedPrefixBoundaryTokens measures the shared block only where a snapshot can
// be admitted (the same predicate the prefill checkpoint uses), so routes that never
// admit snapshots pay no extra tokenization.
func (p *InKernelPlanner) sharedPrefixBoundaryTokens(rendered string, ids []int) int {
	if p == nil || p.tree == nil || p.m == nil || !inKernelPlannerPrefixReuseSupported(p.m, p.backend) {
		return 0
	}
	if p.backend == nil && !inKernelHostSnapshotReuse(p) {
		return 0
	}
	return inKernelSharedPrefixBoundaryTokens(p.tok, rendered, ids)
}
