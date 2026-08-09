package model

// gemma4_session.go — the Session bridge that puts the DEDICATED gemma4 forward on the
// resident-quant path (issue #5495).
//
// The problem it closes. gemma4 interleaves two attention regimes with DIFFERENT per-layer
// head_dim (gemma4.go headDimForLayer), which is the whole reason a dedicated forward
// exists. Session.Prefill / PrefillNoLogits / Step used to fall through to the generic
// uniform-geometry lanes (tokenHidden / tokenHiddenQ -> blockStep), whose every shape — the
// q/k/v widths, the KV stride, the RoPE table and, fatally, the qk-norm band's
// applyQKNormCfg argument — is the SCALAR cfg.HeadDim. So a gemma4 checkpoint could only be
// served through the cacheless Model.Forward, and a resident-quant Session refused by name
// (*ResidentQuantUnsupportedError, issue #4274). f32-only is not a real serving option: a
// 26B-class checkpoint at f32 is ~100 GB of weights, so the dedicated forward existed and no
// real checkpoint could reach it on a single card in the <=48 GB class.
//
// What the bridge does. It routes a gemma4 Session to layerGemma4 — the dedicated forward —
// and that path already reads every projection through residentKernel -> residentMatRows,
// which dispatches by NAME over the resident stores (f32, Q8_0, int4, Q4_K, raw k-quant,
// Q2_0, GPTQ). So "wiring gemma4 onto the resident-quant path" needs no new decode kernel and
// no new quant store: it is exactly this dispatch. Every per-layer shape inside it comes from
// headDimForLayer / numKVHeadsForLayer / ropeDimForLayer, so applyQKNormCfg receives the
// PER-LAYER head_dim (gemma4.go:161,164) — the band panic that motivated the refusal is not
// on this path at all.
//
// What it deliberately does NOT do: build KV state. gemma4 has no cached Session (issue
// #5496 owns that: per-layer KV extents driven by Config.Window, so the sliding-attention
// majority bounds its allocation at the window). This bridge is therefore a RECOMPUTE
// session — it carries the token history and re-runs the cacheless forward over the whole
// prefix on each ingest, which is O(n^2) over a generation and is exactly what
// `diagtok --cacheless` does today, promoted from a diagnostic flag into the Session API so a
// serve path can reach it. Because it holds no K/V rows it leaves s.Cache untouched:
// Cache.Len() stays 0.
//
// Eviction is therefore inert — there are no rows to drop. Prefix reuse is NOT, and saying so
// was the #5548 defect: a reuser that CLONES s.Cache clones nothing, so the clone silently
// omits gemma4Hist, which is the entire prefix. A planner that admitted that empty cache and
// then prefilled only the divergent suffix against it served wrong logits with no error and no
// panic. Reuse is fail-CLOSED here instead — KVPrefixReuseSupported below is the one predicate
// every reuser must ask before treating a *KVCache as a complete session prefix — and becomes
// available, correctly, when #5496 lands the cached per-layer-window path.
//
// Note this is NOT a decode-store change: it touches neither of the two q4 stores (the
// GGUF-native Q4_K super-block with its ggml nibble interleave, nor the re-quantized int4
// q4Tensor with consecutive elements per byte). It only chooses which forward runs; each
// store keeps its own decoder, reached unchanged through residentMatRows.

// KVPrefixReuseSupported reports whether a *KVCache is a COMPLETE prefix for a Session over
// this Config — i.e. whether cloning the cache carries the whole of what the session already
// ingested. It is true for every cached architecture, whose per-layer K/V rows are the entire
// state, and false for the gemma4 recompute bridge, whose state is the token history
// (gemma4Hist) and whose cache stays empty.
//
// It exists because "the cache is empty" is indistinguishable from "the cache is a valid
// zero-length prefix" at every consumer: radixkv.truncatePrefix returns a non-nil zero-length
// clone, a nil-guard passes, and the radix tree matches on TOKEN IDS rather than on KV depth.
// So a reuser holding only a *KVCache cannot detect the difference and must ask the Config
// (#5548). Splitting the rule out of the reuser keeps it from being re-derived — and re-missed
// — at each of the several places that build a session from a cached prefix.
func (c Config) KVPrefixReuseSupported() bool { return !c.isGemma4() }

// gemma4SessionModeWired reports whether this Session's execution mode is one the gemma4
// bridge actually runs. The bridge is the HOST resident path (residentKernel over whatever
// store the model loaded as). A device session (Backend), the Metal prefill lanes, and
// dynamic whole-token precision are hand-copies of the uniform-geometry PreNorm block and
// have no gemma4 twin, so they are NOT wired and must fail closed instead of silently
// running the wrong geometry.
func (s *Session) gemma4SessionModeWired() bool {
	return s.Backend == nil && !s.Metal && !s.MetalQ4K && s.PrecisionPolicy == nil
}

// requireGemma4Session fails a gemma4 Session closed, by name, for the execution modes the
// dedicated forward is not wired for. It mirrors requireMiniMaxSession's idiom: a loud,
// readable boundary at the entry instead of a cryptic panic deep inside a band that assumed
// a uniform geometry. It carries the same typed *ResidentQuantUnsupportedError the generic
// band raises, so an operator (or a preflight) reads ONE inspectable type for "this gemma4
// session cannot run here", with Format naming the resident-quant store in play.
func (s *Session) requireGemma4Session() {
	if s.gemma4SessionModeWired() {
		return
	}
	panic(&ResidentQuantUnsupportedError{
		Arch:   s.M.Cfg.archFamilyKey(),
		Format: s.residentQuantForwardFormat(),
		Mode:   s.unwiredGemma4ModeName(),
	})
}

// unwiredGemma4ModeName names the FIRST unwired execution mode this session carries, so the
// refusal says which one to drop rather than listing every mode it is not.
func (s *Session) unwiredGemma4ModeName() string {
	switch {
	case s.Backend != nil:
		return "compute.Backend (device HAL)"
	case s.Metal:
		return "Metal prefill"
	case s.MetalQ4K:
		return "Metal Q4_K prefill"
	case s.PrecisionPolicy != nil:
		return "dynamic whole-token precision"
	default:
		return ""
	}
}

// gemma4Ingest appends ids to this session's token history. It is the whole of a
// KV-advancing prefill for the recompute bridge: the dedicated gemma4 forward is cacheless,
// so the ONLY state an ingest can advance is the history itself — there is no per-layer K/V
// to fill. That is why PrefillNoLogits costs nothing here rather than running a forward whose
// every output would be discarded.
func (s *Session) gemma4Ingest(ids []int) {
	s.requireGemma4Session()
	s.gemma4Hist = append(s.gemma4Hist, ids...)
}

// gemma4StackLast ingests ids and returns the LAST position's hidden state after the whole
// dedicated gemma4 decoder stack, recomputed over the entire retained history. It is
// Model.forwardHiddenRows's gemma4 arm with the per-position logit fill dropped: a Session
// consumes only the final position's distribution, and the LM head is the single largest GEMM
// in the model, so filling it at every position would be pure waste. The final norm is NOT
// applied here — logitsFromHidden owns it, and sharing that one tail is what keeps this
// bit-exact with Forward.
func (s *Session) gemma4StackLast(ids []int) []float32 {
	s.gemma4Ingest(ids)
	m := s.M
	x := m.embedBand(s.gemma4Hist)
	ropeFreqs := m.gemma4RopeFreqs()
	for l := 0; l < m.Cfg.NumLayers; l++ {
		m.layerGemma4(l, x, ropeFreqs)
	}
	return x[len(x)-1]
}

// prefillGemma4 ingests a prompt through the dedicated gemma4 forward and returns the logits
// of its LAST token — Session.Prefill's contract, served by the gemma4 path.
func (s *Session) prefillGemma4(ids []int) []float32 {
	return s.M.logitsFromHidden(s.gemma4StackLast(ids))
}

// stepGemma4 decodes one already-chosen token. On a recompute session a decode step is an
// ingest of one id followed by the same tail, so Prefill and Step cannot diverge.
func (s *Session) stepGemma4(id int) []float32 {
	return s.prefillGemma4([]int{id})
}
