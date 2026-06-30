package model

// quant_q4k_resident.go — the LM-head resolution for the resident Q4_K path. The rest of
// the path (kernel, eligibility, loader, Session.Q4K wiring) lives in quant_q4k.go,
// q4k_loader.go, kv.go, quant_forward.go; this file owns only what those do not: a stable
// head resolver. headName() keys off the f32 manifest / q8w and would MISS an lm_head held
// raw in q4kw (resolving instead to the tied-embedding key and mis-reading), so the Q4_K
// path resolves the head against whichever resident store actually carries it.

// IsQuantWeight is the exported alias for the "is this a resident matmul weight"
// predicate, so ggufload can classify tensors without duplicating the suffix list.
func IsQuantWeight(name string) bool { return isQuantWeight(name) }

// HasQ4K / HasQ8 are the exported resident-store membership checks for tooling and tests
// (q4kdiag, loader tests) that need to see which store a weight landed in without reaching
// into the unexported maps.
func (m *Model) HasQ4K(name string) bool { return m.q4kw != nil && m.q4kw[name] != nil }
func (m *Model) HasQ8(name string) bool  { return m.q8w != nil && m.q8w[name] != nil }
func (m *Model) q4kHeadName() string {
	if _, ok := m.q4kw["lm_head.weight"]; ok {
		return "lm_head.weight"
	}
	return "model.embed_tokens.weight"
}

// kqHeadName resolves the LM head when it loaded as a resident k-quant (Q5_K/Q6_K) — the
// q4_k_m untied lm_head lands in kqw under "lm_head.weight". Returns "" if no k-quant head.
func (m *Model) kqHeadName() string {
	if m.kqw != nil && m.kqw["lm_head.weight"] != nil {
		return "lm_head.weight"
	}
	return ""
}

// headKQuant applies a resident k-quant (Q5_K/Q6_K) LM head to a post-final-norm hidden vector.
// The k-quant GEMV (kQuantMatRows) is byte-identical to the f32 dequant-then-dot, so logits match
// the dequant path. Mirrors headQ4K for the kqw store.
func (s *Session) headKQuant(xf []float32) []float32 {
	y, t := s.headLogitsBuf()
	qt := s.M.kqw[s.M.kqHeadName()]
	kQuantMatRowsInto(qt, xf, y)
	logitScaleInPlace(y, s.M.Cfg)
	s.phaseEnd("lm_head_kquant", t)
	return y
}

// headQ4K applies the resident Q4_K LM head to a post-final-norm hidden vector.
func (s *Session) headQ4K(xf []float32) []float32 {
	y, t := s.headLogitsBuf()
	qt := s.M.q4khead
	if qt == nil {
		qt = s.M.q4kw[s.M.q4kHeadName()]
	}
	q4kMatRowsInto(qt, xf, y)
	logitScaleInPlace(y, s.M.Cfg)
	s.phaseEnd("lm_head_q4k", t)
	return y
}

// headResident picks the LM head matching whatever resident format this session's model
// carries: q4k (lm_head quantized Q4_K) → q4 → q8 (lm_head quantized Q6_K, or untied Q8)
// → GPTQ → f32 (tied embedding). This makes resident-quant paths' heads correct regardless
// of how the source quantized the head, where headName()-only resolution would miss a raw
// resident lm_head.
func (s *Session) headResident(xf []float32) []float32 {
	m := s.M
	if m.q4khead != nil || m.q4kw[m.q4kHeadName()] != nil {
		return s.headQ4K(xf)
	}
	// A q4_k_m untied lm_head loads Q6_K into kqw (the resident-k-quant dense fast path); resolve
	// it here so the head reads from kqw instead of falling through to the tied-embedding f32 head
	// (which would mis-read). kQuantMatRows is byte-identical to the dequant path.
	if m.kqHeadName() != "" {
		return s.headKQuant(xf)
	}
	if m.q4head != nil {
		return s.headQ4(xf)
	}
	if _, hasQ8 := m.q8w[m.headName()]; hasQ8 || m.q8head != nil {
		return s.headQ(xf)
	}
	if m.hasGPTQ("lm_head.weight") {
		return s.headGPTQ(xf)
	}
	return s.head(xf)
}
