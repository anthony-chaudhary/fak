package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// This file is the issue #109 audit: a deterministic, always-runnable witness for
// the NON-prefix KV segment reuse claim boundary. The repo's local proof already
// shows exact PREFIX/radix KV reuse is bit-identical (kvreuse_test.go); the open
// question — refuted by ON-DEMAND-CONTEXT-KV-REUSE-2026-06-19.md but never put on a
// recompute oracle locally — is whether reusing a MIDDLE span S (prefilled inside
// source context A+S) inside a different target P+S+Q can be exact.
//
// The audit runs four candidates against a full-recompute oracle and reports, for
// each, the last-logit max|Δ|, argmax equality, and a TYPED verdict. The whole point
// of issue #109's acceptance gate is the last assertion: naive non-prefix segment
// reuse must NEVER be served as a silent HIT — it must fault to RECOMPUTE/REFUSE.
//
// Why synthetic weights (NewSynthetic), not the 538MB HF export. The property under
// audit — "does a spliced non-prefix segment's KV equal a full recompute" — is
// STRUCTURAL (it follows from the causal-attention dependency graph), not numeric.
// A synthetic checkpoint runs the SAME real Session KV path (Prefill / token / RoPE
// / Clone) the HF-verified model uses, so it is a faithful witness for this boundary
// AND it never SKIPs — unlike the real-weights oracle, which skips without the export.
// The HF oracle separately certifies the numerics; this audit certifies the boundary.

// nonPrefixVerdict is the typed materialization verdict for a candidate segment, the
// closed vocabulary issue #109 demands instead of a boolean "hit". A candidate may be
// served (HIT) only when it is bit-exact to the recompute oracle; otherwise it must
// fault — RECOMPUTE when an exact repair path exists, REFUSE when it does not.
type nonPrefixVerdict string

const (
	verdictHIT       nonPrefixVerdict = "HIT"       // bit-exact to oracle; safe to serve
	verdictRECOMPUTE nonPrefixVerdict = "RECOMPUTE" // not exact; must fault to exact recompute
	verdictREFUSE    nonPrefixVerdict = "REFUSE"    // not exact and no repair; deny the reuse
)

// classifyAgainstOracle is the audit's HIT-or-fault rule: a candidate is a HIT only
// when its last-token logits are bit-identical to the full-recompute oracle (within
// the cross-path FMA noise floor) AND its argmax matches. Anything else is a fault.
// This is the gate that makes "naive non-prefix reuse" impossible to serve silently.
func classifyAgainstOracle(candidate, oracle []float32, repairAvailable bool) (nonPrefixVerdict, float64, bool) {
	d, _ := maxAbsDiff(candidate, oracle)
	argEq := argmax(candidate) == argmax(oracle)
	if d <= fmaCrossPathTol && argEq {
		return verdictHIT, d, argEq
	}
	if repairAvailable {
		return verdictRECOMPUTE, d, argEq
	}
	return verdictREFUSE, d, argEq
}

// nonPrefixAuditCfg is a small GQA decoder (2 layers, 4 query / 2 kv heads, head_dim
// 8) with RoPE on — exactly the synthetic shape the KV-quarantine and SWA tests use,
// so the spliced-segment path exercises real RoPE-rotated K. EOS=-1 so greedy never
// short-circuits the continuation.
func nonPrefixAuditCfg() Config {
	return Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         97,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
}

// spliceSegmentKV copies the contiguous K/V (post-RoPE K, pre-RoPE Kraw, V) for the
// source-cache positions [srcFrom, srcFrom+n) onto the END of dst, then records each
// spliced position's logical index. This is the "naive segment reuse" primitive: it
// takes a segment's cached attention state verbatim from the context it was computed
// in and pastes it into a new cache, WITHOUT recomputing its hidden state under the
// new prefix. It deliberately does the one thing prefix reuse never has to do —
// relocate a span whose causal dependencies have changed — so the audit can measure
// whether that is exact. (dst.pos is left holding source absolute positions for the
// spliced rows; this models a relocation candidate before any reposition repair.)
func spliceSegmentKV(dst, src *KVCache, srcFrom, n int) {
	w := src.kvStride()
	for l := 0; l < src.cfg.NumLayers; l++ {
		dst.K[l] = append(dst.K[l], src.K[l][srcFrom*w:(srcFrom+n)*w]...)
		dst.Kraw[l] = append(dst.Kraw[l], src.Kraw[l][srcFrom*w:(srcFrom+n)*w]...)
		dst.V[l] = append(dst.V[l], src.V[l][srcFrom*w:(srcFrom+n)*w]...)
	}
	for i := 0; i < n; i++ {
		dst.pos = append(dst.pos, src.pos[srcFrom+i])
	}
}

// TestNonPrefixKVSegmentReuseAudit is the issue #109 audit. It compares four
// candidates for materializing the target sequence P + S + Q against a full-recompute
// oracle, where S is a MIDDLE (non-prefix) segment that was originally prefilled
// inside a DIFFERENT context A + S:
//
//	(1) full recompute        — the oracle itself; the reference last-logit.
//	(2) exact prefix baseline — clone P's KV, prefill S+Q; the known-exact control.
//	(3) naive segment reuse   — splice S's KV (computed under A) after P, prefill Q.
//	(4) selective recompute   — fault candidate (3) to exact recompute under P.
//
// Reported per candidate: last-logit max|Δ| vs the oracle, argmax equality, and the
// typed verdict (HIT / RECOMPUTE / REFUSE). Cache metadata records the model /
// tokenizer / serializer id, precision regime, and position/RoPE mode for the
// candidate segment, per the #109 acceptance checklist.
//
// The load-bearing assertions:
//   - the exact prefix baseline is a HIT (bit-exact) — proves the harness is not
//     vacuously failing everything;
//   - naive segment reuse is NOT a silent HIT — it must be flagged RECOMPUTE (this
//     is the #109 acceptance gate: a failed exactness budget yields a typed fault,
//     never a HIT);
//   - naive segment reuse actually DIVERGES from the oracle — proves the audit is
//     non-vacuous (the reuse genuinely perturbs the output);
//   - the selective-recompute repair path re-equals the oracle — proves the fault is
//     to an EXACT recompute, the only sound way to materialize a non-prefix segment.
func TestNonPrefixKVSegmentReuseAudit(t *testing.T) {
	cfg := nonPrefixAuditCfg()
	m := NewSynthetic(cfg)

	// Disjoint A and P so S sits under genuinely different preceding context in the
	// source (A+S) vs the target (P+S+Q). S is the reused middle segment; Q is the tail.
	A := []int{7, 41, 19, 2}      // source-only prefix
	P := []int{53, 11, 88, 3, 27} // target prefix (different length + content than A)
	S := []int{31, 5, 62}         // the non-prefix middle segment under audit
	Q := []int{17, 44}            // target tail after S

	target := concatInts(P, S, Q)
	posS := len(P)    // S's intended position range in the target: [posS, posS+len(S))
	srcSPos := len(A) // S's position range in the SOURCE A+S: [srcSPos, srcSPos+len(S))
	t.Logf("layout: target=P+S+Q len=%d; S target-positions=[%d,%d) source-positions=[%d,%d) (relocation Δ=%d)",
		len(target), posS, posS+len(S), srcSPos, srcSPos+len(S), posS-srcSPos)

	// ---- candidate (1): full recompute = the ORACLE -----------------------------
	oracle := m.Forward(target).Logits[len(target)-1]

	// ---- candidate (2): exact PREFIX baseline (known-exact control) --------------
	base := m.NewSession()
	base.PrefillNoLogits(P)
	prefixReuse := m.SessionFromPrefix(base.Cache)
	prefixReuse.PrefillNoLogits(S)
	lPrefix := lastLogitsAfter(prefixReuse, Q)

	// ---- candidate (3): NAIVE non-prefix segment reuse ---------------------------
	// Prefill the source A+S, then splice S's cached KV (computed under A) onto a
	// fresh target cache that has only P prefilled — i.e. reuse S verbatim at a new
	// position with a new causal prefix, recomputing nothing. Then prefill the tail Q
	// on top of the spliced cache and read the last-token logits.
	source := m.NewSession()
	source.PrefillNoLogits(concatInts(A, S))

	naive := m.NewSession()
	naive.PrefillNoLogits(P)
	spliceSegmentKV(naive.Cache, source.Cache, srcSPos, len(S))
	if naive.Cache.Len() != len(P)+len(S) {
		t.Fatalf("after splice, cache len = %d, want %d", naive.Cache.Len(), len(P)+len(S))
	}
	lNaive := lastLogitsAfter(naive, Q)

	// ---- candidate (4): SELECTIVE recompute / repair (fault to exact) ------------
	// The repair for a non-prefix segment is to recompute S (and everything causally
	// downstream of the changed prefix) under the correct prefix P. With S in the
	// middle that is exactly: clone P, prefill S then Q. This is the exact prefix
	// baseline's KV path, so it must re-converge to the oracle bit-for-bit.
	repair := m.NewSession()
	repair.PrefillNoLogits(P)
	repairReuse := m.SessionFromPrefix(repair.Cache)
	repairReuse.PrefillNoLogits(S)
	lRepair := lastLogitsAfter(repairReuse, Q)

	// ---- record cache metadata for the candidate segment (#109 checklist) --------
	// Every candidate segment's reuse must carry its model/tokenizer/serializer id,
	// precision regime, and position/RoPE mode, so a lookup can refuse a cross-axis
	// reuse before it ever reaches this numeric audit.
	segMeta := nonPrefixSegmentManifest(cfg, S)
	if segMeta.PositionConvention != cachemeta.PositionRecomputeRequired {
		t.Fatalf("non-prefix middle segment must be tagged %q, got %q",
			cachemeta.PositionRecomputeRequired, segMeta.PositionConvention)
	}
	t.Logf("candidate-segment axes: model=%q tokenizer=%q serializer=%q precision=%q position_mode=%q tokens=%d",
		segMeta.ModelID, segMeta.TokenizerID, segMeta.AdapterID,
		segMeta.Precision, segMeta.PositionConvention, segMeta.Tokens)

	// ---- classify every candidate against the oracle -----------------------------
	type row struct {
		name          string
		logits        []float32
		repairExists  bool
		wantVerdict   nonPrefixVerdict
		wantExact     bool // must this candidate be bit-exact to the oracle?
		wantDivergent bool // must this candidate DIFFER from the oracle (non-vacuity)?
	}
	rows := []row{
		{"full_recompute(oracle)", oracle, false, verdictHIT, true, false},
		{"exact_prefix_baseline", lPrefix, false, verdictHIT, true, false},
		{"naive_segment_reuse", lNaive, true, verdictRECOMPUTE, false, true},
		{"selective_recompute_repair", lRepair, false, verdictHIT, true, false},
	}

	for _, r := range rows {
		got, d, argEq := classifyAgainstOracle(r.logits, oracle, r.repairExists)
		mustFault := got != verdictHIT
		t.Logf("%-28s verdict=%-9s last-logit max|Δ|=%.3e argmax_eq=%v must_fault_to_recompute=%v",
			r.name, got, d, argEq, mustFault)

		if got != r.wantVerdict {
			t.Errorf("%s verdict = %s, want %s (max|Δ|=%.3e argmax_eq=%v)",
				r.name, got, r.wantVerdict, d, argEq)
		}
		if r.wantExact && d > fmaCrossPathTol {
			t.Errorf("%s must be bit-exact to oracle but max|Δ|=%.3e > tol %.0e",
				r.name, d, fmaCrossPathTol)
		}
		if r.wantDivergent && d <= fmaCrossPathTol {
			t.Errorf("%s must DIVERGE from the oracle (audit would be vacuous) but max|Δ|=%.3e",
				r.name, d)
		}
	}

	// ---- the #109 acceptance gate, stated directly -------------------------------
	// Naive non-prefix segment reuse must never be served as a HIT.
	if v, _, _ := classifyAgainstOracle(lNaive, oracle, true); v == verdictHIT {
		t.Errorf("ACCEPTANCE VIOLATION: naive non-prefix segment reuse was classified HIT; " +
			"a failed exactness budget must produce RECOMPUTE/REFUSE, never HIT")
	}
}

// TestNonPrefixSegmentMetaRefusedAcrossAxes proves the metadata recorded for a
// candidate segment is actually load-bearing: a reuse request whose precision regime
// or position mode disagrees with the cached segment is refused at the metadata
// layer (a typed cachemeta FAULT), BEFORE the numeric audit above runs. This is the
// #109 requirement that model/tokenizer/serializer/precision/position be recorded for
// every candidate AND that a cross-axis reuse be refused rather than silently served.
func TestNonPrefixSegmentMetaRefusedAcrossAxes(t *testing.T) {
	cfg := nonPrefixAuditCfg()
	S := []int{31, 5, 62}
	seg := nonPrefixSegmentManifest(cfg, S)

	// Every #109 axis must be recorded on the candidate segment.
	if seg.PositionConvention == cachemeta.PositionPrefixAligned {
		t.Fatalf("non-prefix middle segment was tagged prefix_aligned; it is not")
	}
	if seg.Precision == "" {
		t.Fatalf("candidate segment has no precision regime recorded")
	}
	if seg.ModelID == "" || seg.TokenizerID == "" || seg.AdapterID == "" {
		t.Fatalf("candidate segment missing model/tokenizer/serializer id: %+v", seg)
	}

	// An EXACT-axis, signature-verified claim against the segment's own manifest is the
	// control: it must be a HIT, proving the gate is not vacuously refusing everything.
	exact := residentClaimFor(seg, true)
	if v := cachemeta.CheckResidentClaim(exact, seg); v.Kind != cachemeta.LookupHit {
		t.Fatalf("exact-axis claim should be a HIT, got %s/%s", v.Kind, v.Reason)
	}

	// A reuse request whose PRECISION disagrees with the cached segment must FAULT, not
	// HIT — the cross-axis refusal #109 requires, evaluated before any numeric audit.
	wrongPrec := residentClaimFor(seg, true)
	wrongPrec.Precision = "int8"
	if v := cachemeta.CheckResidentClaim(wrongPrec, seg); v.Kind == cachemeta.LookupHit {
		t.Errorf("precision-mismatched claim was a HIT; cross-axis reuse must FAULT")
	} else {
		t.Logf("precision-mismatch refused: kind=%s reason=%s", v.Kind, v.Reason)
	}

	// A reuse request under a different POSITION convention (prefix_aligned vs the
	// segment's recompute_required) must likewise FAULT — a non-prefix span cannot be
	// served as if it were a relocatable prefix.
	wrongPos := residentClaimFor(seg, true)
	wrongPos.PositionConvention = cachemeta.PositionPrefixAligned
	if v := cachemeta.CheckResidentClaim(wrongPos, seg); v.Kind == cachemeta.LookupHit {
		t.Errorf("position-mismatched claim was a HIT; cross-axis reuse must FAULT")
	} else {
		t.Logf("position-mismatch refused: kind=%s reason=%s", v.Kind, v.Reason)
	}

	// An UNSIGNED claim (digest alone) must FAULT regardless of axis match — imported KV
	// is performance material, never reusable on identity alone (manifest refusal rule 8).
	unsigned := residentClaimFor(seg, false)
	if v := cachemeta.CheckResidentClaim(unsigned, seg); v.Kind == cachemeta.LookupHit {
		t.Errorf("unsigned claim was a HIT; a digest-only reuse must FAULT")
	} else {
		t.Logf("unsigned claim refused: kind=%s reason=%s", v.Kind, v.Reason)
	}
}

// nonPrefixSegmentManifest builds the cachemeta KVManifest that would accompany a
// non-prefix candidate segment. The position convention is recompute_required by
// construction: a middle span's deep-layer KV depends on its preceding context, so it
// cannot be relocated to a new prefix without recompute — exactly what the audit
// measures. The manifest records every #109 identity axis (model/tokenizer/serializer
// id via AdapterID, precision regime, position/RoPE mode, span length + digest) and a
// named signature so a resident-claim checker can refuse a cross-axis reuse.
func nonPrefixSegmentManifest(cfg Config, S []int) cachemeta.KVManifest {
	_ = cfg
	return cachemeta.KVManifest{
		SourceDigest:       cachemeta.DigestTokenIDs(S),
		SpanDigest:         cachemeta.DigestTokenIDs(S),
		Tokens:             int64(len(S)),
		ModelID:            "synthetic-nonprefix-audit",
		TokenizerID:        "synthetic-tok",
		AdapterID:          "fak-f32-raw", // serializer id axis for this synthetic span
		Precision:          "f32",
		PositionConvention: cachemeta.PositionRecomputeRequired,
		Producer:           "issue-109-audit",
		ProducerKeyID:      "audit-key",
		AccessPolicy:       "fleet-internal", // §2.4 access-control axis — admissibility precondition for any third-party KV
		IntegrityChecksum:  "audit-checksum",
		Signature:          cachemeta.ManifestSignature{Algorithm: "hmac-sha256", Value: "deadbeef"},
	}
}

// residentClaimFor builds the resident-span claim that exactly matches a manifest's
// binding axes; verified toggles whether the integrator verified the signature. It is
// the input to cachemeta.CheckResidentClaim, which is HIT only on a full axis match
// with a verified signature and a typed FAULT otherwise.
func residentClaimFor(m cachemeta.KVManifest, verified bool) cachemeta.ResidentClaim {
	return cachemeta.ResidentClaim{
		ModelID:            m.ModelID,
		TokenizerID:        m.TokenizerID,
		AdapterID:          m.AdapterID,
		Precision:          m.Precision,
		PositionConvention: m.PositionConvention,
		Producer:           m.Producer,
		SpanDigest:         m.SpanDigest,
		Tokens:             m.Tokens,
		IntegrityChecksum:  m.IntegrityChecksum,
		SignatureVerified:  verified,
	}
}

// lastLogitsAfter prefills tail ids on a session that has already been positioned and
// returns the last-token logits. With an empty tail it Steps the final cached token's
// successor is not needed — callers always pass a non-empty tail here.
func lastLogitsAfter(s *Session, tail []int) []float32 {
	if len(tail) == 0 {
		panic("lastLogitsAfter: empty tail")
	}
	return s.Prefill(tail)
}

func concatInts(parts ...[]int) []int {
	var out []int
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}
