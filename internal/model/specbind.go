package model

// specbind.go — the ABOVE-leaf request-path binding (#5098, follow-up to #4877). The
// engine-agnostic loop (internal/polymodel.SpecDecode) and the single-pass verify
// (Session.VerifyForward) both already exist; what was unwired was the caller that binds
// them onto LIVE model.Session weights. This file is that caller:
//
//   - the Verifier closure is built from Session.VerifyForward — argmax per logits row, with
//     the already-known current-position logits PREPENDED so the returned slice honors the
//     Verifier's len(draft)+1 contract (index 0 = the target's next token with no draft
//     applied; index i>0 = the target's next token after committed+draft[:i]);
//   - the Rollback closure is bound to KVCache.Evict at base+accepted, rolling the rejected
//     draft suffix out of BOTH the target AND the drafter session bit-exactly, so the caches
//     are left as if the rejected tokens were never drafted;
//   - the Drafter closure greedily decodes the co-resident drafter Session, which owns its
//     OWN context (the loop hands it committed token ids, never the target's session — the
//     #4877 "DSpark drafter inheriting the main model's context size" bug cannot occur here);
//   - the request-path entry (SpecDecodeGreedyResolved) resolves the co-resident drafter via
//     polymodel.PickDrafter + polymodel.BridgeRoles and gates the WHOLE path on
//     polymodel.Enabled() (FAK_POLYMODEL, default off).
//
// LOSSLESS BY CONSTRUCTION. The emitted stream is TOKEN-IDENTICAL to plain greedy decode of
// the target (Session.Generate) for ANY drafter — draft quality changes only the round count
// (throughput), never the output. This holds because VerifyForward's chain shape is
// bit-identical to sequential Step (TestVerifyForwardChainMatchesSerial) and KVCache.Evict is
// bit-exact; the drafter is gated token-by-token, so a wrong proposal costs a rollback, never
// correctness. The returned SpecDecodeRun carries MeanAcceptanceLength — the mean real tokens
// committed per verify pass, > 1 exactly when drafting bought throughput.

import "github.com/anthony-chaudhary/fak/internal/polymodel"

// SpecDecodeGreedy runs greedy speculative decoding of the target Session using the drafter
// Session as the co-resident proposer, driven by polymodel.SpecDecode. It is the pure engine
// binding — it does NOT consult polymodel.Enabled(); the gated request-path entry is
// SpecDecodeGreedyResolved. The output is token-identical to target.Generate(prompt, n) for
// any drafter (see the package doc). n caps emitted tokens (the decode budget); k caps the
// per-round draft length. Both sessions are prefilled with the prompt here, so pass fresh
// sessions.
func SpecDecodeGreedy(target, drafter *Session, prompt []int, n, k int) (polymodel.SpecDecodeRun, error) {
	// Target: the single-pass verify. tl threads the current-position logits (the token after
	// the committed context), so the Verifier can prepend the already-known argmax.
	tl := target.Prefill(prompt)
	var tBase, draftLen int
	verify := func(committed, draft []int) []int {
		// Feed any committed tokens the target session has not yet seen — the previous round's
		// correction/bonus token is committed by the loop but never fed through the session.
		for _, t := range committed[target.Cache.Len():] {
			tl = target.Step(t)
		}
		tBase = target.Cache.Len()
		draftLen = len(draft)
		argmax := make([]int, 0, len(draft)+1)
		argmax = append(argmax, argmaxF32(tl)) // position 0: already known from tl
		for _, row := range target.VerifyForward(draft, nil, nil) {
			argmax = append(argmax, argmaxF32(row))
		}
		return argmax
	}

	// Drafter: greedy propose k tokens from its own session. It owns its own context; the loop
	// only supplies the committed ids.
	dl := drafter.Prefill(prompt)
	var dBase int
	propose := func(committed []int) []int {
		for _, t := range committed[drafter.Cache.Len():] {
			dl = drafter.Step(t)
		}
		dBase = drafter.Cache.Len()
		drafts := make([]int, 0, k)
		for j := 0; j < k; j++ {
			dj := argmaxF32(dl)
			drafts = append(drafts, dj)
			dl = drafter.Step(dj)
		}
		return drafts
	}

	return polymodel.SpecDecode(prompt, propose, verify, polymodel.SpecDecodeConfig{
		MaxNewTokens: n,
		MaxDraft:     k,
		Rollback: func(evictKV int) {
			// Roll the rejected draft suffix out of both caches bit-exactly. base+accepted ==
			// base+(draftLen-evictKV): AcceptGreedy's EvictKV is the rejected count draftLen-accepted.
			target.evictKV(tBase+(draftLen-evictKV), evictKV)
			drafter.evictKV(dBase+(draftLen-evictKV), evictKV)
		},
	})
}

// SpecDecodeGreedyWithDrafter runs greedy speculative decoding with a
// model-independent token proposer. The target session remains the sole authority: every
// proposal is checked by VerifyForward and rejected KV positions are evicted before the
// next round. Pass a fresh target session; this function prefills it with prompt.
func SpecDecodeGreedyWithDrafter(target *Session, prompt []int, n, k int, drafter polymodel.Drafter) (polymodel.SpecDecodeRun, error) {
	tl := target.Prefill(prompt)
	var targetBase, draftLen int
	verify := func(committed, draft []int) []int {
		for _, token := range committed[target.Cache.Len():] {
			tl = target.Step(token)
		}
		targetBase = target.Cache.Len()
		draftLen = len(draft)
		argmax := make([]int, 0, len(draft)+1)
		argmax = append(argmax, argmaxF32(tl))
		for _, row := range target.VerifyForward(draft, nil, nil) {
			argmax = append(argmax, argmaxF32(row))
		}
		return argmax
	}

	return polymodel.SpecDecode(prompt, drafter, verify, polymodel.SpecDecodeConfig{
		MaxNewTokens: n,
		MaxDraft:     k,
		Rollback: func(evictKV int) {
			target.evictKV(targetBase+(draftLen-evictKV), evictKV)
		},
	})
}

// SpecDecodeGreedyResolved is the GATED request-path entry. It runs greedy speculative decode
// of `verifier` ONLY when polymodel.Enabled() (FAK_POLYMODEL, default off) AND a co-resident
// drafter resolves against the residency Pool: polymodel.PickDrafter picks the cheapest
// same-family warm peer, and polymodel.BridgeRoles confirms the pair is genuinely co-resident
// (same non-empty Family AND prefill-shareable). Otherwise it returns ok=false and the caller
// falls back to plain self-decode (Session.Generate) — never a wrong answer, just no
// speculation. `sessions` maps each resident ModelID to its live Session; the verifier and the
// resolved drafter must both be present. On a speculative run it returns the resolved drafter
// id and the SpecDecodeRun (whose MeanAcceptanceLength is the metric to emit on a live serve).
func SpecDecodeGreedyResolved(prompt []int, n, k int, verifier polymodel.ModelID, pool *polymodel.Pool, sessions map[polymodel.ModelID]*Session) (run polymodel.SpecDecodeRun, drafter polymodel.ModelID, ok bool, err error) {
	if !polymodel.Enabled() {
		return polymodel.SpecDecodeRun{}, "", false, nil // gate off: caller self-decodes
	}
	d := polymodel.PickDrafter(verifier, pool)
	if d == "" {
		return polymodel.SpecDecodeRun{}, "", false, nil // no co-resident drafter warm
	}
	if _, berr := polymodel.BridgeRoles(d, verifier, pool); berr != nil {
		return polymodel.SpecDecodeRun{}, "", false, nil // pair not co-resident spec-decodable
	}
	ts, ds := sessions[verifier], sessions[d]
	if ts == nil || ds == nil {
		return polymodel.SpecDecodeRun{}, "", false, nil // a resolved id has no live session
	}
	run, err = SpecDecodeGreedy(ts, ds, prompt, n, k)
	if err != nil {
		return polymodel.SpecDecodeRun{}, d, false, err
	}
	return run, d, true, nil
}
