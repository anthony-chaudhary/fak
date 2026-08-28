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

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// ErrQwen35MTPSpecDecodeUnsupported marks an explicit fail-closed boundary for
// the opt-in production binding. It never authorizes a fallback runtime.
var ErrQwen35MTPSpecDecodeUnsupported = errors.New("model: Qwen3.8 MTP speculative decode is unsupported")

// Qwen35MTPSpecDecodeUnsupportedError names why the opt-in binding refused
// execution before mutating a target or constructing speculative state.
type Qwen35MTPSpecDecodeUnsupportedError struct {
	Reason string
}

func (e *Qwen35MTPSpecDecodeUnsupportedError) Error() string {
	if e == nil || e.Reason == "" {
		return ErrQwen35MTPSpecDecodeUnsupported.Error()
	}
	return fmt.Sprintf("%v: %s", ErrQwen35MTPSpecDecodeUnsupported, e.Reason)
}

func (e *Qwen35MTPSpecDecodeUnsupportedError) Unwrap() error {
	return ErrQwen35MTPSpecDecodeUnsupported
}

type qwen35MTPDrafterBuilder func(*Model, int, Qwen35MTPTargetHidden, Qwen35MTPTokenEmbedding) (*Qwen35MTPDrafter, error)

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

// SpecDecodeGreedyQwen35MTP is the explicit native Qwen3.8 MTP entry point.
// It is intentionally opt-in and never changes ordinary Generate/Prefill behavior.
//
// Each proposal reconstructs the committed prefix in a fresh hidden-source
// session before the MTP drafter may read it. Each verify round likewise uses a
// fresh target session. Rejected drafts are therefore discarded by closing the
// round session rather than claiming recurrent-state rollback support that the
// Qwen hybrid target does not have. The caller's target session remains untouched.
func SpecDecodeGreedyQwen35MTP(target *Session, prompt []int, n, k int) (polymodel.SpecDecodeRun, error) {
	return specDecodeGreedyQwen35MTP(target, prompt, n, k, NewQwen35MTPDrafter)
}

func specDecodeGreedyQwen35MTP(target *Session, prompt []int, n, k int, build qwen35MTPDrafterBuilder) (polymodel.SpecDecodeRun, error) {
	if target == nil || target.M == nil {
		return polymodel.SpecDecodeRun{}, errors.New("model: Qwen3.8 MTP target session is required")
	}
	if len(prompt) == 0 {
		return polymodel.SpecDecodeRun{}, ErrQwen35MTPEmptyPrefix
	}
	if k <= 0 {
		return polymodel.SpecDecodeRun{}, ErrQwen35MTPInvalidDraftLength
	}
	if k != 1 {
		return polymodel.SpecDecodeRun{}, &Qwen35MTPSpecDecodeUnsupportedError{Reason: "the retained one-layer MTP head supports exactly one draft token per round"}
	}
	if target.Cache == nil || target.Cache.Len() != 0 {
		return polymodel.SpecDecodeRun{}, &Qwen35MTPSpecDecodeUnsupportedError{Reason: "target session must be fresh; pass the complete prompt explicitly"}
	}
	if target.Backend != nil || target.Quant || target.Q4 || target.Q4K || target.F16 || target.GPTQ ||
		target.Metal || target.MetalQ4K || target.PrecisionPolicy != nil {
		return polymodel.SpecDecodeRun{}, &Qwen35MTPSpecDecodeUnsupportedError{Reason: "only the native f32 target path has exact pre-final-norm hidden capture"}
	}
	mode, err := target.M.Qwen35MTPMode(false)
	if err != nil {
		return polymodel.SpecDecodeRun{}, err
	}
	if !mode.Enabled {
		return polymodel.SpecDecodeRun{}, &Qwen35MTPSpecDecodeUnsupportedError{Reason: mode.Reason}
	}
	if build == nil {
		return polymodel.SpecDecodeRun{}, errors.New("model: Qwen3.8 MTP drafter builder is required")
	}
	for _, token := range prompt {
		if _, err := target.TokenEmbedding(token); err != nil {
			return polymodel.SpecDecodeRun{}, err
		}
	}

	var hiddenSource *Session
	var hiddenCommitted []int
	d, err := build(target.M, k, func(prefix []int) ([]float32, error) {
		if hiddenSource == nil || len(prefix) == 0 || !tokenPrefix(prefix, hiddenCommitted) {
			return nil, fmt.Errorf("model: target hidden for unevaluated prefix length %d is unavailable", len(prefix))
		}
		pos := len(prefix) - 1
		hidden, err := hiddenSource.TargetHiddenAt(pos)
		if err != nil {
			return nil, fmt.Errorf("committed prefix position %d: %w", pos, err)
		}
		return hidden, nil
	}, target.TokenEmbedding)
	if err != nil {
		return polymodel.SpecDecodeRun{}, err
	}
	defer d.Close()

	var runtimeErr error
	propose := func(committed []int) []int {
		if runtimeErr != nil || d.Err() != nil {
			return nil
		}
		source, err := newQwen35MTPIsolatedSession(target.M)
		if err != nil {
			runtimeErr = err
			return nil
		}
		defer source.Close()
		if err := evaluateQwen35MTPTargetPrefix(source, committed); err != nil {
			runtimeErr = err
			return nil
		}
		hiddenSource = source
		hiddenCommitted = append(hiddenCommitted[:0], committed...)
		draft := d.Propose(committed)
		hiddenSource = nil
		hiddenCommitted = hiddenCommitted[:0]
		if err := d.Err(); err != nil {
			runtimeErr = err
		}
		return draft
	}
	verify := func(committed, draft []int) []int {
		if runtimeErr != nil || d.Err() != nil {
			return nil
		}
		argmax, err := verifyQwen35MTPRound(target.M, committed, draft)
		if err != nil {
			runtimeErr = err
			return nil
		}
		return argmax
	}

	run, decodeErr := polymodel.SpecDecode(prompt, propose, verify, polymodel.SpecDecodeConfig{
		MaxNewTokens: n,
		MaxDraft:     k,
		// No rollback hook: every verifier is isolated to one round and is closed
		// after producing target argmax. Rejection discards the whole session.
	})
	if drafterErr := d.Err(); drafterErr != nil {
		return run, drafterErr
	}
	if runtimeErr != nil {
		return run, runtimeErr
	}
	return run, decodeErr
}

func newQwen35MTPIsolatedSession(m *Model) (session *Session, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			session = nil
			err = fmt.Errorf("model: create isolated Qwen3.8 MTP target session: %v", recovered)
		}
	}()
	session = m.NewSession()
	if session == nil {
		return nil, errors.New("model: could not create isolated Qwen3.8 MTP target session")
	}
	return session, nil
}

func evaluateQwen35MTPTargetPrefix(session *Session, committed []int) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("model: evaluate Qwen3.8 MTP target hidden prefix: %v", recovered)
		}
	}()
	session.captureTargetHidden = true
	for _, token := range committed {
		session.tokenHidden(token, session.Cache.Len())
	}
	return nil
}

func verifyQwen35MTPRound(m *Model, committed, draft []int) (argmax []int, err error) {
	session, err := newQwen35MTPIsolatedSession(m)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	defer func() {
		if recovered := recover(); recovered != nil {
			argmax = nil
			err = fmt.Errorf("model: verify Qwen3.8 MTP round: %v", recovered)
		}
	}()

	logits := session.Prefill(committed)
	if len(logits) == 0 {
		return nil, errors.New("model: Qwen3.8 MTP verifier returned empty prompt logits")
	}
	argmax = make([]int, 0, len(draft)+1)
	argmax = append(argmax, argmaxF32(logits))
	for _, token := range draft {
		logits = session.Step(token)
		if len(logits) == 0 {
			return nil, errors.New("model: Qwen3.8 MTP verifier returned empty decode logits")
		}
		argmax = append(argmax, argmaxF32(logits))
	}
	return argmax, nil
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
