package polymodel

// specdecode.go — the LIVE draft→verify→accept→rollback speculative-decode loop
// (#4877, epic #4867). This is the piece docs/serving/polymodel-prefill-share-plan.md
// named the "DEFERRED engine wiring": the accept DECISION (AcceptGreedy), the residency
// PICK (PickDrafter), the role BRIDGE (BridgeRoles), and the model-side single-pass
// VERIFY (model.Session.VerifyForward) all already exist — what was missing was the loop
// that turns them into a running decode: propose k tokens, verify them in one pass, keep
// the longest correct prefix, and roll back the rejected suffix's KV, over and over until
// the sequence is done.
//
// WHY IT LIVES IN THIS TIER-1 LEAF (and why it takes closures). polymodel imports nothing
// internal (architest: "polymodel ... imports nothing internal") so it can never drag a
// backend onto the request path. The engine pieces it drives — the drafter model, the
// target's verify forward, the KV rollback — therefore cross the seam as CLOSURES the
// caller binds, exactly the way BridgeRoles takes model ids as plain values and
// modelroute.SpecAccept takes the accept call as an AcceptFunc. The loop is thus the pure
// CONTROL STRUCTURE of speculative decoding; the host wires:
//
//	run, _ := polymodel.SpecDecode(prompt,
//	    func(committed []int) []int { return drafter.Propose(committed) },   // a co-resident small model
//	    func(committed, draft []int) []int {                                 // the target's verify pass
//	        rows := targetSession.VerifyForward(draft, nil, nil)             // one batched forward
//	        argmax := make([]int, 0, len(draft)+1)
//	        argmax = append(argmax, mathx.ArgmaxF32(targetLogits))           // position 0: already known
//	        for _, r := range rows { argmax = append(argmax, mathx.ArgmaxF32(r)) }
//	        return argmax
//	    },
//	    polymodel.SpecDecodeConfig{MaxNewTokens: n, MaxDraft: k,
//	        Rollback: func(evict int) { targetSession.Cache.Evict(base+accepted, evict) }})
//
// LOSSLESS BY CONSTRUCTION, FOR ANY DRAFTER. The emitted stream is TOKEN-IDENTICAL to
// plain sequential greedy decode of the target, no matter what the drafter proposes — a
// perfect drafter, a noisy one, or pure garbage all yield the same output, differing only
// in how many verify passes it took. That is the whole point of greedy speculative
// decoding: a drafted token is committed ONLY when it equals the target's own argmax at
// that position, and the first divergence is replaced by the target's own token (the
// correction). Correctness never depends on the draft being right — only speed does. The
// SpecDecodeLossless witness proves this against a deterministic target oracle across four
// drafter qualities, and reports the mean acceptance length the drafting bought.
//
// THE DRAFTER OWNS ITS OWN CONTEXT (the #4877 confusion-risk). The loop hands the drafter
// the committed token history and nothing else, so a co-resident DSpark/MTP drafter is
// free to run its OWN context config (a smaller window, its own KV session) rather than
// inheriting the target's — the "DSpark drafter inheriting the main model's context size"
// bug cannot occur through this seam, because the seam passes tokens, never a session.

import "errors"

// Drafter proposes up to K speculative token ids to continue the committed history. It is
// the co-resident small model (or a schema fast-forward, or any proposer): it may propose
// ANYTHING — the verify pass gates every token, so a wrong proposal costs a rollback, never
// correctness. Returning nil (or an empty slice) means "no draft this round", which the
// loop handles as a plain single-token decode. The drafter owns its own context/session;
// the loop only supplies the committed ids.
type Drafter func(committed []int) []int

// Verifier runs the target model's single verify pass over the K draft tokens and returns
// the target's OWN argmax at the K+1 panel positions, given the committed prefix:
//
//	index 0    — the target's next token after `committed` (with NO draft applied);
//	index i>0  — the target's next token after committed + draft[:i].
//
// So len(result) MUST be len(draft)+1. This is exactly what model.Session.VerifyForward
// produces (per-position logits → argmax) with the already-known current-position logits
// prepended; a pure test binds it to a deterministic oracle. The correction/bonus token
// the loop commits when a draft diverges (or is fully accepted) is result[accepted], which
// the len(draft)+1 contract guarantees is in range.
type Verifier func(committed, draft []int) []int

// Rollback rolls back the n rejected speculative KV positions (the SpecResult.EvictKV
// count) after a verify pass — the engine binds model.KVCache.Evict so the rejected drafts
// leave the cache bit-exactly as if never drafted. It is optional: a pure loop with no KV
// (the witness) passes nil, and the loop still tracks the counts.
type Rollback func(evictKV int)

// SpecDecodeConfig configures one SpecDecode run.
type SpecDecodeConfig struct {
	// MaxNewTokens caps how many tokens the run emits (the decode budget). The run stops
	// as soon as this many tokens are committed, mid-round if necessary.
	MaxNewTokens int
	// MaxDraft caps the per-round draft length K. A drafter that proposes more than this
	// has its proposal truncated (the extra tokens are simply not verified this round). 0
	// or negative means "no cap" — the full drafter proposal is used.
	MaxDraft int
	// StopToken, when StopEnabled, ends the run after this token id is committed (an EOS).
	StopToken   int
	StopEnabled bool
	// Rollback, if non-nil, is called with EvictKV after each round that rejected drafts,
	// so the engine can roll back the rejected suffix's KV. nil for a KV-less pure loop.
	Rollback Rollback
}

// SpecDecodeRun is the outcome of a SpecDecode run: the emitted tokens plus the accounting
// that makes the throughput honest.
type SpecDecodeRun struct {
	// Output is the emitted token ids — TOKEN-IDENTICAL to plain greedy decode of the
	// target for the same budget, for any drafter.
	Output []int
	// Rounds is the number of verify passes performed (the bandwidth-bound cost unit).
	Rounds int
	// DraftedTokens is the total speculative tokens proposed across all rounds.
	DraftedTokens int
	// AcceptedDrafts is the total drafted tokens that matched the target's argmax and were
	// committed (excludes the per-round correction token).
	AcceptedDrafts int
	// AcceptanceProfile retains accepted/proposed counts by zero-based draft position.
	AcceptanceProfile []AcceptancePosition
	// EvictKV is the total rejected speculative KV positions rolled back.
	EvictKV int
	// MeanAcceptanceLength is the mean REAL tokens committed per verify pass
	// (len(Output)/Rounds). It is > 1 exactly when drafting bought throughput: a plain
	// decode (no accepted drafts ever) yields 1.0, and a perfect drafter at depth K yields
	// K+1. This is the "mean acceptance length" the #4877 done-condition reports.
	MeanAcceptanceLength float64
}

// Speculative-decode loop errors.
var (
	// ErrNoVerifier is returned when SpecDecode is called without a Verifier — there is no
	// target to gate the drafts against, so no token can be committed.
	ErrNoVerifier = errors.New("polymodel: SpecDecode needs a Verifier (bind model.Session.VerifyForward)")
	// ErrVerifierStalled is returned when a Verifier returns an empty argmax vector, so a
	// round commits no token and the loop would spin forever. A contract-honoring Verifier
	// (len == len(draft)+1 ≥ 1) never triggers it; it is the guard against a broken binding.
	ErrVerifierStalled = errors.New("polymodel: Verifier returned no argmax; a round cannot advance (want len(draft)+1)")
)

// SpecDecode runs the live greedy speculative-decode loop and returns the emitted tokens
// plus acceptance accounting. Each round: the Drafter proposes k tokens (capped at
// MaxDraft), the Verifier returns the target's argmax at the k+1 panel positions,
// AcceptGreedy keeps the longest matching prefix, the rejected suffix's KV is rolled back
// (Rollback), and the accepted drafts plus the target's correction token are committed. It
// repeats until MaxNewTokens tokens are emitted or the StopToken is committed.
//
// The output is token-identical to plain sequential greedy decode of the target for ANY
// Drafter — the loop's correctness is independent of draft quality (only Rounds, hence
// speed, depends on it). draft may be nil (every round is then a plain single-token
// decode). It reports MeanAcceptanceLength = emitted/Rounds, the honest throughput the
// drafting bought.
// commitToken emits one token: it appends tok to BOTH the emitted output and the
// committed context, and reports whether that token is the configured stop token.
// Every commit site in both speculative loops goes through it — the accepted-prefix
// loop and the correction/bonus token here, the accepted-path loop and the correction
// token in SpecDecodeTree — because "emit a token" has exactly one meaning: the two
// streams advance together. Advancing them by hand at four sites is what would let
// `out` and `committed` drift apart, and a drifted context silently stops being the
// greedy prefix the next verify() round is contracted to score against.
func commitToken(out, committed *[]int, tok int, stopEnabled bool, stopToken int) bool {
	*out = append(*out, tok)
	*committed = append(*committed, tok)
	return stopEnabled && tok == stopToken
}

func SpecDecode(prompt []int, draft Drafter, verify Verifier, cfg SpecDecodeConfig) (SpecDecodeRun, error) {
	var run SpecDecodeRun
	var profile acceptanceProfile
	if verify == nil {
		return run, ErrNoVerifier
	}
	max := cfg.MaxNewTokens
	if max <= 0 {
		return run, nil // empty budget: nothing to decode
	}
	committed := append([]int(nil), prompt...)
	out := make([]int, 0, max)

	for len(out) < max {
		var d []int
		if draft != nil {
			d = draft(committed)
		}
		if cfg.MaxDraft > 0 && len(d) > cfg.MaxDraft {
			d = d[:cfg.MaxDraft]
		}

		targetArgmax := verify(committed, d)
		if len(targetArgmax) == 0 {
			return run, ErrVerifierStalled // no token to commit → would spin forever
		}

		res := AcceptGreedy(d, targetArgmax)
		run.Rounds++
		run.DraftedTokens += len(d)
		run.AcceptedDrafts += res.Accepted
		profile.record(len(d), res.Accepted)
		if res.EvictKV > 0 {
			run.EvictKV += res.EvictKV
			if cfg.Rollback != nil {
				cfg.Rollback(res.EvictKV)
			}
		}

		// Commit the accepted drafts, then the target's correction/bonus token. Each
		// committed token equals the target's own argmax at that position, so the stream
		// stays byte-for-byte greedy. Honor MaxNewTokens and the optional StopToken
		// mid-commit.
		stop := false
		for j := 0; j < res.Accepted && len(out) < max; j++ {
			if commitToken(&out, &committed, d[j], cfg.StopEnabled, cfg.StopToken) {
				stop = true
				break
			}
		}
		// The correction/bonus token is targetArgmax[res.Accepted]; the len(draft)+1
		// contract keeps res.Accepted (≤ len(draft)) in range. A contract-violating
		// verifier that returns exactly len(draft) with everything accepted carries no
		// correction — guard the index so a broken binding cannot panic.
		if !stop && len(out) < max && res.Accepted < len(targetArgmax) {
			if commitToken(&out, &committed, targetArgmax[res.Accepted], cfg.StopEnabled, cfg.StopToken) {
				stop = true
			}
		}
		if stop {
			break
		}
	}

	run.Output = out
	run.AcceptanceProfile = profile.snapshot()
	if run.Rounds > 0 {
		run.MeanAcceptanceLength = float64(len(out)) / float64(run.Rounds)
	}
	return run, nil
}
