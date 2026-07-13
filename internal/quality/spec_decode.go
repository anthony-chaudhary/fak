package quality

import (
	"fmt"
	"strings"
)

// spec_decode.go is the speculative-decoding child of the quality spine (#4539):
// speculative decoding is an ACCELERATION, not a different decode — a draft model
// proposes tokens ahead and the target model verifies them, and the accept/reject
// rule must be exactness-preserving, so the emitted stream is IDENTICAL to what
// target-only decode would have produced. This file models a tiny deterministic
// target model, a draft that sometimes proposes tokens the target rejects, a
// speculative decode loop whose faithful accept rule falls back to the target
// token on every rejection, and a differential oracle ("speculative-parity")
// that judges the speculative stream against target-only decode token by token.
// The injected defect class is a LENIENT accept rule — one that keeps a draft
// token the target would reject — which the oracle localizes to exactly the
// first wrongly-kept token.

// specDecodeVocab is the small fixed vocabulary the deterministic target model
// emits from. Eight entries keep the token space tiny while making an accidental
// collision between a rejected draft token and the target token impossible under
// rotation (see specDecodeRotate).
var specDecodeVocab = []string{"amber", "birch", "cedar", "dune", "elm", "fern", "grove", "heath"}

// specDecodeDraw maps (seed, step) to one pseudo-random draw via splitmix64: the
// step counter advances the state by the golden-gamma constant and the finalizer
// mixes it. The draw is a pure function of (seed, step) — no carried state, no
// ambient entropy — so the toy target model decodes identically on every run.
func specDecodeDraw(seed int64, step int) uint64 {
	z := uint64(seed) + (uint64(step)+1)*0x9e3779b97f4a7c15
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// specTargetToken is the target model's greedy token at step: what target-only
// decode emits there, and the verdict the verify step of speculative decoding
// must defer to on a rejection.
func specTargetToken(seed int64, step int) string {
	return specDecodeVocab[specDecodeDraw(seed, step)%uint64(len(specDecodeVocab))]
}

// specDecodeRotate returns the vocab entry after tok, wrapping. Rotation by one
// in a vocab of eight can never map a token to itself, so a drifted draft
// proposal is GUARANTEED to differ from the target token at its step — the
// mutant cannot accidentally agree with the target.
func specDecodeRotate(tok string) string {
	for i, v := range specDecodeVocab {
		if v == tok {
			return specDecodeVocab[(i+1)%len(specDecodeVocab)]
		}
	}
	return specDecodeVocab[0]
}

// specDraftDrifts reports whether the draft model's proposal disagrees with the
// target at step. Every third step (2, 5, 8, ...) drifts, so a decode of a few
// tokens is certain to exercise the reject-and-fallback path — a speculative run
// in which the draft never disagreed would prove nothing about the accept rule.
func specDraftDrifts(step int) bool { return step%3 == 2 }

// specFirstDriftStep is the first step at which the draft disagrees with the
// target: the exact index a lenient accept rule first keeps a token the target
// rejects, and where the parity oracle must pin the first divergence.
const specFirstDriftStep = 2

// specDraftToken is the draft model's proposal at step: the target token on
// agreeing steps, a rotated (guaranteed-different) token on drift steps.
func specDraftToken(seed int64, step int) string {
	t := specTargetToken(seed, step)
	if specDraftDrifts(step) {
		return specDecodeRotate(t)
	}
	return t
}

// specDecodeBlock is the draft window: how many tokens the draft proposes ahead
// before the target verifies them. Rejection discards the rest of the block, as
// in real speculative decoding.
const specDecodeBlock = 3

// specTargetDecode is the reference path: target-only greedy decode for steps
// tokens under seed. It is the golden trace speculative decoding must reproduce
// exactly — acceleration may change WHEN tokens are produced, never WHICH.
func specTargetDecode(seed int64, steps int) Trace {
	toks := make([]string, 0, steps)
	for i := 0; i < steps; i++ {
		toks = append(toks, specTargetToken(seed, i))
	}
	return Trace{Tokens: toks, Text: strings.Join(toks, " ")}
}

// specSpeculativeDecode runs the speculative loop: the draft proposes a block of
// up to specDecodeBlock tokens, the target verifies each in order, an agreeing
// proposal is accepted, and the first disagreement in a block is a REJECTION —
// the faithful rule discards the rest of the block and emits the target token
// instead, which is exactly why the faithful stream equals target-only decode.
// With lenient=true the accept rule is broken in the classic exactness-losing
// way: it keeps the rejected draft token (still discarding the rest of the
// block), so the emitted stream departs from target-only decode at precisely the
// first wrongly-kept token.
func specSpeculativeDecode(seed int64, steps int, lenient bool) Trace {
	toks := make([]string, 0, steps)
	pos := 0
	for pos < steps {
		for j := 0; j < specDecodeBlock && pos < steps; j++ {
			draft := specDraftToken(seed, pos)
			target := specTargetToken(seed, pos)
			if draft == target {
				// Verified: the draft proposal is exactly the target's token.
				toks = append(toks, draft)
				pos++
				continue
			}
			// The target rejects the draft proposal at this position.
			if lenient {
				toks = append(toks, draft) // DEFECT: keep a token the target rejects.
			} else {
				toks = append(toks, target) // faithful fallback: emit the target token
			}
			pos++
			break // the rest of the draft block is discarded after a rejection
		}
	}
	return Trace{Tokens: toks, Text: strings.Join(toks, " ")}
}

// SpecDecodeRunner is the speculative engine adapter: it decodes the case's
// Params.MaxTokens steps under Params.Seed through the speculative loop. The
// zero value is a faithful engine; the defect field (set via SpecDecodeEngine)
// injects the lenient accept rule. Real speculative engine adapters wire in
// behind the same Runner interface and are judged the same way.
type SpecDecodeRunner struct {
	Label  string
	defect string
}

func (s SpecDecodeRunner) Name() string {
	if s.Label != "" {
		return s.Label
	}
	return "speculative-engine"
}

func (s SpecDecodeRunner) Run(c QualityCase) (Trace, error) {
	t := specSpeculativeDecode(c.Params.Seed, c.Params.MaxTokens, s.defect == "lenient-accept")
	t.Runner = s.Name()
	return t, nil
}

// SpecDecodeEngine returns a speculative engine runner with an optional injected
// defect: "" verifies faithfully (rejections fall back to the target token, so
// the stream equals target-only decode); "lenient-accept" keeps draft tokens the
// target would reject, so the stream first departs at specFirstDriftStep. This
// is the deterministic mutant source the tests use to prove the parity gate
// trips.
func SpecDecodeEngine(defect string) SpecDecodeRunner {
	if defect == "lenient-accept" {
		return SpecDecodeRunner{Label: "engine-lenient-accept", defect: defect}
	}
	return SpecDecodeRunner{Label: "engine-speculative-clean"}
}

// SpecDecodeCase builds the speculative-parity case: a greedy (temperature-zero)
// decode pinned to seed, whose reference trace is target-only decode under that
// seed. MaxTokens spans several draft blocks and multiple drift steps, so both
// the accept path and the reject-fallback path are exercised.
func SpecDecodeCase(seed int64) QualityCase {
	params := SamplingParams{Temperature: 0, MaxTokens: 8, Seed: seed}
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "spec-decode-demo",
		Version:   1,
		Prompt:    "Decode greedily; the speculative path must match target-only decode token for token.",
		Params:    params,
		Reference: specTargetDecode(seed, params.MaxTokens),
		Oracles:   []string{"speculative-parity"},
	}
}

// SpecDecodeParity is the speculative-decoding differential oracle (#4539): the
// engine's speculative token stream must equal the target-only reference stream
// exactly, because a correct accept/reject rule is exactness-preserving. Any
// mismatch — a wrongly-kept draft token, a dropped fallback, a truncated block —
// is reported as the FIRST divergence, so "speculative decode changed the
// output" localizes to "token 2 kept 'fern' where target-only decode emits
// 'elm'".
type SpecDecodeParity struct{}

func (SpecDecodeParity) Name() string { return "speculative-parity" }
func (SpecDecodeParity) Kind() string { return "differential" }

func init() { Register(SpecDecodeParity{}) }

func (SpecDecodeParity) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "speculative-parity", Kind: "differential", Pass: true}
	n := len(ref.Tokens)
	if len(eng.Tokens) < n {
		n = len(eng.Tokens)
	}
	for i := 0; i < n; i++ {
		if ref.Tokens[i] != eng.Tokens[i] {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: eng.Tokens[i]}
			v.Detail = fmt.Sprintf(
				"speculative decode diverged at token %d: target-only decode emits %q, engine kept %q — the accept rule retained a token the target rejects",
				i, ref.Tokens[i], eng.Tokens[i])
			return v
		}
	}
	if len(ref.Tokens) != len(eng.Tokens) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(eng.Tokens, n)}
		v.Detail = fmt.Sprintf(
			"speculative decode length diverged at %d: target-only reference has %d tokens, engine has %d",
			n, len(ref.Tokens), len(eng.Tokens))
		return v
	}
	v.Detail = fmt.Sprintf("speculative decode matched target-only decode exactly: %d tokens", len(ref.Tokens))
	return v
}
