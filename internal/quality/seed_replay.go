package quality

import (
	"fmt"
	"strings"
)

// seed_replay.go is the seeded-replay child of the quality spine (#4529): it makes
// seeded STOCHASTIC generation replayable evidence. A temperature-zero case is
// deterministic by construction; a sampled case is only replayable if the engine
// honors the pinned SamplingParams.Seed, so the same seed reproduces the same token
// sequence exactly. This file models a tiny deterministic sampler as the reference
// decode path and registers a differential oracle that asserts seed-exact replay,
// localizing the first step at which an engine's sampling departed from it.

// seedReplayVocab is the small fixed vocabulary the deterministic sampler draws
// from. Eight entries keep the token space tiny while making an accidental
// collision between two different seeds' sequences vanishingly unlikely over a
// handful of steps.
var seedReplayVocab = []string{"aurora", "basalt", "cirrus", "delta", "ember", "flux", "granite", "harbor"}

// seedReplayDraw maps (seed, step) to one pseudo-random draw via splitmix64: the
// step counter advances the splitmix state by the golden-gamma constant and the
// finalizer mixes it. Because the draw is a pure function of (seed, step) — no
// carried state, no ambient entropy — token i of a decode depends only on the
// pinned seed and its position, which is exactly the replay contract the oracle
// enforces.
func seedReplayDraw(seed int64, step int) uint64 {
	z := uint64(seed) + (uint64(step)+1)*0x9e3779b97f4a7c15
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// seedReplaySample runs the deterministic sampler: steps draws under seed, each
// mapped onto the fixed vocab. It is the shared decode path both the reference and
// a faithful engine run — same seed in, byte-identical Trace out.
func seedReplaySample(seed int64, steps int) Trace {
	toks := make([]string, 0, steps)
	for i := 0; i < steps; i++ {
		toks = append(toks, seedReplayVocab[seedReplayDraw(seed, i)%uint64(len(seedReplayVocab))])
	}
	return Trace{Tokens: toks, Text: strings.Join(toks, " ")}
}

// seedReplayRotate returns the vocab entry after tok, wrapping. Rotation by one in
// a vocab of eight can never map a token to itself, so an injected step bug is
// GUARANTEED to diverge at exactly its step for any seed — the mutant cannot
// accidentally reproduce the reference.
func seedReplayRotate(tok string) string {
	for i, v := range seedReplayVocab {
		if v == tok {
			return seedReplayVocab[(i+1)%len(seedReplayVocab)]
		}
	}
	return seedReplayVocab[0]
}

// seedReplayBugStep is the step the "step-bug" mutant corrupts: mid-sequence, so
// the passing prefix proves the localization is doing work (the failure is pinned
// to the step, not to "the whole run looked wrong").
const seedReplayBugStep = 3

// SeededRunner decodes by running the deterministic sampler for the case's
// Params.MaxTokens steps under Params.Seed. The zero value is a faithful engine;
// the defect field (set via SeedReplayEngine) injects a nondeterminism bug. It is
// the ScriptedRunner-style adapter for the stochastic replay seam: a real sampled
// engine wires in behind the same Runner interface and is judged the same way.
type SeededRunner struct {
	Label  string
	defect string
}

func (s SeededRunner) Name() string {
	if s.Label != "" {
		return s.Label
	}
	return "seeded-engine"
}

func (s SeededRunner) Run(c QualityCase) (Trace, error) {
	var t Trace
	switch s.defect {
	case "seed-drift":
		// The classic unseeded-RNG defect: the engine ignores the pinned seed and
		// samples under some other one (modeled as seed+1).
		t = seedReplaySample(c.Params.Seed+1, c.Params.MaxTokens)
	case "step-bug":
		// A step-dependent defect: the decode is faithful until one step where the
		// engine's sampling departs (modeled by rotating that step's token).
		t = seedReplaySample(c.Params.Seed, c.Params.MaxTokens)
		if seedReplayBugStep < len(t.Tokens) {
			t.Tokens[seedReplayBugStep] = seedReplayRotate(t.Tokens[seedReplayBugStep])
			t.Text = strings.Join(t.Tokens, " ")
		}
	default:
		t = seedReplaySample(c.Params.Seed, c.Params.MaxTokens)
	}
	t.Runner = s.Name()
	return t, nil
}

// SeedReplayEngine returns a seeded engine runner with an optional injected
// nondeterminism defect: "" decodes faithfully under the case's pinned seed;
// "seed-drift" ignores the seed (samples under seed+1) so the decode diverges from
// the first differing draw; "step-bug" corrupts the single token at step 3 so the
// first divergence localizes mid-sequence. This is the deterministic mutant source
// the tests use to prove the replay gate trips.
func SeedReplayEngine(defect string) SeededRunner {
	switch defect {
	case "seed-drift":
		return SeededRunner{Label: "engine-seed-drift", defect: defect}
	case "step-bug":
		return SeededRunner{Label: "engine-step-bug", defect: defect}
	default:
		return SeededRunner{Label: "engine-seeded-clean"}
	}
}

// SeedReplayCase builds a stochastic replay case pinned to seed: temperature above
// zero (this is sampling, not greedy decode), a fixed step budget, and a reference
// trace produced by the deterministic sampler itself under that seed. Replay is
// SEED-SCOPED: the case asserts that the SAME seed reproduces the same sequence,
// not that generation is globally constant — a different seed may (and for this
// vocab does) produce a different sequence.
func SeedReplayCase(seed int64) QualityCase {
	params := SamplingParams{Temperature: 0.8, TopK: len(seedReplayVocab), MaxTokens: 6, Seed: seed}
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "seed-replay-demo",
		Version:   1,
		Prompt:    "Continue the sequence under the pinned sampling seed.",
		Params:    params,
		Reference: seedReplaySample(seed, params.MaxTokens),
		Oracles:   []string{"seed-replay"},
	}
}

// SeedReplay is the differential oracle for seeded replay (#4529): the engine's
// sampled token stream must equal the reference stream token by token, because
// both were pinned to the same seed. Any mismatch — a drifted seed, a
// step-dependent bug, a truncated decode — is reported as the FIRST divergence, so
// "the sampler is nondeterministic" localizes to "step 3 drew 'flux' where the
// reference drew 'ember'".
type SeedReplay struct{}

func (SeedReplay) Name() string { return "seed-replay" }
func (SeedReplay) Kind() string { return "differential" }

func (SeedReplay) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "seed-replay", Kind: "differential", Pass: true}
	n := len(ref.Tokens)
	if len(eng.Tokens) < n {
		n = len(eng.Tokens)
	}
	for i := 0; i < n; i++ {
		if ref.Tokens[i] != eng.Tokens[i] {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: eng.Tokens[i]}
			v.Detail = fmt.Sprintf("seed %d replay diverged at token %d: reference %q, engine %q",
				c.Params.Seed, i, ref.Tokens[i], eng.Tokens[i])
			return v
		}
	}
	if len(ref.Tokens) != len(eng.Tokens) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(eng.Tokens, n)}
		v.Detail = fmt.Sprintf("seed %d replay length diverged at %d: reference has %d tokens, engine has %d",
			c.Params.Seed, n, len(ref.Tokens), len(eng.Tokens))
		return v
	}
	v.Detail = fmt.Sprintf("seed %d replayed exactly: %d tokens matched the reference", c.Params.Seed, len(ref.Tokens))
	return v
}

func init() {
	Register(SeedReplay{})
}
