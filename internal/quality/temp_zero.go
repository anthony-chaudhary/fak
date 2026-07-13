package quality

import (
	"encoding/json"
	"fmt"
	"strings"
)

// temp_zero.go is the temperature-zero determinism child of the quality spine
// (#4525): temperature=0 must decode deterministically AND identically across
// request surfaces — non-streaming, streaming, and batched requests of the same
// case must all emit the greedy reference sequence exactly. A temp-0 path that
// still consults the sampler's RNG on one surface (the classic "streaming path
// forgot the greedy shortcut" defect) produces a fluent answer that silently
// disagrees with its non-streaming twin; this oracle localizes that to the
// offending SURFACE and the first divergent token index.

// tzSurfaces are the request surfaces a faithful engine must agree across. The
// engine runner decodes the case once per surface and serializes all streams
// into Trace.Text as a JSON envelope (the Judge signature stays untouched).
var tzSurfaces = []string{"non-streaming", "streaming", "batched"}

// tzVocab is the small fixed vocabulary the modeled greedy decoder emits from.
// Eight entries keep the token space tiny while making an accidental collision
// between a greedy token and an injected sampled token impossible under
// tzRotate (rotation by one can never map a token to itself).
var tzVocab = []string{"ledger", "monsoon", "nickel", "orchid", "prairie", "quartz", "russet", "signal"}

// tzNoisySurface and tzNoiseStep pin where the "sampling-noise" mutant injects
// its defect: mid-sequence on the streaming surface, so the passing prefix and
// the two clean surfaces prove the localization is doing work.
const (
	tzNoisySurface = "streaming"
	tzNoiseStep    = 2
)

// tzMix is the splitmix64 finalizer: a pure bijective mixer, so the modeled
// argmax below is a pure function of (prompt, step) — no carried state, no
// ambient entropy, which is exactly the temperature-zero contract.
func tzMix(x uint64) uint64 {
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// tzGreedyToken models the greedy argmax at one decode step: FNV-1a over the
// prompt selects the "model state", the step counter advances it by the
// golden-gamma constant, and the mix picks the winning vocab entry. Determinism
// regardless of SamplingParams.Seed is by construction — the seed is never an
// input, mirroring how a real temp-0 decode must never touch the sampler RNG.
func tzGreedyToken(prompt string, step int) string {
	h := uint64(14695981039346656037)
	for i := 0; i < len(prompt); i++ {
		h ^= uint64(prompt[i])
		h *= 1099511628211
	}
	z := tzMix(h + (uint64(step)+1)*0x9e3779b97f4a7c15)
	return tzVocab[z%uint64(len(tzVocab))]
}

// tzGreedyDecode runs the modeled greedy decoder for steps tokens. It is the
// single decode path every request surface of a faithful engine shares — same
// prompt in, byte-identical Trace out, on every surface, under every seed.
func tzGreedyDecode(prompt string, steps int) Trace {
	toks := make([]string, 0, steps)
	for i := 0; i < steps; i++ {
		toks = append(toks, tzGreedyToken(prompt, i))
	}
	return Trace{Tokens: toks, Text: strings.Join(toks, " ")}
}

// tzRotate returns the vocab entry after tok, wrapping. In a vocab of eight,
// rotation by one can never map a token to itself, so the injected sampling
// noise is GUARANTEED to diverge at exactly its step for any prompt.
func tzRotate(tok string) string {
	for i, v := range tzVocab {
		if v == tok {
			return tzVocab[(i+1)%len(tzVocab)]
		}
	}
	return tzVocab[0]
}

// tzSurfaceTrace is one request surface's captured token stream.
type tzSurfaceTrace struct {
	Surface string   `json:"surface"`
	Tokens  []string `json:"tokens"`
}

// tzEnvelope is the JSON the multi-surface engine serializes into Trace.Text:
// every surface's stream travels inside the one Trace the Judge signature
// carries, keeping the extension additive.
type tzEnvelope struct {
	Surfaces []tzSurfaceTrace `json:"surfaces"`
}

// tzParseSurfaces recovers the per-surface streams from an engine trace. A
// trace without the envelope (a plain single-path engine) degrades to one
// surface named after its runner, so the oracle still judges ordinary traces.
func tzParseSurfaces(eng Trace) []tzSurfaceTrace {
	var env tzEnvelope
	if err := json.Unmarshal([]byte(eng.Text), &env); err == nil && len(env.Surfaces) > 0 {
		return env.Surfaces
	}
	name := eng.Runner
	if name == "" {
		name = "engine"
	}
	return []tzSurfaceTrace{{Surface: name, Tokens: eng.Tokens}}
}

// tzFirstDiff returns the first index at which two token streams disagree
// (including a length divergence), with the tokens each side holds there.
func tzFirstDiff(ref, eng []string) (int, string, string, bool) {
	n := len(ref)
	if len(eng) < n {
		n = len(eng)
	}
	for i := 0; i < n; i++ {
		if ref[i] != eng[i] {
			return i, ref[i], eng[i], true
		}
	}
	if len(ref) != len(eng) {
		return n, tokenAt(ref, n), tokenAt(eng, n), true
	}
	return 0, "", "", false
}

func tzSurfaceNames(surfaces []tzSurfaceTrace) string {
	names := make([]string, len(surfaces))
	for i, s := range surfaces {
		names[i] = s.Surface
	}
	return strings.Join(names, ", ")
}

// TZSurfaceRunner is the multi-surface engine adapter: it decodes the case once
// per request surface and returns all streams in one enveloped Trace. The zero
// value is a faithful engine; the defect field (set via TZEngine) models a
// temp-0 path that still injects sampling noise on one surface.
type TZSurfaceRunner struct {
	Label  string
	defect string
}

func (r TZSurfaceRunner) Name() string {
	if r.Label != "" {
		return r.Label
	}
	return "surface-engine"
}

func (r TZSurfaceRunner) Run(c QualityCase) (Trace, error) {
	env := tzEnvelope{}
	for _, s := range tzSurfaces {
		t := tzGreedyDecode(c.Prompt, c.Params.MaxTokens)
		if r.defect == "sampling-noise" && s == tzNoisySurface && tzNoiseStep < len(t.Tokens) {
			// The defect class this child exists for: the surface's temp-0 path
			// still draws from the sampler, overriding greedy argmax at one step.
			toks := append([]string(nil), t.Tokens...)
			toks[tzNoiseStep] = tzRotate(toks[tzNoiseStep])
			t.Tokens = toks
		}
		env.Surfaces = append(env.Surfaces, tzSurfaceTrace{Surface: s, Tokens: t.Tokens})
	}
	b, err := json.Marshal(env)
	if err != nil {
		return Trace{}, err
	}
	// Tokens carries the primary (non-streaming) stream so token-level oracles
	// judging this trace directly still see a token surface.
	return Trace{Runner: r.Name(), Tokens: env.Surfaces[0].Tokens, Text: string(b)}, nil
}

// TZEngine returns a multi-surface engine with an optional injected defect:
// "" decodes every surface through the shared greedy path (clean pass);
// "sampling-noise" injects a sampled token on the streaming surface at step
// tzNoiseStep so exactly one surface departs from the greedy reference. This is
// the deterministic mutant source the tests use to prove the gate trips.
func TZEngine(defect string) TZSurfaceRunner {
	switch defect {
	case "sampling-noise":
		return TZSurfaceRunner{Label: "engine-temp0-sampling-noise", defect: defect}
	default:
		return TZSurfaceRunner{Label: "engine-temp0-clean"}
	}
}

// TZCase builds the temperature-zero surface-parity case: temperature pinned to
// zero, a nonzero seed ON PURPOSE (a faithful temp-0 decode must be identical
// regardless of seed — the seed exists so a defective path has an RNG to leak
// from), and a greedy reference produced by the shared decode path.
func TZCase() QualityCase {
	params := SamplingParams{Temperature: 0, MaxTokens: 6, Seed: 7}
	prompt := "Report the deterministic rollup at temperature zero."
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "temp-zero-surface-parity",
		Version:   1,
		Prompt:    prompt,
		Params:    params,
		Reference: tzGreedyDecode(prompt, params.MaxTokens),
		Oracles:   []string{"temp-zero-determinism"},
	}
}

// TZDeterminism is the temperature-zero determinism oracle (#4525): every
// request surface in the engine trace must equal the greedy reference token by
// token. Any mismatch — injected sampling noise, a surface-specific decode
// path, a truncated stream — is reported as the FIRST divergence with the
// offending surface named, so "streaming sometimes answers differently at
// temp 0" localizes to "surface streaming, token 2".
type TZDeterminism struct{}

func (TZDeterminism) Name() string { return "temp-zero-determinism" }
func (TZDeterminism) Kind() string { return "differential" }

func init() { Register(TZDeterminism{}) }

func (TZDeterminism) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "temp-zero-determinism", Kind: "differential", Pass: true}
	if c.Params.Temperature != 0 {
		v.Pass = false
		v.Detail = fmt.Sprintf("case pins temperature=%g; the temp-zero determinism oracle only judges temperature-zero cases", c.Params.Temperature)
		return v
	}
	surfaces := tzParseSurfaces(eng)
	for _, s := range surfaces {
		if idx, refTok, engTok, diverged := tzFirstDiff(ref.Tokens, s.Tokens); diverged {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: idx, Reference: refTok, Engine: engTok}
			v.Detail = fmt.Sprintf("surface %q diverged from the greedy reference at token %d: reference %q, engine %q",
				s.Surface, idx, refTok, engTok)
			return v
		}
	}
	v.Detail = fmt.Sprintf("temperature-zero decode identical across %d surface(s) (%s): %d tokens matched the greedy reference on each",
		len(surfaces), tzSurfaceNames(surfaces), len(ref.Tokens))
	return v
}
