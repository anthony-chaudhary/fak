package quality

import (
	"fmt"
	"strings"
)

// prefill_parity.go is the chunked-prefill parity child of the quality spine
// (#4536): processing a prompt as ONE monolithic prefill pass versus splitting it
// into chunks (processed sequentially, accumulating state across chunk
// boundaries) must yield the identical generated output. This file models a tiny
// deterministic engine whose generated tokens are a pure function of a running
// hash over ALL prefill tokens seen so far — so a chunk-boundary bug that resets
// or mis-accumulates the carried state changes the output — provides a monolithic
// (reference) runner and a chunked (engine) runner behind the shared Runner seam,
// and registers the "prefill-parity" differential oracle that pins the first
// generated token the two paths disagree on.

// prefillVocab is the small fixed vocabulary the deterministic decode draws from.
// Eight entries keep the token space tiny while making an accidental full-output
// collision between two different prefill states vanishingly unlikely over a
// handful of steps.
var prefillVocab = []string{"anchor", "beacon", "cobalt", "drift", "ember", "fathom", "garnet", "helix"}

// prefillStateSeed is the initial prefill accumulator state both paths start
// from. The state-reset defect snaps back to this value at each chunk boundary —
// exactly the "forgot everything before this chunk" state bug the gate exists to
// catch.
const prefillStateSeed uint64 = 0x9e3779b97f4a7c15

// prefillFold folds one prompt token into the running prefill state: FNV-1a over
// the token bytes plus a token-boundary marker, so "ab"+"c" and "a"+"bc" cannot
// alias. The fold is sequential and order-sensitive — state after token i depends
// on every token up to and including i — which is what makes chunk boundaries
// observable at all.
func prefillFold(state uint64, tok string) uint64 {
	const fnvPrime = 0x100000001b3
	for i := 0; i < len(tok); i++ {
		state ^= uint64(tok[i])
		state *= fnvPrime
	}
	state ^= 0x1f // token-boundary marker
	state *= fnvPrime
	return state
}

// prefillFoldAll folds a token slice into state in order. Folding chunk by chunk
// with CARRIED state is, by construction, this same computation — the parity the
// oracle asserts holds exactly when the engine actually carries its accumulator
// across chunk boundaries.
func prefillFoldAll(state uint64, toks []string) uint64 {
	for _, tok := range toks {
		state = prefillFold(state, tok)
	}
	return state
}

// prefillDecode maps the accumulated prefill state to steps generated tokens via
// a splitmix64-style finalizer over (state, step). Every generated token depends
// on the FULL accumulated state, so a prefill mis-accumulation perturbs the whole
// decode rather than hiding in a suffix.
func prefillDecode(state uint64, steps int) Trace {
	toks := make([]string, 0, steps)
	for i := 0; i < steps; i++ {
		z := state + (uint64(i)+1)*0x9e3779b97f4a7c15
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		z ^= z >> 31
		toks = append(toks, prefillVocab[z%uint64(len(prefillVocab))])
	}
	return Trace{Tokens: toks, Text: strings.Join(toks, " ")}
}

// prefillTokens is the trivial whitespace tokenizer both runners share: parity is
// about how the SAME prefill token sequence is processed, so tokenization must be
// identical on both paths.
func prefillTokens(prompt string) []string { return strings.Fields(prompt) }

// prefillChunks splits toks into consecutive chunks of size n (the last may be
// shorter). n <= 0 or n >= len(toks) yields a single chunk — the degenerate split
// where chunked prefill IS monolithic prefill.
func prefillChunks(toks []string, n int) [][]string {
	if len(toks) == 0 {
		return nil
	}
	if n <= 0 || n >= len(toks) {
		return [][]string{toks}
	}
	var out [][]string
	for start := 0; start < len(toks); start += n {
		end := start + n
		if end > len(toks) {
			end = len(toks)
		}
		out = append(out, toks[start:end])
	}
	return out
}

// prefillDefectReset names the injected chunk-boundary defect: the engine resets
// its accumulated prefill state to the initial seed at every chunk boundary, so
// only the final chunk's tokens influence generation.
const prefillDefectReset = "state-reset"

// PrefillMonolithicRunner is the reference path for #4536: it prefill-processes
// the whole prompt in one pass and decodes Params.MaxTokens generated tokens from
// the accumulated state. It is the "what correct looks like" side of the parity
// differential.
type PrefillMonolithicRunner struct{}

func (PrefillMonolithicRunner) Name() string { return "prefill-monolithic" }

func (r PrefillMonolithicRunner) Run(c QualityCase) (Trace, error) {
	t := prefillDecode(prefillFoldAll(prefillStateSeed, prefillTokens(c.Prompt)), c.Params.MaxTokens)
	t.Runner = r.Name()
	return t, nil
}

// PrefillChunkedRunner is the engine path: it splits the prompt tokens into
// chunks of ChunkSize and prefill-processes them sequentially, folding the
// carried state chunk by chunk before decoding. With no defect it is a faithful
// engine (carried state makes it identical to monolithic at ANY split point); the
// defect field (set via PrefillEngine) injects the boundary bug.
type PrefillChunkedRunner struct {
	Label     string
	ChunkSize int
	defect    string
}

func (r PrefillChunkedRunner) Name() string {
	if r.Label != "" {
		return r.Label
	}
	return "prefill-chunked"
}

func (r PrefillChunkedRunner) Run(c QualityCase) (Trace, error) {
	state := prefillStateSeed
	for ci, chunk := range prefillChunks(prefillTokens(c.Prompt), r.ChunkSize) {
		if r.defect == prefillDefectReset && ci > 0 {
			// The injected boundary bug: state accumulated over earlier chunks is
			// dropped, as if the engine restarted prefill for this chunk.
			state = prefillStateSeed
		}
		state = prefillFoldAll(state, chunk)
	}
	t := prefillDecode(state, c.Params.MaxTokens)
	t.Runner = r.Name()
	return t, nil
}

// PrefillEngine returns a chunked engine runner that splits the prompt at
// chunkSize with an optional injected defect: "" folds carried state faithfully
// across chunk boundaries (parity holds at every split point); "state-reset"
// resets the accumulated state at each chunk boundary so any multi-chunk split
// diverges from the monolithic decode. This is the deterministic mutant source
// the tests use to prove the parity gate trips.
func PrefillEngine(chunkSize int, defect string) PrefillChunkedRunner {
	switch defect {
	case prefillDefectReset:
		return PrefillChunkedRunner{Label: "engine-prefill-reset", ChunkSize: chunkSize, defect: defect}
	default:
		return PrefillChunkedRunner{Label: "engine-prefill-chunked", ChunkSize: chunkSize}
	}
}

// PrefillParityCase builds the deterministic parity case: a fixed multi-token
// prompt, a temperature-zero decode budget, and a reference trace produced by the
// monolithic prefill path itself. Chunk geometry deliberately does NOT appear in
// the case — parity must hold for EVERY split point, so the split is engine
// configuration, not case data.
func PrefillParityCase() QualityCase {
	params := SamplingParams{Temperature: 0, MaxTokens: 6}
	prompt := "throughput ledger closed twelve percent higher than plan across all regions"
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "prefill-parity-demo",
		Version:   1,
		Prompt:    prompt,
		Params:    params,
		Reference: prefillDecode(prefillFoldAll(prefillStateSeed, prefillTokens(prompt)), params.MaxTokens),
		Oracles:   []string{"prefill-parity"},
	}
}

// PrefillParity is the differential oracle for #4536: the chunked engine's
// generated token stream must equal the monolithic reference stream exactly. Any
// mismatch — a reset boundary, a mis-carried accumulator, a truncated decode — is
// reported as the FIRST divergent generated token, so "chunked prefill broke
// generation" localizes to "output token 2 was 'drift' where monolithic emitted
// 'ember'".
type PrefillParity struct{}

func (PrefillParity) Name() string { return "prefill-parity" }
func (PrefillParity) Kind() string { return "differential" }

func init() { Register(PrefillParity{}) }

func (PrefillParity) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "prefill-parity", Kind: "differential", Pass: true}
	n := len(ref.Tokens)
	if len(eng.Tokens) < n {
		n = len(eng.Tokens)
	}
	for i := 0; i < n; i++ {
		if ref.Tokens[i] != eng.Tokens[i] {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: eng.Tokens[i]}
			v.Detail = fmt.Sprintf("chunked prefill diverged from monolithic at output token %d: reference (monolithic) %q, engine (chunked) %q",
				i, ref.Tokens[i], eng.Tokens[i])
			return v
		}
	}
	if len(ref.Tokens) != len(eng.Tokens) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(eng.Tokens, n)}
		v.Detail = fmt.Sprintf("chunked prefill output length diverged at %d: monolithic reference has %d tokens, chunked engine has %d",
			n, len(ref.Tokens), len(eng.Tokens))
		return v
	}
	v.Detail = fmt.Sprintf("chunked prefill matched monolithic prefill: %d output tokens identical", len(ref.Tokens))
	return v
}
