package quality

import (
	"fmt"
	"strings"
)

// streaming_equiv.go is the streaming-equivalence child of the quality spine
// (#4531): concatenating the chunks a STREAMING decode flushes must reproduce
// the NON-STREAMING full output exactly — the same token sequence and the same
// assembled text. Streaming is a delivery mode, not a different generation: any
// token dropped or duplicated at a chunk/flush boundary, or any text glued
// incorrectly across a flush, is an engine defect. The non-streaming full
// decode is the reference; the chunk-flushing decode is the engine; the oracle
// localizes the first divergence to its exact token index (the boundary is the
// usual suspect).

// strmVocab is the small fixed vocabulary the deterministic decoder emits from.
// It is disjoint from the other spine vocabularies so a streaming trace is
// never confused with a sibling oracle's trace in a failure bundle.
var strmVocab = []string{"ingot", "jetty", "kelp", "lagoon", "mesa", "nimbus", "onyx", "prairie"}

// strmDraw maps (seed, step) to one pseudo-random draw via splitmix64: a pure
// function of its inputs, so the decode is deterministic and hermetic.
func strmDraw(seed int64, step int) uint64 {
	z := uint64(seed) + (uint64(step)+1)*0x9e3779b97f4a7c15
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// strmDecode is the non-streaming full decode: steps tokens under seed, text
// assembled with single-space joins. Consecutive tokens are guaranteed DISTINCT
// (the draw selects from the vocab minus the previous token), so a token
// dropped or duplicated at a boundary can never accidentally reproduce the
// reference there — the first divergence lands at exactly the boundary index.
func strmDecode(seed int64, steps int) Trace {
	toks := make([]string, 0, steps)
	prev := -1
	for i := 0; i < steps; i++ {
		var idx int
		if prev < 0 {
			idx = int(strmDraw(seed, i) % uint64(len(strmVocab)))
		} else {
			idx = int(strmDraw(seed, i) % uint64(len(strmVocab)-1))
			if idx >= prev {
				idx++
			}
		}
		toks = append(toks, strmVocab[idx])
		prev = idx
	}
	return Trace{Tokens: toks, Text: strings.Join(toks, " ")}
}

// strmFlushEvery is the streaming engine's chunk size: it flushes after every
// third token, so an N-token decode crosses multiple flush boundaries.
const strmFlushEvery = 3

// strmDefectChunk is the chunk whose leading flush boundary the injected
// defects corrupt. Chunk 2's boundary sits mid-sequence (token index 6 with
// strmFlushEvery=3), so the intact prefix proves the localization is doing
// work — the failure pins to the boundary token, not to "the stream was wrong".
const strmDefectChunk = 2

// strmBoundaryIndex is the token index of the corrupted flush boundary.
const strmBoundaryIndex = strmDefectChunk * strmFlushEvery

// strmFullRunner is the reference path: the whole decode in one non-streaming
// run. It never chunks or flushes, so its trace is what the output looks like
// when delivery cannot lose or repeat anything.
type strmFullRunner struct{}

func (strmFullRunner) Name() string { return "streaming-full" }

func (r strmFullRunner) Run(c QualityCase) (Trace, error) {
	t := strmDecode(c.Params.Seed, c.Params.MaxTokens)
	t.Runner = r.Name()
	return t, nil
}

// strmStreamRunner is the engine path: it runs the same decode but delivers it
// as flushed chunks of strmFlushEvery tokens, concatenating the flushes into
// the trace it reports. The zero defect is a faithful stream; the defect field
// (set via strmStreamingEngine) injects a chunk/flush-boundary bug.
type strmStreamRunner struct {
	label  string
	defect string
}

func (r strmStreamRunner) Name() string {
	if r.label != "" {
		return r.label
	}
	return "streaming-engine"
}

func (r strmStreamRunner) Run(c QualityCase) (Trace, error) {
	full := strmDecode(c.Params.Seed, c.Params.MaxTokens)
	var out []string
	var parts []string // one assembled text piece per flush
	for start := 0; start < len(full.Tokens); start += strmFlushEvery {
		end := start + strmFlushEvery
		if end > len(full.Tokens) {
			end = len(full.Tokens)
		}
		chunk := append([]string(nil), full.Tokens[start:end]...)
		atDefectBoundary := start == strmBoundaryIndex && start > 0
		switch {
		case r.defect == "drop-boundary" && atDefectBoundary:
			// The boundary token is decoded but its emission is lost in the
			// flush handoff: the stream is one token short from the boundary on.
			chunk = chunk[1:]
		case r.defect == "dup-boundary" && atDefectBoundary:
			// The last token of the previous flush is re-emitted at the start of
			// this one — the retry replays an already-delivered token.
			chunk = append([]string{out[len(out)-1]}, chunk...)
		}
		out = append(out, chunk...)
		parts = append(parts, strings.Join(chunk, " "))
	}
	text := strings.Join(parts, " ")
	if r.defect == "glue-boundary" && strmDefectChunk < len(parts) {
		// Tokens are all delivered, but the flush handoff glues the defect
		// chunk's text onto the previous flush with no separator — the
		// assembled text differs while the token stream matches.
		text = strings.Join(parts[:strmDefectChunk], " ") + strings.Join(parts[strmDefectChunk:], " ")
	}
	return Trace{Runner: r.Name(), Tokens: out, Text: text}, nil
}

// strmStreamingEngine returns a streaming engine runner with an optional
// injected chunk/flush-boundary defect: "" streams faithfully;
// "drop-boundary" loses the first token of the defect chunk's flush;
// "dup-boundary" re-emits the last token of the previous flush;
// "glue-boundary" delivers every token but assembles the text across the
// boundary with no separator. These are the deterministic mutants the tests
// use to prove the equivalence gate trips at the boundary.
func strmStreamingEngine(defect string) strmStreamRunner {
	switch defect {
	case "drop-boundary", "dup-boundary", "glue-boundary":
		return strmStreamRunner{label: "streaming-engine-" + defect, defect: defect}
	default:
		return strmStreamRunner{label: "streaming-engine-clean"}
	}
}

// strmStreamingCase builds a streaming-equivalence case: a deterministic
// decode of steps tokens under seed whose reference trace is the
// non-streaming full output. The case says nothing about chunk size — a
// correct stream must be invisible in the delivered output for any chunking.
func strmStreamingCase(seed int64, steps int) QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "streaming-equivalence-demo",
		Version:   1,
		Prompt:    "Stream the decode in flushed chunks; the delivered output must equal the non-streaming run.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: steps, Seed: seed},
		Reference: strmDecode(seed, steps),
		Oracles:   []string{"streaming-equivalence"},
	}
}

// strmFirstTextDiff returns the first byte index at which a and b differ, or
// -1 when they are equal. The vocab is ASCII, so byte indexes are character
// indexes here.
func strmFirstTextDiff(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

// strmSnippet returns a short window of s around byte index i for a readable
// divergence detail.
func strmSnippet(s string, i int) string {
	start := i - 8
	if start < 0 {
		start = 0
	}
	end := i + 8
	if end > len(s) {
		end = len(s)
	}
	return s[start:end]
}

// strmEquivalence is the streaming-equivalence differential oracle (#4531):
// the concatenation of the engine's streamed chunks must equal the
// non-streaming reference in BOTH the token sequence and the assembled text.
// A token dropped or duplicated at a chunk/flush boundary is reported as the
// first divergence at that token index; a text-assembly defect with an intact
// token stream still fails, localized to the first differing byte.
type strmEquivalence struct{}

func (strmEquivalence) Name() string { return "streaming-equivalence" }
func (strmEquivalence) Kind() string { return "differential" }

func init() { Register(strmEquivalence{}) }

func (strmEquivalence) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "streaming-equivalence", Kind: "differential", Pass: true}
	n := len(ref.Tokens)
	if len(eng.Tokens) < n {
		n = len(eng.Tokens)
	}
	for i := 0; i < n; i++ {
		if ref.Tokens[i] != eng.Tokens[i] {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: eng.Tokens[i]}
			v.Detail = fmt.Sprintf("streamed output diverged at token %d: reference %q, engine %q — a token was dropped or duplicated at a chunk/flush boundary",
				i, ref.Tokens[i], eng.Tokens[i])
			return v
		}
	}
	if len(ref.Tokens) != len(eng.Tokens) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(eng.Tokens, n)}
		v.Detail = fmt.Sprintf("streamed output length diverged at %d: reference has %d tokens, engine has %d — a boundary flush lost or repeated a token",
			n, len(ref.Tokens), len(eng.Tokens))
		return v
	}
	if ref.Text != eng.Text {
		v.Pass = false
		i := strmFirstTextDiff(ref.Text, eng.Text)
		v.Detail = fmt.Sprintf("token streams match but assembled text diverged at byte %d: reference ...%q..., engine ...%q... — chunk flushes were glued into the text incorrectly",
			i, strmSnippet(ref.Text, i), strmSnippet(eng.Text, i))
		return v
	}
	v.Detail = fmt.Sprintf("streamed chunks reassembled the non-streaming output exactly: %d tokens and the assembled text match", len(ref.Tokens))
	return v
}
