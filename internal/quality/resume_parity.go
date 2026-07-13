package quality

import (
	"encoding/binary"
	"fmt"
	"strings"
)

// resume_parity.go is the retry/resume child of the quality spine (#4538): it
// proves that interrupting a generation, serializing its state, restoring the
// state from bytes, and resuming to completion preserves semantics EXACTLY. An
// uninterrupted decode is the reference; a decode interrupted at step k and
// resumed from a byte snapshot is the engine. Correct resume means the two
// token sequences are identical — any state lost, dropped, or duplicated across
// the interrupt/restore boundary surfaces as a first divergence at (or after)
// the resume boundary, localized to its exact token index.

// resumeVocab is the small fixed vocabulary the deterministic stateful decoder
// emits from. It is disjoint from the other spine vocabularies so a resume
// trace is never confused with a sibling oracle's trace in a failure bundle.
var resumeVocab = []string{"alloy", "breaker", "cinder", "dune", "eddy", "fjord", "grove", "haze"}

// resumeState is the explicit, serializable decode state carried between steps:
// a PRNG stream state advanced by a fixed gamma each step, and an accumulator
// that folds every prior draw back in. Because the accumulator is genuinely
// CARRIED (token i depends on the whole history, not just (seed, i)), a resume
// path that fails to round-trip any field of this struct decodes differently
// from the first post-resume step — exactly the defect class #4538 gates.
type resumeState struct {
	rng  uint64 // splitmix64-style stream state, advanced each step
	acc  uint64 // running fold of every prior draw (history dependence)
	step uint64 // steps taken so far
}

// resumeNewState returns the decode state at step zero for a pinned seed.
func resumeNewState(seed int64) resumeState {
	return resumeState{rng: uint64(seed)*0x9e3779b97f4a7c15 + 0x243f6a8885a308d3}
}

// next advances the state one step and returns the emitted token: the PRNG
// state advances by the golden gamma, is mixed with the carried accumulator,
// and the finalized draw both selects the token and folds into the accumulator.
func (s *resumeState) next() string {
	s.rng += 0x9e3779b97f4a7c15
	z := s.rng ^ s.acc
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z ^= z >> 31
	s.acc = s.acc*0x100000001b3 + z
	s.step++
	return resumeVocab[z%uint64(len(resumeVocab))]
}

// resumeSnapshotLen is the exact byte length of a serialized resumeState.
const resumeSnapshotLen = 24

// marshal serializes the full decode state to bytes — the snapshot an
// interrupted generation persists. Every field is included; a snapshot that
// omitted the accumulator would BE the state-loss defect this child detects.
func (s resumeState) marshal() []byte {
	b := make([]byte, resumeSnapshotLen)
	binary.BigEndian.PutUint64(b[0:8], s.rng)
	binary.BigEndian.PutUint64(b[8:16], s.acc)
	binary.BigEndian.PutUint64(b[16:24], s.step)
	return b
}

// resumeRestore reconstructs a decode state from snapshot bytes. It refuses a
// snapshot of the wrong length rather than guessing — a truncated snapshot
// restored as zeroes would silently reproduce the reseed defect.
func resumeRestore(b []byte) (resumeState, error) {
	if len(b) != resumeSnapshotLen {
		return resumeState{}, fmt.Errorf("resume snapshot is %d bytes, want %d", len(b), resumeSnapshotLen)
	}
	return resumeState{
		rng:  binary.BigEndian.Uint64(b[0:8]),
		acc:  binary.BigEndian.Uint64(b[8:16]),
		step: binary.BigEndian.Uint64(b[16:24]),
	}, nil
}

// resumeDecode runs the uninterrupted decode: seed in, steps straight through,
// no interrupt. It is the ground truth both the reference runner and every
// faithful resumed decode must reproduce token for token.
func resumeDecode(seed int64, steps int) Trace {
	st := resumeNewState(seed)
	toks := make([]string, 0, steps)
	for i := 0; i < steps; i++ {
		toks = append(toks, st.next())
	}
	return Trace{Tokens: toks, Text: strings.Join(toks, " ")}
}

// ResumeUninterruptedRunner is the reference path: it decodes the case's full
// Params.MaxTokens steps in one uninterrupted run under Params.Seed. It never
// snapshots or restores, so its trace is what generation looks like when
// nothing is ever interrupted.
type ResumeUninterruptedRunner struct{}

func (ResumeUninterruptedRunner) Name() string { return "resume-uninterrupted" }

func (r ResumeUninterruptedRunner) Run(c QualityCase) (Trace, error) {
	t := resumeDecode(c.Params.Seed, c.Params.MaxTokens)
	t.Runner = r.Name()
	return t, nil
}

// ResumeRunner is the engine path: it decodes k steps, snapshots the live
// state to bytes (the interrupt), restores a FRESH state value from those
// bytes, and resumes to completion. The zero defect is a faithful resume; the
// defect field (set via ResumeEngine) injects a state-loss bug at the
// interrupt/restore boundary.
type ResumeRunner struct {
	Label     string
	Interrupt int
	defect    string
}

func (r ResumeRunner) Name() string {
	if r.Label != "" {
		return r.Label
	}
	return "resume-engine"
}

func (r ResumeRunner) Run(c QualityCase) (Trace, error) {
	n := c.Params.MaxTokens
	k := r.Interrupt
	if k < 0 {
		k = 0
	}
	if k > n {
		k = n
	}

	// Phase 1: decode up to the interrupt.
	st := resumeNewState(c.Params.Seed)
	toks := make([]string, 0, n)
	for i := 0; i < k; i++ {
		toks = append(toks, st.next())
	}

	// The interrupt: serialize the live state to bytes, then restore from those
	// bytes into a fresh value. The in-memory state is dead past this point —
	// only what round-tripped through the snapshot survives, as in a real
	// process restart.
	snap := st.marshal()
	restored, err := resumeRestore(snap)
	if err != nil {
		return Trace{}, err
	}
	st = restored

	// Injected boundary defects — each a real-world resume failure mode.
	target := n
	switch r.defect {
	case "reseed":
		// State lost on restore: the resumed process re-seeds from scratch, so
		// generation silently replays from token 0 at position k.
		st = resumeNewState(c.Params.Seed)
	case "drop-boundary":
		// The boundary token is decoded but its emission is lost in the
		// handoff: the resumed stream is one token short from index k on.
		if k < n {
			st.next()
			target = n - 1
		}
	case "dup-boundary":
		// The last pre-snapshot token is re-emitted after restore — the retry
		// replays an already-delivered step.
		if k > 0 {
			toks = append(toks, toks[len(toks)-1])
		}
	}

	// Phase 2: resume to completion from the restored state.
	for len(toks) < target {
		toks = append(toks, st.next())
	}
	return Trace{Runner: r.Name(), Tokens: toks, Text: strings.Join(toks, " ")}, nil
}

// ResumeEngine returns a resuming engine runner interrupted at step k with an
// optional injected defect: "" resumes faithfully from the byte snapshot;
// "reseed" loses the restored state and restarts generation from scratch;
// "drop-boundary" loses the boundary token's emission; "dup-boundary"
// re-emits the last pre-snapshot token. These are the deterministic mutants
// the tests use to prove the resume gate trips at the boundary.
func ResumeEngine(k int, defect string) ResumeRunner {
	switch defect {
	case "reseed", "drop-boundary", "dup-boundary":
		return ResumeRunner{Label: "resume-engine-" + defect, Interrupt: k, defect: defect}
	default:
		return ResumeRunner{Label: "resume-engine-faithful", Interrupt: k}
	}
}

// ResumeParityCase builds a resume-parity case: a deterministic stateful decode
// of steps tokens under seed, whose reference trace is the uninterrupted run.
// The case itself says nothing about WHERE the engine is interrupted — a
// correct resume must be invisible in the output for any k.
func ResumeParityCase(seed int64, steps int) QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "resume-parity-demo",
		Version:   1,
		Prompt:    "Resume mid-generation from a serialized snapshot and complete the sequence.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: steps, Seed: seed},
		Reference: resumeDecode(seed, steps),
		Oracles:   []string{"resume-parity"},
	}
}

// ResumeParity is the differential oracle for retry/mid-generation resume
// (#4538): an interrupted-then-resumed generation must emit exactly the token
// sequence the uninterrupted reference emits. Any state lost, dropped, or
// duplicated across the interrupt/restore boundary is reported as the FIRST
// divergence — the resume boundary is the usual suspect, so "the retry came
// back different" localizes to "token 3 was 'dune' where the reference emitted
// 'grove'".
type ResumeParity struct{}

func (ResumeParity) Name() string { return "resume-parity" }
func (ResumeParity) Kind() string { return "differential" }

func init() { Register(ResumeParity{}) }

func (ResumeParity) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "resume-parity", Kind: "differential", Pass: true}
	n := len(ref.Tokens)
	if len(eng.Tokens) < n {
		n = len(eng.Tokens)
	}
	for i := 0; i < n; i++ {
		if ref.Tokens[i] != eng.Tokens[i] {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: eng.Tokens[i]}
			v.Detail = fmt.Sprintf("resumed generation diverged at token %d: reference %q, engine %q — state was not preserved across the interrupt/restore boundary",
				i, ref.Tokens[i], eng.Tokens[i])
			return v
		}
	}
	if len(ref.Tokens) != len(eng.Tokens) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(eng.Tokens, n)}
		v.Detail = fmt.Sprintf("resumed generation length diverged at %d: reference has %d tokens, engine has %d — a token was dropped or duplicated at the resume boundary",
			n, len(ref.Tokens), len(eng.Tokens))
		return v
	}
	v.Detail = fmt.Sprintf("resume preserved semantics: %d tokens identical to the uninterrupted reference", len(ref.Tokens))
	return v
}
