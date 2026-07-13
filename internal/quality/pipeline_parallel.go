package quality

import (
	"encoding/binary"
	"fmt"
	"strings"
)

// pipeline_parallel.go is the pipeline-parallel child of the quality spine
// (#4543): it proves that splitting a model's layers across pipeline stages —
// with activations handed off at every stage boundary, and a mid-generation
// resume that snapshots/restores state AT a boundary — preserves output
// EXACTLY. The single-stage (monolithic) decode is the reference; the staged
// decode is the engine. Correct pipelining is invisible in the output: any
// activation dropped or duplicated in a stage handoff, or any stage state lost
// across a boundary resume, surfaces as a first divergence localized to the
// affected token.

// ppVocab is the small fixed vocabulary the deterministic layered decoder
// emits from. It is disjoint from the other spine vocabularies so a
// pipeline-parity trace is never confused with a sibling oracle's trace in a
// failure bundle.
var ppVocab = []string{"arbor", "bastion", "cairn", "dolmen", "esker", "fell", "gully", "heath"}

// ppNumLayers is the depth of the toy layered model, and ppSplit is the stage
// boundary: layers [0, ppSplit) run on stage one, layers [ppSplit,
// ppNumLayers) on stage two. One boundary is enough — every handoff defect
// class this child gates lives at a boundary, and more boundaries are more of
// the same seam.
const (
	ppNumLayers = 4
	ppSplit     = 2
)

// ppLayerConst gives each layer a distinct mixing constant, so the layers are
// not interchangeable: applying a stage's layers twice, or skipping one,
// cannot cancel out to the reference computation.
var ppLayerConst = [ppNumLayers]uint64{
	0x9e3779b97f4a7c15,
	0xbf58476d1ce4e5b9,
	0x94d049bb133111eb,
	0x2545f4914f6cdd1d,
}

// ppMix is the splitmix64-style finalizer shared by every layer: a bijective
// avalanche so near-miss activations decode to unrelated draws.
func ppMix(z uint64) uint64 {
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// ppLayerStep applies layer l to an incoming activation given the layer's
// carried state, returning the outgoing activation and the updated state. The
// state genuinely CARRIES across tokens (a KV-cache stand-in): token i's
// output depends on every prior token that flowed through the layer, so a
// stage whose state is corrupted at a boundary decodes differently from the
// affected token on.
func ppLayerStep(l int, act, state uint64) (uint64, uint64) {
	z := ppMix(act ^ state ^ ppLayerConst[l])
	return z, state*0x100000001b3 + z
}

// ppModelState is the full decode state: each layer's carried state plus the
// carry (the previous token's final activation, fed forward as the next
// token's embedding) and the step count. The single-stage reference owns all
// of it in one place; the pipeline engine partitions layer[0:ppSplit] to
// stage one and layer[ppSplit:] to stage two.
type ppModelState struct {
	layer [ppNumLayers]uint64
	carry uint64
	step  uint64
}

// ppNewModelState returns the decode state at step zero for a pinned seed.
func ppNewModelState(seed int64) ppModelState {
	return ppModelState{carry: uint64(seed)*0x9e3779b97f4a7c15 + 0x243f6a8885a308d3}
}

// ppEmbed produces token step's initial activation from the carried final
// activation of the previous token — the input both the single-stage and the
// staged decode feed to layer 0.
func ppEmbed(carry uint64) uint64 {
	return ppMix(carry + 0x9e3779b97f4a7c15)
}

// next advances the single-stage decode one token: the embedded activation
// flows through ALL layers in one process — no boundary, no handoff.
func (s *ppModelState) next() string {
	act := ppEmbed(s.carry)
	for l := 0; l < ppNumLayers; l++ {
		act, s.layer[l] = ppLayerStep(l, act, s.layer[l])
	}
	s.carry = act
	s.step++
	return ppVocab[act%uint64(len(ppVocab))]
}

// ppDecodeSingle runs the uninterrupted single-stage decode: seed in, steps
// straight through every layer in one process. It is the ground truth every
// staged decode — however split, and wherever resumed — must reproduce token
// for token.
func ppDecodeSingle(seed int64, steps int) Trace {
	st := ppNewModelState(seed)
	toks := make([]string, 0, steps)
	for i := 0; i < steps; i++ {
		toks = append(toks, st.next())
	}
	return Trace{Tokens: toks, Text: strings.Join(toks, " ")}
}

// ppSnapshotLen is the exact byte length of a serialized boundary snapshot:
// every layer's carried state, the carry, the IN-FLIGHT activation sitting at
// the stage boundary, and the step count. The in-flight activation is the
// field unique to a stage-boundary resume — a snapshot taken between stages
// mid-token must persist the half-computed token's activation too.
const ppSnapshotLen = 8*ppNumLayers + 24

// ppMarshalSnapshot serializes the decode state plus the in-flight boundary
// activation. Every field is included; a snapshot that omitted a stage's
// layer states would BE the state-loss defect this child detects.
func ppMarshalSnapshot(s ppModelState, act uint64) []byte {
	b := make([]byte, ppSnapshotLen)
	for l := 0; l < ppNumLayers; l++ {
		binary.BigEndian.PutUint64(b[8*l:], s.layer[l])
	}
	binary.BigEndian.PutUint64(b[8*ppNumLayers:], s.carry)
	binary.BigEndian.PutUint64(b[8*ppNumLayers+8:], act)
	binary.BigEndian.PutUint64(b[8*ppNumLayers+16:], s.step)
	return b
}

// ppRestoreSnapshot reconstructs the decode state and in-flight activation
// from snapshot bytes. It refuses a snapshot of the wrong length rather than
// guessing — a truncated snapshot restored as zeroes would silently reproduce
// the state-loss defect.
func ppRestoreSnapshot(b []byte) (ppModelState, uint64, error) {
	if len(b) != ppSnapshotLen {
		return ppModelState{}, 0, fmt.Errorf("pipeline snapshot is %d bytes, want %d", len(b), ppSnapshotLen)
	}
	var s ppModelState
	for l := 0; l < ppNumLayers; l++ {
		s.layer[l] = binary.BigEndian.Uint64(b[8*l:])
	}
	s.carry = binary.BigEndian.Uint64(b[8*ppNumLayers:])
	act := binary.BigEndian.Uint64(b[8*ppNumLayers+8:])
	s.step = binary.BigEndian.Uint64(b[8*ppNumLayers+16:])
	return s, act, nil
}

// ppDefectToken is the token whose stage handoff the drop/dup mutants
// corrupt: mid-sequence, so the passing prefix proves the localization is
// doing work (the failure pins to the handoff's token, not to "the whole
// staged run looked wrong").
const ppDefectToken = 3

// PPPipelineRunner is the engine path: it decodes with the layers split
// across two stages, handing the activation across the boundary for every
// token, and — when Resume names a token in range — interrupting AT the stage
// boundary of that token: stage one has run, stage two has not, and only what
// round-trips through the byte snapshot survives. The zero defect is a
// faithful pipeline; the defect field (set via PPPipelineEngine) injects a
// boundary bug.
type PPPipelineRunner struct {
	Label  string
	Resume int
	defect string
}

func (p PPPipelineRunner) Name() string {
	if p.Label != "" {
		return p.Label
	}
	return "pp-pipeline-engine"
}

func (p PPPipelineRunner) Run(c QualityCase) (Trace, error) {
	n := c.Params.MaxTokens
	st := ppNewModelState(c.Params.Seed)
	toks := make([]string, 0, n)
	for i := 0; i < n; i++ {
		// Stage one: embed the carry and run layers [0, ppSplit).
		act := ppEmbed(st.carry)
		for l := 0; l < ppSplit; l++ {
			act, st.layer[l] = ppLayerStep(l, act, st.layer[l])
		}

		// The stage boundary: the activation is handed from stage one to
		// stage two. This is where the injected handoff defects live.
		switch {
		case p.defect == "drop-handoff" && i == ppDefectToken:
			// The handoff message is lost: stage two reads a zeroed buffer
			// where token i's activation should be.
			act = 0
		case p.defect == "dup-handoff" && i == ppDefectToken:
			// The handoff is delivered twice: stage two folds the same
			// activation through its layers a second time (the extra pass
			// below), corrupting both the token and stage two's carried state.
			for l := ppSplit; l < ppNumLayers; l++ {
				act, st.layer[l] = ppLayerStep(l, act, st.layer[l])
			}
		}

		// A resume at THIS stage boundary: serialize the live state plus the
		// in-flight activation to bytes, then restore into fresh values. The
		// in-memory state is dead past this point — only what round-tripped
		// through the snapshot survives, as in a real stage restart.
		if i == p.Resume {
			snap := ppMarshalSnapshot(st, act)
			restored, restoredAct, err := ppRestoreSnapshot(snap)
			if err != nil {
				return Trace{}, err
			}
			st, act = restored, restoredAct
			if p.defect == "resume-state-loss" {
				// Stage two's carried state fails to restore at the boundary:
				// its layers resume from scratch while stage one's survive.
				for l := ppSplit; l < ppNumLayers; l++ {
					st.layer[l] = 0
				}
			}
		}

		// Stage two: layers [ppSplit, ppNumLayers) complete the token.
		for l := ppSplit; l < ppNumLayers; l++ {
			act, st.layer[l] = ppLayerStep(l, act, st.layer[l])
		}
		st.carry = act
		st.step++
		toks = append(toks, ppVocab[act%uint64(len(ppVocab))])
	}
	return Trace{Runner: p.Name(), Tokens: toks, Text: strings.Join(toks, " ")}, nil
}

// PPPipelineEngine returns a staged engine runner that resumes at the stage
// boundary of token resume (a resume out of [0, MaxTokens) never triggers),
// with an optional injected defect: "" pipelines and resumes faithfully;
// "drop-handoff" loses token ppDefectToken's activation in the stage handoff;
// "dup-handoff" delivers that activation to stage two twice;
// "resume-state-loss" restores the boundary snapshot without stage two's
// carried state. These are the deterministic mutants the tests use to prove
// the parity gate trips at the affected token.
func PPPipelineEngine(resume int, defect string) PPPipelineRunner {
	switch defect {
	case "drop-handoff", "dup-handoff", "resume-state-loss":
		return PPPipelineRunner{Label: "pp-engine-" + defect, Resume: resume, defect: defect}
	default:
		return PPPipelineRunner{Label: "pp-engine-faithful", Resume: resume}
	}
}

// PPParityCase builds a pipeline-parity case: a deterministic layered decode
// of steps tokens under seed, whose reference trace is the single-stage run.
// The case says nothing about HOW the engine is staged or WHERE it resumes —
// a correct pipeline split and a correct boundary resume must both be
// invisible in the output.
func PPParityCase(seed int64, steps int) QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "pp-parity-demo",
		Version:   1,
		Prompt:    "Decode across pipeline stages; the stage split and any boundary resume must be invisible in the output.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: steps, Seed: seed},
		Reference: ppDecodeSingle(seed, steps),
		Oracles:   []string{"pp-parity"},
	}
}

// PPParity is the differential oracle for pipeline-parallel execution
// (#4543): a decode split across stages — including one resumed from a
// stage-boundary snapshot — must emit exactly the token sequence the
// single-stage reference emits. Any activation dropped or duplicated in a
// handoff, or any stage state lost across a boundary resume, is reported as
// the FIRST divergence, so "the sharded engine reads wrong" localizes to
// "token 3 was 'cairn' where the reference emitted 'fell'".
type PPParity struct{}

func (PPParity) Name() string { return "pp-parity" }
func (PPParity) Kind() string { return "differential" }

func init() { Register(PPParity{}) }

func (PPParity) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "pp-parity", Kind: "differential", Pass: true}
	n := len(ref.Tokens)
	if len(eng.Tokens) < n {
		n = len(eng.Tokens)
	}
	for i := 0; i < n; i++ {
		if ref.Tokens[i] != eng.Tokens[i] {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: eng.Tokens[i]}
			v.Detail = fmt.Sprintf("pipeline-parallel decode diverged at token %d: reference %q, engine %q — state was dropped, duplicated, or lost at a stage boundary",
				i, ref.Tokens[i], eng.Tokens[i])
			return v
		}
	}
	if len(ref.Tokens) != len(eng.Tokens) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(eng.Tokens, n)}
		v.Detail = fmt.Sprintf("pipeline-parallel decode length diverged at %d: reference has %d tokens, engine has %d — a stage handoff dropped or duplicated a token",
			n, len(ref.Tokens), len(eng.Tokens))
		return v
	}
	v.Detail = fmt.Sprintf("pipeline split and boundary resume were invisible: %d tokens identical to the single-stage reference", len(ref.Tokens))
	return v
}
