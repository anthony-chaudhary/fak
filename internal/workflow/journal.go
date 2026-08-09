package workflow

// journal.go — the append-only step journal and the witness-keyed resume it feeds (#2444).
//
// A run journals one row per settled step: the step id, the hash of the inputs it ran on,
// the hash of the epoch it ran under, and the hash of its structured output. Folding those
// rows is pure — the same rows plus the same injected clock in, the same state out, with
// zero I/O and zero clock reads — the discipline internal/toolproc's Fold keeps.
//
// Resume is the half that refuses to take the journal's word for it. A row saying a step
// finished is the run's own narration; it is not evidence that the step's effect is still
// there. So a step is skipped only when its completion is corroborated:
//
//   - a pure step re-derives its own evidence: the journaled output still hashes to the
//     journaled output hash, under an unchanged inputs hash and an unchanged epoch hash.
//   - an effectful step must carry a claim in the dos_verify grammar (ancestor:, committed:,
//     grep:, ...) that the injected Corroborate confirms against evidence the run did not
//     author — git ancestry, a tracked path, a ship-stamp in history.
//
// Everything else re-executes: an unjournaled step, drifted inputs, a changed epoch, a
// claim nobody corroborates, or a step whose upstream is itself re-running. Fail-closed: a
// nil Corroborate re-executes every effectful step rather than trusting the narration, and
// a commit that was reverted stops corroborating the moment the caller's oracle reads git
// again — which is the whole point of re-deriving "done" instead of replaying it.
//
// The package is tier 1, so this file stays stdlib-only: the evidence oracle is a func the
// caller supplies, letting cmd/fak back it with the real git-evidence resolver while a test
// drives it from a table.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// JournalSchema is the row schema tag. A row that names another schema is refused at the
// boundary rather than coerced, so a format change can never read as a silent cache hit.
const JournalSchema = "fak.workflow-journal/1"

// StepKind is the CLOSED effect vocabulary. It decides which evidence a resume demands:
// a pure step re-derives its own from the journal, an effectful step needs an outside
// corroboration. Parsing fails closed on anything else.
type StepKind string

const (
	StepPure      StepKind = "pure"
	StepEffectful StepKind = "effectful"
)

// Entry is one appended journal row: what a step ran on, what epoch it ran under, and what
// it produced. Claim is the dos_verify-grammar claim an effectful step's completion must be
// corroborated by; a pure step leaves it empty.
type Entry struct {
	Schema     string   `json:"schema"`
	Run        string   `json:"run"`
	Step       string   `json:"step"`
	Kind       StepKind `json:"kind"`
	InputsHash string   `json:"inputs_hash"`
	EpochHash  string   `json:"epoch_hash"`
	OutputHash string   `json:"output_hash"`
	Output     string   `json:"output,omitempty"`
	Claim      string   `json:"claim,omitempty"`
	TSMS       int64    `json:"ts_ms"`
}

// State is a folded journal: the last row per step, the steps in a deterministic order,
// and the injected fold time. It carries no live clock read of its own.
type State struct {
	Run      string           `json:"run"`
	Steps    map[string]Entry `json:"steps"`
	Order    []string         `json:"order"`
	Rows     int              `json:"rows"`
	FoldedMS int64            `json:"folded_ms"`
}

// Fold reduces journal rows to the state a resume reads. It is pure: nowMS is injected,
// never read, and the same rows fold to the same state byte for byte. Later rows for a
// step supersede earlier ones (a re-execution overwrites its own cache line), and the
// step order is sorted so the fold is independent of arrival order.
func Fold(entries []Entry, nowMS int64) (State, error) {
	st := State{Steps: make(map[string]Entry, len(entries)), FoldedMS: nowMS, Rows: len(entries)}
	for i, e := range entries {
		if e.Schema != "" && e.Schema != JournalSchema {
			return State{}, fmt.Errorf("workflow: journal row %d: unknown schema %q", i, e.Schema)
		}
		if strings.TrimSpace(e.Step) == "" {
			return State{}, fmt.Errorf("workflow: journal row %d: empty step id", i)
		}
		switch e.Kind {
		case StepPure, StepEffectful:
		default:
			return State{}, fmt.Errorf("workflow: journal row %d (step %q): unknown kind %q", i, e.Step, e.Kind)
		}
		if st.Run == "" {
			st.Run = e.Run
		}
		e.Schema = JournalSchema
		st.Steps[e.Step] = e
	}
	st.Order = make([]string, 0, len(st.Steps))
	for id := range st.Steps {
		st.Order = append(st.Order, id)
	}
	sort.Strings(st.Order)
	return st, nil
}

// ReadJournal decodes an append-only JSONL journal. Unknown fields are rejected so a
// typo'd or foreign row is a loud refusal, not a silent skip that reads as a cache hit.
func ReadJournal(r io.Reader) ([]Entry, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var out []Entry
	for {
		var e Entry
		err := dec.Decode(&e)
		if err == io.EOF {
			return out, nil
		}
		if err != nil {
			return nil, fmt.Errorf("workflow: read journal: %w", err)
		}
		out = append(out, e)
	}
}

// AppendEntry writes one row in the append-only form ReadJournal parses.
func AppendEntry(w io.Writer, e Entry) error {
	e.Schema = JournalSchema
	line, err := json.Marshal(e)
	if err != nil {
		return fmt.Errorf("workflow: append journal: %w", err)
	}
	if _, err := w.Write(append(line, '\n')); err != nil {
		return fmt.Errorf("workflow: append journal: %w", err)
	}
	return nil
}

// HashOutput is the structured-output hash a pure step's cache line is keyed by.
func HashOutput(out string) string {
	sum := sha256.Sum256([]byte(out))
	return hex.EncodeToString(sum[:])
}

// StepInputsHash hashes everything that decides a step's output: its own declaration and
// the output hashes of the steps it needs, folded in sorted order. An upstream that
// produced something different lands here as a different hash, so its dependents cannot
// be served from the cache.
func StepInputsHash(n Node, depHashes map[string]string) string {
	var b strings.Builder
	b.WriteString("id=" + n.ID + "\x00op=" + n.Op + "\x00payload=" + n.Payload)
	b.WriteString("\x00retries=" + strconv.Itoa(n.Retries))
	needs := append([]string(nil), n.Needs...)
	sort.Strings(needs)
	for _, d := range needs {
		b.WriteString("\x00need=" + d + "=" + depHashes[d])
	}
	return HashOutput(b.String())
}

// GraphEpoch hashes the epoch a run's steps are cached under: the compiled graph plus the
// caller's epoch label (its policy revision). Change either and every cache line drifts,
// so a re-policied run re-executes instead of replaying decisions made under the old one.
func GraphEpoch(g *Graph, label string) string {
	var b strings.Builder
	b.WriteString("epoch=" + label + "\x00workflow=")
	if g != nil {
		b.WriteString(g.Name)
		for _, n := range g.Nodes {
			b.WriteString("\x00" + StepInputsHash(n, nil))
		}
	}
	return HashOutput(b.String())
}

// Disposition is the CLOSED resume vocabulary: a step is either served from its cache line
// or handed to the runner. There is no third state — an uncertain step re-executes.
type Disposition string

const (
	DispSkip    Disposition = "skip"
	DispExecute Disposition = "execute"
)

// The closed reason vocabulary an execute disposition cites. A step never re-executes for
// free-text reasons, so a report can be folded by a machine.
const (
	ReasonUnjournaled     = "unjournaled"
	ReasonEpochDrift      = "epoch_drift"
	ReasonInputsDrift     = "inputs_drift"
	ReasonOutputMismatch  = "output_hash_mismatch"
	ReasonClaimMissing    = "claim_missing"
	ReasonClaimUnverified = "claim_unverified"
	ReasonUpstreamRerun   = "upstream_rerun"
)

// Corroborate answers whether an effectful step's claim is confirmed by evidence the run
// did not author, returning the source that answered (for the report) alongside the
// verdict. It is the caller's seam onto the dos_verify ladder — a registry row, git
// ancestry, a ship-stamp grep. A false verdict always re-executes the step.
type Corroborate func(ctx context.Context, step, claim string) (source string, ok bool)

// StepVerdict is one step's resume disposition. Source is filled only on a skip (naming
// the evidence that allowed it) and Reason only on an execute.
type StepVerdict struct {
	Step        string      `json:"step"`
	Kind        StepKind    `json:"kind,omitempty"`
	Disposition Disposition `json:"disposition"`
	Source      string      `json:"source,omitempty"`
	Reason      string      `json:"reason,omitempty"`
	Output      string      `json:"-"`
}

// Resumption is the whole resume decision for a graph, in the graph's own deterministic
// topological order.
type Resumption struct {
	Run    string        `json:"run"`
	Epoch  string        `json:"epoch"`
	Steps  []StepVerdict `json:"steps"`
	Skips  int           `json:"skips"`
	Reruns int           `json:"reruns"`
}

// Resume decides, for every step of g, whether its journaled completion is corroborated
// well enough to serve from the cache or must be re-executed. It is pure: all evidence
// arrives through st and corr.
func Resume(ctx context.Context, g *Graph, st State, epoch string, corr Corroborate) Resumption {
	out := Resumption{Run: st.Run, Epoch: epoch}
	if g == nil {
		return out
	}
	if out.Run == "" {
		out.Run = g.Name
	}
	hashes := make(map[string]string, len(g.Nodes)) // step -> the output hash a dependent may rely on
	rerun := make(map[string]bool, len(g.Nodes))
	for _, n := range g.Nodes { // Compile ordered these topologically, so upstreams are settled
		v := StepVerdict{Step: n.ID, Disposition: DispExecute}
		row, journaled := st.Steps[n.ID]
		if journaled {
			v.Kind = row.Kind
		}
		switch {
		case upstreamRerun(n, rerun):
			v.Reason = ReasonUpstreamRerun
		case !journaled:
			v.Reason = ReasonUnjournaled
		case row.EpochHash != epoch:
			v.Reason = ReasonEpochDrift
		case row.InputsHash != StepInputsHash(n, hashes):
			v.Reason = ReasonInputsDrift
		case row.Kind == StepPure && HashOutput(row.Output) != row.OutputHash:
			v.Reason = ReasonOutputMismatch
		case row.Kind == StepPure:
			v.Disposition, v.Source, v.Output = DispSkip, "journal-hash:"+shortHash(row.OutputHash), row.Output
		case strings.TrimSpace(row.Claim) == "":
			v.Reason = ReasonClaimMissing
		default:
			src, ok := "", false
			if corr != nil { // fail-closed: no oracle means no corroboration
				src, ok = corr(ctx, n.ID, row.Claim)
			}
			if !ok {
				v.Reason = ReasonClaimUnverified
				break
			}
			if src == "" {
				src = "dos_verify:" + row.Claim
			}
			v.Disposition, v.Source, v.Output = DispSkip, src, row.Output
		}
		if v.Disposition == DispSkip {
			out.Skips++
			hashes[n.ID] = row.OutputHash
		} else {
			out.Reruns++
			rerun[n.ID] = true
		}
		out.Steps = append(out.Steps, v)
	}
	return out
}

// upstreamRerun reports whether any step this one needs is itself re-executing. Its output
// is then unknown at decision time, so serving this step from a cache line keyed on the old
// output would be trusting a value nothing has produced yet.
func upstreamRerun(n Node, rerun map[string]bool) bool {
	for _, d := range n.Needs {
		if rerun[d] {
			return true
		}
	}
	return false
}

func shortHash(h string) string {
	if len(h) > 12 {
		return h[:12]
	}
	return h
}

// ResumeRunner is the Runner a resume drives the engine through: a step the resumption
// skips returns its journaled output without ever reaching Base, and every other step is
// handed to Base. It records what it actually did, so a caller reports measured counts
// rather than the decision's intent — a step the engine never reached (an earlier failure
// aborted the run) is counted by neither.
type ResumeRunner struct {
	Base Runner

	verdicts map[string]StepVerdict
	mu       sync.Mutex
	skipped  map[string]bool
	executed map[string]bool
}

// NewResumeRunner binds a resumption to the runner that performs the steps it re-executes.
func NewResumeRunner(r Resumption, base Runner) *ResumeRunner {
	rr := &ResumeRunner{
		Base:     base,
		verdicts: make(map[string]StepVerdict, len(r.Steps)),
		skipped:  map[string]bool{},
		executed: map[string]bool{},
	}
	for _, v := range r.Steps {
		rr.verdicts[v.Step] = v
	}
	return rr
}

// Run implements Runner.
func (r *ResumeRunner) Run(ctx context.Context, in RunInput) (string, error) {
	if v, ok := r.verdicts[in.Node.ID]; ok && v.Disposition == DispSkip {
		r.mark(r.skipped, in.Node.ID)
		return v.Output, nil
	}
	r.mark(r.executed, in.Node.ID)
	if r.Base == nil {
		return "", fmt.Errorf("workflow: step %q must re-execute but no runner is bound", in.Node.ID)
	}
	return r.Base.Run(ctx, in)
}

// Verdict returns the resume disposition recorded for a step.
func (r *ResumeRunner) Verdict(step string) (StepVerdict, bool) {
	v, ok := r.verdicts[step]
	return v, ok
}

// Skipped and Executed report what the run actually did, in sorted order.
func (r *ResumeRunner) Skipped() []string  { return r.sorted(r.skipped) }
func (r *ResumeRunner) Executed() []string { return r.sorted(r.executed) }

func (r *ResumeRunner) mark(set map[string]bool, id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	set[id] = true
}

func (r *ResumeRunner) sorted(set map[string]bool) []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, 0, len(set))
	for id := range set {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}
