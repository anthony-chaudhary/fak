package quality

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// This file closes the one acceptance criterion of the portable failure bundle
// (#4515) that run.go did not carry: "one command replays an injected failure
// from its bundle". run.go can EMIT a scrubbed, replay-complete FailureBundle;
// until now nothing could CONSUME one, so the ladder's replay promise was
// asserted rather than exercised — a bundle could quietly lose the evidence a
// replay needs and no gate would notice.
//
// Three properties make a replay trustworthy:
//
//   - The bundle is the SOLE input. Replay reads no case file, no environment,
//     no live engine: it re-runs RunCase — the same orchestrator that produced
//     the bundle — over the case and traces the bundle carries. A replay that
//     could fall back to ambient state would prove nothing about the artifact.
//   - A bundle that cannot be replayed is reported INCONCLUSIVE, never as a
//     pass. Scrubbing is lossy by design, so evidence that was load-bearing for
//     a divergence can legitimately be gone; the answer to that is to say so,
//     which is the same "missing or inconclusive evidence is never pass" rule
//     every oracle in the package already holds to.
//   - Reproduction is judged on a SIGNATURE — the failing oracle, its kind, and
//     the first divergence — not on "it failed again". A bundle that replays to
//     a different first divergence has not reproduced its failure, and saying it
//     did would launder a drifted artifact into a green replay.

// ReplaySchema is the versioned envelope tag for a replay verdict. Consumers pin
// the major exactly as they do for the case, result, and manifest schemas.
const ReplaySchema = "fak-quality-replay/1"

// ReplaySignature is the identity of a failure: which oracle failed, of which
// kind, and where the token streams first diverged. It is what a replay must
// reproduce for the bundle to be considered replay-complete — the same
// localizing evidence the epic exists to produce, compared rather than narrated.
type ReplaySignature struct {
	FailingOracle   string      `json:"failing_oracle"`
	FailingKind     string      `json:"failing_kind"`
	FirstDivergence *Divergence `json:"first_divergence,omitempty"`
}

func (s ReplaySignature) equal(o ReplaySignature) bool {
	if s.FailingOracle != o.FailingOracle || s.FailingKind != o.FailingKind {
		return false
	}
	switch {
	case s.FirstDivergence == nil && o.FirstDivergence == nil:
		return true
	case s.FirstDivergence == nil || o.FirstDivergence == nil:
		return false
	default:
		return *s.FirstDivergence == *o.FirstDivergence
	}
}

// String renders a signature as the one line an operator reads to compare an
// expected failure against a replayed one.
func (s ReplaySignature) String() string {
	if s.FirstDivergence == nil {
		return fmt.Sprintf("%s (%s), no token divergence", s.FailingOracle, s.FailingKind)
	}
	d := s.FirstDivergence
	return fmt.Sprintf("%s (%s), first divergence at token %d: reference %q, engine %q",
		s.FailingOracle, s.FailingKind, d.Index, d.Reference, d.Engine)
}

func signatureOf(f FailureBundle) ReplaySignature {
	s := ReplaySignature{FailingOracle: f.FailingOracle, FailingKind: f.FailingKind}
	if f.FirstDivergence != nil {
		d := *f.FirstDivergence
		s.FirstDivergence = &d
	}
	return s
}

// ReplayVerdict is the machine-readable outcome of replaying one failure bundle.
// Reproduced is the only green state; Inconclusive marks a bundle that could not
// be replayed at all (as opposed to one that replayed to a different or absent
// failure), so a CI consumer can tell "this artifact is broken" from "this
// artifact no longer describes the defect it claims".
type ReplayVerdict struct {
	Schema       string           `json:"schema"`
	CaseID       string           `json:"case_id"`
	Reproduced   bool             `json:"reproduced"`
	Inconclusive bool             `json:"inconclusive"`
	Reason       string           `json:"reason"`
	Expected     ReplaySignature  `json:"expected"`
	Observed     *ReplaySignature `json:"observed,omitempty"`
	// Result is the freshly replayed run, present whenever the replay actually
	// executed. It carries its own manifest and provenance, so a replay is itself
	// an auditable artifact rather than a bare boolean.
	Result *Result `json:"result,omitempty"`
}

// ReplayComplete reports whether the bundle carries everything Replay needs, and
// names the first missing piece when it does not. It is the admission gate for a
// replay: a bundle that is short of evidence is refused with a reason rather than
// replayed to a misleading verdict.
func (f FailureBundle) ReplayComplete() error {
	if strings.TrimSpace(f.CaseID) == "" {
		return fmt.Errorf("bundle names no case_id")
	}
	if f.Case.ID != f.CaseID {
		return fmt.Errorf("bundle case_id %q does not match its embedded case id %q", f.CaseID, f.Case.ID)
	}
	if ok, why := f.Case.Valid(); !ok {
		return fmt.Errorf("embedded case is not runnable: %s", why)
	}
	if !f.Scrubbed {
		return fmt.Errorf("bundle is not marked scrubbed; refusing to replay an artifact the spine did not redact")
	}
	if strings.TrimSpace(f.FailingOracle) == "" {
		return fmt.Errorf("bundle names no failing oracle")
	}
	if traceEmpty(f.Reference) {
		return fmt.Errorf("bundle carries no reference trace to judge against")
	}
	if traceEmpty(f.Engine) {
		return fmt.Errorf("bundle carries no engine trace to judge")
	}
	if _, err := Lookup(f.Case.Oracles); err != nil {
		return fmt.Errorf("embedded case names an unrunnable oracle: %w", err)
	}
	// A bundle may attribute its failure to a judge the embedded case does not
	// list — a pre-oracle check, or an oracle the case was edited to drop. Replay
	// re-runs the case's own oracles and nothing else, so such a bundle cannot
	// reproduce its recorded signature here. Saying so is the honest answer;
	// replaying it anyway would report a DIFFERENT failure as if it were the one.
	if !namesOracle(f.Case.Oracles, f.FailingOracle) {
		return fmt.Errorf("bundle blames oracle %q, which the embedded case does not run (case runs: %s)",
			f.FailingOracle, strings.Join(f.Case.Oracles, ", "))
	}
	return nil
}

func traceEmpty(t Trace) bool { return len(t.Tokens) == 0 && strings.TrimSpace(t.Text) == "" }

func namesOracle(names []string, want string) bool {
	for _, n := range names {
		if n == want {
			return true
		}
	}
	return false
}

// Replay reproduces a recorded failure from its bundle and nothing else (#4515).
// It re-runs the embedded case through the bundle's own captured reference and
// engine traces, using the same RunCase orchestrator that emitted the bundle, and
// compares the failure it observes against the failure the bundle recorded. It
// never returns an error: every outcome — replay-incomplete, replayed-clean,
// replayed-different, reproduced — is a verdict a gate can route on, and only
// reproduction is green.
func Replay(b FailureBundle) ReplayVerdict {
	v := ReplayVerdict{Schema: ReplaySchema, CaseID: b.CaseID, Expected: signatureOf(b)}
	if err := b.ReplayComplete(); err != nil {
		v.Inconclusive = true
		v.Reason = "bundle is not replay-complete: " + err.Error()
		return v
	}
	oracles, err := Lookup(b.Case.Oracles)
	if err != nil {
		v.Inconclusive = true
		v.Reason = "embedded case names an unrunnable oracle: " + err.Error()
		return v
	}
	// Both sides replay from the traces the bundle CAPTURED, stamped with the
	// runner names that produced them: the reference is re-read from the bundle
	// rather than re-derived from the case, so a bundle whose two copies of the
	// reference disagree replays the one that actually ran.
	ref := ScriptedRunner{Label: runnerLabel(b.Reference.Runner, "reference"), Trace: b.Reference}
	eng := ScriptedRunner{Label: runnerLabel(b.Engine.Runner, "engine"), Trace: b.Engine}
	res, err := RunCase(b.Case, ref, eng, oracles)
	if err != nil {
		v.Inconclusive = true
		v.Reason = "replaying the embedded case failed: " + err.Error()
		return v
	}
	v.Result = &res
	if res.Pass || res.FailureBundle == nil {
		v.Reason = fmt.Sprintf("replay of the bundle's own evidence PASSED; the recorded failure — %s — did not reproduce", v.Expected)
		return v
	}
	obs := signatureOf(*res.FailureBundle)
	v.Observed = &obs
	if !obs.equal(v.Expected) {
		v.Reason = fmt.Sprintf("replayed to a different failure: bundle recorded %s, replay observed %s", v.Expected, obs)
		return v
	}
	v.Reproduced = true
	v.Reason = fmt.Sprintf("reproduced from the bundle alone: %s", obs)
	return v
}

func runnerLabel(recorded, fallback string) string {
	if s := strings.TrimSpace(recorded); s != "" {
		return s
	}
	return fallback
}

// LoadBundle decodes a failure bundle from JSON. It accepts either a bare bundle
// or the full Result that `fak quality run --json` emits, so the artifact CI
// already stores replays without being unwrapped by hand. Unknown fields and
// trailing documents are refused: a bundle from a schema the reader does not
// understand must not be replayed as if it were understood.
func LoadBundle(data []byte) (FailureBundle, error) {
	var probe struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return FailureBundle{}, fmt.Errorf("decode failure bundle: %w", err)
	}
	if probe.Schema == ResultSchema {
		var r Result
		if err := strictDecodeJSON(data, &r); err != nil {
			return FailureBundle{}, fmt.Errorf("decode quality result: %w", err)
		}
		if r.FailureBundle == nil {
			return FailureBundle{}, fmt.Errorf("result for case %q recorded a pass and carries no failure bundle to replay", r.CaseID)
		}
		return *r.FailureBundle, nil
	}
	var b FailureBundle
	if err := strictDecodeJSON(data, &b); err != nil {
		return FailureBundle{}, fmt.Errorf("decode failure bundle: %w", err)
	}
	return b, nil
}

func strictDecodeJSON(data []byte, into any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(into); err != nil {
		return err
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return fmt.Errorf("trailing document")
		}
		return err
	}
	return nil
}

// ExplainReplay renders a replay verdict as the human half of the replay verb:
// the state, the failure the bundle claimed, the failure the replay observed, and
// — when the replay ran — the full first-failure localization of the replayed
// run, so an operator reads the same explanation they would have read at the
// original failure site.
func ExplainReplay(v ReplayVerdict) string {
	var b strings.Builder
	state := "NOT-REPRODUCED"
	switch {
	case v.Reproduced:
		state = "REPRODUCED"
	case v.Inconclusive:
		state = "INCONCLUSIVE"
	}
	fmt.Fprintf(&b, "%s  case %s\n", state, v.CaseID)
	fmt.Fprintf(&b, "  bundle recorded: %s\n", v.Expected)
	if v.Observed != nil {
		fmt.Fprintf(&b, "  replay observed: %s\n", *v.Observed)
	}
	fmt.Fprintf(&b, "  %s\n", v.Reason)
	if v.Result != nil {
		fmt.Fprint(&b, Explain(*v.Result))
	}
	return b.String()
}
