// Package eveparity is the CI-runnable witness for issue #2605: it runs a fixture
// Eve-shaped eval suite once "raw" (against a fixture model directly) and once
// "fak-routed" (the same suite with every model call flowing through fak's real
// gateway proxy), then proves the two arms agree — and, crucially, that fak never
// silently downgrades a hard Eve gate FAILURE into a soft observation.
//
// Why a package and not a shell script: the regression #2605 exists to catch lives
// in the ADAPTER that folds an eval result into fak's artifact, not in the transport
// (fak's proxy transparency already has extensive gateway tests). The dangerous move
// is reclassifying a hard gate ("t.calledTool must fire", "content must equal X")
// into a soft, non-failing score. So the load-bearing code here is Evaluate (which
// keeps the hard/soft distinction) and Compare (which refuses to let a raw hard-gate
// FAIL become a fak pass/soft) — both pure, deterministic, and unit-testable.
//
// Honesty boundary (read before quoting a number): this harness models the eval
// SEMANTICS the Eve docs specify (t.succeeded, t.calledTool, an exact content gate,
// and --strict soft thresholds) with a deterministic in-repo evaluator. It is NOT the
// upstream `eve` npm CLI — that arm is host-gated (needs `eve` installed) and is named
// as the residual in docs/benchmarks/EVE-EVAL-PARITY-RUNBOOK.md. What this proves is
// the fak-mediation invariant: routing the identical fixture suite through fak's
// gateway preserves every gate verdict, reason, and strict threshold byte-for-byte.
package eveparity

import "fmt"

// WitnessSchema is the stable id of the parity witness artifact. It matches the schema
// the runbook (docs/benchmarks/EVE-EVAL-PARITY-RUNBOOK.md) proposed for #2605.
const WitnessSchema = "fak.eve-eval-parity-contract.v1"

// Issue is the GitHub issue this witness answers.
const Issue = 2605

// CheckKind is the Eve distinction fak must never blur: a HardGate is pass/fail (a
// gate the case MUST clear), a SoftScore is a 0..1 quality number that only fails a
// case when --strict is set and the score is below the case's threshold. Downgrading
// a HardGate to a SoftScore is the exact silent-pass #2605 guards against.
type CheckKind int

const (
	// HardGate is a mandatory pass/fail assertion (t.succeeded, t.calledTool, an exact
	// content match). A failed hard gate fails the case regardless of --strict.
	HardGate CheckKind = iota
	// SoftScore is a 0..1 quality score. It fails the case only under --strict, and
	// only when Score < Threshold; without --strict it is reported but never fails.
	SoftScore
)

// String converts the CheckKind enum value into its canonical JSON tag representation.
func (k CheckKind) String() string {
	if k == SoftScore {
		return "soft_score"
	}
	return "hard_gate"
}

// MarshalJSON emits the kind as a self-documenting string ("hard_gate"/"soft_score")
// so the witness artifact reads legibly — the whole point of the golden is to show, at
// a glance, that the deliberately-failing gate stayed a HARD gate.
func (k CheckKind) MarshalJSON() ([]byte, error) {
	return []byte(`"` + k.String() + `"`), nil
}

// UnmarshalJSON accepts the string form (and, for forward-compat, the legacy integer).
func (k *CheckKind) UnmarshalJSON(b []byte) error {
	switch string(b) {
	case `"soft_score"`, "1":
		*k = SoftScore
	default:
		*k = HardGate
	}
	return nil
}

// SoftSpec declares a case's single soft score. Score is the deterministic fixture
// value the scorer would return (no model sampling), Threshold is the --strict cutoff.
//
// Precondition: Threshold must be defined within the inclusive normalized range of zero to one.
type SoftSpec struct {
	Name      string  `json:"name"`
	Score     float64 `json:"score"`
	Threshold float64 `json:"threshold"`
}

// Case is one fixture eval task and the checks Eve would apply to it. An empty
// ExpectToolCall / ExpectContent means that gate is not part of the case.
type Case struct {
	ID     string   `json:"id"`
	Prompt string   `json:"prompt"`
	Tools  []string `json:"tools,omitempty"`
	// ExpectSucceeded is the t.succeeded hard gate: the agent turn must complete
	// without a transport/model error.
	ExpectSucceeded bool `json:"expect_succeeded"`
	// ExpectToolCall is the t.calledTool hard gate: the agent must propose this tool
	// at least once. Empty disables the gate.
	ExpectToolCall string `json:"expect_tool_call,omitempty"`
	// ExpectContent is the deterministic content hard gate: the agent's final text
	// must equal this exactly. Empty disables the gate.
	ExpectContent string `json:"expect_content,omitempty"`
	// Soft is the optional soft score (SoftScore kind). nil disables it.
	Soft *SoftSpec `json:"soft,omitempty"`
}

// Suite represents an ordered collection of eval test cases validated across execution arms.
type Suite struct {
	Name  string `json:"name"`
	Cases []Case `json:"cases"`
}

// Transcript is what ONE arm produced for ONE case by driving the model. Both the
// raw arm and the fak-routed arm produce transcripts of this identical shape; the
// evaluator is blind to which arm produced it, so any verdict difference is caused
// by fak's mediation, never by a different scorer.
type Transcript struct {
	CaseID    string   `json:"case_id"`
	SessionID string   `json:"session_id"` // the eval session/response id (#2605: must be preserved)
	FinalText string   `json:"final_text"`
	ToolCalls []string `json:"tool_calls"`
	Succeeded bool     `json:"succeeded"`
	// PromptTokens/CompletionTokens are the timing/token metadata #2605 requires the
	// fak artifact to preserve rather than drop.
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
	Err              string `json:"err,omitempty"`
}

// CheckResult is one applied check's outcome. Reason is the Eve failure reason,
// preserved verbatim so the fak artifact can be proven byte-identical to the raw one.
type CheckResult struct {
	Name   string    `json:"name"`
	Kind   CheckKind `json:"kind"`
	Passed bool      `json:"passed"`
	Score  float64   `json:"score,omitempty"`
	Reason string    `json:"reason,omitempty"`
}

// CaseOutcome is a case's folded verdict: the ordered checks, whether the case passed,
// and (on failure) the first failing check's reason — the string Eve surfaces.
type CaseOutcome struct {
	CaseID     string        `json:"case_id"`
	Passed     bool          `json:"passed"`
	FailReason string        `json:"fail_reason,omitempty"`
	FailKind   CheckKind     `json:"fail_kind,omitempty"`
	Checks     []CheckResult `json:"checks"`
}

// Evaluate applies a case's checks to a transcript under the given strict flag and
// folds them into a CaseOutcome. The check order is fixed (succeeded, calledTool,
// content, soft) so the outcome is deterministic. A case passes iff every hard gate
// passes AND, under strict, every soft score is at least its threshold. This is the
// single scorer both arms share — the hard/soft distinction lives here and nowhere
// else, so it cannot be silently blurred per-arm.
//
// Invariant: Evaluation ordering is strictly monotonic across gates to prevent unhandled early exits.
// Postcondition: CaseOutcome records deterministic gate verdicts without mutating the supplied transcript.
func Evaluate(c Case, tr Transcript, strict bool) CaseOutcome {
	out := CaseOutcome{CaseID: c.ID, Passed: true}
	record := func(cr CheckResult) {
		out.Checks = append(out.Checks, cr)
		// The first FAILING check decides the case reason (Eve reports the first gate
		// failure). A soft score below threshold only fails the case under --strict.
		failsCase := !cr.Passed && (cr.Kind == HardGate || strict)
		if failsCase && out.Passed {
			out.Passed = false
			out.FailReason = cr.Reason
			out.FailKind = cr.Kind
		}
	}

	if c.ExpectSucceeded {
		ok := tr.Succeeded && tr.Err == ""
		reason := ""
		if !ok {
			reason = "t.succeeded: agent did not complete"
			if tr.Err != "" {
				reason = "t.succeeded: " + tr.Err
			}
		}
		record(CheckResult{Name: "t.succeeded", Kind: HardGate, Passed: ok, Reason: reason})
	}
	if c.ExpectToolCall != "" {
		ok := containsStr(tr.ToolCalls, c.ExpectToolCall)
		reason := ""
		if !ok {
			reason = fmt.Sprintf("t.calledTool(%q): tool not called", c.ExpectToolCall)
		}
		record(CheckResult{Name: fmt.Sprintf("t.calledTool(%q)", c.ExpectToolCall), Kind: HardGate, Passed: ok, Reason: reason})
	}
	if c.ExpectContent != "" {
		ok := tr.FinalText == c.ExpectContent
		reason := ""
		if !ok {
			reason = fmt.Sprintf("content: got %q, want %q", tr.FinalText, c.ExpectContent)
		}
		record(CheckResult{Name: "content", Kind: HardGate, Passed: ok, Reason: reason})
	}
	if c.Soft != nil {
		ok := c.Soft.Score >= c.Soft.Threshold
		reason := ""
		if !ok {
			reason = fmt.Sprintf("%s: score %.2f below --strict threshold %.2f", c.Soft.Name, c.Soft.Score, c.Soft.Threshold)
		}
		record(CheckResult{Name: c.Soft.Name, Kind: SoftScore, Passed: ok, Score: c.Soft.Score, Reason: reason})
	}
	return out
}

// ArmResult is a whole arm's run: which arm, the command line that produced it, and
// the per-case outcomes.
type ArmResult struct {
	Arm     string        `json:"arm"`
	Command string        `json:"command"`
	Cases   []CaseOutcome `json:"cases"`
	// SessionIDsPresent / TokenMetadataPresent record that this arm's transcripts
	// carried the session ids and timing/token metadata #2605 requires be preserved.
	SessionIDsPresent    bool `json:"session_ids_present"`
	TokenMetadataPresent bool `json:"token_metadata_present"`
}

// CaseParity is the per-case comparison between the two arms.
type CaseParity struct {
	CaseID      string    `json:"case_id"`
	RawPassed   bool      `json:"raw_passed"`
	FakPassed   bool      `json:"fak_passed"`
	RawReason   string    `json:"raw_fail_reason,omitempty"`
	FakReason   string    `json:"fak_fail_reason,omitempty"`
	RawFailKind CheckKind `json:"raw_fail_kind,omitempty"`
	FakFailKind CheckKind `json:"fak_fail_kind,omitempty"`
	// Agree is true when both arms reached the same pass/fail with the same reason.
	Agree bool `json:"agree"`
	// Downgraded is the #2605 regression bit: the raw arm FAILED a HARD gate but the
	// fak arm either passed the case or reclassified the failing check as a soft score.
	// A parity witness with any Downgraded case is a divergence, never a pass.
	Downgraded bool `json:"downgraded"`
}

// Witness is the artifact #2605 asks the adapter to write: both arms' results, the
// per-case parity, and the top-line verdicts. Harness names the honesty boundary.
type Witness struct {
	Schema                   string       `json:"schema"`
	Issue                    int          `json:"issue"`
	Suite                    string       `json:"suite"`
	Strict                   bool         `json:"strict"`
	Raw                      ArmResult    `json:"raw"`
	Fak                      ArmResult    `json:"fak"`
	Cases                    []CaseParity `json:"cases"`
	ParityVerdict            string       `json:"parity_verdict"` // "pass" | "divergent"
	GateFailurePreserved     bool         `json:"gate_failure_preserved"`
	StrictThresholdPreserved bool         `json:"strict_threshold_preserved"`
	SessionIDsPreserved      bool         `json:"session_ids_preserved"`
	TokenMetadataPreserved   bool         `json:"token_metadata_preserved"`
	Harness                  string       `json:"harness"`
}

// Compare folds a raw arm and a fak-routed arm into the parity witness. It is the
// no-downgrade gate: parity is "pass" ONLY when every case agrees on pass/fail with a
// byte-identical reason AND no hard-gate failure was downgraded. Any disagreement or
// downgrade yields "divergent" — a fak result that turned a hard gate FAIL into a pass
// or a soft score is caught here, not waved through.
//
// Precondition: Both raw and fak arm results must be evaluated under the same strictness configuration.
// Postcondition: Returns a pass verdict only when every case agrees and no hard gate was downgraded.
func Compare(suite string, strict bool, raw, fak ArmResult) Witness {
	w := Witness{
		Schema: WitnessSchema, Issue: Issue, Suite: suite, Strict: strict,
		Raw: raw, Fak: fak,
		ParityVerdict:            "pass",
		GateFailurePreserved:     true,
		StrictThresholdPreserved: true,
		SessionIDsPreserved:      raw.SessionIDsPresent && fak.SessionIDsPresent,
		TokenMetadataPreserved:   raw.TokenMetadataPresent && fak.TokenMetadataPresent,
		Harness:                  "in-repo eve-semantics evaluator; fak arm routed through internal/gateway proxy; upstream `eve` CLI arm host-gated (see EVE-EVAL-PARITY-RUNBOOK.md)",
	}
	rawByID := indexByCase(raw.Cases)
	fakByID := indexByCase(fak.Cases)
	for _, rc := range raw.Cases {
		fc, ok := fakByID[rc.CaseID]
		cp := CaseParity{
			CaseID:      rc.CaseID,
			RawPassed:   rc.Passed,
			RawReason:   rc.FailReason,
			RawFailKind: rc.FailKind,
		}
		if !ok {
			// A case the fak arm dropped entirely is a divergence.
			cp.Agree = false
			w.ParityVerdict = "divergent"
			w.Cases = append(w.Cases, cp)
			continue
		}
		cp.FakPassed = fc.Passed
		cp.FakReason = fc.FailReason
		cp.FakFailKind = fc.FailKind
		cp.Agree = rc.Passed == fc.Passed && rc.FailReason == fc.FailReason

		// The downgrade check: the raw arm failed a HARD gate, but the fak arm either
		// passed the case or reports the failure under a soft-score kind.
		rawHardFail := !rc.Passed && rc.FailKind == HardGate
		if rawHardFail && (fc.Passed || fc.FailKind == SoftScore) {
			cp.Downgraded = true
			w.GateFailurePreserved = false
		}
		// A hard-gate failure whose reason string changed is also not "preserved".
		if rawHardFail && fc.FailReason != rc.FailReason {
			w.GateFailurePreserved = false
		}
		if !cp.Agree || cp.Downgraded {
			w.ParityVerdict = "divergent"
		}
		// Strict threshold preservation: a soft-score case must reach the same verdict
		// on both arms (the strict cutoff applied identically).
		if isSoftCase(rc) && rc.Passed != fc.Passed {
			w.StrictThresholdPreserved = false
		}
		w.Cases = append(w.Cases, cp)
	}
	// Any fak case with no raw counterpart is also a divergence.
	for _, fc := range fak.Cases {
		if _, ok := rawByID[fc.CaseID]; !ok {
			w.ParityVerdict = "divergent"
			w.Cases = append(w.Cases, CaseParity{CaseID: fc.CaseID, FakPassed: fc.Passed, FakReason: fc.FailReason})
		}
	}
	return w
}

func indexByCase(cs []CaseOutcome) map[string]CaseOutcome {
	m := make(map[string]CaseOutcome, len(cs))
	for _, c := range cs {
		m[c.CaseID] = c
	}
	return m
}

func isSoftCase(c CaseOutcome) bool {
	for _, ck := range c.Checks {
		if ck.Kind == SoftScore {
			return true
		}
	}
	return false
}

func containsStr(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}
