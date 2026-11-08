package optimize

// Schemas bind each emitted artifact to its stable shape. Consumers switch on
// the schema tag before trusting any field; a body without it is foreign.
const (
	PlanSchema    = "fak-optimize-plan/v1"
	ReceiptSchema = "fak-optimize-receipt/v1"
	VerdictSchema = "fak-optimize-verdict/v1"
)

// OptimizeRefusal classes are the closed vocabulary every typed `fak optimize`
// OptimizeRefusal carries. Tests and CI consumers switch on Class, never on prose.
const (
	ClassNoReadyLever            = "no_ready_lever"
	ClassUnknownLever            = "unknown_lever"
	ClassUnknownEnvelope         = "unknown_envelope"
	ClassTraceUnreadable         = "trace_unreadable"
	ClassNoArmMapping            = "no_arm_mapping"
	ClassInvalidInput            = "invalid_input"
	ClassNonPositiveIterations   = "non_positive_iterations"
	ClassNoPlanningBand          = "no_planning_band"
	ClassCriterionUnresolved     = "criterion_unresolved"
	ClassReceiptLeverMismatch    = "receipt_lever_mismatch"
	ClassReceiptEnvelopeMismatch = "receipt_envelope_mismatch"
)

// OptimizeRefusal is one typed, structured refusal: a closed-vocabulary class plus a
// human detail. It implements error so the leaves return it through their
// normal error paths and the CLI maps it to the sub-verb's OptimizeRefusal exit code.
type OptimizeRefusal struct {
	Class  string `json:"class"`
	Detail string `json:"detail,omitempty"`
}

// NewOptimizeRefusal builds a typed OptimizeRefusal in class + detail form.
func NewOptimizeRefusal(class, detail string) *OptimizeRefusal {
	return &OptimizeRefusal{Class: class, Detail: detail}
}

func (r *OptimizeRefusal) Error() string {
	if r.Detail == "" {
		return "optimize: " + r.Class
	}
	return "optimize: " + r.Class + ": " + r.Detail
}

// MetricDecodeTokS is the ledger metric axis every loop line reports on.
const MetricDecodeTokS = "decode_tok_s"

// OptimizePlan verdicts and quality states are closed vocabularies.
const (
	VerdictAccept = "accept"
	VerdictReject = "reject"

	QualityPass   = "pass"
	QualityFail   = "fail"
	QualityAbsent = "absent"
)

// Ledger verdicts extend the accept/reject pair with the loop-only outcomes:
// planned (dry run stopped after OptimizePlan), refused (a typed OptimizeRefusal ended the
// iteration).
const (
	LedgerPlanned = "planned"
	LedgerRefused = "refused"
)

// OptimizePlan is the read-only output of `fak optimize plan`: the next dependency-ready
// unwitnessed lever plus the exact follow-on invocations. ExpectedFloor/Roof are
// the lever's planning band projected from the graph rung planningRungs binds it
// to; nil means the lever plans without a registered band (hardware levers whose
// gain is not yet banded in the committed graph).
type OptimizePlan struct {
	Schema          string   `json:"schema"`
	LeverID         string   `json:"lever_id"`
	Envelope        string   `json:"envelope"`
	ExpectedFloor   *float64 `json:"expected_floor,omitempty"`
	ExpectedRoof    *float64 `json:"expected_roof,omitempty"`
	DependencyReady bool     `json:"dependency_ready"`
	MeasureCmd      string   `json:"measure_cmd"`
	VerifyCmd       string   `json:"verify_cmd"`
	GeneratedAt     string   `json:"generated_at"`
}

// OptimizeReceipt is the measured output of `fak optimize measure`: either one
// frozen-trace A/B sweep of the lever's mapped ablate arm (BaselineTokS is the
// all-off arm, CandidateTokS the single-lever arm, Reps the trace replays per
// arm) or the fold of a nativeperf baseline/candidate receipt pair captured on
// real hardware (the means over each arm's repetitions). Every number comes
// from bench counters over a real replay or from a decoded native receipt —
// nothing here is fabricated, and an unmeasurable or missing arm refuses
// instead of reporting zero. The revision/artifact/forward-path fields are
// populated only by the --receipt path and stay out of the encoded JSON for
// trace sweeps, keeping the emitted keys byte-stable.
type OptimizeReceipt struct {
	Schema         string  `json:"schema"`
	LeverID        string  `json:"lever_id"`
	EngineID       string  `json:"engine_id"`
	TracePath      string  `json:"trace_path"`
	TraceSHA256    string  `json:"trace_sha256"`
	BaselineTokS   float64 `json:"baseline_tok_s"`
	CandidateTokS  float64 `json:"candidate_tok_s"`
	DeltaPct       float64 `json:"delta_pct"`
	Reps           int     `json:"reps"`
	MeasuredAt     string  `json:"measured_at"`
	Revision       string  `json:"revision,omitempty"`
	ArtifactSHA256 string  `json:"artifact_sha256,omitempty"`
	ForwardPath    string  `json:"forward_path,omitempty"`
}

// OptimizeVerdict is the output of `fak optimize verify`: the folded accept/reject
// decision, the delta it was made on, the closed reasons, and the quality gate
// state (pass | fail | absent).
type OptimizeVerdict struct {
	Schema   string   `json:"schema"`
	Verdict  string   `json:"verdict"`
	DeltaPct float64  `json:"delta_pct"`
	Reasons  []string `json:"reasons"`
	Quality  string   `json:"quality"`
}

// OptimizeLedgerLine is one JSONL row of `fak optimize loop`. The throughput fields are
// pointers so a refused or planned iteration omits them instead of printing a
// fabricated zero.
type OptimizeLedgerLine struct {
	Iter      int      `json:"iter"`
	LeverID   string   `json:"lever_id"`
	Metric    string   `json:"metric"`
	Baseline  *float64 `json:"baseline,omitempty"`
	Candidate *float64 `json:"candidate,omitempty"`
	DeltaPct  *float64 `json:"delta_pct,omitempty"`
	Verdict   string   `json:"verdict"`
	Refusal   string   `json:"refusal,omitempty"`
	Date      string   `json:"date"`
}
