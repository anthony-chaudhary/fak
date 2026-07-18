package shipgate

// crossaudit_closure.go — issue #3860 (epic #3846): require an INDEPENDENT
// cross-model audit receipt before a HIGH-RISK issue closure is allowed.
//
// The two existing halves of shipgate decide keep-or-revert for one RSI candidate
// (shipgate.go) and lift a ship-shaped tool call to RequireWitness (adjudicate.go).
// This file is the third: a pure, fail-closed policy that a closure driver consults
// before it lets a HIGH-RISK issue closure land. It never takes a model's word — the
// receipt it inspects is a projection of a hash-verified modelroute.IssueAuditReceipt
// (built + chained + verified in internal/modelroute), and this policy checks that the
// receipt is a PASS, bound to the exact subject, produced by a CALIBRATED and
// INDEPENDENT auditor, at the current calibration version, and still FRESH — or it
// blocks the closure.
//
// WHY A VIEW, NOT THE RECEIPT ITSELF. shipgate is a stdlib-only decision layer (like
// canary.go); it deliberately does not import the receipt/ledger crypto. The closure
// driver (cmd/fak, out of this leaf's lane) maps an already-verified
// modelroute.IssueAuditReceipt into AuditReceiptView and hands it here. The verdict
// token vocabulary (PASS/REFUTE/INCONCLUSIVE/UNAVAILABLE) is kept identical to
// modelroute.CrossAuditVerdict so the two agree by construction.
//
// STRUCTURAL-DENY PRECEDENCE. A calibrated independent PASS can NEVER flip a
// structural deny (a red test suite or a DOS/structural refusal) to allowed — the
// model is not permitted to override structure. Structural deny is evaluated first and
// break-glass cannot override it either.
//
// STAGED ENABLEMENT. Enforcement fails closed only once its prerequisites are met:
// enough INDEPENDENT calibrated auditors (#3854) AND a green dogfood rollout (#3859).
// Until then the gate runs in DRY-RUN — it computes what it WOULD do and reports it,
// but never blocks a closure. That is the encoded form of the issue's out-of-scope
// rule "no enablement when calibration/dogfood prerequisites are not met": it is a
// property of the type, not a comment. The committed adoption report
// (CrossAuditAdoptionReport / testdata) records the measured prerequisite evidence and
// the resulting recommended stage.

import "strings"

// ClosureRisk classifies an issue closure. Only RiskHigh closures are gated; the
// ordinary RiskLow path is unchanged (no new gate for day-to-day work). The zero
// value is RiskLow so an unclassified closure is never accidentally forced through the
// high-risk gate — but it is also never forced to BLOCK, since the low-risk path
// allows.
type ClosureRisk uint8

const (
	RiskLow  ClosureRisk = iota // ordinary work — the audit gate is a no-op
	RiskHigh                    // trust-floor / security / CI-spec change — gated
)

// String renders the risk class as a stable token.
func (r ClosureRisk) String() string {
	if r == RiskHigh {
		return "high"
	}
	return "low"
}

// AuditVerdictToken mirrors modelroute.CrossAuditVerdict's closed vocabulary. Only the
// literal PASS opens the gate; REFUTE, INCONCLUSIVE, and UNAVAILABLE all block. The
// zero value is the empty token, which is never PASS — a missing verdict fails closed.
type AuditVerdictToken string

const (
	AuditPass         AuditVerdictToken = "PASS"
	AuditRefute       AuditVerdictToken = "REFUTE"
	AuditInconclusive AuditVerdictToken = "INCONCLUSIVE"
	AuditUnavailable  AuditVerdictToken = "UNAVAILABLE"
)

// AuditReceiptView is the minimal, transport-agnostic projection of a verified
// cross-audit receipt the closure policy needs. The closure driver builds it from a
// modelroute.IssueAuditReceipt that has ALREADY been hash-verified against its ledger;
// this policy never re-implements the receipt schema or its chain crypto. Present=false
// models "no admitted independent receipt at all".
type AuditReceiptView struct {
	Present             bool              `json:"present"`
	SubjectDigest       string            `json:"subject_digest"`         // digest of the audited subject (must match the closure subject)
	Verdict             AuditVerdictToken `json:"verdict"`                // PASS | REFUTE | INCONCLUSIVE | UNAVAILABLE
	AuditorFamily       string            `json:"auditor_family"`         // model family that audited (independence + allowlist key)
	AuthorFamily        string            `json:"author_family"`          // model family that authored the closure (independence check)
	CalibrationVersion  string            `json:"calibration_version"`    // policy/prompt version the auditor was calibrated at
	CompletedAtUnixNano int64             `json:"completed_at_unix_nano"` // receipt completion time (freshness)
}

// BreakGlass is an explicit, time-bound, AUDITED emergency override. It can waive the
// audit-RECEIPT requirement only — it can never override a structural deny. A blank
// operator/reason/ledger reference or an expired window makes it invalid, and an
// invalid break-glass never overrides anything.
type BreakGlass struct {
	Operator       string `json:"operator"`             // who authorized the override
	Reason         string `json:"reason"`               // why (recorded, not free-passed)
	LedgerRef      string `json:"ledger_ref"`           // reference proving it was written to the audited override ledger
	ExpiresAtUnixN int64  `json:"expires_at_unix_nano"` // override is invalid at/after this instant
}

// valid reports whether a break-glass entry is explicit, audited, and unexpired.
func (b *BreakGlass) valid(nowUnixNano int64) bool {
	if b == nil {
		return false
	}
	return strings.TrimSpace(b.Operator) != "" &&
		strings.TrimSpace(b.Reason) != "" &&
		strings.TrimSpace(b.LedgerRef) != "" &&
		b.ExpiresAtUnixN > 0 && nowUnixNano < b.ExpiresAtUnixN
}

// Prerequisites gates default-on enforcement. Enforcement fires only when there are at
// least MinIndependent (>= 2) calibrated INDEPENDENT auditor families (#3854) AND the
// dogfood rollout is green (#3859). MinIndependent < 2 can never satisfy the gate —
// cross-model independence is meaningless with a single family.
type Prerequisites struct {
	CalibratedAuditorFamilies int  `json:"calibrated_auditor_families"` // count of distinct calibrated families (#3854)
	MinIndependent            int  `json:"min_independent"`             // required minimum (>= 2)
	DogfoodGreen              bool `json:"dogfood_green"`               // #3859 rollout report green (live loop not dark)
}

// Met reports whether the calibration + dogfood prerequisites for default-on
// enforcement are satisfied.
func (p Prerequisites) Met() bool {
	return p.MinIndependent >= 2 &&
		p.CalibratedAuditorFamilies >= p.MinIndependent &&
		p.DogfoodGreen
}

// CrossAuditPolicy is the calibrated enforcement policy: which auditor families are
// admitted, at what calibration version, how fresh a receipt must be, and whether the
// staged-enablement prerequisites are met. It is data, not a hard-coded provider
// monopoly — a policy with a two-family allowlist admits either family.
type CrossAuditPolicy struct {
	RequiredCalibrationVersion string        `json:"required_calibration_version"`
	CalibratedAuditorFamilies  []string      `json:"calibrated_auditor_families"` // allowlist (from #3854)
	MaxReceiptAgeNanos         int64         `json:"max_receipt_age_nanos"`       // freshness window (0 = no freshness bound)
	Prereqs                    Prerequisites `json:"prereqs"`
}

// calibrated reports whether family is on the policy's calibrated allowlist
// (case-insensitive; a blank family is never calibrated).
func (pol CrossAuditPolicy) calibrated(family string) bool {
	family = strings.ToLower(strings.TrimSpace(family))
	if family == "" {
		return false
	}
	for _, f := range pol.CalibratedAuditorFamilies {
		if strings.ToLower(strings.TrimSpace(f)) == family {
			return true
		}
	}
	return false
}

// ClosureReason is the closed-vocabulary reason a closure was allowed or blocked —
// never free text, so a driver (and the tests) can match on it exactly.
type ClosureReason string

const (
	ReasonLowRiskExempt       ClosureReason = "LOW_RISK_EXEMPT"               // ordinary work — gate is a no-op
	ReasonStructuralDeny      ClosureReason = "STRUCTURAL_DENY"               // tests/DOS red — precedence over any PASS or break-glass
	ReasonReceiptMissing      ClosureReason = "AUDIT_RECEIPT_MISSING"         // no admitted independent receipt
	ReasonReceiptNonPass      ClosureReason = "AUDIT_RECEIPT_NONPASS"         // REFUTE / INCONCLUSIVE / UNAVAILABLE
	ReasonSubjectMismatch     ClosureReason = "AUDIT_SUBJECT_DIGEST_MISMATCH" // receipt bound to a different subject
	ReasonAuditorUncalibrated ClosureReason = "AUDITOR_UNCALIBRATED"          // auditor family not on the calibrated allowlist
	ReasonCalibrationMismatch ClosureReason = "CALIBRATION_VERSION_MISMATCH"  // stale/wrong calibration version
	ReasonNotIndependent      ClosureReason = "AUDITOR_NOT_INDEPENDENT"       // auditor family == author family
	ReasonReceiptStale        ClosureReason = "AUDIT_RECEIPT_STALE"           // receipt older than the freshness window
	ReasonAuditPass           ClosureReason = "AUDIT_PASS_INDEPENDENT"        // valid calibrated independent PASS — allowed
	ReasonBreakGlass          ClosureReason = "BREAK_GLASS_OVERRIDE"          // explicit audited override waived the receipt requirement
	ReasonPrereqsDryRun       ClosureReason = "PREREQS_UNMET_DRYRUN"          // enforcement disabled by staged-enablement — reported, not blocked
)

// ClosureDecision is the typed outcome of the high-risk closure gate.
type ClosureDecision struct {
	Issue      int           `json:"issue"`
	Risk       ClosureRisk   `json:"risk"`
	Enforced   bool          `json:"enforced"`    // false => staged-enablement dry-run (never blocks)
	Allowed    bool          `json:"allowed"`     // whether the closure may proceed
	WouldBlock bool          `json:"would_block"` // in dry-run, whether enforcement WOULD have blocked
	Reason     ClosureReason `json:"reason"`
	Detail     string        `json:"detail,omitempty"`
}

// AdjudicateClosure decides whether a HIGH-RISK issue closure may land. It fails
// closed: any missing, non-PASS, mismatched, uncalibrated, same-family, or stale
// receipt blocks a high-risk closure when enforcement is on. Structural deny takes
// precedence over everything (no model or break-glass override). The RiskLow path is
// unchanged. When the staged-enablement prerequisites are not met, the gate runs in
// DRY-RUN: it reports what enforcement WOULD do (WouldBlock) but always allows.
func AdjudicateClosure(req ClosureRequest, pol CrossAuditPolicy) ClosureDecision {
	d := ClosureDecision{Issue: req.Issue, Risk: req.Risk, Enforced: pol.Prereqs.Met()}

	// The low-risk path is a no-op: ordinary work is never newly gated.
	if req.Risk != RiskHigh {
		d.Allowed, d.Reason = true, ReasonLowRiskExempt
		return d
	}

	allowed, reason, detail := evaluateHighRisk(req, pol)

	if !d.Enforced {
		// Staged enablement: report the would-be verdict, never block.
		d.Allowed = true
		d.WouldBlock = !allowed
		d.Reason = ReasonPrereqsDryRun
		d.Detail = "enforcement disabled until calibration+dogfood prerequisites are met; would-be reason: " + string(reason)
		if detail != "" {
			d.Detail += " (" + detail + ")"
		}
		return d
	}

	d.Allowed, d.Reason, d.Detail = allowed, reason, detail
	return d
}

// evaluateHighRisk computes the enforce-mode verdict for a high-risk closure,
// independent of whether staged enablement has turned enforcement on.
func evaluateHighRisk(req ClosureRequest, pol CrossAuditPolicy) (allowed bool, reason ClosureReason, detail string) {
	// (1) Structural-deny precedence: a red suite / DOS refusal blocks unconditionally.
	// A calibrated independent PASS cannot flip it, and break-glass cannot override it.
	if req.StructuralDeny {
		return false, ReasonStructuralDeny, "a failing test suite or DOS/structural refusal blocks closure regardless of any audit receipt"
	}

	// (2) The receipt rungs. A break-glass can waive these — but only these.
	valid, rr, rd := receiptAdmits(req, pol)
	if valid {
		return true, ReasonAuditPass, ""
	}
	if req.BreakGlass.valid(req.NowUnixNano) {
		return true, ReasonBreakGlass, "operator=" + strings.TrimSpace(req.BreakGlass.Operator) +
			" ledger=" + strings.TrimSpace(req.BreakGlass.LedgerRef) + " waived: " + string(rr)
	}
	return false, rr, rd
}

// receiptAdmits reports whether the receipt is a valid, calibrated, independent,
// fresh PASS bound to the closure subject. On failure it returns the specific
// closed-vocabulary reason, checked in a fixed order so the verdict is deterministic.
func receiptAdmits(req ClosureRequest, pol CrossAuditPolicy) (ok bool, reason ClosureReason, detail string) {
	r := req.Receipt
	if !r.Present {
		return false, ReasonReceiptMissing, "no admitted independent audit receipt for a high-risk closure"
	}
	if r.Verdict != AuditPass {
		return false, ReasonReceiptNonPass, "audit verdict is " + string(r.Verdict) + ", not PASS"
	}
	if strings.TrimSpace(r.SubjectDigest) == "" || r.SubjectDigest != req.SubjectDigest {
		return false, ReasonSubjectMismatch, "receipt subject digest is not bound to this closure subject"
	}
	if !pol.calibrated(r.AuditorFamily) {
		return false, ReasonAuditorUncalibrated, "auditor family " + displayFamily(r.AuditorFamily) + " is not on the calibrated allowlist"
	}
	if strings.TrimSpace(r.CalibrationVersion) != strings.TrimSpace(pol.RequiredCalibrationVersion) {
		return false, ReasonCalibrationMismatch, "receipt calibration " + displayFamily(r.CalibrationVersion) +
			" != required " + displayFamily(pol.RequiredCalibrationVersion)
	}
	if sameFamily(r.AuditorFamily, r.AuthorFamily) {
		return false, ReasonNotIndependent, "auditor and author share family " + displayFamily(r.AuditorFamily)
	}
	if pol.MaxReceiptAgeNanos > 0 {
		age := req.NowUnixNano - r.CompletedAtUnixNano
		if age > pol.MaxReceiptAgeNanos {
			return false, ReasonReceiptStale, "receipt is older than the freshness window"
		}
	}
	return true, ReasonAuditPass, ""
}

// sameFamily reports whether two non-blank family names are equal (case-insensitive).
// Two blanks are NOT "the same family" — a missing author family is not proof of
// independence, so that case is left to the allowlist/calibration rungs.
func sameFamily(a, b string) bool {
	a = strings.ToLower(strings.TrimSpace(a))
	b = strings.ToLower(strings.TrimSpace(b))
	return a != "" && a == b
}

// displayFamily renders a possibly-blank identifier for a reason string.
func displayFamily(v string) string {
	if strings.TrimSpace(v) == "" {
		return "<none>"
	}
	return v
}

// ClosureRequest is the input to the high-risk closure gate.
type ClosureRequest struct {
	Issue          int              `json:"issue"`
	Risk           ClosureRisk      `json:"risk"`
	SubjectDigest  string           `json:"subject_digest"`  // digest of the exact closure subject the receipt must bind
	StructuralDeny bool             `json:"structural_deny"` // tests red or a DOS/structural refusal
	Receipt        AuditReceiptView `json:"receipt"`
	NowUnixNano    int64            `json:"now_unix_nano"` // "now", for the freshness window
	BreakGlass     *BreakGlass      `json:"break_glass,omitempty"`
}
