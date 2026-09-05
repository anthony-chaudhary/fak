package shipgate

import "strings"

// ClosureRisk classifies the risk tier of an issue closure.
type ClosureRisk uint8

// ClosureRisk constants define the risk tiers for issue closures.
const (
	RiskLow  ClosureRisk = iota
	RiskHigh
)

// String renders the risk classification as a stable token.
func (r ClosureRisk) String() string {
	if r == RiskHigh {
		return "high"
	}
	return "low"
}

// AuditVerdictToken mirrors the closed vocabulary for audit verdicts.
type AuditVerdictToken string

// AuditVerdictToken constants define supported verdict outcomes from audit receipts.
const (
	AuditPass         AuditVerdictToken = "PASS"
	AuditRefute       AuditVerdictToken = "REFUTE"
	AuditInconclusive AuditVerdictToken = "INCONCLUSIVE"
	AuditUnavailable  AuditVerdictToken = "UNAVAILABLE"
)

// AuditReceiptView provides the projection of a verified audit receipt needed for closure gating.
type AuditReceiptView struct {
	Present             bool              `json:"present"`
	SubjectDigest       string            `json:"subject_digest"`
	Verdict             AuditVerdictToken `json:"verdict"`
	AuditorFamily       string            `json:"auditor_family"`
	AuthorFamily        string            `json:"author_family"`
	CalibrationVersion  string            `json:"calibration_version"`
	CompletedAtUnixNano int64             `json:"completed_at_unix_nano"`
}

// BreakGlass represents an explicit, audited override for high-risk closure requirements.
type BreakGlass struct {
	Operator       string `json:"operator"`
	Reason         string `json:"reason"`
	LedgerRef      string `json:"ledger_ref"`
	ExpiresAtUnixN int64  `json:"expires_at_unix_nano"`
}

func (b *BreakGlass) valid(nowUnixNano int64) bool {
	if b == nil {
		return false
	}
	return strings.TrimSpace(b.Operator) != "" &&
		strings.TrimSpace(b.Reason) != "" &&
		strings.TrimSpace(b.LedgerRef) != "" &&
		b.ExpiresAtUnixN > 0 && nowUnixNano < b.ExpiresAtUnixN
}

// Prerequisites defines requirements for active policy enforcement.
type Prerequisites struct {
	CalibratedAuditorFamilies int  `json:"calibrated_auditor_families"`
	MinIndependent            int  `json:"min_independent"`
	DogfoodGreen              bool `json:"dogfood_green"`
}

// Met reports whether calibration and dogfood rollout requirements are satisfied.
func (p Prerequisites) Met() bool {
	return p.MinIndependent >= 2 &&
		p.CalibratedAuditorFamilies >= p.MinIndependent &&
		p.DogfoodGreen
}

// CrossAuditPolicy defines criteria for gating high-risk issue closures.
type CrossAuditPolicy struct {
	RequiredCalibrationVersion string        `json:"required_calibration_version"`
	CalibratedAuditorFamilies  []string      `json:"calibrated_auditor_families"`
	MaxReceiptAgeNanos         int64         `json:"max_receipt_age_nanos"`
	Prereqs                    Prerequisites `json:"prereqs"`
}

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

// ClosureReason specifies why a closure was allowed or blocked.
type ClosureReason string

// ClosureReason constants specify the rationale for closure outcomes.
const (
	ReasonLowRiskExempt       ClosureReason = "LOW_RISK_EXEMPT"
	ReasonStructuralDeny      ClosureReason = "STRUCTURAL_DENY"
	ReasonReceiptMissing      ClosureReason = "AUDIT_RECEIPT_MISSING"
	ReasonReceiptNonPass      ClosureReason = "AUDIT_RECEIPT_NONPASS"
	ReasonSubjectMismatch     ClosureReason = "AUDIT_SUBJECT_DIGEST_MISMATCH"
	ReasonAuditorUncalibrated ClosureReason = "AUDITOR_UNCALIBRATED"
	ReasonCalibrationMismatch ClosureReason = "CALIBRATION_VERSION_MISMATCH"
	ReasonNotIndependent      ClosureReason = "AUDITOR_NOT_INDEPENDENT"
	ReasonReceiptStale        ClosureReason = "AUDIT_RECEIPT_STALE"
	ReasonAuditPass           ClosureReason = "AUDIT_PASS_INDEPENDENT"
	ReasonBreakGlass          ClosureReason = "BREAK_GLASS_OVERRIDE"
	ReasonPrereqsDryRun       ClosureReason = "PREREQS_UNMET_DRYRUN"
)

// ClosureDecision holds the outcome and rationale of a closure evaluation.
type ClosureDecision struct {
	Issue      int           `json:"issue"`
	Risk       ClosureRisk   `json:"risk"`
	Enforced   bool          `json:"enforced"`
	Allowed    bool          `json:"allowed"`
	WouldBlock bool          `json:"would_block"`
	Reason     ClosureReason `json:"reason"`
	Detail     string        `json:"detail,omitempty"`
}

// AdjudicateClosure determines whether an issue closure request satisfies risk policy.
func AdjudicateClosure(req ClosureRequest, pol CrossAuditPolicy) ClosureDecision {
	d := ClosureDecision{Issue: req.Issue, Risk: req.Risk, Enforced: pol.Prereqs.Met()}

	if req.Risk != RiskHigh {
		d.Allowed, d.Reason = true, ReasonLowRiskExempt
		return d
	}

	allowed, reason, detail := evaluateHighRisk(req, pol)

	if !d.Enforced {
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

func evaluateHighRisk(req ClosureRequest, pol CrossAuditPolicy) (allowed bool, reason ClosureReason, detail string) {
	if req.StructuralDeny {
		return false, ReasonStructuralDeny, "a failing test suite or DOS/structural refusal blocks closure regardless of any audit receipt"
	}

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

func sameFamily(a, b string) bool {
	a = strings.ToLower(strings.TrimSpace(a))
	b = strings.ToLower(strings.TrimSpace(b))
	return a != "" && a == b
}

func displayFamily(v string) string {
	if strings.TrimSpace(v) == "" {
		return "<none>"
	}
	return v
}

// ClosureRequest contains metadata and audit evidence for an issue closure.
type ClosureRequest struct {
	Issue          int              `json:"issue"`
	Risk           ClosureRisk      `json:"risk"`
	SubjectDigest  string           `json:"subject_digest"`
	StructuralDeny bool             `json:"structural_deny"`
	Receipt        AuditReceiptView `json:"receipt"`
	NowUnixNano    int64            `json:"now_unix_nano"`
	BreakGlass     *BreakGlass      `json:"break_glass,omitempty"`
}
