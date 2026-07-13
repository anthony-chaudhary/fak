package journal

import "strings"

// Flaky-quality-eval policy — the durable witness + pure admission kernel for a
// STOCHASTIC quality case (#4569). A quality eval over a model is not deterministic
// like a kernel adjudication: the same case can flicker green/red across samples, so
// a naive "re-run until green" loop silently RETRIES A REAL FAILURE INTO A PASS — the
// exact defect lm-evaluation-harness and HELM exist to prevent. This file is the
// contract that makes that impossible and records the case so a green is reproducible.
//
// THREE PARTS, all in one leaf:
//
//   - A CLOSED classification (FlakeInfraError | FlakeQualityFailure | FlakeInconclusive
//     | FlakePass) that SEPARATES an infrastructure error from a real quality failure —
//     the #4569 scope. Only an infra error (or an evidence-gathering inconclusive) may be
//     retried; a real quality failure may NEVER be (RetryAdmit).
//   - A retry-SAFE fold (FoldAttempts) that collapses a case's ordered attempts into the
//     final verdict such that ANY real quality failure is STICKY — a later green can never
//     flip it — and missing/inconclusive evidence is FAIL-CLOSED (never pass).
//   - A durable, hash-chained QUALITY_QUARANTINE row (AppendQualityQuarantine) carrying the
//     full reproducibility provenance (model, tokenizer, engine, seed/oracle, revision,
//     tolerance/baseline), the owner/expiry/tier/cost the quarantine lease needs, the first
//     actionable divergence, and a SCRUBBED replay handle — so `fak audit verify` covers a
//     quarantine decision exactly like a per-call kernel decision.
//
// Like AppendConfigSwap / AppendLivelock, the row is written DIRECTLY through the chain
// (not the ABI fan-out): a quarantine decision is quality supervision, not a per-call
// kernel adjudication, so routing a synthetic event through the frozen ABI would fan a
// non-decision out to every decision-stream folder. Its chained forensic identity — Kind,
// the case id (Tool), the deciding authority (By), the folded verdict (Verdict) and its
// mirrored class (Reason) — rides the frozen decision fields, so it verifies end-to-end
// with every existing row and needs no format migration; the full correlated record rides
// the non-chained Quality carrier.

// KindQualityQuarantine marks a quarantined flaky-quality-eval case row. It is a genuine
// chained row (it consumes the next Seq and chains onto the prior head) that carries no
// per-call tool verdict, so a decision-folding consumer that keys on the closed verdict set
// skips its verdict accounting while the chain verifier covers it like every row.
const KindQualityQuarantine = "QUALITY_QUARANTINE"

// QualityQuarantineSchema versions the correlated per-case record carried on the row, so a
// later reader can migrate the carrier without touching the frozen chained decision fields.
const QualityQuarantineSchema = "fak.quality.quarantine.v1"

// Quality-eval tiers — the CLOSED vocabulary for WHICH gate a case runs in (#4569 acceptance
// criterion 4: assign the case to an explicit PR, nightly, or release tier). ValidTier is the
// membership check; an unknown tier is refused rather than silently accepted.
const (
	QualityTierPR      = "pr"      // runs on every PR — cheapest, tightest budget
	QualityTierNightly = "nightly" // runs nightly — larger sample budget
	QualityTierRelease = "release" // runs at release — the full, most expensive sweep
)

// Flake classes — the CLOSED vocabulary separating an INFRASTRUCTURE error from a real
// QUALITY failure (#4569 scope). The separation is load-bearing: only an infra error (the
// eval never reached a quality verdict) may be retried; a quality failure (the eval ran and
// the model's output diverged past tolerance) may NOT, because retrying it is exactly how a
// flaky real failure gets laundered into green.
const (
	FlakeInfraError     = "INFRA_ERROR"     // host/setup/network broke; NO quality verdict was reached — retriable
	FlakeQualityFailure = "QUALITY_FAILURE" // the eval ran; output diverged past tolerance — a REAL failure, NOT retriable
	FlakeInconclusive   = "INCONCLUSIVE"    // evidence missing or divergence not localizable — fail-closed, never pass
	FlakePass           = "PASS"            // the eval ran and matched the oracle/baseline within tolerance
)

// Final case verdicts — the CLOSED vocabulary FoldAttempts collapses an attempt log to.
// ERROR_NO_VERDICT is distinct from FAIL: it means the eval never produced a quality signal
// at all (every attempt was infra, or none ran, or a would-be pass lacked provenance), so it
// is reported as "could not evaluate" and NEVER as a pass — but it is not a quality failure
// either.
const (
	QualityPass           = "PASS"
	QualityFail           = "FAIL"
	QualityErrorNoVerdict = "ERROR_NO_VERDICT"
)

// Retry-admission reasons — the CLOSED vocabulary for WHY RetryAdmit admitted or refused a
// re-run. A refusal is a first-class, auditable value: the caller records the token, it never
// re-rolls on a bare boolean.
const (
	RetryAdmitInfraBudget      = "INFRA_RETRY_ADMITTED"          // infra error, within budget — re-run
	RetryAdmitGatherEvidence   = "INCONCLUSIVE_RETRY_ADMITTED"   // inconclusive, within budget — re-run to gather evidence
	RetryRefuseQualityFailure  = "QUALITY_FAILURE_NOT_RETRIABLE" // a real quality failure — the load-bearing refusal
	RetryRefuseBudgetExhausted = "RETRY_BUDGET_EXHAUSTED"        // out of retries — give up, keep the failing verdict
	RetryRefuseAlreadyPassed   = "ALREADY_PASSED"                // nothing to retry
	RetryRefuseUnclassified    = "UNCLASSIFIED_NOT_RETRIABLE"    // unknown class — fail-closed, refuse the re-roll
)

// DefaultMaxInfraRetries bounds how many times an infra-classified (or inconclusive) attempt
// may be re-run before the case is given up. It caps the retry budget so an infra flap cannot
// spin forever; a caller may pass its own bound to RetryAdmit.
const DefaultMaxInfraRetries = 3

// QualityCaseProvenance records every reproducibility axis #4569 acceptance criterion 2
// requires. A case missing any axis cannot yield a TRUSTWORTHY pass — a green with no recorded
// model / seed / revision is unreproducible — so Complete fails it closed, naming the first
// missing axis. "Seed or deterministic oracle": exactly one of Seed / Oracle is required.
type QualityCaseProvenance struct {
	Model     string `json:"model"`            // the model under test (id + revision)
	Tokenizer string `json:"tokenizer"`        // tokenizer id / revision
	Engine    string `json:"engine"`           // engine / backend (cpu-q8, cuda-fp16, ...)
	Seed      string `json:"seed,omitempty"`   // RNG seed — OR ...
	Oracle    string `json:"oracle,omitempty"` // ... a deterministic oracle id (exactly one required)
	Revision  string `json:"revision"`         // code / module revision under test (git sha)
	Tolerance string `json:"tolerance"`        // tolerance / baseline provenance (how pass/fail was bounded)
}

// Complete reports whether the case records every provenance axis a trustworthy pass needs,
// returning the first missing axis when it does not. It is the gate Decide consults before it
// will let a folded PASS stand.
func (p QualityCaseProvenance) Complete() (ok bool, missing string) {
	switch {
	case strings.TrimSpace(p.Model) == "":
		return false, "model"
	case strings.TrimSpace(p.Tokenizer) == "":
		return false, "tokenizer"
	case strings.TrimSpace(p.Engine) == "":
		return false, "engine"
	case strings.TrimSpace(p.Seed) == "" && strings.TrimSpace(p.Oracle) == "":
		return false, "seed_or_oracle"
	case strings.TrimSpace(p.Revision) == "":
		return false, "revision"
	case strings.TrimSpace(p.Tolerance) == "":
		return false, "tolerance"
	}
	return true, ""
}

// ValidTier reports whether tier is one of the closed #4569 tiers.
func ValidTier(tier string) bool {
	switch tier {
	case QualityTierPR, QualityTierNightly, QualityTierRelease:
		return true
	}
	return false
}

// QualityQuarantineRow is the correlated per-case record (QualityQuarantineSchema): the
// reproducibility provenance, the owner / expiry / tier / cost the quarantine lease tracks,
// the ordered attempt log, the folded verdict, the first actionable divergence, and a scrubbed
// replay handle — one durable value. It is journal-local (not a re-export of a harness
// envelope) so the on-disk schema stays self-contained, the same choice LivelockRow makes.
type QualityQuarantineRow struct {
	Schema          string                `json:"schema"`
	Case            string                `json:"case"`                       // stable case id
	Owner           string                `json:"owner,omitempty"`            // who owns triage (#4569 scope: track owner)
	Tier            string                `json:"tier"`                       // QualityTierPR | Nightly | Release (#4569 criterion 4)
	CostMS          int64                 `json:"cost_ms,omitempty"`          // runtime / resource cost in ms (#4569 criterion 4)
	ExpiryUnixNano  int64                 `json:"expiry_unix_nano,omitempty"` // when the quarantine lease lapses (#4569 scope: expiry)
	Provenance      QualityCaseProvenance `json:"provenance"`                 // #4569 criterion 2
	Attempts        []string              `json:"attempts,omitempty"`         // ordered Flake* classes, oldest first
	Verdict         string                `json:"verdict"`                    // QualityPass | Fail | ErrorNoVerdict (set by Decide)
	FirstDivergence string                `json:"first_divergence,omitempty"` // the first actionable divergence (#4569 criterion 3)
	ReplayArtifact  string                `json:"replay_artifact,omitempty"`  // scrubbed replay handle (#4569 criterion 3)
}

// RetryAdmit is the pure admission kernel (#4569 acceptance criterion 1): given the class of a
// just-observed attempt and how many retries the case has already spent, decide whether another
// attempt may run. A real FlakeQualityFailure is NEVER admitted — that is the "an intermittent
// real failure cannot be retried into green" invariant AT ITS SOURCE: once the eval reached a
// quality verdict and it diverged, no re-roll may overwrite it. Only an infra error (or an
// inconclusive evidence-gathering retry) is admitted, and only within maxRetries; maxRetries<=0
// means the DefaultMaxInfraRetries budget. An unknown class is fail-closed (refused). State in,
// decision out, no I/O.
func RetryAdmit(class string, priorRetries, maxRetries int) (admit bool, reason string) {
	if maxRetries <= 0 {
		maxRetries = DefaultMaxInfraRetries
	}
	switch class {
	case FlakeQualityFailure:
		return false, RetryRefuseQualityFailure
	case FlakePass:
		return false, RetryRefuseAlreadyPassed
	case FlakeInfraError:
		if priorRetries >= maxRetries {
			return false, RetryRefuseBudgetExhausted
		}
		return true, RetryAdmitInfraBudget
	case FlakeInconclusive:
		if priorRetries >= maxRetries {
			return false, RetryRefuseBudgetExhausted
		}
		return true, RetryAdmitGatherEvidence
	default:
		return false, RetryRefuseUnclassified
	}
}

// FoldAttempts collapses a case's ordered attempt classes into the final, retry-safe verdict
// (#4569 acceptance criteria 1 + 3), in this order of dominance:
//
//   - STICKY FAIL: if ANY attempt is a real FlakeQualityFailure, the verdict is FAIL — a later
//     PASS can never flip it. This is what makes a flaky real failure impossible to retry green.
//   - FAIL-CLOSED: if any attempt is FlakeInconclusive (or an unknown class), the verdict is
//     FAIL — missing / inconclusive evidence is never pass.
//   - PASS: only if a FlakePass exists AND no quality failure / inconclusive attempt does.
//   - ERROR_NO_VERDICT: every attempt was an infra error (or there were none) — the eval never
//     produced a quality signal, so the case is "could not evaluate", never a pass.
//
// Infra errors are TRANSPARENT: they neither pass nor fail, so an infra flap followed by a clean
// green ([INFRA_ERROR, PASS]) legitimately passes — the real eval only ran once, to a pass.
func FoldAttempts(attempts []string) string {
	sawPass, sawInconclusive, sawQualityVerdict := false, false, false
	for _, a := range attempts {
		switch a {
		case FlakeQualityFailure:
			return QualityFail // sticky: dominates everything, including a later pass
		case FlakePass:
			sawPass = true
			sawQualityVerdict = true
		case FlakeInconclusive:
			sawInconclusive = true
			sawQualityVerdict = true
		case FlakeInfraError:
			// transparent: contributes no quality signal
		default:
			sawInconclusive = true // unknown class is fail-closed evidence
			sawQualityVerdict = true
		}
	}
	switch {
	case sawInconclusive:
		return QualityFail // fail-closed
	case sawPass:
		return QualityPass
	case !sawQualityVerdict:
		return QualityErrorNoVerdict // all infra / empty: could not evaluate
	default:
		return QualityErrorNoVerdict
	}
}

// Decide sets the record's Verdict from its attempt log AND gates a PASS on complete provenance.
// A folded PASS with incomplete provenance is unreproducible, so it is DEMOTED to
// ERROR_NO_VERDICT (never a trustworthy pass) and the missing axis is recorded as the first
// divergence. FAIL and ERROR_NO_VERDICT pass through unchanged — an incomplete case can still be
// known-bad. Idempotent.
func (r *QualityQuarantineRow) Decide() {
	r.Verdict = FoldAttempts(r.Attempts)
	if r.Verdict == QualityPass {
		if ok, missing := r.Provenance.Complete(); !ok {
			r.Verdict = QualityErrorNoVerdict
			if r.FirstDivergence == "" {
				r.FirstDivergence = "incomplete_provenance:" + missing
			}
		}
	}
}

// scrubReplay bounds and redacts a replay-artifact handle so a QUALITY_QUARANTINE row never
// carries raw secrets (#4569 criterion 3: a SCRUBBED replay artifact). It reuses the journal's
// secretish guard — a handle that looks secret-bearing is DROPPED entirely rather than recorded —
// and bounds the length like every other operator-facing label the journal keeps.
func scrubReplay(handle string) string {
	handle = strings.TrimSpace(handle)
	if handle == "" || secretish(handle) {
		return ""
	}
	return boundArgsLabel(handle)
}

// AppendQualityQuarantine records one flaky-quality-eval case decision as a durable, chained
// QUALITY_QUARANTINE row and returns the committed row (with its stamped Seq/hash). It folds the
// attempt log through the retry-safe policy (Decide) and scrubs the replay handle BEFORE writing,
// so the persisted verdict already obeys the "no real failure retried into green" and
// "missing / inconclusive is never pass" invariants regardless of what the caller passed. The
// case id rides Tool, the deciding authority rides By, the folded verdict rides the chained
// Verdict field, and its class is mirrored onto Reason so the row's forensic identity is
// self-describing on the frozen decision fields; the full correlated record rides the non-chained
// Quality carrier. It is a no-op returning the zero Row on a nil receiver, so a caller that
// guarded the journal on may call it unconditionally.
func (j *Journal) AppendQualityQuarantine(rec QualityQuarantineRow) Row {
	if j == nil {
		return Row{}
	}
	if rec.Schema == "" {
		rec.Schema = QualityQuarantineSchema
	}
	rec.ReplayArtifact = scrubReplay(rec.ReplayArtifact)
	rec.Decide() // fold attempts + provenance gate into the retry-safe verdict
	row := Row{
		Kind:    KindQualityQuarantine,
		Tool:    rec.Case,
		By:      "quality-policy",
		Verdict: rec.Verdict,
		Reason:  rec.Verdict,
		Witness: rec.FirstDivergence,
		Quality: &rec,
	}
	j.mu.Lock()
	j.appendLocked(row)
	committed := j.recent[len(j.recent)-1]
	j.mu.Unlock()
	return committed
}
