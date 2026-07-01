package fleetmon

import (
	"fmt"
	"strings"
	"time"
)

// ReplaceSchema tags the replacement decision payload.
const ReplaceSchema = "fak-fleet-replace/1"

// eligibleClasses is the closed set of monitor classifications for which a
// replacement is permitted. A healthy or completed worker is never replaced; a
// stale-child-command worker is the janitor's job, not the launcher's.
var eligibleClasses = map[Classification]bool{
	ClassDead:            true,
	ClassAuthRateBlocked: true,
	ClassStaleTranscript: true,
}

// ReplaceRequest bundles the inputs to a replacement decision.
type ReplaceRequest struct {
	Worker   PlanWorker     // the failed worker's plan entry (issue, url, area, account)
	Class    Classification // the monitor's classification of the original
	Index    int            // replacement index (1 for the first replacement)
	Account  string         // account/config bucket override ("" keeps the worker's)
	Template string         // an optional corrected prompt template ("" uses the default)
	Force    bool           // operator override: treat the original as explicitly unrecoverable
	RunID    string
	Now      time.Time
}

// ReplaceDecision is the machine-readable verdict: whether a replacement is
// permitted, the generated session name, the rendered prompt, and the ledger row
// that supersedes the original.
type ReplaceDecision struct {
	Schema     string     `json:"schema"`
	Eligible   bool       `json:"eligible"`
	Reason     string     `json:"reason"`
	Class      string     `json:"class"`
	NewSession string     `json:"new_session,omitempty"`
	Account    string     `json:"account,omitempty"`
	Prompt     string     `json:"prompt,omitempty"`
	LedgerRow  *LedgerRow `json:"ledger_row,omitempty"`
}

// EvaluateReplace decides whether a stuck worker may be replaced and, if so,
// renders the replacement session name, the safe prompt, and the superseding
// ledger row. It refuses a healthy/completed worker (and any class outside the
// eligible set) unless Force is set — the guard against auto-relaunching a
// slow-but-healthy worker.
func EvaluateReplace(req ReplaceRequest) ReplaceDecision {
	d := ReplaceDecision{Schema: ReplaceSchema, Class: string(req.Class)}
	idx := req.Index
	if idx < 1 {
		idx = 1
	}
	account := firstNonEmpty(req.Account, req.Worker.Account)
	d.Account = account

	if !eligibleClasses[req.Class] && !req.Force {
		d.Eligible = false
		d.Reason = fmt.Sprintf("refused: monitor classifies the original as %q — a replacement is only allowed for dead, auth-or-rate-blocked, or stale-transcript (or with --force for an explicitly unrecoverable worker)", req.Class)
		return d
	}

	d.Eligible = true
	if eligibleClasses[req.Class] {
		d.Reason = fmt.Sprintf("original is %q — replacement permitted", req.Class)
	} else {
		d.Reason = fmt.Sprintf("original is %q but --force treats it as explicitly unrecoverable", req.Class)
	}
	d.NewSession = fmt.Sprintf("issue-%d-replacement-%d", req.Worker.Issue, idx)
	d.Prompt = RenderReplacementPrompt(req.Worker, req.Template)

	row := LedgerRow{
		Schema:       RunLedgerSchema,
		RunID:        req.RunID,
		Issue:        req.Worker.Issue,
		Session:      req.Worker.Session,
		Outcome:      string(OutcomeSuperseded),
		SupersededBy: d.NewSession,
		FollowUp:     fmt.Sprintf("replaced (original %s) after monitor class %q", req.Worker.Session, req.Class),
		RecordedAt:   req.Now.UTC().Format(time.RFC3339),
	}
	d.LedgerRow = &row
	return d
}

// defaultPromptTemplate is the safe replacement prompt. It carries forward the
// run's non-negotiable discipline so a replacement worker cannot lose the
// shared-tree warnings, the no-push/no-reset rules, or the final-report
// requirement. Placeholders: {{issue}}, {{issue_url}}, {{area}}.
const defaultPromptTemplate = `You are a REPLACEMENT worker for issue #{{issue}}.

Issue: {{issue_url}}
Expected area: {{area}}

A prior worker on this issue became unrecoverable (dead, auth/rate-blocked, or
stale past the threshold). Pick up the issue from a clean read of its current
state — do not assume the prior worker's partial work is correct or even present.

SHARED-TREE DISCIPLINE (these are enforced below the agent layer — do not fight them):
- The working tree is shared and DIRTY with other agents' in-flight work. Never
  ` + "`git add -A`" + `; commit only by explicit path (` + "`git commit -s -- <paths>`" + `).
- Work directly on the trunk (main). Never open a feature branch or a new worktree.
- Never ` + "`git reset --hard`" + `, never ` + "`git checkout -- <peer files>`" + `, never force-push.
- If a guard refuses (OFF_TRUNK, a peer MERGE_HEAD in flight), reconcile in place or STOP.

FINAL-REPORT REQUIREMENT:
End with a final report that states, explicitly:
- the outcome (patch-with-witness / blocked-scoped / read-only audit),
- the exact files you changed (if any),
- the witness command you ran and its result (the proof the fix works), and
- if blocked, the single smallest follow-up needed to unblock.

Do not report "done" without a captured witness. If you cannot prove it, say
"not yet" with the missing witness and the next checkable step.`

// RenderReplacementPrompt fills a replacement prompt template with the worker's
// issue, url, and area. An empty template uses the safe default.
func RenderReplacementPrompt(w PlanWorker, template string) string {
	tpl := template
	if strings.TrimSpace(tpl) == "" {
		tpl = defaultPromptTemplate
	}
	url := w.IssueURL
	if url == "" {
		url = fmt.Sprintf("https://github.com/anthony-chaudhary/fak/issues/%d", w.Issue)
	}
	area := w.Area
	if area == "" {
		area = "(unspecified — read the issue to scope it)"
	}
	r := strings.NewReplacer(
		"{{issue}}", itoa(w.Issue),
		"{{issue_url}}", url,
		"{{area}}", area,
	)
	return r.Replace(tpl)
}
