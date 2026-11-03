package adjudicator

import (
	"crypto/subtle"
	"fmt"
	"regexp"
	"strings"
)

// ReversibilityClass is the preview-gate label for a pending tool call.
type ReversibilityClass string

const (
	ReversibilityReversible    ReversibilityClass = "reversible"
	ReversibilityIrreversible  ReversibilityClass = "irreversible"
	ReversibilityOutwardFacing ReversibilityClass = "outward-facing"
)

// ReversibilityConfirmArg is the reserved argument key a caller must echo to
// confirm an irreversible/outward-facing preview. It is namespaced so ordinary
// tools are unlikely to define it accidentally.
const ReversibilityConfirmArg = "_fak_confirm"

// ReversibilityEnvelope is the bounded preview surfaced before an
// irreversible/outward-facing call is allowed to execute.
type ReversibilityEnvelope struct {
	Class        ReversibilityClass `json:"class"`
	Preview      string             `json:"preview"`
	ConfirmToken string             `json:"confirm_token,omitempty"`
	DryRunHint   string             `json:"dry_run_hint,omitempty"`
	// RewriteCommand / RewriteTool name the sanctioned sidestep this call could be
	// auto-substituted with, and the tool it would run under. They are populated ONLY
	// for the SAFE-subset push (a bare `git push` / `git push <remote>` with no branch)
	// and are empty for every other call — including every dangerous push.
	//
	// This is the CLASSIFIER half only. The opt-in actuation half (a Policy.AutoRepairSidestep knob that turns the
	// reversibility HOLD into an in-flight TRANSFORM) is landed in decide.go, where the rung reads RewriteCommand
	// to perform safe-subset substitutions when auto-repair is opted in.
	// The invariant this half establishes ahead of it is the one that makes the
	// eventual gate safe to write as a bare `env.RewriteCommand != ""`:
	// the safe-subset test lives HERE, so a rung that trusts a non-empty
	// RewriteCommand can never be handed a force/delete/refspec/named-branch push.
	RewriteCommand string `json:"rewrite_command,omitempty"`
	RewriteTool    string `json:"rewrite_tool,omitempty"`
}

var previewSecretRE = regexp.MustCompile(`(?i)(password|token|secret|api[_-]?key|authorization)(=|:)[^\s]+`)

// ClassifyReversibility labels a pending tool call by the durability of its
// visible effect. The classifier is intentionally cheap and structural: it never
// shells out, never consults tool-specific code, and only escalates the known
// outward/destructive families that need a preview-confirm pause.
func ClassifyReversibility(tool string, args map[string]any) ReversibilityEnvelope {
	class, hint := classifyReversibility(tool, args)
	preview := reversibilityPreview(class, tool, args)
	env := ReversibilityEnvelope{Class: class, Preview: preview, DryRunHint: hint}
	if class != ReversibilityReversible {
		env.ConfirmToken = ReversibilityConfirmToken(class, tool, args)
	}
	// Populate the safe-subset sidestep target. Only a bare `git push` /
	// `git push <remote>` with no branch and no dangerous flag carries a rewrite; a
	// force/delete/tags/all/mirror/refspec/-u/named-branch/sibling push does not, so a
	// future `env.RewriteCommand != ""` gate can never see a dangerous push. Reads
	// the same sink-scoped view the classifier does (#5898): a declared tool's
	// payload prose can no more offer a sidestep than it can classify.
	if cmd := actionCommandText(tool, args); cmd != "" {
		if safeBareGitPush(cmd) {
			env.RewriteCommand = gitPushSidestepCommand
			env.RewriteTool = tool
		} else if safeBareGitPull(cmd) {
			env.RewriteCommand = gitPullSidestepCommand
			env.RewriteTool = tool
		}
	}
	return env
}

// ReversibilityConfirmed returns the call's preview envelope and whether it may
// proceed without another preview pause. Reversible calls are always confirmed;
// irreversible/outward-facing calls must echo the deterministic confirm token.
func ReversibilityConfirmed(tool string, args map[string]any) (ReversibilityEnvelope, bool) {
	env := ClassifyReversibility(tool, args)
	if env.Class == ReversibilityReversible {
		return env, true
	}
	got := confirmationToken(args)
	if got == "" {
		return env, false
	}
	return env, subtle.ConstantTimeCompare([]byte(got), []byte(env.ConfirmToken)) == 1
}

func classifyReversibility(tool string, args map[string]any) (ReversibilityClass, string) {
	// Action rules read the sink-scoped argument view (#5898): a tool that
	// declares which argument keys reach an execution sink exposes only those
	// to the family matchers, so payload prose (a commit message, an issue
	// body) cannot classify. An undeclared tool keeps the full scan.
	cmd := actionCommandText(tool, args)
	// The dry-run scan reads the quote-stripped payload view, not the raw
	// command: a QUOTED "--dry-run" is a mention, and must not launder a
	// destructive command past the preview gate (#2752).
	if hasDryRunPreview(payloadScanView(cmd), args) {
		return ReversibilityReversible, ""
	}
	return classifyAgainstFamilies(reversibilityFamilies, tool, cmd)
}

func reversibilityPreview(class ReversibilityClass, tool string, args map[string]any) string {
	if cmd := commandText(args); cmd != "" {
		return string(class) + " command: " + previewSnippet(cmd)
	}
	if target := targetPath(args); target != "" {
		return fmt.Sprintf("%s tool %q targeting %s", class, tool, previewSnippet(target))
	}
	return fmt.Sprintf("%s tool %q", class, tool)
}

func previewSnippet(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	s = previewSecretRE.ReplaceAllString(s, "$1=[REDACTED]")
	const max = 180
	if len(s) <= max {
		return s
	}
	return s[:max-3] + "..."
}
