package adjudicator

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode"
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

// gitPushSidestepCommand is the sanctioned safe-push verb offered in place of a bare
// `git push`: a non-force, current-branch push the kernel admits because its command
// head is `fak`, not `git push`. It is pinned to the SAME string the git-push
// reversibility family's redirect hint advertises (familyRemedyCommands["git-push"][0])
// — TestSidestepRewriteMatchesRemedyTable binds the two so the machine-readable
// rewrite target and the human-facing advisory hint can never drift apart.
const gitPushSidestepCommand = "fak sync push"

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
	if cmd := actionCommandText(tool, args); cmd != "" && safeBareGitPush(cmd) {
		env.RewriteCommand = gitPushSidestepCommand
		env.RewriteTool = tool
	}
	return env
}

// safeBareGitPush reports whether cmd is the SAFE push subset a sidestep may substitute:
// a single command (no sibling riding along a `&&`/`;`/`|`) whose head is `git push`
// with, at most, a bare remote name and NO branch, NO refspec, and NO flag. Every
// departure — a force/delete/tags/all/mirror/prune flag, an upstream `-u`, a
// `local:remote` refspec, a named target branch (`git push origin main`), or a second
// segment — differs from `fak sync push` (non-force, current-branch, fast-forward-only),
// so laundering it into that verb would silently change intent; those return false and
// stay on the preview-confirm hold.
//
// It parses the RAW command (dashes intact, unlike commandSegments' family-matcher
// normalization) so a flag is visible, strips the same leading env assignments and
// wrapper heads, and splits on commandSegmentRE. That split is deliberately NOT the
// quote-aware one commandSegments uses: a sequencing byte inside a quoted payload
// manufactures an extra segment here and pushes the call OUT of the safe subset. For a
// family matcher that would be a false positive; for a safe-subset admission it is the
// conservative direction — the call keeps its preview-confirm hold.
func safeBareGitPush(cmd string) bool {
	var nonEmpty []string
	for _, raw := range commandSegmentRE.Split(cmd, -1) {
		if strings.TrimSpace(raw) != "" {
			nonEmpty = append(nonEmpty, raw)
		}
	}
	if len(nonEmpty) != 1 { // a sibling command disqualifies the whole line
		return false
	}
	fields := strings.Fields(nonEmpty[0])
	for len(fields) > 0 &&
		(envAssignmentRE.MatchString(fields[0]) || commandWrapperHeads[strings.ToLower(fields[0])]) {
		fields = fields[1:]
	}
	if len(fields) < 2 || strings.ToLower(fields[0]) != "git" || strings.ToLower(fields[1]) != "push" {
		return false
	}
	positional := 0
	for _, tok := range fields[2:] {
		if strings.HasPrefix(tok, "-") {
			return false // any flag (force, -u, --tags, --all, --delete, --mirror, --prune, …)
		}
		if strings.Contains(tok, ":") {
			return false // an explicit refspec (local:remote)
		}
		positional++
		if positional > 1 {
			return false // a second positional names a specific branch, not the current one
		}
	}
	return true
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

// ReversibilityConfirmToken derives the stable preview token. Confirmation keys
// AND incidental keys (a Bash call's free-text "description", and the client
// supervision knobs "timeout"/"timeout_ms"/"run_in_background" — see
// isIncidentalTokenKey) are excluded from the hashed payload, and the
// action-bearing command text is whitespace-normalized (see normalizeCommandForToken),
// so the token binds only to the call's effect-bearing subject and is stable across a
// re-proposal even when the client's confirm key, prose annotation, supervision knobs,
// OR cosmetic command spacing drift between proposals.
//
// The annotation exclusion is the fix for the confirm-loop that wedged a fleet
// session (docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md): Claude Code's Bash
// tool carries a mandatory "description" the model regenerates every turn, so a
// re-proposal of the identical destructive command still hashed to a FRESH token
// and the advertised "re-propose byte-identical + add _fak_confirm" recovery
// could never converge. Excluding annotations cannot weaken any floor — they
// never influence the call's effect, the classifier never reads them, and hard
// denies still win before this rung.
func ReversibilityConfirmToken(class ReversibilityClass, tool string, args map[string]any) string {
	canon, _ := json.Marshal(argsForToken(args))
	sum := sha256.Sum256([]byte(string(class) + "\x00" + strings.ToLower(tool) + "\x00" + string(canon)))
	return "fak-" + hex.EncodeToString(sum[:8])
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

// familyMatchInput carries the precomputed views of one pending call that
// family matchers key on: the payload-scan view of the command (quoted
// mentions blanked, live client payloads kept — #2752) raw and lowered, the
// quote-aware operator-split segments, the view's word stream, and the
// lowered tool name.
type familyMatchInput struct {
	cmd       string
	lowerCmd  string
	segs      [][]string
	words     []string
	lowerTool string
}

// reversibilityFamily declares one escalated family in a single place: every
// surface that blocks it AND the redirect surfaced with the escalation. The
// classifier and the dry-run hint both read the SAME entry, so a family
// cannot be added to the block without deciding its redirect, and a redirect
// cannot name a surface the block never fires on (#2748, #2746).
type reversibilityFamily struct {
	name  string
	class ReversibilityClass
	// Block surfaces — any non-zero surface participates in the match.
	heads        []string                       // command segment head verbs
	prefixes     [][]string                     // command segment word prefixes
	cmdContains  []string                       // lowered whole-command substrings
	toolContains []string                       // lowered tool-name substrings
	matchCmd     func(in familyMatchInput) bool // payload-shaped matchers (curl flags, SQL words)
	// Redirect surfaced when this family blocks ("" = no sanctioned sidestep).
	hint string
}

func (f *reversibilityFamily) matches(in familyMatchInput) bool {
	if segmentHeadIs(in.segs, f.heads...) {
		return true
	}
	for _, p := range f.prefixes {
		if segmentHasPrefix(in.segs, p...) {
			return true
		}
	}
	for _, sub := range f.cmdContains {
		if strings.Contains(in.lowerCmd, sub) {
			return true
		}
	}
	for _, sub := range f.toolContains {
		if strings.Contains(in.lowerTool, sub) {
			return true
		}
	}
	return f.matchCmd != nil && f.matchCmd(in)
}

// reversibilityClassOrder fixes the escalation precedence: a call matching
// both an outward-facing family and an irreversible family previews as
// outward-facing, whatever the table order.
var reversibilityClassOrder = []ReversibilityClass{ReversibilityOutwardFacing, ReversibilityIrreversible}

func classifyAgainstFamilies(families []reversibilityFamily, tool, cmd string) (ReversibilityClass, string) {
	// Payload-shaped scans (curl flags, SQL words, cmdContains substrings) read
	// the quote-stripped view so a trigger mentioned inside a quoted argument —
	// a commit message, a grep pattern, an echoed string — is not a live token,
	// while a quoted payload handed to a DB/shell client stays live (#2752).
	view := payloadScanView(cmd)
	in := familyMatchInput{
		cmd:       view,
		lowerCmd:  strings.ToLower(view),
		segs:      commandSegments(cmd),
		words:     commandWords(view),
		lowerTool: strings.ToLower(tool),
	}
	for _, class := range reversibilityClassOrder {
		for i := range families {
			if families[i].class != class {
				continue
			}
			if families[i].matches(in) {
				return class, families[i].hint
			}
		}
	}
	return ReversibilityReversible, ""
}

// reversibilityFamilies is the single source of truth for the escalated
// outward/destructive families (#2748). Within a class, earlier entries win
// the redirect when a call matches several families, so the hinted families
// stay first, in redirect-priority order.
//
// Most of the `gh` surface is deliberately NOT a family (operator decision,
// 2026-07-05): issue/pr/release create·comment·edit·close·reopen·merge·upload
// targets the operator's OWN authenticated GitHub and is reversible in practice
// (issues/PRs edit·close·reopen; a release can be deleted), so the preview-confirm
// pause was pure friction on routine fleet work — the #2650/#2651 confirm-loop
// lesson — while the Claude Code allow-list already admits `Bash(gh …)`.
//
// The ONE carve-out is the gh-write family below (#3560): `gh api` with a write
// method (--method/-X POST|PUT|PATCH|DELETE) and `gh repo fork|rename|delete` are
// re-escalated to the SAME outward-facing class as `git push`, because those
// escape the relaxation's premise. `gh api` mutations reach ARBITRARY third-party
// repos and the Git Data API (author commits/branches/trees, open cross-repo PRs,
// PATCH-rename or delete a repo) and `gh repo fork|rename|delete` always mutate —
// a strictly LARGER blast radius than the git-CLI writes the relaxation was
// matching, so leaving them allowed reroutes capability DOWNWARD past the git-push
// gate. A `gh api` READ (GET / no --method) stays reversible. curl/httpie POSTs
// likewise stay gated (http-write below): those reach ARBITRARY hosts, not the
// authenticated gh surface, so the outbound floor still holds for the general case.
var reversibilityFamilies = []reversibilityFamily{
	// An MCP tool literally named create_issue / issue_create is still
	// escalated; the redirect points at the compiled `fak issue create` verb
	// (cmd/fak/issue_create.go), a trusted-binary sidestep the kernel admits.
	// Raw `gh issue create` is no longer escalated (see the gh note above), so
	// a Bash gh call never reaches this family.
	{
		name:         "issue-create-tool",
		class:        ReversibilityOutwardFacing,
		toolContains: []string{"create_issue", "issue_create"},
		hint:         "file it with the sanctioned compiled verb: fak issue create --title … --body-file … (a trusted-binary path the kernel admits)",
	},
	// A Bash `slack …` HEAD (or an MCP tool whose name contains "slack") is
	// escalated; the redirect points at the compiled `fak slack send` verb
	// (cmd/fak/slack.go). Matched on the segment HEAD — not a bare substring —
	// so `git push origin slack-feature` keeps the git-push redirect below,
	// and sendmail/mail/mutt (no fak equivalent) stay a separate family with
	// no redirect.
	{
		name:         "slack",
		class:        ReversibilityOutwardFacing,
		heads:        []string{"slack"},
		toolContains: []string{"slack"},
		hint:         "send it with the sanctioned compiled verb: fak slack send (a trusted-binary path the kernel admits)",
	},
	// git push is escalated; the redirect names the compiled sidestep FIRST —
	// a safe, non-force push the kernel admits because its command head is
	// `fak`, not `git push` — with the --dry-run preview kept as the secondary
	// option. Before #2651's pattern was generalized here, this hint named
	// only `git push --dry-run`, which previews nothing and funnels the agent
	// straight back to the gated real push (the confirm loop in
	// docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md).
	{
		name:         "git-push",
		class:        ReversibilityOutwardFacing,
		prefixes:     [][]string{{"git", "push"}},
		toolContains: []string{"git_push"},
		hint:         "push with the safe compiled verb: fak sync push (a trusted-binary non-force push the kernel admits), or preview first with git push --dry-run",
	},
	// The gh-write carve-out (#3560, see the gh note above): a `gh api` mutation
	// (--method/-X POST|PUT|PATCH|DELETE) or `gh repo fork|rename|delete` is the
	// SAME outward-facing class as git push — a larger-blast-radius write than the
	// git-CLI path, so it may not slip under the gh relaxation. A `gh api` read
	// (GET / no --method) is untouched and stays reversible.
	{
		name:     "gh-write",
		class:    ReversibilityOutwardFacing,
		matchCmd: func(in familyMatchInput) bool { return ghWriteMutation(in.cmd) },
		hint:     "gh api writes and gh repo fork/rename/delete reach arbitrary GitHub repos with a larger blast radius than the git-CLI path; read instead (gh api with GET / no --method), or review the exact repo, endpoint, and method before confirming the live mutation",
	},
	{
		name:     "npm-publish",
		class:    ReversibilityOutwardFacing,
		prefixes: [][]string{{"npm", "publish"}},
		hint:     "preview the package publish first: npm publish --dry-run",
	},
	{
		name:  "mail",
		class: ReversibilityOutwardFacing,
		heads: []string{"sendmail", "mail", "mutt"},
		hint:  "review the exact recipient and body in a draft before confirming the live mail send",
	},
	{
		name:         "webhook",
		class:        ReversibilityOutwardFacing,
		cmdContains:  []string{"webhook"},
		toolContains: []string{"webhook"},
		hint:         "send to a test endpoint or use the endpoint's dry-run mode before confirming the live webhook",
	},
	{
		name:         "registry-publish",
		class:        ReversibilityOutwardFacing,
		prefixes:     [][]string{{"docker", "push"}, {"cargo", "publish"}, {"gem", "push"}, {"twine", "upload"}},
		toolContains: []string{"publish", "upload"},
		hint:         "use the registry's check or dry-run path before publishing, such as cargo publish --dry-run or twine check",
	},
	{
		name:  "http-write",
		class: ReversibilityOutwardFacing,
		matchCmd: func(in familyMatchInput) bool {
			return curlWrites(in.cmd) || httpieWrites(in.segs)
		},
		hint: "use a read-only request, a test endpoint, or the service's dry-run mode before confirming the live HTTP write",
	},
	{
		name:         "messaging-tool",
		class:        ReversibilityOutwardFacing,
		toolContains: []string{"send_email", "sendemail", "email", "send_mail", "post_message"},
		hint:         "use the sanctioned compiled verb when one exists, such as fak slack send for Slack, or review recipient/body before confirming",
	},
	{
		name:         "pr-create-tool",
		class:        ReversibilityOutwardFacing,
		toolContains: []string{"create_pr", "pr_create"},
		hint:         "preview the pull request title, body, and target branch with the host's draft or dry-run path before confirming creation",
	},
	{
		name:  "fs-destroy",
		class: ReversibilityIrreversible,
		heads: []string{"rm", "rmdir", "del", "erase", "shred", "truncate", "mkfs", "remove-item"},
		hint:  "use a preview or quarantine path first, such as Remove-Item -WhatIf or git clean -nd for git cleanup",
	},
	{
		name:     "git-destroy",
		class:    ReversibilityIrreversible,
		prefixes: [][]string{{"git", "clean"}, {"git", "reset", "hard"}},
		hint:     "inspect first with git status and a non-mutating preview such as git clean -nd; preserve shared work instead of resetting it",
	},
	{
		name:     "infra-destroy",
		class:    ReversibilityIrreversible,
		prefixes: [][]string{{"terraform", "destroy"}, {"kubectl", "delete"}},
		hint:     "preview infrastructure deletion first with terraform plan -destroy or kubectl diff/delete --dry-run=server",
	},
	// Payload scans stay whole-command on purpose: SQL statements and dd
	// targets arrive as arguments to a client binary, never as the head.
	{
		name:  "sql-drop",
		class: ReversibilityIrreversible,
		matchCmd: func(in familyMatchInput) bool {
			return orderedWords(in.words, "drop", "database") || orderedWords(in.words, "drop", "table")
		},
		hint: "inspect or back up the schema first, then use a transaction that rolls back before confirming the DROP",
	},
	{
		name:  "sql-truncate",
		class: ReversibilityIrreversible,
		matchCmd: func(in familyMatchInput) bool {
			return orderedWords(in.words, "truncate", "table") || (containsWord(in.words, "truncate") && (containsWord(in.words, "table") || containsWord(in.words, "database")))
		},
		hint: "inspect or back up data first, then use a transaction with rollback before confirming the TRUNCATE",
	},
	{
		name:  "sql-alter",
		class: ReversibilityIrreversible,
		matchCmd: func(in familyMatchInput) bool {
			return orderedWords(in.words, "alter", "table") || orderedWords(in.words, "alter", "database")
		},
		hint: "verify schema compatibility or capture a snapshot first, then use a transaction with rollback before confirming the ALTER",
	},
	{
		name:  "db-migration-down",
		class: ReversibilityIrreversible,
		matchCmd: func(in familyMatchInput) bool {
			return (containsWord(in.words, "prisma") && containsWord(in.words, "reset")) ||
				(containsWord(in.words, "alembic") && containsWord(in.words, "downgrade")) ||
				(containsWord(in.words, "goose") && (containsWord(in.words, "down") || containsWord(in.words, "reset"))) ||
				(containsWord(in.words, "flyway") && containsWord(in.words, "clean"))
		},
		hint: "downward database migration or database reset is irreversible; capture a database snapshot before confirming",
	},
	{
		name:  "dd-device-write",
		class: ReversibilityIrreversible,
		matchCmd: func(in familyMatchInput) bool {
			return containsWord(in.words, "dd") && strings.Contains(in.lowerCmd, "of=/dev/")
		},
		hint: "write to an image file first and inspect the target device before confirming a raw device write",
	},
	{
		name:         "destructive-tool",
		class:        ReversibilityIrreversible,
		toolContains: []string{"delete", "remove", "destroy", "truncate", "unlink", "rmdir"},
		hint:         "use the tool's preview or dry-run mode, or move the target to quarantine first, before confirming destruction",
	},
}

func commandText(args map[string]any) string {
	for _, key := range []string{"command", "cmd", "shell", "script"} {
		if s, ok := stringArg(args, key); ok {
			return s
		}
	}
	for _, key := range []string{"argv", "args"} {
		if joined, ok := stringSliceArg(args, key); ok {
			return joined
		}
	}
	return ""
}

func hasDryRunPreview(cmd string, args map[string]any) bool {
	for _, key := range []string{"dry_run", "dryRun", "preview"} {
		if b, ok := boolArg(args, key); ok && b {
			return true
		}
	}
	lower := strings.ToLower(cmd)
	if strings.Contains(lower, "--dry-run") || strings.Contains(lower, "--preview") {
		return true
	}
	// PowerShell's -WhatIf is a universal no-op preview: a ShouldProcess cmdlet
	// carrying it reports what WOULD happen and performs nothing. So the
	// "Remove-Item -WhatIf" the fs-destroy redirect recommends is a preview, not
	// a deletion — recognize it, or the guard gates its OWN sanctioned dry-run
	// and the agent loops (the self-refuting-remedy class of the confirm-gate
	// deadlock, docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md).
	if hasWhatIfFlag(cmd) {
		return true
	}
	// `git clean -n` / `git push -n` (the short spelling of --dry-run) preview
	// without mutating, even combined with -d/-f (e.g. "git clean -nd", the
	// fs-destroy/git-destroy redirect; or "git push -nf"). -n is recognized ONLY
	// inside a `git clean`/`git push` segment, never globally: a bare -n means
	// unrelated things elsewhere (`kubectl delete -n <ns>` is a REAL delete that
	// must stay gated).
	if gitDryRunPreview(cmd) {
		return true
	}
	return false
}

// whatIfFlagRE matches PowerShell's -WhatIf common parameter as a standalone
// token (optionally -WhatIf:$true), anchored to a segment/word boundary so a
// path like ./-whatiffoo or a substring never trips it.
var whatIfFlagRE = regexp.MustCompile(`(?i)(^|[\s;&|(])-whatif($|[\s:;&|)])`)

// hasWhatIfFlag reports whether the command carries PowerShell's -WhatIf. It
// scans the RAW command (dashes intact): commandWords()/commandSegments() strip
// leading dashes, so a flag-shaped recognizer must not go through them.
func hasWhatIfFlag(cmd string) bool {
	return whatIfFlagRE.MatchString(cmd)
}

// gitDryRunPreview reports whether a `git clean` or `git push` invocation carries
// a dry-run flag: --dry-run, or a single-dash short-flag cluster containing 'n'
// (-n, -nd, -dn, -nf). Both subcommands spell dry-run identically, and for BOTH,
// 'n' is the only short flag that means dry-run — so any cluster containing 'n'
// is a genuine preview. A cluster WITHOUT 'n' is a real mutation and stays gated:
// `git clean -fd`/`-f`/`-x` (a real removal), and `git push -f` (force) or the
// destructive `git push -d`/`--delete <branch>` (a remote-branch delete) all lack
// 'n' and are untouched. The short `git push -n` is the documented equivalent of
// the `git push --dry-run` the git-push redirect itself recommends — recognizing
// it closes the same self-refuting-remedy gap the -WhatIf / git-clean-`-n` cases
// did (docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md): the long form was already
// reversible via the --dry-run substring, but the short form was gated as a live
// push. The flag is recognized ONLY inside these two git subcommands, never
// globally: a bare -n means unrelated things elsewhere (`kubectl delete -n <ns>`
// is a REAL delete that must stay gated). It parses the RAW command (dash-
// preserving), splitting on the same sequencing operators and stripping the same
// env/wrapper heads as commandSegments, but keeping flag dashes that commandWords
// drops.
func gitDryRunPreview(cmd string) bool {
	for _, raw := range commandSegmentRE.Split(cmd, -1) {
		fields := strings.Fields(raw)
		for len(fields) > 0 &&
			(envAssignmentRE.MatchString(fields[0]) || commandWrapperHeads[strings.ToLower(fields[0])]) {
			fields = fields[1:]
		}
		if len(fields) < 2 || strings.ToLower(fields[0]) != "git" {
			continue
		}
		switch strings.ToLower(fields[1]) {
		case "clean", "push":
		default:
			continue
		}
		for _, tok := range fields[2:] {
			lt := strings.ToLower(tok)
			if lt == "--dry-run" {
				return true
			}
			if strings.HasPrefix(lt, "-") && !strings.HasPrefix(lt, "--") && strings.ContainsRune(lt[1:], 'n') {
				return true
			}
		}
	}
	return false
}

var curlWriteFlagRE = regexp.MustCompile(`(?i)(^|[;&|()]|[[:space:]])curl(\.exe)?\b.*([[:space:]]-[Xx][[:space:]]*(POST|PUT|PATCH|DELETE)\b|[[:space:]]--request(=|[[:space:]]+)(POST|PUT|PATCH|DELETE)\b|[[:space:]](-d|--data|--data-raw|--data-binary|--form)(=|[[:space:]]|$))`)

func curlWrites(cmd string) bool {
	return curlWriteFlagRE.MatchString(cmd)
}

func httpieWrites(segs [][]string) bool {
	for _, seg := range segs {
		if seg[0] != "http" && seg[0] != "https" {
			continue
		}
		for _, w := range seg[1:] {
			if httpWriteVerb(w) {
				return true
			}
		}
	}
	return false
}

func httpWriteVerb(w string) bool {
	switch strings.ToLower(w) {
	case "post", "put", "patch", "delete":
		return true
	default:
		return false
	}
}

// ghWriteMutation reports whether the command invokes a MUTATING gh surface that
// the operator relaxation (see the gh note on reversibilityFamilies) must NOT
// cover (#3560): `gh api` with a write method (--method/-X POST|PUT|PATCH|DELETE,
// the flag anywhere in argv), or `gh repo fork|rename|delete` (always mutating).
// A `gh api` READ (GET / no --method) is a read and returns false. It reads the
// payload-scan view — quoted MENTIONS are already blanked (so a commit message or
// grep pattern naming `gh api -X POST` does not classify), while a live-payload
// statement stays visible — and parses raw segments so flag dashes survive
// (commandWords strips them). It splits on the same sequencing operators and
// strips the same leading env/wrapper heads as commandSegments/gitDryRunPreview,
// so a `sudo`/`FOO=1` prefix does not hide the gh call.
func ghWriteMutation(cmd string) bool {
	for _, raw := range commandSegmentRE.Split(cmd, -1) {
		fields := strings.Fields(raw)
		for len(fields) > 0 &&
			(envAssignmentRE.MatchString(fields[0]) || commandWrapperHeads[strings.ToLower(fields[0])]) {
			fields = fields[1:]
		}
		if len(fields) < 2 || ghHead(fields[0]) != "gh" {
			continue
		}
		switch strings.ToLower(fields[1]) {
		case "api":
			if ghAPIArgsWrite(fields[2:]) {
				return true
			}
		case "repo":
			if len(fields) >= 3 && ghRepoSubMutates(fields[2]) {
				return true
			}
		}
	}
	return false
}

// ghHead normalizes a command head to its bare program name so `gh`, `gh.exe`,
// and a path like /usr/bin/gh all read as "gh".
func ghHead(tok string) string {
	h := strings.ToLower(tok)
	h = strings.TrimSuffix(h, ".exe")
	if k := strings.LastIndexAny(h, `/\`); k >= 0 {
		h = h[k+1:]
	}
	return h
}

// ghAPIArgsWrite reports whether a `gh api` argument list carries a write method
// flag — `--method POST`, `--method=POST`, `-X POST`, or the glued `-XPOST` — for
// any of POST/PUT/PATCH/DELETE, robust to the flag appearing ANYWHERE in argv. The
// default gh api method is GET (a read), so no method flag means not a write.
func ghAPIArgsWrite(args []string) bool {
	const eq = "--method="
	for i, a := range args {
		switch {
		case strings.EqualFold(a, "--method"), strings.EqualFold(a, "-X"):
			if i+1 < len(args) && ghMethodWrites(args[i+1]) {
				return true
			}
		case len(a) > len(eq) && strings.EqualFold(a[:len(eq)], eq):
			if ghMethodWrites(a[len(eq):]) {
				return true
			}
		case len(a) > 2 && (strings.HasPrefix(a, "-X") || strings.HasPrefix(a, "-x")):
			if ghMethodWrites(a[2:]) {
				return true
			}
		}
	}
	return false
}

func ghMethodWrites(m string) bool {
	switch strings.ToUpper(m) {
	case "POST", "PUT", "PATCH", "DELETE":
		return true
	default:
		return false
	}
}

func ghRepoSubMutates(sub string) bool {
	switch strings.ToLower(sub) {
	case "fork", "rename", "delete":
		return true
	default:
		return false
	}
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

var (
	commandSegmentRE = regexp.MustCompile(`\|\||&&|[;|&\r\n]`)
	envAssignmentRE  = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*=`)
)

var commandWrapperHeads = map[string]bool{
	"sudo":    true,
	"env":     true,
	"nice":    true,
	"time":    true,
	"command": true,
	"xargs":   true,
	"doas":    true,
}

// commandSegments splits a shell command on UNQUOTED sequencing/pipe
// operators and returns each non-empty segment's word list with leading env
// assignments (NAME=value) and wrapper heads (sudo, env, ...) stripped, so
// index 0 is the command actually being invoked. Family matchers anchor here
// instead of scanning the whole token stream, so trigger words inside quoted
// payloads (grep patterns, commit messages) no longer classify — the
// quote-aware split closes the residual #2752 path where an operator byte
// INSIDE a quoted payload manufactured a spurious segment head.
func commandSegments(cmd string) [][]string {
	var segs [][]string
	for _, raw := range quoteAwareSegmentTexts(cmd) {
		text := segmentMentionStrip(raw)
		if quotedPayloadIsLive(raw) {
			text = unquoteSpans(raw)
		}
		fields := strings.Fields(text)
		for len(fields) > 0 &&
			(envAssignmentRE.MatchString(fields[0]) || commandWrapperHeads[strings.ToLower(fields[0])]) {
			fields = fields[1:]
		}
		if len(fields) == 0 {
			continue
		}
		head := strings.ToLower(strings.TrimSuffix(unquoteSpans(fields[0]), ".exe"))
		if i := strings.LastIndexAny(head, `/\`); i >= 0 {
			head = head[i+1:]
		}
		if head == "" {
			continue
		}
		words := []string{head}
		words = append(words, commandWords(strings.Join(fields[1:], " "))...)
		segs = append(segs, words)
	}
	return segs
}

// quotedSpanEnd returns the index just past the quoted span opening at s[i]
// (s[i] is ' or "), honoring backslash escapes inside double quotes and
// inside $'…' (dollar=true for the ANSI-C form; POSIX single quotes take no
// escapes). An unterminated span runs to the end of the string — the shell
// would refuse to run such a command, so hiding its tail fails toward
// reversible on text that cannot execute.
func quotedSpanEnd(s string, i int, dollar bool) int {
	q := s[i]
	j := i + 1
	for j < len(s) {
		c := s[j]
		if c == '\\' && (q == '"' || dollar) && j+1 < len(s) {
			j += 2
			continue
		}
		if c == q {
			return j + 1
		}
		j++
	}
	return j
}

// quoteAwareSegmentTexts splits cmd on unquoted sequencing/pipe operators
// (`;`, `|`, `&`, newlines — `||`/`&&` fall out of the single-byte split) and
// returns each non-blank segment's raw text, quotes intact. A backslash
// escapes the following byte outside quotes, so an unquoted `\|` (a grep
// alternation) does not split either.
func quoteAwareSegmentTexts(cmd string) []string {
	var segs []string
	var b strings.Builder
	flush := func() {
		if strings.TrimSpace(b.String()) != "" {
			segs = append(segs, b.String())
		}
		b.Reset()
	}
	for i := 0; i < len(cmd); {
		c := cmd[i]
		switch {
		case c == '\\' && i+1 < len(cmd):
			b.WriteString(cmd[i : i+2])
			i += 2
		case c == '\'' || c == '"':
			dollar := c == '\'' && i > 0 && cmd[i-1] == '$'
			end := quotedSpanEnd(cmd, i, dollar)
			b.WriteString(cmd[i:end])
			i = end
		case c == ';' || c == '|' || c == '&' || c == '\n' || c == '\r':
			flush()
			i++
		default:
			b.WriteByte(c)
			i++
		}
	}
	flush()
	return segs
}

// quoteAwareFields splits a segment on unquoted whitespace, keeping quote
// bytes inside each token so a token's quoting is still inspectable after
// the split.
func quoteAwareFields(s string) []string {
	var fields []string
	var b strings.Builder
	flush := func() {
		if b.Len() > 0 {
			fields = append(fields, b.String())
		}
		b.Reset()
	}
	for i := 0; i < len(s); {
		c := s[i]
		switch {
		case c == '\\' && i+1 < len(s):
			b.WriteString(s[i : i+2])
			i += 2
		case c == '\'' || c == '"':
			dollar := c == '\'' && i > 0 && s[i-1] == '$'
			end := quotedSpanEnd(s, i, dollar)
			b.WriteString(s[i:end])
			i = end
		case c == ' ' || c == '\t':
			flush()
			i++
		default:
			b.WriteByte(c)
			i++
		}
	}
	flush()
	return fields
}

// unquoteSpans removes the quote BYTES from every quoted span in s, keeping
// the quoted content in place. Live-payload segments read this view — the
// quoted statement executes, so its text must stay visible with clean word
// boundaries (`bash -c "curl -X POST …"` must still match the curl write
// regex, which anchors on a space before curl, not a quote byte).
func unquoteSpans(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		c := s[i]
		switch {
		case c == '\\' && i+1 < len(s):
			b.WriteString(s[i : i+2])
			i += 2
		case c == '\'' || c == '"':
			dollar := c == '\'' && i > 0 && s[i-1] == '$'
			end := quotedSpanEnd(s, i, dollar)
			stop := end
			if end > i+1 && s[end-1] == c {
				stop = end - 1
			}
			b.WriteString(s[i+1 : stop])
			i = end
		default:
			b.WriteByte(c)
			i++
		}
	}
	return b.String()
}

// segmentMentionStrip returns the non-live segment text the matchers read:
// the effective command token (past env assignments and wrapper heads) is
// UNQUOTED — the shell resolves a quoted head ("rm") and runs it exactly
// like an unquoted one, so hiding it would turn quoting the verb into a
// guard bypass — while every other quoted span is blanked as an inert
// mention (#2752).
func segmentMentionStrip(segText string) string {
	toks := quoteAwareFields(segText)
	out := make([]string, 0, len(toks))
	headDone := false
	for _, tok := range toks {
		if !headDone {
			bare := unquoteSpans(tok)
			if envAssignmentRE.MatchString(bare) || commandWrapperHeads[strings.ToLower(bare)] {
				// An env assignment's quoted VALUE is data: blank it, keep the
				// NAME= husk for the callers' existing prefix-strip loop.
				out = append(out, strings.Fields(stripQuotedSpans(tok))...)
				continue
			}
			headDone = true
			out = append(out, strings.Fields(bare)...)
			continue
		}
		out = append(out, strings.Fields(stripQuotedSpans(tok))...)
	}
	return strings.Join(out, " ")
}

// stripQuotedSpans replaces every quoted span in s (single, double, $'…')
// with a single space, leaving the unquoted command skeleton. The payload
// scans read this view so a quoted mention is inert (#2752).
func stripQuotedSpans(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		c := s[i]
		switch {
		case c == '\\' && i+1 < len(s):
			b.WriteString(s[i : i+2])
			i += 2
		case c == '\'' || c == '"':
			dollar := c == '\'' && i > 0 && s[i-1] == '$'
			b.WriteByte(' ')
			i = quotedSpanEnd(s, i, dollar)
		default:
			b.WriteByte(c)
			i++
		}
	}
	return b.String()
}

// quotedPayloadLiveHeads are the client binaries whose QUOTED argument is a
// live statement executed elsewhere — a SQL client's -c/-e payload, a shell's
// -c script, a remote command handed to ssh — never an inert mention. Their
// segments keep quoted contents in the payload-scan view, so
// `psql -c "drop table t"` still escalates while `git commit -m "drop table
// t"` does not (#2752's explicit both-directions rule).
var quotedPayloadLiveHeads = map[string]bool{
	"psql": true, "mysql": true, "mariadb": true, "sqlite3": true, "sqlcmd": true,
	"mongo": true, "mongosh": true, "duckdb": true, "clickhouse-client": true, "redis-cli": true,
	"sh": true, "bash": true, "zsh": true, "dash": true, "ksh": true, "fish": true,
	"pwsh": true, "powershell": true, "cmd": true, "eval": true, "ssh": true,
}

// quotedPayloadIsLive reports whether a raw segment's effective head (env
// assignments and wrapper heads stripped, path and .exe suffix trimmed) is a
// client whose quoted argument executes as a statement.
func quotedPayloadIsLive(segText string) bool {
	fields := quoteAwareFields(segText)
	for len(fields) > 0 {
		bare := unquoteSpans(fields[0])
		if !envAssignmentRE.MatchString(bare) && !commandWrapperHeads[strings.ToLower(bare)] {
			break
		}
		fields = fields[1:]
	}
	if len(fields) == 0 {
		return false
	}
	// A quoted client name still executes: "psql" -c … is as live as psql.
	head := strings.ToLower(unquoteSpans(fields[0]))
	head = strings.TrimSuffix(head, ".exe")
	if k := strings.LastIndexAny(head, `/\`); k >= 0 {
		head = head[k+1:]
	}
	return quotedPayloadLiveHeads[head]
}

// payloadScanView is the whole-command text the payload-shaped scans (curl
// flags, SQL words, cmdContains substrings, the dry-run flag scan) read:
// quoted spans are blanked segment-by-segment so an inert mention cannot
// classify, EXCEPT in segments whose head is a live-payload client, where the
// quoted text is the statement that executes and must stay visible.
func payloadScanView(cmd string) string {
	if cmd == "" {
		return ""
	}
	segs := quoteAwareSegmentTexts(cmd)
	parts := make([]string, 0, len(segs))
	for _, seg := range segs {
		if quotedPayloadIsLive(seg) {
			parts = append(parts, unquoteSpans(seg))
			continue
		}
		parts = append(parts, segmentMentionStrip(seg))
	}
	return strings.Join(parts, " ; ")
}

func segmentHeadIs(segs [][]string, names ...string) bool {
	for _, seg := range segs {
		for _, name := range names {
			if seg[0] == name {
				return true
			}
		}
	}
	return false
}

func segmentHasPrefix(segs [][]string, want ...string) bool {
	for _, seg := range segs {
		if len(seg) < len(want) {
			continue
		}
		match := true
		for i, w := range want {
			if seg[i] != w {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

func commandWords(cmd string) []string {
	var words []string
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		words = append(words, strings.ToLower(b.String()))
		b.Reset()
	}
	for _, r := range cmd {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '-':
			b.WriteRune(unicode.ToLower(r))
		default:
			flush()
		}
	}
	flush()
	for i, w := range words {
		words[i] = strings.TrimLeft(w, "-")
	}
	return words
}

func containsWord(words []string, want string) bool {
	for _, w := range words {
		if w == want {
			return true
		}
	}
	return false
}

func orderedWords(words []string, want ...string) bool {
	pos := 0
	for _, w := range words {
		if w == want[pos] {
			pos++
			if pos == len(want) {
				return true
			}
		}
	}
	return false
}

func stringArg(args map[string]any, key string) (string, bool) {
	if args == nil {
		return "", false
	}
	v, ok := args[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok && strings.TrimSpace(s) != ""
}

func stringSliceArg(args map[string]any, key string) (string, bool) {
	if args == nil {
		return "", false
	}
	v, ok := args[key]
	if !ok {
		return "", false
	}
	switch xs := v.(type) {
	case []string:
		if len(xs) == 0 {
			return "", false
		}
		return strings.Join(xs, " "), true
	case []any:
		parts := make([]string, 0, len(xs))
		for _, x := range xs {
			s, ok := x.(string)
			if !ok {
				return "", false
			}
			parts = append(parts, s)
		}
		if len(parts) == 0 {
			return "", false
		}
		return strings.Join(parts, " "), true
	default:
		return "", false
	}
}

func boolArg(args map[string]any, key string) (bool, bool) {
	if args == nil {
		return false, false
	}
	v, ok := args[key]
	if !ok {
		return false, false
	}
	b, ok := v.(bool)
	return b, ok
}

func argsWithoutConfirmation(args map[string]any) map[string]any {
	if len(args) == 0 {
		return map[string]any{}
	}
	out := make(map[string]any, len(args))
	for k, v := range args {
		if isConfirmationKey(k) {
			continue
		}
		out[k] = v
	}
	return out
}

// argsForToken canonicalizes a call for the confirm-token hash: it drops both
// the confirmation keys (so echoing _fak_confirm does not change the token) and
// the incidental annotation keys (so a reworded description does not either).
// The result binds the token to exactly the effect-bearing arguments. This is
// strictly narrower than the dispatch-time strip (argsWithoutConfirmation, which
// keeps the description so the tool still sees it): only the token hash ignores
// annotations, never dispatch.
func argsForToken(args map[string]any) map[string]any {
	if len(args) == 0 {
		return map[string]any{}
	}
	out := make(map[string]any, len(args))
	for k, v := range args {
		if isConfirmationKey(k) || isIncidentalTokenKey(k) {
			continue
		}
		// Cosmetic whitespace in the action-bearing command text (the model
		// re-indenting or padding runs of spaces between the SAME words) rotated
		// the nonce just like a reworded description did, so normalize it here —
		// the second half of the #2777 convergence fix. Only command-bearing
		// string keys are touched; every other argument is hashed verbatim.
		if s, ok := v.(string); ok && isCommandBearingKey(k) {
			out[k] = normalizeCommandForToken(s)
			continue
		}
		out[k] = v
	}
	return out
}

// commandBearingTokenKeys are the argument keys whose value IS the action-bearing
// command text (mirrors commandText's string-key list). Only these are whitespace-
// normalized in the confirm token; a benign non-command field that happens to
// carry incidental spacing is left byte-exact.
var commandBearingTokenKeys = map[string]bool{
	"command": true, "cmd": true, "shell": true, "script": true,
}

func isCommandBearingKey(k string) bool {
	return commandBearingTokenKeys[strings.ToLower(k)]
}

// normalizeCommandForToken collapses cosmetic whitespace in a command for the
// confirm-token hash: line-continuation reflows are unwrapped, leading/trailing
// whitespace is trimmed, and every run of UNQUOTED spaces/tabs between tokens
// becomes a single space, while whitespace INSIDE a quoted span is preserved
// byte-for-byte. Collapsing only unquoted runs is semantically faithful — the
// shell already word-splits on unquoted spacing, so `rm  foo` and `rm foo` are
// the same action — and cannot let two different commands collide, because quoted
// content (the only whitespace the shell keeps) is left untouched. This is why
// the fix does not weaken the token's binding while still letting a reflowed
// re-proposal converge (#2777).
func normalizeCommandForToken(cmd string) string {
	return strings.Join(quoteAwareFields(foldLineContinuations(cmd)), " ")
}

// foldLineContinuations deletes shell line-continuation sequences — a backslash
// immediately followed by a newline (LF or CRLF) OUTSIDE a quoted span — exactly
// as the shell does when it joins the wrapped lines, so a long command re-proposed
// wrapped across lines normalizes to its unwrapped form for the confirm token. A
// backslash-newline INSIDE quotes is literal content and is left intact, and a
// BARE unquoted newline — a command SEPARATOR, not whitespace — is never touched
// (folding it would merge two commands into one and could transplant a confirm
// across distinct actions). Other backslash escapes pass through unchanged, matching
// quoteAwareFields, which reads this view next.
func foldLineContinuations(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		c := s[i]
		switch {
		case c == '\'' || c == '"':
			dollar := c == '\'' && i > 0 && s[i-1] == '$'
			end := quotedSpanEnd(s, i, dollar)
			b.WriteString(s[i:end])
			i = end
		case c == '\\' && i+1 < len(s) && s[i+1] == '\n':
			i += 2
		case c == '\\' && i+2 < len(s) && s[i+1] == '\r' && s[i+2] == '\n':
			i += 3
		case c == '\\' && i+1 < len(s):
			b.WriteString(s[i : i+2])
			i += 2
		default:
			b.WriteByte(c)
			i++
		}
	}
	return b.String()
}

// isIncidentalTokenKey reports whether an argument is incidental to the call's
// EFFECT — carried by the client for its own bookkeeping, never changing what the
// call does to the outside world. Clients re-emit these fields per turn, so folding
// them into the confirm token rotated it on every re-proposal and made the
// preview-confirm handshake unsatisfiable. Two categories qualify:
//
//   - Human-readable annotation: the free-text "description" Claude Code attaches to
//     every Bash call, or the "explanation" other harnesses use (#2777).
//   - Client supervision knobs: how long the CALLER waits ("timeout"/"timeout_ms")
//     and whether it detaches ("run_in_background"). These say how the caller
//     SUPERVISES the call, not what it does — `git push` pushes the same commits to
//     the same remote at any timeout, foreground or background — and none is read by
//     the classifier or rendered into the preview the operator acknowledges. They are
//     the residual axis of the confirm-loop (fak-private#21): the gate advertises
//     "re-propose byte-identical … the free-text description need not match", but a
//     model that re-proposed the identical outward-facing publish while nudging its timeout
//     (the ordinary reaction to a slow or timed-out call over a flaky remote bridge)
//     drew a FRESH token, so the advertised recovery could never converge and the
//     operator's repeated approval never became executable.
//
// Excluding a key here cannot weaken the floor: the token must bind to exactly the
// effect-bearing subject the preview shows, and every argument that steers the effect
// — a command's text, an MCP git_push `force`/`branch`, a `dangerouslyDisableSandbox`,
// a `workdir` — is still hashed verbatim, so a confirm can never be transplanted onto
// a materially different call. Hashing everything by default and excluding a NAMED
// incidental is deliberate: a novel effect-bearing arg binds automatically.
func isIncidentalTokenKey(k string) bool {
	switch strings.ToLower(k) {
	case "description", "explanation":
		return true
	case "timeout", "timeout_ms", "run_in_background":
		return true
	default:
		return false
	}
}

func hasConfirmationArg(args map[string]any) bool {
	for k := range args {
		if isConfirmationKey(k) {
			return true
		}
	}
	return false
}

func confirmationToken(args map[string]any) string {
	for k, v := range args {
		if !isConfirmationKey(k) {
			continue
		}
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func isConfirmationKey(k string) bool {
	switch strings.ToLower(k) {
	case ReversibilityConfirmArg, "_fak_confirm_token", "confirm_token", "confirmation_token":
		return true
	default:
		return false
	}
}

// ReasonProductionDBBlock is the policy refusal token emitted when an action targets a remote database.
const ReasonProductionDBBlock = "POLICY_BLOCK (PRODUCTION_DB_BLOCK)"

var allowedLocalDBHosts = map[string]bool{
	"localhost":            true,
	"127.0.0.1":            true,
	"::1":                  true,
	"0.0.0.0":              true,
	"host.docker.internal": true,
	"test-db":              true,
	"postgres":             true,
	"mysql":                true,
	"mariadb":              true,
	"redis":                true,
	"db":                   true,
	"database":             true,
}

var (
	dbURLRE    = regexp.MustCompile(`(?i)\b(postgres|postgresql|mysql|mariadb|redis|rediss|mongodb|clickhouse)://([^\s"']+)`)
	cliHostRE  = regexp.MustCompile(`(?i)(?:^|[\s;&|])(?:psql|mysql|mariadb|redis-cli|mongosh|clickhouse-client)\b.*?(?:-h\s+|--host[=\s]+)([^\s;&|]+)`)
	dbTargetRE = regexp.MustCompile(`(?i)\bDATABASE_URL=([^\s]+)`)
)

// ClassifyDatabaseDestination inspects tool arguments or command text for database connection endpoints.
// If a non-local/production destination is detected, it returns the target host, true, and the refusal reason.
func ClassifyDatabaseDestination(tool string, args map[string]any) (string, bool, string) {
	for _, key := range []string{"database_url", "db_url", "connection_string", "url", "endpoint"} {
		if val, ok := args[key]; ok {
			if s, ok := val.(string); ok && s != "" {
				if host, blocked := checkDBConnectionString(s); blocked {
					return host, true, ReasonProductionDBBlock
				}
			}
		}
	}
	for _, key := range []string{"host", "hostname", "db_host"} {
		if val, ok := args[key]; ok {
			if s, ok := val.(string); ok && s != "" {
				if !isLocalDBHost(s) {
					return s, true, ReasonProductionDBBlock
				}
			}
		}
	}
	cmd := commandText(args)
	if cmd == "" {
		return "", false, ""
	}
	if m := dbTargetRE.FindStringSubmatch(cmd); len(m) > 1 {
		if host, blocked := checkDBConnectionString(m[1]); blocked {
			return host, true, ReasonProductionDBBlock
		}
	}
	for _, m := range dbURLRE.FindAllStringSubmatch(cmd, -1) {
		if len(m) > 0 {
			if host, blocked := checkDBConnectionString(m[0]); blocked {
				return host, true, ReasonProductionDBBlock
			}
		}
	}
	if m := cliHostRE.FindStringSubmatch(cmd); len(m) > 1 {
		host := strings.Trim(m[1], `"'`)
		if host != "" && !isLocalDBHost(host) {
			return host, true, ReasonProductionDBBlock
		}
	}
	return "", false, ""
}

func isLocalDBHost(host string) bool {
	h := strings.ToLower(strings.TrimSpace(host))
	if idx := strings.LastIndex(h, ":"); idx != -1 {
		if !strings.Contains(h[:idx], ":") || strings.HasPrefix(h, "[") {
			h = strings.Trim(h[:idx], "[]")
		}
	}
	return allowedLocalDBHosts[h]
}

func checkDBConnectionString(raw string) (string, bool) {
	idx := strings.Index(raw, "://")
	if idx == -1 {
		return "", false
	}
	rest := raw[idx+3:]
	if at := strings.LastIndex(rest, "@"); at != -1 {
		rest = rest[at+1:]
	}
	if slash := strings.Index(rest, "/"); slash != -1 {
		rest = rest[:slash]
	}
	if q := strings.Index(rest, "?"); q != -1 {
		rest = rest[:q]
	}
	h := rest
	if colon := strings.LastIndex(h, ":"); colon != -1 {
		h = h[:colon]
	}
	if h == "" {
		return "", false
	}
	if !isLocalDBHost(h) {
		return h, true
	}
	return h, false
}
