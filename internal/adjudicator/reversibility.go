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

// ReversibilityConfirmToken derives the stable preview token. Confirmation keys
// AND incidental annotation keys (a Bash call's free-text "description") are
// excluded from the hashed payload, so the token binds only to the call's
// effect-bearing arguments and is stable across byte-identical *commands* even
// when the client's confirm key or prose annotation drifts between proposals.
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
	cmd := commandText(args)
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
// `gh` is deliberately NOT a family here (operator decision, 2026-07-05). Every gh
// write — issue/pr/release create·comment·edit·close·reopen·merge·upload and `gh api`
// mutations — targets the operator's OWN authenticated GitHub and is reversible in
// practice (issues/PRs edit·close·reopen; a release can be deleted), so the
// preview-confirm pause was pure friction on routine fleet work — the #2650/#2651
// confirm-loop lesson — while the Claude Code allow-list already admits `Bash(gh …)`.
// curl/httpie POSTs stay gated (http-write below): those reach ARBITRARY hosts, not
// the authenticated gh surface, so the outbound floor still holds for the general case.
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
	// `git clean -n` (the short spelling of --dry-run) previews without deleting,
	// even combined with -d/-f (e.g. "git clean -nd", the fs-destroy/git-destroy
	// redirect). -n is recognized ONLY inside a `git clean` segment, never
	// globally: a bare -n means unrelated things elsewhere (`kubectl delete -n
	// <ns>` is a REAL delete that must stay gated).
	if gitCleanDryRun(cmd) {
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

// gitCleanDryRun reports whether a `git clean` invocation carries a dry-run
// flag: --dry-run, or a single-dash short-flag cluster containing 'n' (-n, -nd,
// -dn, -nf). A cluster WITHOUT 'n' (-fd, -f, -x) is a real removal and is NOT a
// dry-run, so it stays gated. It parses the RAW command (dash-preserving),
// splitting on the same sequencing operators and stripping the same env/wrapper
// heads as commandSegments, but keeping flag dashes that commandWords drops.
func gitCleanDryRun(cmd string) bool {
	for _, raw := range commandSegmentRE.Split(cmd, -1) {
		fields := strings.Fields(raw)
		for len(fields) > 0 &&
			(envAssignmentRE.MatchString(fields[0]) || commandWrapperHeads[strings.ToLower(fields[0])]) {
			fields = fields[1:]
		}
		if len(fields) < 2 || strings.ToLower(fields[0]) != "git" || strings.ToLower(fields[1]) != "clean" {
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
		if words := commandWords(strings.Join(fields, " ")); len(words) > 0 {
			segs = append(segs, words)
		}
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
		out[k] = v
	}
	return out
}

// isIncidentalTokenKey reports whether an argument is a pure human-readable
// annotation with no execution effect — the free-text "description" Claude Code
// attaches to every Bash call, or the "explanation" other harnesses use. Clients
// regenerate these each turn, so folding them into the confirm token rotated it
// on every re-proposal and made the preview-confirm handshake unsatisfiable.
func isIncidentalTokenKey(k string) bool {
	switch strings.ToLower(k) {
	case "description", "explanation":
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
