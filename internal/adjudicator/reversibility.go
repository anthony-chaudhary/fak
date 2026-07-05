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
// are excluded so re-proposing the same call with _fak_confirm does not change
// the token being checked.
func ReversibilityConfirmToken(class ReversibilityClass, tool string, args map[string]any) string {
	canon, _ := json.Marshal(argsWithoutConfirmation(args))
	sum := sha256.Sum256([]byte(string(class) + "\x00" + strings.ToLower(tool) + "\x00" + string(canon)))
	return "fak-" + hex.EncodeToString(sum[:8])
}

func classifyReversibility(tool string, args map[string]any) (ReversibilityClass, string) {
	cmd := commandText(args)
	if hasDryRunPreview(cmd, args) {
		return ReversibilityReversible, ""
	}
	return classifyAgainstFamilies(reversibilityFamilies, tool, cmd)
}

// familyMatchInput carries the precomputed views of one pending call that
// family matchers key on: the raw and lowered command, its operator-split
// segments, its whole-command word stream, and the lowered tool name.
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
	in := familyMatchInput{
		cmd:       cmd,
		lowerCmd:  strings.ToLower(cmd),
		segs:      commandSegments(cmd),
		words:     commandWords(cmd),
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
		hint:     "try npm publish --dry-run first",
	},
	// Outward families below carry no sanctioned sidestep yet (hint "").
	{
		name:  "mail",
		class: ReversibilityOutwardFacing,
		heads: []string{"sendmail", "mail", "mutt"},
	},
	{
		name:         "webhook",
		class:        ReversibilityOutwardFacing,
		cmdContains:  []string{"webhook"},
		toolContains: []string{"webhook"},
	},
	{
		name:         "registry-publish",
		class:        ReversibilityOutwardFacing,
		prefixes:     [][]string{{"docker", "push"}, {"cargo", "publish"}, {"gem", "push"}, {"twine", "upload"}},
		toolContains: []string{"publish", "upload"},
	},
	{
		name:  "http-write",
		class: ReversibilityOutwardFacing,
		matchCmd: func(in familyMatchInput) bool {
			return curlWrites(in.cmd) || httpieWrites(in.segs)
		},
	},
	{
		name:         "messaging-tool",
		class:        ReversibilityOutwardFacing,
		toolContains: []string{"send_email", "sendemail", "email", "send_mail", "post_message"},
	},
	{
		name:         "pr-create-tool",
		class:        ReversibilityOutwardFacing,
		toolContains: []string{"create_pr", "pr_create"},
	},
	{
		name:  "fs-destroy",
		class: ReversibilityIrreversible,
		heads: []string{"rm", "rmdir", "del", "erase", "shred", "truncate", "mkfs", "remove-item"},
	},
	{
		name:     "git-destroy",
		class:    ReversibilityIrreversible,
		prefixes: [][]string{{"git", "clean"}, {"git", "reset", "hard"}},
	},
	{
		name:     "infra-destroy",
		class:    ReversibilityIrreversible,
		prefixes: [][]string{{"terraform", "destroy"}, {"kubectl", "delete"}},
	},
	// Payload scans stay whole-command on purpose: SQL statements and dd
	// targets arrive as arguments to a client binary, never as the head.
	{
		name:  "sql-drop",
		class: ReversibilityIrreversible,
		matchCmd: func(in familyMatchInput) bool {
			return orderedWords(in.words, "drop", "database") || orderedWords(in.words, "drop", "table")
		},
	},
	{
		name:  "dd-device-write",
		class: ReversibilityIrreversible,
		matchCmd: func(in familyMatchInput) bool {
			return containsWord(in.words, "dd") && strings.Contains(in.lowerCmd, "of=/dev/")
		},
	},
	{
		name:         "destructive-tool",
		class:        ReversibilityIrreversible,
		toolContains: []string{"delete", "remove", "destroy", "truncate", "unlink", "rmdir"},
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
	return strings.Contains(lower, "--dry-run") || strings.Contains(lower, "--preview")
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

// commandSegments splits a shell command on sequencing/pipe operators and
// returns each non-empty segment's word list with leading env assignments
// (NAME=value) and wrapper heads (sudo, env, ...) stripped, so index 0 is the
// command actually being invoked. Family matchers anchor here instead of
// scanning the whole token stream, so trigger words inside quoted payloads
// (grep patterns, commit messages) no longer classify.
func commandSegments(cmd string) [][]string {
	var segs [][]string
	for _, raw := range commandSegmentRE.Split(cmd, -1) {
		fields := strings.Fields(raw)
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
