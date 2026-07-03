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
	if commandOutwardFacing(cmd) || toolOutwardFacing(tool) {
		return ReversibilityOutwardFacing, dryRunHint(tool, cmd)
	}
	if commandIrreversible(cmd) || toolIrreversible(tool) {
		return ReversibilityIrreversible, dryRunHint(tool, cmd)
	}
	return ReversibilityReversible, ""
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

func commandOutwardFacing(cmd string) bool {
	segs := commandSegments(cmd)
	lower := strings.ToLower(cmd)
	switch {
	case segmentHeadIs(segs, "slack", "sendmail", "mail", "mutt"):
		return true
	case strings.Contains(lower, "webhook"):
		return true
	case segmentHasPrefix(segs, "git", "push"):
		return true
	case segmentHasPrefix(segs, "docker", "push"), segmentHasPrefix(segs, "npm", "publish"),
		segmentHasPrefix(segs, "cargo", "publish"), segmentHasPrefix(segs, "gem", "push"),
		segmentHasPrefix(segs, "twine", "upload"):
		return true
	case segmentHasPrefix(segs, "gh", "issue", "create"), segmentHasPrefix(segs, "gh", "issue", "comment"),
		segmentHasPrefix(segs, "gh", "issue", "edit"), segmentHasPrefix(segs, "gh", "issue", "close"),
		segmentHasPrefix(segs, "gh", "issue", "reopen"), segmentHasPrefix(segs, "gh", "pr", "create"),
		segmentHasPrefix(segs, "gh", "pr", "comment"), segmentHasPrefix(segs, "gh", "pr", "merge"),
		segmentHasPrefix(segs, "gh", "release", "create"), segmentHasPrefix(segs, "gh", "release", "upload"):
		return true
	case curlWrites(cmd), httpieWrites(segs), ghAPIWrites(segs):
		return true
	default:
		return false
	}
}

func commandIrreversible(cmd string) bool {
	segs := commandSegments(cmd)
	words := commandWords(cmd)
	lower := strings.ToLower(cmd)
	switch {
	case segmentHeadIs(segs, "rm", "rmdir", "del", "erase", "shred", "truncate", "mkfs", "remove-item"):
		return true
	case segmentHasPrefix(segs, "git", "clean"), segmentHasPrefix(segs, "git", "reset", "hard"):
		return true
	case segmentHasPrefix(segs, "terraform", "destroy"), segmentHasPrefix(segs, "kubectl", "delete"):
		return true
	// Payload scans stay whole-command on purpose: SQL statements and dd
	// targets arrive as arguments to a client binary, never as the head.
	case orderedWords(words, "drop", "database"), orderedWords(words, "drop", "table"):
		return true
	case containsWord(words, "dd") && strings.Contains(lower, "of=/dev/"):
		return true
	default:
		return false
	}
}

func toolOutwardFacing(tool string) bool {
	lower := strings.ToLower(tool)
	if strings.Contains(lower, "slack") || strings.Contains(lower, "webhook") ||
		strings.Contains(lower, "publish") || strings.Contains(lower, "upload") ||
		strings.Contains(lower, "send_email") || strings.Contains(lower, "sendemail") ||
		strings.Contains(lower, "email") || strings.Contains(lower, "send_mail") ||
		strings.Contains(lower, "post_message") || strings.Contains(lower, "git_push") ||
		strings.Contains(lower, "create_issue") || strings.Contains(lower, "issue_create") ||
		strings.Contains(lower, "create_pr") || strings.Contains(lower, "pr_create") {
		return true
	}
	return false
}

func toolIrreversible(tool string) bool {
	lower := strings.ToLower(tool)
	return strings.Contains(lower, "delete") || strings.Contains(lower, "remove") ||
		strings.Contains(lower, "destroy") || strings.Contains(lower, "truncate") ||
		strings.Contains(lower, "unlink") || strings.Contains(lower, "rmdir")
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

func ghAPIWrites(segs [][]string) bool {
	for _, seg := range segs {
		if len(seg) < 2 || seg[0] != "gh" || seg[1] != "api" {
			continue
		}
		for i, w := range seg {
			if (w == "x" || w == "method") && i+1 < len(seg) && httpWriteVerb(seg[i+1]) {
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

func dryRunHint(tool, cmd string) string {
	lowerTool, lowerCmd := strings.ToLower(tool), strings.ToLower(cmd)
	switch {
	case strings.Contains(lowerCmd, "git push") || strings.Contains(lowerTool, "git_push"):
		return "try git push --dry-run first"
	case strings.Contains(lowerCmd, "npm publish"):
		return "try npm publish --dry-run first"
	default:
		return ""
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
