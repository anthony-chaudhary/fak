// guard.go — the pure, interpreter-free core of the repo-guard PreToolUse hook.
//
// This is a faithful Go port of tools/repo_guard.py. It exists because the hook
// fires on EVERY Bash/Write/Edit tool call, and the Python original re-spawned a
// fresh interpreter each time — a per-decision subprocess on the live request
// path, exactly the boundary DIRECTION.md says must stay Go-only and
// interpreter-free (architest TestHotPathHasNoExec). Compiled to one binary, the
// guard runs with no Python spawn.
//
// The semantics mirror the Python byte-for-byte (the ported test table in
// guard_test.go is the witness): resolve every destructive/write target against
// the workspace root and refuse the ones that land OUTSIDE it (and outside an
// allow-listed scratch root — the OS temp dirs, ~/.cache, the agent's own
// ~/.claude state tree, and the workspace's <ws>-private companion). Pure: no
// filesystem access in the classification core (only the context helpers stat),
// so it is hermetically testable.
package repoguard

import (
	"os"
	"slices"
	"sort"
	"strings"
)

const (
	guardSchema = "fak-repo-guard/1"
	guardReason = "OUT_OF_TREE_WRITE"

	// Schema is the JSON protocol version emitted by the repo-guard check surface.
	Schema = guardSchema
	// Reason is the structured refusal token for out-of-workspace effects.
	Reason = guardReason
)

// Verb classes — kept identical to tools/repo_guard.py.
var (
	// deleteVerbs: every non-flag operand is a DELETE target.
	deleteVerbs = setOf("rm", "rmdir", "unlink", "shred", "trash", "trash-put")
	// destVerbs: the LAST non-flag operand is a WRITE destination.
	destVerbs = setOf("cp", "mv", "install", "rsync", "ln")
	// writeVerbs: every non-flag operand is a WRITE target.
	writeVerbs = setOf("tee", "truncate")
	// outputFlags: --output/-out almost always name a real output file.
	outputFlags = setOf("--output", "-out")
	// buildVerbs: the verbs for which a bare -o names an output file.
	buildVerbs = setOf("go", "gcc", "g++", "cc", "clang", "clang++", "ld", "rustc", "gccgo", "tcc", "zig")
	// nullDevices: the POSIX null / std-stream device sinks. Writing or redirecting
	// to one of these can never harm a sibling repo, so they are exempt even though
	// they resolve outside the workspace — otherwise the universal `... > /dev/null`
	// idiom trips the guard and pushes an operator to disable the WHOLE gate. (Windows
	// `NUL` resolves relative-to-cwd and is already in-tree.)
	nullDevices = setOf(
		"/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom",
		"/dev/stdout", "/dev/stderr", "/dev/tty",
	)
)

func setOf(items ...string) map[string]bool {
	m := make(map[string]bool, len(items))
	for _, s := range items {
		m[s] = true
	}
	return m
}

// Violation is one out-of-tree finding. The JSON tags match the Python dict keys
// so a consumer (or a parity diff against the Python --json output) sees the same
// shape.
type Violation struct {
	Reason   string `json:"reason"`
	Op       string `json:"op"`
	Target   string `json:"target"`
	Resolved string `json:"resolved"`
	Why      string `json:"why"`
}

// target is an (op, raw-path) pair extracted from a command.
type target struct {
	op  string
	raw string
}

// --------------------------------------------------------------------------- //
// Path normalization — the load-bearing primitive (git-bash + Windows aware)
// --------------------------------------------------------------------------- //

// normalize maps a path string to forward-slash form with an upper-case drive.
// Mirrors repo_guard.normalize: strip surrounding quotes, backslash->slash, then
// MSYS /c/ -> C:/ and a lower drive letter -> upper. Pure string work (not
// resolved against the filesystem).
func normalize(path string) string {
	p := strings.TrimSpace(path)
	p = strings.Trim(p, "\"")
	p = strings.Trim(p, "'")
	p = strings.ReplaceAll(p, "\\", "/")
	// MSYS drive form: /c/Users/... -> C:/Users/...
	if len(p) >= 3 && p[0] == '/' && p[2] == '/' && isAlpha(p[1]) {
		p = strings.ToUpper(p[1:2]) + ":" + p[2:]
	}
	// Upper-case a leading drive letter: c:/x -> C:/x
	if len(p) >= 2 && p[1] == ':' && isAlpha(p[0]) {
		p = strings.ToUpper(p[0:1]) + p[1:]
	}
	return p
}

func isAlpha(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

func isAbsolute(p string) bool {
	return strings.HasPrefix(p, "/") || (len(p) >= 2 && p[1] == ':')
}

// toAbs resolves raw (a path operand) to a normalized absolute path string,
// relative to normalized base. The bool is false for an UNRESOLVABLE target
// (shell variable / glob / command substitution / ~), so the caller falls back to
// a conservative textual check instead of guessing.
func toAbs(raw, base string) (string, bool) {
	if raw == "" {
		return "", false
	}
	if strings.ContainsAny(raw, "$*?`~") {
		return "", false
	}
	n := normalize(raw)
	b := normalize(base)
	var joined string
	if isAbsolute(n) {
		joined = n
	} else {
		joined = strings.TrimRight(b, "/") + "/" + n
	}
	var parts []string
	for _, seg := range strings.Split(joined, "/") {
		switch seg {
		case "", ".":
			continue
		case "..":
			if len(parts) > 0 && parts[len(parts)-1] != ".." {
				parts = parts[:len(parts)-1]
			}
			continue
		default:
			parts = append(parts, seg)
		}
	}
	// Preserve drive (C:) or leading slash.
	if len(joined) >= 2 && joined[1] == ':' {
		if len(parts) > 1 {
			return parts[0] + "/" + strings.Join(parts[1:], "/"), true
		}
		return parts[0] + "/", true
	}
	return "/" + strings.Join(parts, "/"), true
}

// isUnder reports whether normalized-absolute child is parent or below it. Mirrors
// repo_guard.is_under (PurePosixPath semantics): drive letters are ordinary path
// components, only a leading "/" is "absolute", and ".." is NOT resolved here
// (callers pass already-resolved paths). No filesystem access.
func isUnder(child, parent string) bool {
	if child == "" || parent == "" {
		return false
	}
	cAbs, cParts := splitPosix(child)
	pAbs, pParts := splitPosix(parent)
	if cAbs != pAbs {
		return false
	}
	if len(pParts) == len(cParts) {
		return slices.Equal(cParts, pParts) // child == parent
	}
	if len(pParts) < len(cParts) { // parent is an ancestor of child
		return slices.Equal(cParts[:len(pParts)], pParts)
	}
	return false
}

func splitPosix(s string) (bool, []string) {
	abs := strings.HasPrefix(s, "/")
	var parts []string
	for _, seg := range strings.Split(s, "/") {
		if seg == "" || seg == "." {
			continue
		}
		parts = append(parts, seg)
	}
	return abs, parts
}

// --------------------------------------------------------------------------- //
// Pure core: extract write/delete targets from a command, classify each
// --------------------------------------------------------------------------- //

// splitSegments splits a compound command into simple-command segments on
// ; | && || & and newlines, without a full parser. Mirrors repo_guard._split_segments.
func splitSegments(command string) []string {
	var out []string
	var cur strings.Builder
	r := []rune(command)
	i, n := 0, len(r)
	for i < n {
		if i+1 < n {
			two := string(r[i : i+2])
			if two == "||" || two == "&&" {
				out = append(out, cur.String())
				cur.Reset()
				i += 2
				continue
			}
		}
		ch := r[i]
		if ch == ';' || ch == '|' || ch == '&' || ch == '\n' {
			out = append(out, cur.String())
			cur.Reset()
			i++
			continue
		}
		cur.WriteRune(ch)
		i++
	}
	out = append(out, cur.String())
	res := out[:0]
	for _, s := range out {
		if strings.TrimSpace(s) != "" {
			res = append(res, s)
		}
	}
	return res
}

// shlexSplit mirrors Python's shlex.split(s, posix=True) closely enough for
// command-target extraction. The bool is false on an unbalanced quote or a
// dangling escape (Python raises ValueError there and the caller falls back to a
// plain whitespace split).
func shlexSplit(s string) ([]string, bool) {
	var tokens []string
	var cur []rune
	started := false // cur is a real token (incl. an empty quoted "")
	r := []rune(s)
	i, n := 0, len(r)
	for i < n {
		c := r[i]
		switch {
		case c == '\\': // outside quotes: escape any next char
			if i+1 >= n {
				return nil, false // dangling escape
			}
			cur = append(cur, r[i+1])
			started = true
			i += 2
		case c == '\'': // single quote: literal until next '
			j := i + 1
			for j < n && r[j] != '\'' {
				j++
			}
			if j >= n {
				return nil, false // no closing quote
			}
			cur = append(cur, r[i+1:j]...)
			started = true
			i = j + 1
		case c == '"': // double quote: until next ", \ escapes only " and \
			j := i + 1
			closed := false
			for j < n {
				if r[j] == '\\' && j+1 < n && (r[j+1] == '"' || r[j+1] == '\\') {
					cur = append(cur, r[j+1])
					j += 2
					continue
				}
				if r[j] == '"' {
					closed = true
					break
				}
				cur = append(cur, r[j])
				j++
			}
			if !closed {
				return nil, false // no closing quote
			}
			started = true
			i = j + 1
		case c == ' ' || c == '\t' || c == '\r' || c == '\n':
			if started {
				tokens = append(tokens, string(cur))
				cur = cur[:0]
				started = false
			}
			i++
		default:
			cur = append(cur, c)
			started = true
			i++
		}
	}
	if started {
		tokens = append(tokens, string(cur))
	}
	return tokens, true
}

// basename returns the final path component, splitting on both / and \ so the
// verb of `/usr/bin/rm` or `C:\tools\go.exe` resolves the same on any OS.
func basename(p string) string {
	if i := strings.LastIndexAny(p, "/\\"); i >= 0 {
		return p[i+1:]
	}
	return p
}

// extractTargets returns the write/delete targets a command acts on. Mirrors
// repo_guard.extract_targets.
func extractTargets(command string) []target {
	var targets []target
	for _, seg := range splitSegments(command) {
		// Redirections: > file / >> file (skip >&2; >/dev/null is handled by the
		// scratch allow-list downstream). Scanned on the raw whitespace split.
		toksRaw := strings.Fields(seg)
		for j, t := range toksRaw {
			switch {
			case (t == ">" || t == ">>") && j+1 < len(toksRaw):
				targets = append(targets, target{"redirect", toksRaw[j+1]})
			case strings.HasPrefix(t, ">") && len(t) > 1 && !strings.HasPrefix(t, ">&"):
				targets = append(targets, target{"redirect", strings.TrimLeft(t, ">")})
			}
		}
		toks, ok := shlexSplit(seg)
		if !ok {
			toks = toksRaw
		}
		if len(toks) == 0 {
			continue
		}
		verb := basename(toks[0])
		operands := toks[1:]
		// -o / --output PATH (and --output=PATH); -o only for build verbs.
		for k, t := range operands {
			switch {
			case outputFlags[t] && k+1 < len(operands):
				targets = append(targets, target{"output-flag", operands[k+1]})
			case t == "-o" && buildVerbs[verb] && k+1 < len(operands):
				targets = append(targets, target{"output-flag", operands[k+1]})
			case strings.HasPrefix(t, "--output="):
				targets = append(targets, target{"output-flag", strings.SplitN(t, "=", 2)[1]})
			case strings.HasPrefix(t, "of=") && verb == "dd":
				targets = append(targets, target{"dd", strings.SplitN(t, "=", 2)[1]})
			}
		}
		var nonFlags []string
		for _, t := range operands {
			if !strings.HasPrefix(t, "-") {
				nonFlags = append(nonFlags, t)
			}
		}
		switch {
		case deleteVerbs[verb], writeVerbs[verb]:
			for _, t := range nonFlags {
				targets = append(targets, target{verb, t})
			}
		case destVerbs[verb] && len(nonFlags) > 0:
			targets = append(targets, target{verb, nonFlags[len(nonFlags)-1]}) // destination is last
		}
	}
	return targets
}

// textualEscape is a conservative escape signal for an UNRESOLVABLE target: it
// literally starts a parent traversal or names an absolute path. Used only when
// toAbs cannot resolve. Mirrors repo_guard._textual_escape.
func textualEscape(raw string) bool {
	n := normalize(raw)
	return strings.HasPrefix(n, "../") || strings.Contains(n, "/../") || isAbsolute(n)
}

// classifyCommand returns violations for destructive/write targets that escape the
// workspace into a non-scratch location. Pure: no filesystem access.
func classifyCommand(command, workspaceRoot string, safeRoots []string) []Violation {
	ws := normalize(workspaceRoot)
	safe := make([]string, len(safeRoots))
	for i, s := range safeRoots {
		safe[i] = normalize(s)
	}
	var violations []Violation
	for _, tg := range extractTargets(command) {
		absTarget, ok := toAbs(tg.raw, ws)
		if !ok {
			if textualEscape(tg.raw) {
				violations = append(violations, violation(tg.op, tg.raw, "<unresolved>", "parent/absolute escape"))
			}
			continue
		}
		if isUnder(absTarget, ws) {
			continue // in-repo
		}
		if nullDevices[absTarget] {
			continue // the null / std-stream device sinks — harmless, never a sibling
		}
		if underAny(absTarget, safe) {
			continue // scratch
		}
		violations = append(violations, violation(tg.op, tg.raw, absTarget, whyOutside(absTarget, ws)))
	}
	return violations
}

// ClassifyCommand returns out-of-tree write/delete violations for a shell command.
func ClassifyCommand(command, workspaceRoot string, safeRoots []string) []Violation {
	return classifyCommand(command, workspaceRoot, safeRoots)
}

// classifyWritePath is the same idea for a Write/Edit/NotebookEdit file_path.
func classifyWritePath(filePath, workspaceRoot string, safeRoots []string) []Violation {
	ws := normalize(workspaceRoot)
	absTarget, ok := toAbs(filePath, ws)
	if !ok {
		if textualEscape(filePath) {
			return []Violation{violation("write", filePath, "<unresolved>", "parent/absolute escape")}
		}
		return nil
	}
	if isUnder(absTarget, ws) {
		return nil
	}
	if nullDevices[absTarget] {
		return nil // the null / std-stream device sinks — harmless, never a sibling
	}
	for _, s := range safeRoots {
		if isUnder(absTarget, normalize(s)) {
			return nil
		}
	}
	return []Violation{violation("write", filePath, absTarget, whyOutside(absTarget, ws))}
}

// ClassifyWritePath returns out-of-tree violations for a direct Write/Edit file path.
func ClassifyWritePath(filePath, workspaceRoot string, safeRoots []string) []Violation {
	return classifyWritePath(filePath, workspaceRoot, safeRoots)
}

func underAny(absTarget string, roots []string) bool {
	for _, s := range roots {
		if isUnder(absTarget, s) {
			return true
		}
	}
	return false
}

func whyOutside(absTarget, ws string) string {
	if isSibling(absTarget, ws) {
		return "sibling of workspace"
	}
	return "outside workspace"
}

func isSibling(absTarget, ws string) bool {
	parent := posixParent(ws)
	return isUnder(absTarget, parent) && !isUnder(absTarget, ws)
}

// posixParent returns the parent path the way PurePosixPath(ws).parent would.
func posixParent(p string) string {
	abs, parts := splitPosix(p)
	if len(parts) > 0 {
		parts = parts[:len(parts)-1]
	}
	if abs {
		return "/" + strings.Join(parts, "/")
	}
	if len(parts) == 0 {
		return "."
	}
	return strings.Join(parts, "/")
}

func violation(op, raw, resolved, why string) Violation {
	return Violation{Reason: guardReason, Op: op, Target: raw, Resolved: resolved, Why: why}
}

// --------------------------------------------------------------------------- //
// Context (workspace root + scratch allow-list) — minimal stat-walk, no spawns
// --------------------------------------------------------------------------- //

// findRepoRoot walks up from start to the nearest dir containing .git; falls back
// to start. No subprocess (a stat walk, not a `git` spawn). Mirrors
// repo_guard.find_repo_root.
func findRepoRoot(start string) string {
	cur := normalize(start)
	for _, cand := range append([]string{cur}, posixParents(cur)...) {
		if _, err := os.Stat(cand + "/.git"); err == nil {
			return cand
		}
	}
	return cur
}

// FindRepoRoot walks up from start to the nearest directory containing .git.
func FindRepoRoot(start string) string {
	return findRepoRoot(start)
}

// posixParents yields the ancestors of p the way PurePosixPath(p).parents does
// (including a trailing "." for a relative/drive path).
func posixParents(p string) []string {
	abs, parts := splitPosix(p)
	var out []string
	for len(parts) > 0 {
		parts = parts[:len(parts)-1]
		if abs {
			out = append(out, "/"+strings.Join(parts, "/"))
		} else if len(parts) == 0 {
			out = append(out, ".")
		} else {
			out = append(out, strings.Join(parts, "/"))
		}
	}
	return out
}

// isAgentStateDir reports whether a home-level entry is the agent's own Claude
// Code state tree: ".claude", ".claude-<acct>", or ".claude.<x>". A STRUCTURED
// match, not a loose prefix — ".claudex" is some other dir and must NOT match.
func isAgentStateDir(name string) bool {
	return name == ".claude" || strings.HasPrefix(name, ".claude-") || strings.HasPrefix(name, ".claude.")
}

// agentStateRoots returns the agent's own state trees under home — ~/.claude plus
// any per-account variant present. entries is injectable for tests; the default is
// a stat-walk listing of home. ~/.claude is always included even if absent.
func agentStateRoots(home string, entries []string) []string {
	roots := []string{home + "/.claude"}
	if entries == nil {
		if listed, err := os.ReadDir(home); err == nil {
			for _, e := range listed {
				entries = append(entries, e.Name())
			}
		}
	}
	names := append([]string(nil), entries...)
	sort.Strings(names)
	for _, name := range names {
		if isAgentStateDir(name) {
			roots = append(roots, home+"/"+name)
		}
	}
	return dedup(roots)
}

// AgentStateRoots returns the agent-owned state roots under home.
func AgentStateRoots(home string, entries []string) []string {
	return agentStateRoots(home, entries)
}

// privateCompanionRoots returns the workspace's OWN private companion repo — the
// same-named <ws>-private sibling (fak -> fak-private). Bounded ON PURPOSE: only
// the same-named "-private" sibling is admitted, never an arbitrary sibling
// project. When the workspace IS already the private repo (…-private) there is no
// further companion — return nil: a <ws>-private-private does not exist, so
// synthesizing it would silently admit an out-of-tree write to a nonexistent
// sibling (and writing back into the PUBLIC sibling is not auto-safe).
func privateCompanionRoots(workspaceRoot string) []string {
	ws := strings.TrimRight(normalize(workspaceRoot), "/")
	if ws == "" {
		return nil
	}
	if strings.HasSuffix(basename(ws), "-private") {
		return nil
	}
	return []string{ws + "-private"}
}

// PrivateCompanionRoots returns the same-named private companion root, if any.
func PrivateCompanionRoots(workspaceRoot string) []string {
	return privateCompanionRoots(workspaceRoot)
}

// defaultSafeRoots is the scratch allow-list: the OS temp dirs, ~/.cache,
// ~/Downloads, and the agent's own state tree(s). The <ws>-private companion is
// added at the call sites that know the workspace root.
func defaultSafeRoots() []string {
	home := normalize(userHome())
	roots := []string{"/tmp", "/var/tmp", home + "/.cache", home + "/Downloads"}
	roots = append(roots, agentStateRoots(home, nil)...)
	for _, v := range []string{"TMPDIR", "TEMP", "TMP"} {
		if val := os.Getenv(v); val != "" {
			roots = append(roots, normalize(val))
		}
	}
	return dedup(roots)
}

// DefaultSafeRoots returns the standard scratch and agent-state roots.
func DefaultSafeRoots() []string {
	return defaultSafeRoots()
}

// SafeRootsForWorkspace returns the standard safe roots plus the workspace companion.
func SafeRootsForWorkspace(workspaceRoot string) []string {
	return append(defaultSafeRoots(), privateCompanionRoots(workspaceRoot)...)
}

func userHome() string {
	if h, err := os.UserHomeDir(); err == nil {
		return h
	}
	return ""
}

func dedup(in []string) []string {
	seen := make(map[string]bool, len(in))
	out := in[:0]
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

// --------------------------------------------------------------------------- //
// Evaluate one tool call (used by both --check and --hook)
// --------------------------------------------------------------------------- //

func evaluate(toolName string, toolInput map[string]any, workspaceRoot string, safeRoots []string) []Violation {
	switch toolName {
	case "Bash":
		return classifyCommand(stringField(toolInput, "command"), workspaceRoot, safeRoots)
	case "Write", "Edit", "MultiEdit", "NotebookEdit":
		fp := stringField(toolInput, "file_path")
		if fp == "" {
			fp = stringField(toolInput, "notebook_path")
		}
		return classifyWritePath(fp, workspaceRoot, safeRoots)
	}
	return nil
}

// Evaluate classifies one tool call using the repo-guard structural rules.
func Evaluate(toolName string, toolInput map[string]any, workspaceRoot string, safeRoots []string) []Violation {
	return evaluate(toolName, toolInput, workspaceRoot, safeRoots)
}

func stringField(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func renderReason(violations []Violation) string {
	parts := make([]string, len(violations))
	for i, v := range violations {
		parts[i] = v.Op + " -> " + v.Target + " (" + v.Why + ": " + v.Resolved + ")"
	}
	return guardReason + ": a destructive/write op targets a path OUTSIDE this repo. " +
		strings.Join(parts, "; ") +
		". Operate inside the workspace, or write scratch to a temp dir. " +
		"If this is intentional, re-run with FAK_REPO_GUARD=warn (advisory) or off."
}

// RenderReason formats the human-readable denial message for violations.
func RenderReason(violations []Violation) string {
	return renderReason(violations)
}
