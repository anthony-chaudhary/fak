// searchhazard.go — the recursive-search-at-a-broad-root rung of the
// repo-guard PreToolUse hook (the grep-spam root cause, 2026-11-07).
//
// A recursive `grep -r` / `find` issued from the repo root crawls the
// gitignored `.worktrees/` peer-checkout farm and `_scratch` — grep and find do
// NOT honor .gitignore, so the walk is the ~787k-file workspace (19.7M files
// with ignore off, 25x). The Bash tool times out at <=10 min but does not kill
// the grep child, and Git Bash on Windows does not reliably propagate the kill
// to grandchildren, so the search orphans and runs for hours, saturating the
// host at 97-99% CPU and stalling publish/land gates on deadline (the 2026-11-07
// incident: 40 /usr/bin/grep processes, 21 orphans 7-9h old).
//
// This is the Claude Code Bash-command surface's counterpart of the two
// search-scoping guards that already exist on the OTHER surfaces: pi's
// rg-guard.ts (rewrites recursive grep->rg) and the private hookscope
// PreToolUse hook (rewrites the structured Glob/Grep tool's tool_input.path).
// repoguard is the guard on the Bash-command path, where a `grep -r` typed into
// the Bash tool actually runs — and it had no search-hazard rung.
//
// Curation principle (mirrors foregroundloop.go / cdmap.go): fire ONLY when it
// is provably the mistake. That is a RECURSIVE search (grep -r/-R/--recursive,
// or a find without -maxdepth) whose search root is BROAD — the workspace root
// (where the Bash tool's shell starts, so a bare `.` is the whole workspace),
// an ancestor of it, the .worktrees farm, or a drive/POSIX/home root. A
// recursive search scoped to a subdirectory (`grep -r foo internal/`) is NOT
// the mistake and is never flagged; a non-recursive grep (`grep -o ../foo
// internal/policy/x.go`) is not either. ripgrep (`rg`) is deliberately NOT
// flagged: it honors .gitignore, so `rg -r foo .` skips the .worktrees farm.
//
// The decision is an ADVISORY (SeverityWarn by default): repoguard's PreToolUse
// protocol can only advise (stderr) or deny — it cannot rewrite the command the
// way hookscope/rg-guard do. The fix-hint steers the agent to the Grep tool
// (ripgrep) or a bounded subdirectory search. Pure string work: no filesystem
// access, hermetically testable like the rest of the core.
package repoguard

import "strings"

// ReasonSearchHazard is the structured advisory token for a recursive search
// (grep -r / find) at a broad root that would crawl the .worktrees farm.
const ReasonSearchHazard = "SEARCH_HAZARD"

// ClassifySearchHazard returns SEARCH_HAZARD advisories for a shell command
// that runs a recursive search at a broad root.
func ClassifySearchHazard(command, workspaceRoot string) []Violation {
	return classifySearchHazard(command, workspaceRoot)
}

func classifySearchHazard(command, workspaceRoot string) []Violation {
	// A backgrounded search (`… &`) does not hold the turn open — the whole
	// point of the fix — so it is never the mistake this rung names.
	if hasBackgroundAmpersand(command) {
		return nil
	}
	var out []Violation
	for _, seg := range splitSegments(command) {
		verb, operands, _ := tokenizeSegment(seg)
		var v *Violation
		switch verb {
		case "grep":
			v = grepHazard(operands, workspaceRoot)
		case "find":
			v = findHazard(operands, workspaceRoot)
		}
		if v != nil {
			v.Target = strings.TrimSpace(seg)
			out = append(out, *v)
		}
	}
	return out
}

// grepHazard flags a recursive grep whose search root is broad. parseGrep
// splits the operands into (recursive, pattern, files); files are the search
// roots. A nil files means grep reads stdin (not a hazard).
func grepHazard(operands []string, workspaceRoot string) *Violation {
	recursive, _, files := parseGrep(operands)
	if !recursive || len(files) == 0 {
		return nil
	}
	for _, f := range files {
		resolved, ok := toAbs(f, workspaceRoot)
		if !ok {
			continue // unresolvable (glob / $var / ~): stay out of it
		}
		if isBroadSearchRoot(resolved, workspaceRoot) {
			return &Violation{
				Reason:   ReasonSearchHazard,
				Op:       "grep -r",
				Resolved: resolved,
				Why:      "a recursive grep at a broad root crawls the gitignored .worktrees/ peer checkouts (grep does not honor .gitignore) — the ~787k-file workspace walk saturates the host and orphans past the 10-min tool timeout",
				Fix:      "use the Grep tool (ripgrep honors .gitignore) or scope the search to a specific repo/subdirectory, e.g. grep -r <pat> internal/",
			}
		}
	}
	return nil
}

// findHazard flags a find at a broad root with no -maxdepth bound. find's
// syntax is `find [starting-point...] [expression]`; the expression begins at
// the first -flag or `(` token. A -maxdepth anywhere in the expression bounds
// the walk and clears the hazard.
func findHazard(operands []string, workspaceRoot string) *Violation {
	var roots []string
	sawMaxDepth := false
	inExpr := false
	for _, tok := range operands {
		if inExpr {
			if tok == "-maxdepth" {
				sawMaxDepth = true
			}
			continue
		}
		if strings.HasPrefix(tok, "-") || tok == "(" {
			inExpr = true
			if tok == "-maxdepth" {
				sawMaxDepth = true
			}
			continue
		}
		roots = append(roots, tok)
	}
	if sawMaxDepth || len(roots) == 0 {
		return nil
	}
	for _, r := range roots {
		resolved, ok := toAbs(r, workspaceRoot)
		if !ok {
			continue
		}
		if isBroadSearchRoot(resolved, workspaceRoot) {
			return &Violation{
				Reason:   ReasonSearchHazard,
				Op:       "find",
				Resolved: resolved,
				Why:      "a find at a broad root without -maxdepth crawls the gitignored .worktrees/ peer checkouts — the ~787k-file workspace walk saturates the host and orphans past the 10-min tool timeout",
				Fix:      "add -maxdepth to bound the walk, or scope the search to a specific repo/subdirectory, e.g. find internal -name '*.go'",
			}
		}
	}
	return nil
}

// parseGrep splits grep operands into (recursive, pattern, files). recursive is
// true when a -r/-R/--recursive flag is present. pattern is the first non-flag
// operand; files are the search roots (the operands after the pattern). A bare
// `-` (stdin) and the `--` end-of-flags marker are handled; a glob/$var file
// operand is left for toAbs to reject conservatively.
func parseGrep(operands []string) (recursive bool, pattern string, files []string) {
	sawPattern := false
	endFlags := false
	for _, tok := range operands {
		if endFlags {
			if !sawPattern {
				pattern, sawPattern = tok, true
			} else {
				files = append(files, tok)
			}
			continue
		}
		if tok == "--" {
			endFlags = true
			continue
		}
		if strings.HasPrefix(tok, "-") && tok != "-" {
			if tok == "--recursive" {
				recursive = true
			} else if !strings.HasPrefix(tok, "--") {
				// a short flag cluster: -r, -rn, -rl, -rln, -R, ...
				for _, c := range tok[1:] {
					if c == 'r' || c == 'R' {
						recursive = true
					}
				}
			}
			continue
		}
		if !sawPattern {
			pattern, sawPattern = tok, true
		} else {
			files = append(files, tok)
		}
	}
	return recursive, pattern, files
}

// isBroadSearchRoot reports whether a resolved (normalized-absolute) search
// root would crawl the .worktrees farm or the whole workspace: the .worktrees
// farm directly, the workspace root or an ancestor of it (the walk includes the
// gitignored .worktrees/ peer checkouts), or a broad drive/POSIX/home root.
func isBroadSearchRoot(resolved, workspaceRoot string) bool {
	if pathBase(resolved) == ".worktrees" {
		return true
	}
	if ws := normalize(workspaceRoot); ws != "" && isUnder(ws, resolved) {
		return true
	}
	return isBroadRoot(resolved)
}

// pathBase is the final path component after normalization.
func pathBase(p string) string {
	t := normalize(p)
	if i := strings.LastIndex(t, "/"); i >= 0 {
		return t[i+1:]
	}
	return t
}

// isBroadRoot mirrors hookscope's rule (pure, no os.UserHomeDir): an ABSOLUTE
// path with <= 1 component below a drive, POSIX root, or home (C:/work, /, ~,
// C:/Users/<me>) is broad; a relative path never is, and a specific repo
// (C:/work/fak-private, depth 2) is not either.
func isBroadRoot(raw string) bool {
	t := normalize(raw)
	if t == "" {
		return false
	}
	if n := homePrefixLen(t); n > 0 {
		return pathDepth(t[n:]) <= 1
	}
	if !(len(t) >= 2 && t[1] == ':' && isAlpha(t[0])) && !strings.HasPrefix(t, "/") {
		return false
	}
	return pathDepth(t) <= 1
}

// homePrefixLen reports the length of a home prefix (~, $HOME, %USERPROFILE%,
// ...) when t starts with one followed by a separator or end; 0 when not.
func homePrefixLen(t string) int {
	low := strings.ToLower(t)
	for _, p := range []string{"~", "$home", "${home}", "$env:userprofile", "$env:home", "%userprofile%"} {
		if !strings.HasPrefix(low, p) {
			continue
		}
		rest := t[len(p):]
		if rest == "" || rest[0] == '/' {
			return len(p)
		}
	}
	return 0
}

// pathDepth counts components below the drive/POSIX prefix.
func pathDepth(p string) int {
	cleaned := normalize(p)
	if len(cleaned) >= 2 && cleaned[1] == ':' && isAlpha(cleaned[0]) {
		cleaned = cleaned[2:] // strip the drive
	}
	cleaned = strings.Trim(cleaned, "/")
	if cleaned == "" {
		return 0
	}
	n := 0
	for _, part := range strings.Split(cleaned, "/") {
		if part != "" {
			n++
		}
	}
	return n
}

// renderSearchHazardReason formats the advisory block for SEARCH_HAZARD
// findings — the same shape as the other fix-bearing rungs.
func renderSearchHazardReason(violations []Violation) string {
	parts := make([]string, len(violations))
	for i, v := range violations {
		parts[i] = v.Op + " in `" + v.Target + "` — fix: " + v.Fix
	}
	return ReasonSearchHazard + ": a recursive search at a broad root crawls the gitignored .worktrees/ peer checkouts (grep/find do not honor .gitignore) — the ~787k-file workspace walk saturates the host and orphans past the 10-min tool timeout. " +
		strings.Join(parts, "; ") +
		". To silence this per reason set FAK_REPO_GUARD_SEVERITY=" + ReasonSearchHazard + "=record or =off; " +
		"the master switch FAK_REPO_GUARD=warn|off still overrides."
}
