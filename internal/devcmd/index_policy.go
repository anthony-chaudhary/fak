package devcmd

import (
	_ "embed"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// index policy owns the reusable repository-boundary audit that
// finds every BARE dev-tier spelling (`fak <devverb>`) still living in an in-repo
// caller surface, so the migration to the canonical `fak dev <verb>` form has a
// witness instead of a one-off grep. It belongs to fak-dev, not the serving runtime hooks.
//
// The classification authority is C1's tier table (internal/devindex.TierOf), NOT a
// second copy of the verb list: a token is a bare dev spelling iff TierOf resolves it
// (alias spellings included) to TierDev. So a new dev verb is covered the moment it is
// classified, and a frontdoor verb (`fak run`, `fak guard`) or the namespace token
// itself (`fak dev commit` — the token after `fak` is `dev`, untiered) is never flagged.
//
// SCOPE is the exact caller-surface set the issue names, root-anchored so nested
// checkout copies under `.fak/` (and any vendored `.claude/…` mirror) are pruned by the
// anchor, matching the scorecard corpus-scope lesson:
//
//	tools/*.py, tools/*.ps1   — fleet tools + scheduled-task registration scripts
//	Makefile, dos.toml        — build + command-string surfaces
//	.github/workflows/*       — CI
//	.claude/skills/*/SKILL.md — agent skill instructions
//	docs/**/*.md (code blocks only, excl. docs/archive/) — doc command examples
//
// Executable surfaces are scanned line-for-line (a bare spelling anywhere is a caller);
// docs are scanned inside fenced code blocks only (prose that NAMES the old spelling to
// explain the migration is not a caller). Anything that must legitimately keep the bare
// spelling goes on the committed allowlist (bare_dev_allowlist.txt) with a one-line reason.
//
// The gate is registered DefaultOff in HygieneGates(): it is a measurement instrument the
// migration batches run via `fak hygiene --gates BARE_DEV_SPELLING`, and it flips on (the
// one-line C5/DoD gate) once the tree is migrated to zero-outside-allowlist. Wiring it
// always-on today would red `make ci` for the whole fleet against ~1600 unmigrated sites.

// bareDevCallRE matches a bare `fak <verb>` invocation: the literal `fak` token (so a
// `$(FAK)` Makefile variable or an `unfak` word never matches) followed by whitespace and
// a verb token. The captured token is then classified by TierOf — the regex only nominates.
var bareDevCallRE = regexp.MustCompile(`\bfak[ \t]+([a-z][a-z0-9_-]*)`)

//go:embed bare_dev_allowlist.txt
var bareDevAllowlistRaw string

// bareDevAllowlist is the parsed exemption set: exact paths and directory prefixes (a
// trailing `/`) that legitimately keep a bare spelling. Membership suppresses a finding.
type bareDevAllowlist struct {
	exact    map[string]bool
	prefixes []string
}

func (a bareDevAllowlist) allows(file string) bool {
	if a.exact[file] {
		return true
	}
	for _, p := range a.prefixes {
		if strings.HasPrefix(file, p) {
			return true
		}
	}
	return false
}

// parseBareDevAllowlist reads the committed allowlist. Each non-empty, non-`#` line is
// `<path>[<space...># reason]`; the reason after `#` is documentation only. A path ending
// in `/` is a directory prefix. Blank lines and `#`-comment lines are ignored.
func parseBareDevAllowlist(raw string) bareDevAllowlist {
	a := bareDevAllowlist{exact: map[string]bool{}}
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// Drop an inline `# reason` so the entry is just the path.
		if i := strings.Index(line, "#"); i >= 0 {
			line = strings.TrimSpace(line[:i])
		}
		if line == "" {
			continue
		}
		entry := path.Clean(strings.ReplaceAll(line, "\\", "/"))
		if strings.HasSuffix(strings.TrimSpace(strings.ReplaceAll(line, "\\", "/")), "/") {
			a.prefixes = append(a.prefixes, strings.TrimSuffix(entry, "/")+"/")
			continue
		}
		a.exact[entry] = true
	}
	return a
}

// bareDevScanMode returns how an in-scope file is scanned: "full" (every line — a bare
// spelling anywhere is a caller) or "codeblock" (fenced code only — docs prose that names
// the old spelling to explain the migration is not a caller). ok=false means out of scope.
// Every case is root-anchored, so a nested checkout copy under `.fak/…` never matches.
func bareDevScanMode(p string) (mode string, ok bool) {
	if strings.HasPrefix(p, ".fak/") || strings.Contains(p, "/.fak/") {
		return "", false // vendored/checkout copy — the scorecard corpus-scope prune
	}
	switch p {
	case "Makefile", "dos.toml":
		return "full", true
	}
	if path.Dir(p) == "tools" && (strings.HasSuffix(p, ".py") || strings.HasSuffix(p, ".ps1")) {
		return "full", true
	}
	if strings.HasPrefix(p, ".github/workflows/") && (strings.HasSuffix(p, ".yml") || strings.HasSuffix(p, ".yaml")) {
		return "full", true
	}
	if strings.HasPrefix(p, "hooks/") {
		return "full", true // shell hooks, if any are tracked under hooks/
	}
	if strings.HasPrefix(p, ".claude/skills/") && strings.HasSuffix(p, "/SKILL.md") && strings.Count(p, "/") == 3 {
		return "full", true // .claude/skills/<name>/SKILL.md — the agent-facing instruction
	}
	if strings.HasPrefix(p, "docs/") && !strings.HasPrefix(p, "docs/archive/") && strings.HasSuffix(p, ".md") {
		return "codeblock", true
	}
	return "", false
}

// gateBareDevSpellingTree is the DefaultOff `fak hygiene --gates BARE_DEV_SPELLING` gate.

// bareDevSpellingFindings is the pure sweep, taking the allowlist as a parameter so the
// scope/classification/allowlist rules are unit-testable without the embedded file.
func bareDevSpellingFindings(root string, paths []string, allow bareDevAllowlist) []indexPolicyFinding {
	var findings []indexPolicyFinding
	for _, f := range paths {
		mode, ok := bareDevScanMode(f)
		if !ok || allow.allows(f) {
			continue
		}
		body, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(f)))
		if err != nil {
			continue
		}
		inFence := false
		fenceTok := ""
		for i, line := range strings.Split(string(body), "\n") {
			if mode == "codeblock" {
				if tok := fenceToken(line); tok != "" {
					if !inFence {
						inFence, fenceTok = true, tok
					} else if strings.HasPrefix(strings.TrimSpace(line), fenceTok) {
						inFence, fenceTok = false, ""
					}
					continue
				}
				if !inFence {
					continue // prose line in a docs file — not a caller
				}
			}
			for _, m := range bareDevCallRE.FindAllStringSubmatch(line, -1) {
				verb := m[1]
				if tier, known := devindex.TierOf(verb); !known || tier != devindex.TierDev {
					continue // frontdoor / unknown / the `dev` namespace token itself
				}
				findings = append(findings, indexPolicyFinding{
					Reason: "BARE_DEV_SPELLING",
					File:   f,
					Line:   i + 1,
					Detail: fmt.Sprintf("bare dev spelling %q — migrate to \"fak dev %s\" (#2233 C4), or add %q to internal/devcmd/bare_dev_allowlist.txt with a one-line reason", "fak "+verb, verb, f),
				})
			}
		}
	}
	return findings
}

// fenceToken returns the fence marker ("```" or "~~~") if the line opens/closes a fenced
// code block, else "". A fence may be indented and carry an info string after the marker.
func fenceToken(line string) string {
	s := strings.TrimSpace(line)
	switch {
	case strings.HasPrefix(s, "```"):
		return "```"
	case strings.HasPrefix(s, "~~~"):
		return "~~~"
	}
	return ""
}
