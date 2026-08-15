package hooks

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// gate_democommand.go — the DEMO_COMMAND gate, a byte-faithful port of tools/demo_command_audit.py.
// It is the local twin of the hosted-link guard: where that catches broken PUBLIC urls, this catches
// docs that tell a reader to run `go run ./cmd/<demo>`, `bash tools/<x>.sh`, `python tools/<x>.py`,
// or `make <target>` that no longer resolves. Static and network-free — it never executes a demo.
//
// This is a TREE-mode-ONLY gate (issue #928 A4): the Python checker has no `--audit-staged` branch —
// `make hygiene` invokes `python3 tools/demo_command_audit.py` over the whole workspace — so there is
// no staged Gate twin, only the HygieneGate wired in tree.go and run by `fak hygiene` / `make hygiene`.
// The Python checker stays on disk as the fail-open fallback (make hygiene runs it on exit 2) and as
// the parity oracle (gate_democommand_test.go), exactly like the BRAND_CONSISTENCY port.
//
// Parity anchor: tools/demo_command_audit.py collect() / _path_defect() / extract_command_refs() /
// bare_cmd_defects() / browser_demo_coverage_defects(). The default make-hygiene invocation passes no
// --source, so collect() audits default_sources (the two demo docs + every cmd/*/README.md) AND runs
// the browser-demo coverage gate against the demo registry — this gate reproduces both.

// The command-extraction regexes mirror demo_command_audit.py's module-level REs verbatim. RE2 has no
// lookahead, but none of these need it (MAKE_TARGET_RE's `(?!=)` is handled in demoMakeTargets).
var (
	demoEnvIdent  = `[A-Za-z_][A-Za-z0-9_]*`
	demoEnvValue  = `(?:"[^"]*"|'[^']*'|[^\s` + "`" + `<>]+)`
	demoTailCls   = `[^\r\n` + "`" + `<]*`
	demoEnvPrefix = `(?:(?:` + demoEnvIdent + `)=` + demoEnvValue + `\s+)*`

	// GO_CMD_RE: an optional env prefix, `go run|test|build`, an optional flag/arg run ending in
	// whitespace, then `./cmd/<name>`.
	demoGoCmdRE = regexp.MustCompile(`(?P<command>` + demoEnvPrefix +
		`go\s+(?P<verb>run|test|build)\s+(?:[^\r\n` + "`" + `<]*?\s)?\./cmd/(?P<name>[A-Za-z0-9_-]+)/?` + demoTailCls + `)`)
	// GO_C_CMD_RE: the `go -C <dir> run|test|build ... ./cmd/<name>` form.
	demoGoCCmdRE = regexp.MustCompile(`(?P<command>` + demoEnvPrefix +
		`go\s+-C\s+(?P<dir>[^\s` + "`" + `<>]+)\s+(?P<verb>run|test|build)\s+(?:[^\r\n` + "`" + `<]*?\s)?\./cmd/(?P<name>[A-Za-z0-9_-]+)/?` + demoTailCls + `)`)
	demoScriptCmdRE = regexp.MustCompile(`(?P<command>(?:bash|sh)\s+(?P<path>tools/[A-Za-z0-9_./-]+\.sh)` + demoTailCls + `)`)
	demoPyToolCmdRE = regexp.MustCompile(`(?P<command>python(?:3)?\s+(?P<path>tools/[A-Za-z0-9_./-]+\.py)` + demoTailCls + `)`)
	demoMakeCmdRE   = regexp.MustCompile(`(?:^|[^A-Za-z0-9_])(?P<command>make\s+(?P<target>[A-Za-z0-9_.-]+)` + demoTailCls + `)`)
	// MAKE_TARGET_RE without the `(?!=)` lookahead — the caller checks the post-colon byte.
	demoMakeTargetRE = regexp.MustCompile(`^(?P<target>[A-Za-z0-9_.-]+)\s*:`)
	demoBareInlineRE = regexp.MustCompile(`(?:<code>|` + "`" + `)\s*(?P<target>\./cmd/[A-Za-z0-9_-]+)\s*(?:</code>|` + "`" + `)`)
	demoDefaultDocs  = []string{"docs/run-the-demos.md", "docs/demos.html"}
)

// demoBrowserNames is the browser-demo registry from tools/demo_registry.py (DEMOS), in registry
// order. demo_command_audit checks every registered demo is documented with a `go run ./cmd/<name>`
// command. It is now derived from the shared demoRegistry table (demo_registry.go) so the two
// hygiene gates that key on the registry share one source of truth; gate_democommand_test.go pins
// it against the live Python DEMOS so a registry add/remove that skips this file reds `go test`.
var demoBrowserNames = demoRegNames()

// demoCommandRef is one documented command reference — the Go twin of demo_command_audit.CommandRef.
type demoCommandRef struct {
	source  string
	line    int
	kind    string
	target  string
	command string
	goDir   string
}

func (r demoCommandRef) loc() string { return r.source + ":" + itoa(int64(r.line)) }

// reGroup returns the named subgroup from a FindStringSubmatch result, or "".
func reGroup(re *regexp.Regexp, m []string, name string) string {
	for i, n := range re.SubexpNames() {
		if n == name && i < len(m) {
			return m[i]
		}
	}
	return ""
}

// extractDemoCommandRefs ports extract_command_refs: per line, run each command regex and collect the
// refs it finds, in the same regex order the Python appends them.
func extractDemoCommandRefs(source, text string) []demoCommandRef {
	var refs []demoCommandRef
	for lineNo, line := range strings.Split(text, "\n") {
		n := lineNo + 1
		for _, m := range demoGoCmdRE.FindAllStringSubmatch(line, -1) {
			refs = append(refs, demoCommandRef{source, n, "go-" + reGroup(demoGoCmdRE, m, "verb"),
				"cmd/" + reGroup(demoGoCmdRE, m, "name"), reGroup(demoGoCmdRE, m, "command"), ""})
		}
		for _, m := range demoGoCCmdRE.FindAllStringSubmatch(line, -1) {
			refs = append(refs, demoCommandRef{source, n, "go-" + reGroup(demoGoCCmdRE, m, "verb"),
				"cmd/" + reGroup(demoGoCCmdRE, m, "name"), reGroup(demoGoCCmdRE, m, "command"),
				strings.Trim(reGroup(demoGoCCmdRE, m, "dir"), "\"'")})
		}
		for _, m := range demoScriptCmdRE.FindAllStringSubmatch(line, -1) {
			refs = append(refs, demoCommandRef{source, n, "shell-script",
				reGroup(demoScriptCmdRE, m, "path"), reGroup(demoScriptCmdRE, m, "command"), ""})
		}
		for _, m := range demoPyToolCmdRE.FindAllStringSubmatch(line, -1) {
			refs = append(refs, demoCommandRef{source, n, "python-tool",
				reGroup(demoPyToolCmdRE, m, "path"), reGroup(demoPyToolCmdRE, m, "command"), ""})
		}
		for _, m := range demoMakeCmdRE.FindAllStringSubmatch(line, -1) {
			refs = append(refs, demoCommandRef{source, n, "make-target",
				reGroup(demoMakeCmdRE, m, "target"), reGroup(demoMakeCmdRE, m, "command"), ""})
		}
	}
	return refs
}

// demoMakeTargets ports make_targets: the set of Makefile phony/file targets. A recipe/comment line
// (leading tab, space, or #) is skipped; a `name:` header is a target unless the colon is followed by
// `=` (a `:=` assignment) — the RE2-free spelling of the Python `:(?!=)`.
func demoMakeTargets(root string) map[string]bool {
	targets := map[string]bool{}
	body, err := os.ReadFile(filepath.Join(root, "Makefile"))
	if err != nil {
		return targets
	}
	for _, line := range strings.Split(string(body), "\n") {
		if line == "" {
			continue
		}
		if c := line[0]; c == '\t' || c == ' ' || c == '#' {
			continue
		}
		loc := demoMakeTargetRE.FindStringSubmatchIndex(line)
		if loc == nil {
			continue
		}
		// loc[1] is the index just past the matched colon; skip when the next byte is '=' (`:=`).
		if loc[1] < len(line) && line[loc[1]] == '=' {
			continue
		}
		targets[line[loc[2]:loc[3]]] = true // the "target" group
	}
	return targets
}

// demoInsideWorkspace ports _inside_workspace: does rel resolve to a path under root?
func demoInsideWorkspace(root, rel string) bool {
	ra, err1 := filepath.Abs(root)
	ta, err2 := filepath.Abs(filepath.Join(root, filepath.FromSlash(rel)))
	if err1 != nil || err2 != nil {
		return false
	}
	r, err := filepath.Rel(ra, ta)
	if err != nil {
		return false
	}
	return r != ".." && !strings.HasPrefix(r, ".."+string(filepath.Separator))
}

func demoIsDir(root, rel string) bool {
	fi, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
	return err == nil && fi.IsDir()
}

func demoIsFile(root, rel string) bool {
	fi, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
	return err == nil && !fi.IsDir()
}

func demoHasTestGo(root, rel string) bool {
	matches, err := filepath.Glob(filepath.Join(root, filepath.FromSlash(rel), "*_test.go"))
	return err == nil && len(matches) > 0
}

// demoPathDefect ports _path_defect: resolve one command ref against the tree, returning a defect
// message (and true) when its target does not resolve. makeTargets is precomputed once per run.
func demoPathDefect(root string, ref demoCommandRef, makeTargets map[string]bool) (string, bool) {
	if ref.goDir != "" && ref.goDir != "." && ref.goDir != filepath.Base(root) {
		return ref.loc() + " unsupported go -C directory in demo command: " + ref.goDir, true
	}
	for _, p := range strings.Split(ref.target, "/") {
		if p == ".." {
			return ref.loc() + " " + ref.kind + " target escapes audited tree: " + ref.target, true
		}
	}
	if !demoInsideWorkspace(root, ref.target) {
		return ref.loc() + " " + ref.kind + " target resolves outside workspace: " + ref.target, true
	}

	spaced := strings.ReplaceAll(ref.kind, "-", " ") // "go-run" -> "go run"
	switch ref.kind {
	case "go-build", "go-run":
		if !demoIsDir(root, ref.target) {
			return ref.loc() + " " + spaced + " target missing: " + ref.target, true
		}
		if !demoIsFile(root, ref.target+"/main.go") {
			return ref.loc() + " " + spaced + " target has no main.go: " + ref.target, true
		}
		return "", false
	case "go-test":
		if !demoIsDir(root, ref.target) {
			return ref.loc() + " go test target missing: " + ref.target, true
		}
		if !demoHasTestGo(root, ref.target) {
			return ref.loc() + " go test target has no *_test.go: " + ref.target, true
		}
		return "", false
	case "shell-script", "python-tool":
		if !demoIsFile(root, ref.target) {
			return ref.loc() + " " + ref.kind + " target missing: " + ref.target, true
		}
		return "", false
	case "make-target":
		if !makeTargets[ref.target] {
			return ref.loc() + " make target missing from Makefile: " + ref.target, true
		}
		return "", false
	}
	return ref.loc() + " unclassified command kind: " + ref.kind, true
}

// demoBareCmdDefects ports bare_cmd_defects: a `./cmd/<name>` inside inline `code`/<code> is a bare
// (non-runnable) command path — docs should use the `go run ./cmd/<name>` form.
func demoBareCmdDefects(source, text string) []string {
	var defects []string
	for lineNo, line := range strings.Split(text, "\n") {
		for _, m := range demoBareInlineRE.FindAllStringSubmatch(line, -1) {
			target := reGroup(demoBareInlineRE, m, "target")
			defects = append(defects, source+":"+itoa(int64(lineNo+1))+
				" bare cmd path in inline code: "+target+"; use `go run "+target+"` for runnable demo docs")
		}
	}
	return defects
}

// demoDocumentedGoRunPackages ports documented_go_run_packages: the set of cmd/<name> documented with
// a `go run` command.
func demoDocumentedGoRunPackages(refs []demoCommandRef) map[string]bool {
	pkgs := map[string]bool{}
	for _, r := range refs {
		if r.kind == "go-run" && strings.HasPrefix(r.target, "cmd/") {
			pkgs[strings.TrimPrefix(r.target, "cmd/")] = true
		}
	}
	return pkgs
}

// demoBrowserCoverageDefects ports browser_demo_coverage_defects: every registered browser demo must
// be documented with a `go run` command. Emitted in registry order (demo_registry.py DEMOS order).
func demoBrowserCoverageDefects(refs []demoCommandRef) []string {
	documented := demoDocumentedGoRunPackages(refs)
	var defects []string
	for _, name := range demoBrowserNames {
		if !documented[name] {
			defects = append(defects, "browser demo registry entry is not documented with a go run command: cmd/"+name)
		}
	}
	return defects
}

// demoDefaultSources ports default_sources: the two demo docs plus every cmd/*/README.md, sorted.
func demoDefaultSources(root string) []string {
	sources := append([]string{}, demoDefaultDocs...)
	matches, _ := filepath.Glob(filepath.Join(root, "cmd", "*", "README.md"))
	var readmes []string
	for _, m := range matches {
		rel, err := filepath.Rel(root, m)
		if err != nil {
			continue
		}
		readmes = append(readmes, filepath.ToSlash(rel))
	}
	sort.Strings(readmes)
	return append(sources, readmes...)
}

// demoCommandDefects ports collect()'s defect derivation. sources==nil means the default make-hygiene
// invocation: audit default_sources AND run the browser-demo coverage gate. Explicit sources skip the
// coverage gate (matching `collect(..., sources=...)`). Returns the defect list (empty == clean).
func demoCommandDefects(root string, sources []string) []string {
	explicit := sources != nil
	if !explicit {
		sources = demoDefaultSources(root)
	}
	var refs []demoCommandRef
	var defects []string
	for _, source := range sources {
		if !demoInsideWorkspace(root, source) {
			defects = append(defects, "source resolves outside workspace: "+source)
			continue
		}
		body, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(source)))
		if err != nil {
			defects = append(defects, "read "+source+": "+err.Error())
			continue
		}
		text := string(body)
		refs = append(refs, extractDemoCommandRefs(source, text)...)
		defects = append(defects, demoBareCmdDefects(source, text)...)
	}
	if len(refs) == 0 {
		defects = append(defects, "no documented demo commands found in audited sources")
	}
	makeTargets := demoMakeTargets(root)
	for _, ref := range refs {
		if d, bad := demoPathDefect(root, ref, makeTargets); bad {
			defects = append(defects, d)
		}
	}
	if !explicit {
		defects = append(defects, demoBrowserCoverageDefects(refs)...)
	}
	return defects
}

// gateDemoCommandTree is the DEMO_COMMAND --audit-tree gate: run the default (registry-coverage)
// audit over the tree root and surface each documented-command defect as a Finding. The defect
// string already carries its own `source:line` locator, so it is placed whole in Detail.
func gateDemoCommandTree(t *TrackedTree) ([]Finding, error) {
	var findings []Finding
	for _, d := range demoCommandDefects(t.Root, nil) {
		findings = append(findings, Finding{Gate: "DEMO_COMMAND", Detail: d})
	}
	return findings, nil
}
