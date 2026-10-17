package envconfiglint

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os/exec"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ReasonConfigNotEnv is the closed-vocabulary refusal code for an environment read that
// is not a declared secret — the structured, machine-checkable form the ratchet refuses
// with instead of free text, the internal/pythongate NEW_PYTHON_TOOL idiom.
const ReasonConfigNotEnv = "CONFIG_NOT_ENV"

// secretEnvRe matches env-var NAMES that denote a SECRET (legitimately read from the
// environment / .env). The judgment is structural: a name is a secret iff one of its
// underscore-delimited tokens is a credential word. Erring toward "secret" is the safe
// direction — it yields FEWER offenses, so a behavioral read mislabeled secret merely
// stays un-nagged; it is never a false alarm that blocks a peer's commit.
var secretEnvRe = regexp.MustCompile(
	`(?i)(^|_)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS|APIKEY|AUTH|BEARER|PAT|SALT)($|_)`)

// envReadRe extracts the constant NAME argument of os.Getenv / os.LookupEnv calls from Go
// source text. Only string-literal names are linted; a COMPUTED name (a variable or a
// concatenation) cannot be classified structurally and is skipped by construction — the
// documented invalidating assumption in doc.go.
var envReadRe = regexp.MustCompile(`os\.(?:Getenv|LookupEnv)\(\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\)`)

// IsSecretName reports whether an env-var name denotes a declared secret (belongs in the
// environment) rather than behavioral configuration (belongs in the config surface).
func IsSecretName(name string) bool { return secretEnvRe.MatchString(name) }

// ScanGoEnvReads returns the distinct env-var names read via os.Getenv / os.LookupEnv in
// the given Go source text, in first-seen order.
func ScanGoEnvReads(src string) []string {
	var out []string
	seen := map[string]bool{}
	for _, m := range envReadRe.FindAllStringSubmatch(src, -1) {
		if name := m[1]; !seen[name] {
			seen[name] = true
			out = append(out, name)
		}
	}
	return out
}

// scanComputeEnvReads recognizes the known package-local cache wrapper without
// treating same-named helpers in other packages, selectors, or shadowed locals as
// environment reads. Parsing the complete file excludes comments and string data.
// This adds visibility, never an admission: callers still classify every name.
func scanComputeEnvReads(file, src string) ([]string, error) {
	if path.Dir(file) != "internal/compute" || !strings.HasSuffix(file, ".go") || strings.HasSuffix(file, "_test.go") {
		return nil, nil
	}
	f, err := parser.ParseFile(token.NewFileSet(), file, src, 0)
	if err != nil {
		return nil, fmt.Errorf("parse compute env wrapper calls in %s: %w", file, err)
	}
	if f.Name.Name != "compute" {
		return nil, nil
	}
	var names []string
	seen := map[string]bool{}
	ast.Inspect(f, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			return true
		}
		name, ok := call.Fun.(*ast.Ident)
		if !ok || name.Name != "computeEnvValue" {
			return true
		}
		if name.Obj != nil && (name.Obj.Kind != ast.Fun || file != "internal/compute/v41_q3k_token_panel.go") {
			return true
		}
		arg, ok := call.Args[0].(*ast.BasicLit)
		if !ok || arg.Kind != token.STRING {
			return true
		}
		value, err := strconv.Unquote(arg.Value)
		if err == nil && !seen[value] {
			seen[value] = true
			names = append(names, value)
		}
		return true
	})
	return names, nil
}

// Offense is one non-secret environment read that should relocate to the config surface.
type Offense struct {
	Name string // the env-var name, e.g. "FAK_TIMEOUT_MS"
	File string // repo-relative path it was read in; empty when judged from a bare diff
}

// String renders the offense as a one-line, codemod-style report carrying the reason code.
func (o Offense) String() string {
	msg := fmt.Sprintf("env read %s is not a declared secret; move it to the config surface (%s)",
		o.Name, ReasonConfigNotEnv)
	if o.File == "" {
		return msg
	}
	return o.File + ": " + msg
}

// Fix renders the codemod-style SUGGESTION: the concrete relocation an author should make.
// Advisory by contract — this package never rewrites source. An automatic rewrite of a
// behavioral read could silently change runtime behavior, so the fix stays a suggestion a
// human or agent applies (issue #2863's "codemod suggestion", not "codemod").
func (o Offense) Fix() string {
	return fmt.Sprintf(
		"%s: if %s is genuinely a credential, name it as one (…_KEY/_TOKEN/_SECRET/_PASSWORD) so it reads as a declared secret; "+
			"otherwise move it to the config surface and drop the os.Getenv/os.LookupEnv read",
		o.File, o.Name)
}

// Classify returns the offenses among names: those that are neither a declared secret nor
// grandfathered in baseline. This is the pure ratchet core (verify the verifier), mirroring
// internal/pythongate's offensesAgainst. A nil baseline grandfathers nothing.
func Classify(names []string, file string, baseline map[string]bool) []Offense {
	var out []Offense
	for _, n := range names {
		if IsSecretName(n) || baseline[n] {
			continue
		}
		out = append(out, Offense{Name: n, File: file})
	}
	return out
}

// AddedEnvReads returns the distinct env-var names read on the ADDED lines of a unified
// diff — lines starting with '+' that are not the '+++' file header. This is the diff-mode
// core (#2863's literal "diff new env reads against a declared-secret allowlist"): only
// newly-introduced reads are judged, so a grandfathered read appearing as unchanged context
// (' ') or being removed ('-') is never seen.
func AddedEnvReads(unifiedDiff string) []string {
	var added strings.Builder
	for _, line := range strings.Split(unifiedDiff, "\n") {
		if strings.HasPrefix(line, "+++") || !strings.HasPrefix(line, "+") {
			continue // file header, context, deletion, or hunk header — not added source
		}
		added.WriteString(line[1:]) // drop the '+' marker, keep the added source text
		added.WriteByte('\n')
	}
	return ScanGoEnvReads(added.String())
}

// ClassifyDiff returns the offenses a unified diff INTRODUCES: non-secret env reads on its
// added lines. This is the shape a pre-commit or CI shell wraps around `git diff`. No
// baseline is needed — added-line filtering already restricts the verdict to reads the diff
// brings into the tree.
func ClassifyDiff(unifiedDiff, file string) []Offense {
	return Classify(AddedEnvReads(unifiedDiff), file, nil)
}

// ScanTree returns one Offense per env-var name read in repoRoot's COMMITTED Go source that
// is neither a declared secret nor grandfathered, sorted by name (first-seen file attributed).
//
// It reads committed content at HEAD via `git grep`, not the working tree, for two reasons
// this shared checkout makes load-bearing: only source that would actually ship counts (the
// pythongate rule), and a peer's uncommitted or untracked WIP can never red YOUR gate run.
func ScanTree(repoRoot string) ([]Offense, error) { return scanTree(repoRoot, baselineSet()) }

// scanTree is ScanTree with an injectable baseline, so a test can prove the tree scanner
// actually SEES reads (scan with no baseline ⇒ many offenses). Without that negative
// control a broken scanner — a git-grep output-format change, say — would report zero
// offenses and the gate would pass vacuously forever.
func scanTree(repoRoot string, baseline map[string]bool) ([]Offense, error) {
	matches, err := committedEnvReadMatches(repoRoot)
	if err != nil {
		return nil, err
	}
	firstFile := map[string]string{}
	var names []string
	for _, m := range matches {
		for _, n := range m.names {
			if _, seen := firstFile[n]; seen {
				continue
			}
			firstFile[n] = m.file
			names = append(names, n)
		}
	}
	var offenses []Offense
	for _, n := range names {
		if IsSecretName(n) || baseline[n] {
			continue
		}
		offenses = append(offenses, Offense{Name: n, File: firstFile[n]})
	}
	sort.Slice(offenses, func(i, j int) bool { return offenses[i].Name < offenses[j].Name })
	return offenses, nil
}

// envReadMatch binds recognized environment names to their committed source file.
type envReadMatch struct {
	file  string
	names []string
}

// committedEnvReadMatches shells to git for authoritative COMMITTED source.
// envReadRe extracts direct reads; scanComputeEnvReads recognizes the one known
// package-local wrapper. Git's patterns only prefilter candidate source.
//
// _test.go is EXCLUDED. The rule is about the shipped CONFIGURATION surface — what an
// operator must set to run fak — and a test-harness switch (FAK_NEGBENCH_MODEL,
// CODEX_AUDIT_SAMPLE) is neither a credential nor a setting that could relocate to a config
// file: it selects a fixture for a test that only runs under `go test`. Scanning tests made
// the gate demand relocation for reads with nowhere to go, and baked this lint's OWN
// synthetic fixtures (FOO_MODE, BAR_TOKEN) into the baseline. Partitioning by file class is
// the internal/boundarylint precedent — Scan skips _test.go, ScanTests walks only _test.go,
// so a rule runs over exactly the file class its tell is about.
func committedEnvReadMatches(repoRoot string) ([]envReadMatch, error) {
	// One immutable commit binds every subprocess, even if a peer advances HEAD
	// between candidate discovery and reading the complete wrapper files.
	revision, err := gitOutput(repoRoot, "rev-parse", "--verify", "HEAD^{commit}")
	if err != nil {
		return nil, fmt.Errorf("resolve committed env scan revision: %w", err)
	}
	revision = strings.TrimSpace(revision)
	cmd := exec.Command("git", "grep", "-o", "-E",
		`os\.(Getenv|LookupEnv)\("[A-Za-z_][A-Za-z0-9_]*"\)`, revision, "--", "*.go", ":(exclude)*_test.go")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = repoRoot
	out, err := cmd.Output()
	if err != nil {
		// git grep exits 1 with no output when nothing matches — an empty tree, not a failure.
		var ee *exec.ExitError
		if !(errors.As(err, &ee) && ee.ExitCode() == 1 && len(out) == 0) {
			return nil, fmt.Errorf("git grep env reads in %s: %w", repoRoot, err)
		}
	}
	var matches []envReadMatch
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line = strings.TrimSpace(line); line == "" {
			continue
		}
		// `git grep <tree-ish>` emits "<revision>:<path>:<matched text>".
		rest, ok := strings.CutPrefix(line, revision+":")
		if !ok {
			continue
		}
		file, text, ok := strings.Cut(rest, ":")
		if !ok {
			continue
		}
		matches = append(matches, envReadMatch{file: strings.ReplaceAll(file, "\\", "/"), names: ScanGoEnvReads(text)})
	}
	// Read complete committed files for AST-based wrapper recognition. The direct
	// call regex and its public API remain unchanged; this known package-local
	// wrapper is not a global exception for arbitrary helper names.
	cmd = exec.Command("git", "grep", "-l", "-z", "computeEnvValue", revision, "--", "internal/compute/*.go", ":(exclude)*_test.go")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = repoRoot
	out, err = cmd.Output()
	if err != nil {
		var ee *exec.ExitError
		if !(errors.As(err, &ee) && ee.ExitCode() == 1 && len(out) == 0) {
			return nil, fmt.Errorf("git grep compute env wrapper in %s: %w", repoRoot, err)
		}
	}
	for _, entry := range strings.Split(string(out), "\x00") {
		file, ok := strings.CutPrefix(entry, revision+":")
		if !ok {
			continue
		}
		src, err := gitOutput(repoRoot, "show", revision+":"+file)
		if err != nil {
			return nil, fmt.Errorf("read committed wrapper source %s: %w", file, err)
		}
		names, err := scanComputeEnvReads(file, src)
		if err != nil {
			return nil, err
		}
		matches = append(matches, envReadMatch{file: file, names: names})
	}
	return matches, nil
}

// baselineSet materializes the allowed names into a lookup set: the generated freeze
// (grandfathered) plus the explicitly-reasoned post-freeze admissions (admittedPostFreeze).
// The two are unioned here rather than merged into one list so that regenerating baseline.go
// can never silently swallow a re-admission — see admitted.go.
func baselineSet() map[string]bool {
	set := make(map[string]bool, len(grandfathered)+len(admittedPostFreeze))
	for _, p := range grandfathered {
		set[p] = true
	}
	for _, p := range admittedPostFreeze {
		set[p] = true
	}
	return set
}
