package envconfiglint

import (
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"sort"
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
		for _, n := range ScanGoEnvReads(m.text) {
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

// envReadMatch is one `git grep -o` hit: the file it came from and the matched call text.
type envReadMatch struct {
	file string
	text string
}

// committedEnvReadMatches shells to git for the authoritative COMMITTED matches. The ERE
// handed to git is only a coarse prefilter; envReadRe (Go) stays the authoritative extractor,
// so there is exactly one source of truth for what counts as an env read.
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
	cmd := exec.Command("git", "grep", "-o", "-E",
		`os\.(Getenv|LookupEnv)\("[A-Za-z_][A-Za-z0-9_]*"\)`, "HEAD", "--", "*.go", ":(exclude)*_test.go")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = repoRoot
	out, err := cmd.Output()
	if err != nil {
		// git grep exits 1 with no output when nothing matches — an empty tree, not a failure.
		var ee *exec.ExitError
		if errors.As(err, &ee) && ee.ExitCode() == 1 && len(out) == 0 {
			return nil, nil
		}
		return nil, fmt.Errorf("git grep env reads in %s: %w", repoRoot, err)
	}
	var matches []envReadMatch
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line = strings.TrimSpace(line); line == "" {
			continue
		}
		// `git grep <tree-ish>` emits "HEAD:<path>:<matched text>".
		rest, ok := strings.CutPrefix(line, "HEAD:")
		if !ok {
			continue
		}
		file, text, ok := strings.Cut(rest, ":")
		if !ok {
			continue
		}
		matches = append(matches, envReadMatch{file: strings.ReplaceAll(file, "\\", "/"), text: text})
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
