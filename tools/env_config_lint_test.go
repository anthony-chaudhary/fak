package tools_test

// Kernel-enforced env-var-vs-config lint (issue #2863, Track G / #2834).
//
// Hermes' contribution rubric (its AGENTS.md) rejects "new HERMES_* env vars for
// non-secret config": .env is for SECRETS only (keys/tokens/passwords); all
// behavioral settings belong in the config surface (config.yaml). Hermes enforces
// that with a human reviewer reading the diff, so violations still land and get
// walked back later. fak makes the same rule machine-checkable and deterministic:
// an environment-variable READ (os.Getenv / os.LookupEnv) must name a *declared
// secret*, or it is a config-surface violation with a codemod-style fix ("move it
// to the config surface"). Same deny-by-structure instinct as the schema mask and
// the pythongate ratchet, applied to the *configuration* surface.
//
// This file ships the deny-by-structure RULE plus its contract test — the gen/next
// foundation. The classifier (envReadIsSecret), the source scanner (scanGoEnvReads),
// and the ratchet core (classifyEnvReads) are pure and hermetically tested, mirroring
// internal/pythongate's offensesAgainst. The whole-tree scan (TestEnvConfigLintTreeReadout)
// is deliberately ADVISORY — a dogfood readout that never reds the shared trunk —
// because the current tree carries many grandfathered behavioral FAK_* reads. See that
// test for the promotion evidence that turns this into a hard CI gate.

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// secretEnvRe matches env-var NAMES that denote a secret (legitimately in .env). The
// test is structural: a name is a secret iff one of its underscore-delimited tokens is
// a credential word. Erring toward "secret" is the safe direction for this lint — it
// yields FEWER false-positive offenses; a behavioral read mislabeled secret merely
// stays un-nagged, it is never a false alarm.
var secretEnvRe = regexp.MustCompile(
	`(?i)(^|_)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS|APIKEY|AUTH|BEARER|PAT|SALT)($|_)`)

// envReadRe extracts the constant NAME argument of os.Getenv / os.LookupEnv calls from
// Go source text. Only string-literal names are linted; a computed name (a variable or
// concatenation) cannot be classified structurally and is skipped — a documented
// limitation the readout names as an invalidating assumption.
var envReadRe = regexp.MustCompile(`os\.(?:Getenv|LookupEnv)\(\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\)`)

// envReadIsSecret reports whether an env-var name denotes a declared secret (belongs in
// .env) rather than behavioral config (belongs in the config surface).
func envReadIsSecret(name string) bool { return secretEnvRe.MatchString(name) }

// scanGoEnvReads returns the distinct env-var names read via os.Getenv / os.LookupEnv in
// the given Go source text, in first-seen order.
func scanGoEnvReads(src string) []string {
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

// envConfigOffense is a non-secret env read that should relocate to the config surface.
type envConfigOffense struct {
	Name string
	File string
}

// String renders the codemod-style suggestion the CI gate would print.
func (o envConfigOffense) String() string {
	return o.File + ": env read " + o.Name +
		" is not a declared secret; move it to the config surface (CONFIG_NOT_ENV)"
}

// classifyEnvReads returns the offenses among names: those that are neither a declared
// secret nor grandfathered in baseline. This is the pure ratchet core (verify the
// verifier), mirroring internal/pythongate's offensesAgainst.
func classifyEnvReads(names []string, file string, baseline map[string]bool) []envConfigOffense {
	var out []envConfigOffense
	for _, n := range names {
		if envReadIsSecret(n) || baseline[n] {
			continue
		}
		out = append(out, envConfigOffense{Name: n, File: file})
	}
	return out
}

// TestEnvConfigLintContract is the hermetic contract test: it pins the classifier, the
// source scanner, the ratchet core, and the codemod message on synthetic inputs (no
// git, no tree). This is the gen/next proof bar.
func TestEnvConfigLintContract(t *testing.T) {
	// Secret-shaped names are allowed in env.
	for _, n := range []string{"ANTHROPIC_API_KEY", "GITHUB_TOKEN", "DB_PASSWORD", "CLIENT_SECRET", "HMAC_SIGNING_KEY", "WEBHOOK_SECRET"} {
		if !envReadIsSecret(n) {
			t.Errorf("%s should classify as a secret (allowed in env)", n)
		}
	}
	// Behavioral names are NOT secrets — they belong in the config surface.
	for _, n := range []string{"FAK_TIMEOUT_MS", "HERMES_MAX_RETRIES", "LOG_LEVEL", "FAK_NATIVE_GUIDED_DECODE", "PATH"} {
		if envReadIsSecret(n) {
			t.Errorf("%s should NOT classify as a secret", n)
		}
	}
	// The scanner extracts literal names from both read forms and skips a computed name.
	src := `a := os.Getenv("FOO_MODE"); b, ok := os.LookupEnv("BAR_TOKEN"); c := os.Getenv(dynamicName)`
	got := scanGoEnvReads(src)
	if len(got) != 2 || got[0] != "FOO_MODE" || got[1] != "BAR_TOKEN" {
		t.Fatalf("scanGoEnvReads = %v, want [FOO_MODE BAR_TOKEN]", got)
	}
	// Ratchet: FOO_MODE (non-secret, un-baselined) is exactly one offense; BAR_TOKEN (secret) is clean.
	off := classifyEnvReads(got, "x.go", map[string]bool{})
	if len(off) != 1 || off[0].Name != "FOO_MODE" {
		t.Fatalf("classifyEnvReads = %v, want one offense FOO_MODE", off)
	}
	if want := "x.go: env read FOO_MODE is not a declared secret; move it to the config surface (CONFIG_NOT_ENV)"; off[0].String() != want {
		t.Errorf("offense string = %q, want %q", off[0].String(), want)
	}
	// Grandfathering: baselining FOO_MODE clears the offense (the ratchet never nags a known read).
	if off := classifyEnvReads(got, "x.go", map[string]bool{"FOO_MODE": true}); len(off) != 0 {
		t.Fatalf("baselined FOO_MODE: want 0 offenses, got %v", off)
	}
}

// TestEnvConfigLintTreeReadout is the ADVISORY dogfood: it scans the real Go tree and
// REPORTS the non-secret env-read surface. It never fails — the current tree carries
// grandfathered behavioral reads (a hard gate today would red the shared trunk for
// every session). The readout is the promotion evidence.
//
//	Promotion evidence (advisory -> hard gate, moves this toward `now`): freeze the names
//	  printed below as a checked-in grandfather baseline, then flip the loop to t.Errorf on
//	  any name not in {secret, baseline}. That makes a NEW non-secret env read red CI —
//	  the issue's full ask. The remaining wire-up (staged-diff mode so only ADDED reads
//	  are judged) is #2863's follow-up, pairing with the config-surface budget gate (#2862).
//	Demotion/retirement evidence: if the count reaches zero AND the config surface (#2862)
//	  is the sole behavioral-settings home, this lint has done its job and can retire.
//	Invalidating assumption: only STRING-LITERAL env names are seen; a read built from a
//	  computed name (os.Getenv(prefix+key)) is invisible to a regex scanner and would need
//	  an AST/go/analysis pass to catch — the reason this stays advisory, not authoritative.
func TestEnvConfigLintTreeReadout(t *testing.T) {
	root := envLintRepoRoot(t)
	counts := map[string]int{}
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "node_modules", "vendor", "testdata":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		b, rerr := os.ReadFile(path)
		if rerr != nil {
			return nil
		}
		for _, n := range scanGoEnvReads(string(b)) {
			if !envReadIsSecret(n) {
				counts[n]++
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	names := make([]string, 0, len(counts))
	for n := range counts {
		names = append(names, n)
	}
	sort.Strings(names)
	t.Logf("env-config lint (advisory): %d distinct non-secret env-var read name(s) on the current tree", len(names))
	t.Logf("promotion: freeze these as the grandfather baseline, then flip this test to t.Errorf on any NEW name to make it a hard CI gate (#2863)")
}

// envLintRepoRoot walks up from the test working directory to the module root (go.mod).
func envLintRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod found above %s", dir)
		}
		dir = parent
	}
}
