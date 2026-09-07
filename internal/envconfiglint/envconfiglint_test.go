package envconfiglint

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestClassifierAndRatchetCore pins the secret classifier, the source scanner, the ratchet
// core, and the codemod message on synthetic inputs (no git, no tree) — verify the verifier,
// the internal/pythongate TestOffensesAgainst idiom.
func TestClassifierAndRatchetCore(t *testing.T) {
	// Secret-shaped names are legitimately read from the environment.
	for _, n := range []string{"ANTHROPIC_API_KEY", "GITHUB_TOKEN", "DB_PASSWORD", "CLIENT_SECRET", "HMAC_SIGNING_KEY", "WEBHOOK_SECRET"} {
		if !IsSecretName(n) {
			t.Errorf("%s should classify as a declared secret (allowed in env)", n)
		}
	}
	// Behavioral names are NOT secrets — they belong in the config surface.
	for _, n := range []string{"FAK_TIMEOUT_MS", "HERMES_MAX_RETRIES", "LOG_LEVEL", "FAK_NATIVE_GUIDED_DECODE", "PATH"} {
		if IsSecretName(n) {
			t.Errorf("%s should NOT classify as a declared secret", n)
		}
	}
	// The scanner extracts literal names from both read forms and skips a computed name.
	src := `a := os.Getenv("FOO_MODE"); b, ok := os.LookupEnv("BAR_TOKEN"); c := os.Getenv(dynamicName)`
	got := ScanGoEnvReads(src)
	if len(got) != 2 || got[0] != "FOO_MODE" || got[1] != "BAR_TOKEN" {
		t.Fatalf("ScanGoEnvReads = %v, want [FOO_MODE BAR_TOKEN]", got)
	}
	// Ratchet: FOO_MODE (non-secret, un-baselined) is exactly one offense; BAR_TOKEN is clean.
	off := Classify(got, "x.go", map[string]bool{})
	if len(off) != 1 || off[0].Name != "FOO_MODE" {
		t.Fatalf("Classify = %v, want one offense FOO_MODE", off)
	}
	if want := "x.go: env read FOO_MODE is not a declared secret; move it to the config surface (CONFIG_NOT_ENV)"; off[0].String() != want {
		t.Errorf("offense string = %q, want %q", off[0].String(), want)
	}
	// The codemod SUGGESTION names both escape hatches (declare-as-secret, or relocate).
	if fix := off[0].Fix(); !strings.Contains(fix, "FOO_MODE") || !strings.Contains(fix, "config surface") {
		t.Errorf("Fix() = %q, want it to name FOO_MODE and the config surface", fix)
	}
	// Grandfathering: baselining FOO_MODE clears the offense (the ratchet never nags a known read).
	if off := Classify(got, "x.go", map[string]bool{"FOO_MODE": true}); len(off) != 0 {
		t.Fatalf("baselined FOO_MODE: want 0 offenses, got %v", off)
	}
}

// TestDiffMode pins the diff-mode core — #2863's literal ask, "a lint that diffs new env
// reads against a declared-secret allowlist" — on synthetic unified diffs (no git, no tree):
// an ADDED non-secret read is exactly one offense carrying the codemod message, an ADDED
// secret is clean, and a read seen only as unchanged context or a deletion is never judged.
func TestDiffMode(t *testing.T) {
	// A diff that ADDS a non-secret behavioral read is caught, with the codemod suggestion.
	addNonSecret := "--- a/server.go\n+++ b/server.go\n@@ -1,3 +1,4 @@\n func f() {\n+\tttl := os.Getenv(\"FAK_TIMEOUT_MS\")\n \treturn\n }\n"
	off := ClassifyDiff(addNonSecret, "server.go")
	if len(off) != 1 || off[0].Name != "FAK_TIMEOUT_MS" {
		t.Fatalf("ClassifyDiff(add non-secret) = %v, want one offense FAK_TIMEOUT_MS", off)
	}
	if want := "server.go: env read FAK_TIMEOUT_MS is not a declared secret; move it to the config surface (CONFIG_NOT_ENV)"; off[0].String() != want {
		t.Errorf("offense string = %q, want %q", off[0].String(), want)
	}
	// A diff that ADDS a declared secret is clean — secrets legitimately live in the env.
	addSecret := "+++ b/auth.go\n@@\n+\ttok := os.Getenv(\"GITHUB_TOKEN\")\n"
	if off := ClassifyDiff(addSecret, "auth.go"); len(off) != 0 {
		t.Fatalf("ClassifyDiff(add secret) = %v, want 0 offenses", off)
	}
	// A read seen ONLY as unchanged context (' ') or a deletion ('-') introduces nothing new.
	contextOnly := "--- a/old.go\n+++ b/old.go\n@@ -1,2 +1,2 @@\n \tmode := os.Getenv(\"LEGACY_MODE\")\n-\tx := os.Getenv(\"OLD_FLAG\")\n+\treturn mode\n"
	if off := ClassifyDiff(contextOnly, "old.go"); len(off) != 0 {
		t.Fatalf("ClassifyDiff(context/deletion only) = %v, want 0 offenses", off)
	}
	// A diff adding BOTH a secret and a non-secret read yields exactly the non-secret offense.
	addMixed := "+++ b/mix.go\n@@\n+\tk := os.Getenv(\"API_KEY\")\n+\tlvl, _ := os.LookupEnv(\"LOG_LEVEL\")\n"
	if off := ClassifyDiff(addMixed, "mix.go"); len(off) != 1 || off[0].Name != "LOG_LEVEL" {
		t.Fatalf("ClassifyDiff(mixed) = %v, want one offense LOG_LEVEL", off)
	}
}

// TestNoNewNonSecretEnvReads is the LIVE trunk guard and the CI gate that closes #2863:
// scanning the real repo's COMMITTED Go source against the frozen baseline must yield ZERO
// offenses. The day a diff lands a new non-secret env read, this reds the trunk naming the
// variable, the file, and the relocation fix.
func TestNoNewNonSecretEnvReads(t *testing.T) {
	root := repoRoot(t)
	offenses, err := ScanTree(root)
	if err != nil {
		t.Skipf("git grep unavailable (%v); the tree gate needs a git checkout", err)
	}
	if len(offenses) > 0 {
		t.Errorf("%d NEW non-secret env-var read(s) not in the grandfathered baseline:", len(offenses))
		for _, o := range offenses {
			t.Errorf("  %s", o)
			t.Errorf("    fix: %s", o.Fix())
		}
		t.Errorf("secrets belong in the environment; behavioral settings belong in the config surface. " +
			"If you legitimately relocated-and-deleted a grandfathered read, regenerate " +
			"internal/envconfiglint/baseline.go (see doc.go).")
	}
}

// TestTreeScannerIsNotVacuous is the negative control for the gate above. A green ratchet is
// only meaningful if the scanner it rests on actually sees the tree: were `git grep` to
// silently return nothing (an output-format change, a git that lacks -o), ScanTree would
// report zero offenses and TestNoNewNonSecretEnvReads would pass vacuously forever. Scanning
// with NO baseline must therefore surface the tree's large grandfathered behavioral surface —
// proving the green above comes from the baseline, not from a blind scanner.
func TestTreeScannerIsNotVacuous(t *testing.T) {
	root := repoRoot(t)
	offenses, err := scanTree(root, nil)
	if err != nil {
		t.Skipf("git grep unavailable (%v); the tree gate needs a git checkout", err)
	}
	if len(offenses) < 100 {
		t.Fatalf("un-baselined scan found only %d non-secret env reads; the tree carries hundreds — "+
			"the scanner is likely blind, which would make the trunk gate vacuous", len(offenses))
	}
	// And the classifier must be doing real work: no offense may be a declared secret.
	for _, o := range offenses {
		if IsSecretName(o.Name) {
			t.Fatalf("offense %s is secret-shaped; secrets must never be reported as config violations", o.Name)
		}
	}
}

// TestRatchetRedsOnASingleNewRead is the direct witness for #2863's done condition — "a diff
// introducing a new os.Getenv read that is not on the declared-secret allowlist fails CI,
// with a codemod-style suggestion pointing at relocating it to the config surface" — proven
// against the REAL tree rather than a synthetic string. It grandfathers every non-secret name
// the tree carries EXCEPT one, which is exactly the state the tree is in the moment a diff
// lands one new read, and asserts the gate reports precisely that name plus its relocation
// fix. The victim is derived from the scan, so the test never hardcodes a tree detail.
func TestRatchetRedsOnASingleNewRead(t *testing.T) {
	root := repoRoot(t)
	all, err := scanTree(root, nil)
	if err != nil {
		t.Skipf("git grep unavailable (%v); the tree gate needs a git checkout", err)
	}
	if len(all) == 0 {
		t.Skip("tree carries no non-secret env reads; nothing to hold out")
	}
	victim := all[0].Name
	baseline := make(map[string]bool, len(all))
	for _, o := range all {
		if o.Name != victim {
			baseline[o.Name] = true
		}
	}
	got, err := scanTree(root, baseline)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(got) != 1 || got[0].Name != victim {
		t.Fatalf("holding out %s: got %v, want exactly that one offense", victim, got)
	}
	if !strings.Contains(got[0].String(), ReasonConfigNotEnv) || got[0].File == "" {
		t.Errorf("offense %q must carry %s and the offending file", got[0].String(), ReasonConfigNotEnv)
	}
	if !strings.Contains(got[0].Fix(), "config surface") {
		t.Errorf("Fix() = %q, want the codemod-style relocation suggestion", got[0].Fix())
	}
}

// TestAdmittedPostFreezeStaysHonest keeps the re-admission ledger (admitted.go) from
// rotting into a junk drawer. A frozen baseline is only trustworthy if every exception on
// it is still load-bearing, so each admitted name must be:
//
//   - genuinely NON-secret — a secret-shaped name is allowed by IsSecretName anyway, so
//     listing one here would be a meaningless entry implying debt that does not exist; and
//   - still actually READ in the committed non-test tree — once a read is relocated to the
//     config surface (the #2862 endgame) its line must be DELETED here, not left behind.
//     Without this check the list could never be observed to shrink, which is its whole job.
func TestThirdUnwatchedCleanupIsExplicitlyAdmitted(t *testing.T) {
	want := []string{
		"FAK_ROOT_REGISTRATION_ID", "ComSpec", "FAK_MICRO_TASK", "FAK_LEASE_ID",
		"FAK_PROVIDER_ACCOUNT_IDENTITY", "FAK_GUARD_REFUSAL_STATE_DIR", "FAK_CLAUDE_SPEED",
		"FAK_WORK_EFFECT_CALIBRATION_JSON",
		"FAK_EP_COORDINATED_DECODE", "FAK_TOOLCALL_CONTROL_DIR", "FAK_TOOLCALL_CONTROL_MODE",
		"SystemDrive", "FAK_DEV_EXE", "FLEET_CODEX_EXE",
	}
	admitted := map[string]bool{}
	for _, name := range admittedPostFreeze {
		admitted[name] = true
	}
	for _, name := range want {
		if !admitted[name] {
			t.Errorf("stale live offense %s is not explicitly admitted", name)
		}
	}
}

func TestAdmittedPostFreezeCoversTrunkAdvances(t *testing.T) {
	if len(admittedPostFreeze) == 0 {
		t.Fatal("expected non-empty admittedPostFreeze")
	}
}

func TestAdmittedPostFreezeStaysHonest(t *testing.T) {
	root := repoRoot(t)
	matches, err := committedEnvReadMatches(root)
	if err != nil {
		t.Skipf("git grep unavailable (%v); the tree gate needs a git checkout", err)
	}
	live := map[string]bool{}
	for _, m := range matches {
		for _, n := range ScanGoEnvReads(m.text) {
			live[n] = true
		}
	}
	for _, n := range admittedPostFreeze {
		if IsSecretName(n) {
			t.Errorf("admittedPostFreeze carries secret-shaped %s; IsSecretName already allows it — drop the entry", n)
		}
		if !live[n] {
			t.Errorf("admittedPostFreeze carries %s, which is no longer read in the committed non-test tree; "+
				"the read was relocated or deleted, so delete this entry too (the ledger must only shrink)", n)
		}
	}
}

// repoRoot walks up from the test's working directory to the module root (go.mod).
func repoRoot(t *testing.T) string {
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
