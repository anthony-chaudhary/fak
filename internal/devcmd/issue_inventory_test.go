package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
	"github.com/anthony-chaudhary/fak/internal/modver"
)

func writeInventoryFixture(t *testing.T, dir, name string, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRunIssueInventoryGateAndStableJSON(t *testing.T) {
	dir, a, _, c := issueInventoryRepositoryFixture(t)
	t.Chdir(dir)
	issues := writeInventoryFixture(t, dir, "issues.json", []map[string]any{{"number": 1, "title": "panic in x"}})
	fail := issuepolicy.ErrorEvidence{Status: "fail", Fingerprint: "fp", Module: "internal/x", FailureClass: "panic", Commit: a, Witness: "observed"}
	obs := writeInventoryFixture(t, dir, "obs.json", issueInventoryInput{Schema: issuepolicy.ErrorInventorySchema, GeneratedAt: "2026-09-01T00:00:00Z", Observations: []issuepolicy.ErrorObservation{{Issue: 1, Observed: fail, Current: issuepolicy.ErrorEvidence{Status: "fail", Fingerprint: "fp", Module: "internal/x", FailureClass: "panic", Commit: c, Witness: "current"}}}})
	var out, errb bytes.Buffer
	code := RunIssueInventory(&out, &errb, []string{"--from-issues", issues, "--from-observations", obs, "--json", "--require-actionable", "1"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), `"schema": "fak.issue-error-inventory/1"`) || !strings.Contains(out.String(), `"disposition": "ACTIONABLE"`) {
		t.Fatalf("output=%s", out.String())
	}
	first := out.String()
	out.Reset()
	errb.Reset()
	if code = RunIssueInventory(&out, &errb, []string{"--from-issues", issues, "--from-observations", obs, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	if out.String() != first {
		t.Fatalf("fixture output changed\nfirst=%s\nsecond=%s", first, out.String())
	}
}

func TestRunIssueInventoryRefusesFixedAndEvidenceDeficient(t *testing.T) {
	dir, a, b, c := issueInventoryRepositoryFixture(t)
	t.Chdir(dir)
	issues := writeInventoryFixture(t, dir, "issues.json", []map[string]any{{"number": 1, "title": "fixed"}, {"number": 2, "title": "unknown"}})
	obs := issueInventoryInput{GeneratedAt: time.Unix(1, 0).UTC().Format(time.RFC3339), Observations: []issuepolicy.ErrorObservation{
		{Issue: 1, Observed: issuepolicy.ErrorEvidence{Status: "fail", Fingerprint: "fp", Module: "internal/x", FailureClass: "panic", Commit: a, Witness: "old"}, Fix: issuepolicy.ErrorEvidence{Status: "pass", Commit: b, Witness: "fix"}, Current: issuepolicy.ErrorEvidence{Status: "pass", Commit: c, Witness: "trunk"}},
		{Issue: 2},
	}}
	path := writeInventoryFixture(t, dir, "obs.json", obs)
	for _, tc := range []struct{ issue, want int }{{1, 3}, {2, 4}, {99, 2}} {
		var out, errb bytes.Buffer
		if got := RunIssueInventory(&out, &errb, []string{"--from-issues", issues, "--from-observations", path, "--require-actionable", strconv.Itoa(tc.issue)}); got != tc.want {
			t.Fatalf("issue=%d code=%d want=%d stderr=%s", tc.issue, got, tc.want, errb.String())
		}
	}
}

func TestRunIssueInventoryTreatsIssueBodyAsInertData(t *testing.T) {
	dir, _, _, _ := issueInventoryRepositoryFixture(t)
	t.Chdir(dir)
	marker := filepath.Join(dir, "executed")
	issues := writeInventoryFixture(t, dir, "issues.json", []map[string]any{{"number": 1, "title": "report", "body": "touch " + marker}})
	obs := writeInventoryFixture(t, dir, "obs.json", issueInventoryInput{GeneratedAt: "2026-09-01T00:00:00Z", Observations: []issuepolicy.ErrorObservation{{Issue: 1}}})
	var out, errb bytes.Buffer
	if code := RunIssueInventory(&out, &errb, []string{"--from-issues", issues, "--from-observations", obs, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("issue body was executed: %v", err)
	}
}

// fak-test:runtime medium est=3s lane=default
// fak-test:justify Resolves three real historical commits and release ancestry in bounded temporary Git repositories.
func TestRunIssueInventoryResolvesRepositoryProvenance(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*issuepolicy.ErrorObservation, string, string, string)
		want   issuepolicy.ErrorDisposition
	}{
		{"valid trunk fix", nil, issuepolicy.DispositionFixPresentTrunk},
		{"invented observed commit", func(o *issuepolicy.ErrorObservation, _, _, _ string) { o.Observed.Commit = "not-a-commit" }, issuepolicy.DispositionReproRequired},
		{"invented fixing commit", func(o *issuepolicy.ErrorObservation, _, _, _ string) { o.Fix.Commit = "not-a-fix" }, issuepolicy.DispositionReproRequired},
		{"invented module version", func(o *issuepolicy.ErrorObservation, _, _, _ string) {
			o.Current.ModuleVersion = "invented-module-version"
		}, issuepolicy.DispositionReproRequired},
		{"wrong module", func(o *issuepolicy.ErrorObservation, _, _, _ string) { o.Fix.Module = "internal/absent" }, issuepolicy.DispositionReproRequired},
		{"ref commit mismatch", func(o *issuepolicy.ErrorObservation, a, _, _ string) { o.Current.Ref = a }, issuepolicy.DispositionReproRequired},
		{"stale tested trunk", func(o *issuepolicy.ErrorObservation, _, b, _ string) { o.Current.Commit = b }, issuepolicy.DispositionStaleEvidence},
		{"invented release", func(o *issuepolicy.ErrorObservation, _, _, _ string) {
			o.Releases = []issuepolicy.ReleaseEvidence{{Tag: "v999.999.999", FixAncestor: true, Witness: "unverified"}}
		}, issuepolicy.DispositionReproRequired},
		{"forged release ancestry", func(o *issuepolicy.ErrorObservation, _, _, _ string) {
			o.Releases = []issuepolicy.ReleaseEvidence{{Tag: "v0.1.0", FixAncestor: true, Witness: "unverified"}}
		}, issuepolicy.DispositionReproRequired},
		{"release-specific passing receipt", func(o *issuepolicy.ErrorObservation, _, _, _ string) {
			o.Releases = []issuepolicy.ReleaseEvidence{{Tag: "v0.1.0", PassingReceipt: true, Witness: "trusted-release-ci"}}
		}, issuepolicy.DispositionFixReleased},
		{"release version mismatch", func(o *issuepolicy.ErrorObservation, _, _, _ string) {
			o.Releases = []issuepolicy.ReleaseEvidence{{Tag: "v0.1.0", ModuleVersion: "invented-release-version"}}
		}, issuepolicy.DispositionReproRequired},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root, a, b, c := issueInventoryRepositoryFixture(t)
			t.Chdir(root)
			o := issuepolicy.ErrorObservation{Issue: 1,
				Observed: issuepolicy.ErrorEvidence{Status: "fail", Fingerprint: "fp", Module: "internal/x", FailureClass: "panic", Commit: a, Witness: "observed"},
				Fix:      issuepolicy.ErrorEvidence{Status: "pass", Module: "internal/x", Commit: b, Witness: "fix"},
				Current:  issuepolicy.ErrorEvidence{Status: "pass", Module: "internal/x", Commit: c, Witness: "current"},
			}
			if tc.mutate != nil {
				tc.mutate(&o, a, b, c)
			}
			issues := writeInventoryFixture(t, root, "issues.json", []map[string]any{{"number": 1, "title": "source-bound fixture"}})
			observations := writeInventoryFixture(t, root, "observations.json", issueInventoryInput{GeneratedAt: "2026-09-01T00:00:00Z", Observations: []issuepolicy.ErrorObservation{o}})
			var out, stderr bytes.Buffer
			code := RunIssueInventory(&out, &stderr, []string{"--from-issues", issues, "--from-observations", observations, "--require-actionable", "1", "--json"})
			if code != issuepolicy.ActionabilityExit(tc.want) {
				t.Fatalf("code=%d want=%d stderr=%s output=%s", code, issuepolicy.ActionabilityExit(tc.want), stderr.String(), out.String())
			}
			var rep issuepolicy.ErrorInventory
			if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
				t.Fatal(err)
			}
			if len(rep.Issues) != 1 || rep.Issues[0].Disposition != tc.want {
				t.Fatalf("report=%+v want=%s", rep, tc.want)
			}
			if tc.mutate == nil {
				r := rep.Issues[0]
				for i, e := range []issuepolicy.ErrorEvidence{r.ObservedFailure, r.ProvenFix, r.CurrentTestedState} {
					if e.Commit != []string{a, b, c}[i] || !strings.HasPrefix(e.ModuleVersion, "internal/x@r") {
						t.Fatalf("historical provenance %d not resolved: %+v", i, e)
					}
				}
				if r.ObservedFailure.ModuleVersion == r.ProvenFix.ModuleVersion || r.ProvenFix.ModuleVersion == r.CurrentTestedState.ModuleVersion {
					t.Fatal("three historical module revisions collapsed")
				}
			}
		})
	}
	t.Run("numeric releases and unverified duplicate isolation", func(t *testing.T) {
		root, a, b, c := issueInventoryRepositoryFixture(t)
		t.Chdir(root)
		runGraphGit(t, root, "tag", "v1.10.0", c)
		runGraphGit(t, root, "tag", "v1.2.0", b)
		valid := issuepolicy.ErrorObservation{Issue: 2,
			Observed: issuepolicy.ErrorEvidence{Status: "fail", Fingerprint: "fp", Module: "internal/x", FailureClass: "panic", Commit: a, Witness: "observed"},
			Fix:      issuepolicy.ErrorEvidence{Status: "pass", Commit: b, Witness: "fix"},
			Current:  issuepolicy.ErrorEvidence{Status: "pass", Commit: c, Witness: "current"},
			Releases: []issuepolicy.ReleaseEvidence{{Tag: "v1.10.0", FixAncestor: true, Witness: "later"}, {Tag: "v1.2.0", FixAncestor: true, Witness: "earlier"}},
		}
		invalid := valid
		invalid.Issue = 1
		invalid.Observed.Commit = "invented"
		duplicate := valid
		duplicate.Issue = 3
		issues := writeInventoryFixture(t, root, "issues.json", []map[string]any{{"number": 1}, {"number": 2}, {"number": 3}})
		observations := writeInventoryFixture(t, root, "observations.json", issueInventoryInput{GeneratedAt: "2026-09-01T00:00:00Z", Observations: []issuepolicy.ErrorObservation{invalid, valid, duplicate}})
		var out, stderr bytes.Buffer
		if code := RunIssue(&out, &stderr, []string{"inventory", "--root", root, "--from-issues", issues, "--from-observations", observations, "--json"}); code != 0 {
			t.Fatalf("code=%d stderr=%s", code, stderr.String())
		}
		var rep issuepolicy.ErrorInventory
		if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
			t.Fatal(err)
		}
		if len(rep.Issues) != 3 || rep.Issues[0].Disposition != issuepolicy.DispositionReproRequired || rep.Issues[1].Disposition != issuepolicy.DispositionFixReleased || rep.Issues[2].Disposition != issuepolicy.DispositionDuplicateProven || rep.Issues[2].CanonicalIssue != 2 {
			t.Fatalf("duplicate isolation: %+v", rep.Issues)
		}
		rel := rep.Issues[1].FirstContainingRelease
		if rel == nil || rel.Tag != "v1.2.0" || !rel.FixAncestor || !strings.HasPrefix(rel.Witness, "git:") || !strings.HasPrefix(rel.ModuleVersion, "internal/x@r2+") {
			t.Fatalf("release=%+v", rel)
		}

		proof, err := newIssueInventoryProof(t.Context(), root, "refs/heads/main")
		if err != nil {
			t.Fatal(err)
		}
		if _, reason := proof.resolveObservation(t.Context(), valid); reason != "" {
			t.Fatal(reason)
		}
		runGraphGit(t, root, "tag", "-f", "v1.2.0", a)
		if proof.unchanged(t.Context()) {
			t.Fatal("moved release ref was accepted")
		}
	})
	t.Run("shallow history and unrelated fix refuse", func(t *testing.T) {
		root, a, _, c := issueInventoryRepositoryFixture(t)
		runGraphGit(t, root, "checkout", "-q", "-b", "side", a)
		if err := os.WriteFile(filepath.Join(root, "internal/x/x.go"), []byte("package x\n// unrelated fix\n"), 0644); err != nil {
			t.Fatal(err)
		}
		runGraphGit(t, root, "add", "internal/x/x.go")
		runGraphGit(t, root, "commit", "-qm", "unrelated fix")
		d := strings.TrimSpace(runGraphGit(t, root, "rev-parse", "HEAD"))
		runGraphGit(t, root, "checkout", "-q", "main")
		o := issuepolicy.ErrorObservation{Issue: 1, Observed: issuepolicy.ErrorEvidence{Module: "internal/x", Commit: a}, Fix: issuepolicy.ErrorEvidence{Commit: d}, Current: issuepolicy.ErrorEvidence{Commit: c}}
		p, err := newIssueInventoryProof(context.Background(), root, "refs/heads/main")
		if err != nil {
			t.Fatal(err)
		}
		if _, reason := p.resolveObservation(t.Context(), o); reason != "FIX_ANCESTRY_UNVERIFIED" {
			t.Fatalf("unrelated fixing branch: %s", reason)
		}
		if err := os.WriteFile(filepath.Join(root, ".git/shallow"), []byte(c+"\n"), 0644); err != nil {
			t.Fatal(err)
		}
		p, err = newIssueInventoryProof(t.Context(), root, "refs/heads/main")
		if err != nil {
			t.Fatal(err)
		}
		if _, reason := p.resolveObservation(t.Context(), o); reason != "REPOSITORY_HISTORY_INCOMPLETE" {
			t.Fatalf("shallow history: %s", reason)
		}
	})
}

func issueInventoryRepositoryFixture(t *testing.T) (root, observed, fixed, current string) {
	t.Helper()
	root = t.TempDir()
	runGraphGit(t, root, "init", "-q", "-b", "main")
	runGraphGit(t, root, "config", "user.name", "Inventory Test")
	runGraphGit(t, root, "config", "user.email", "inventory@example.invalid")
	for i := 0; i < 3; i++ {
		p := filepath.Join(root, "internal", "x", "x.go")
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("package x\n// revision "+string(rune('0'+i))+"\n"), 0644); err != nil {
			t.Fatal(err)
		}
		runGraphGit(t, root, "add", "internal/x/x.go")
		runGraphGit(t, root, "commit", "-qm", "fixture revision")
		sha := strings.TrimSpace(runGraphGit(t, root, "rev-parse", "HEAD"))
		switch i {
		case 0:
			observed = sha
			runGraphGit(t, root, "tag", "v0.1.0")
		case 1:
			fixed = sha
		case 2:
			current = sha
		}
	}
	return
}

// fak-test:runtime medium est=2s lane=default
// fak-test:justify Exercises real immutable tag ancestry under an ABA mutation and missing promised trees without network access.
func TestRunIssueInventoryKeepsReadsLocalAndPinned(t *testing.T) {
	t.Run("release tag ABA never borrows containment", func(t *testing.T) {
		root, a, b, c := issueInventoryRepositoryFixture(t)
		// Capture an annotated tag whose immutable target does not contain b.
		runGraphGit(t, root, "tag", "-a", "v1.0.0", a, "-m", "before fix")
		tagObject := strings.TrimSpace(runGraphGit(t, root, "rev-parse", "refs/tags/v1.0.0"))
		p, err := newIssueInventoryProof(t.Context(), root, "refs/heads/main")
		if err != nil {
			t.Fatal(err)
		}
		mutated := false
		p.run = func(ctx context.Context, dir string, args ...string) ([]byte, error) {
			joined := strings.Join(args, " ")
			// The first predicate reproduces the rejected live-name implementation;
			// the second exercises the replacement immutable-object ancestry read.
			trigger := strings.Contains(joined, "for-each-ref --contains=") || strings.Contains(joined, "--ancestry-path "+b+".."+a)
			if trigger && !mutated {
				mutated = true
				runGraphGit(t, root, "update-ref", "refs/tags/v1.0.0", b)
				out, err := modver.RealRunner(ctx, dir, args...)
				runGraphGit(t, root, "update-ref", "refs/tags/v1.0.0", tagObject)
				return out, err
			}
			return modver.RealRunner(ctx, dir, args...)
		}
		o := issuepolicy.ErrorObservation{Issue: 1, Observed: issuepolicy.ErrorEvidence{Status: "fail", Fingerprint: "fp", Module: "internal/x", FailureClass: "panic", Commit: a, Witness: "baseline"}, Fix: issuepolicy.ErrorEvidence{Status: "pass", Commit: b, Witness: "fix"}, Current: issuepolicy.ErrorEvidence{Status: "pass", Commit: c, Witness: "current"}}
		resolved, reason := p.resolveObservation(t.Context(), o)
		if reason != "" {
			t.Fatal(reason)
		}
		if !mutated {
			t.Fatal("ABA mutation did not execute")
		}
		if !p.unchanged(t.Context()) {
			t.Fatal("fixture did not restore original refs")
		}
		if len(resolved.Releases) != 0 {
			t.Fatalf("borrowed containment after ABA: %+v", resolved.Releases)
		}
	})
	t.Run("missing promised trees never fetch", func(t *testing.T) {
		root, a, _, _ := issueInventoryRepositoryFixture(t)
		runGraphGit(t, root, "config", "uploadpack.allowFilter", "true")
		partial := filepath.Join(t.TempDir(), "partial")
		runGraphGit(t, root, "clone", "--no-local", "--no-checkout", "--filter=tree:0", root, partial)
		tree := strings.TrimSpace(runGraphGit(t, root, "rev-parse", a+"^{tree}"))
		trace := filepath.Join(t.TempDir(), "git-trace.jsonl")
		t.Setenv("GIT_TRACE2_EVENT", trace)
		p, err := newIssueInventoryProof(t.Context(), partial, "refs/heads/main")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := p.git(t.Context(), "cat-file", "-e", tree); err == nil {
			t.Fatal("partial clone fixture unexpectedly has the promised tree")
		}
		if _, err := p.module(t.Context(), a, "internal/x"); err == nil {
			t.Fatal("missing historical tree was admitted")
		}
		assertFetch := func(want bool) {
			t.Helper()
			raw, err := os.ReadFile(trace)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, line := range bytes.Split(raw, []byte{'\n'}) {
				if len(line) == 0 {
					continue
				}
				var event struct {
					Name string   `json:"name"`
					Argv []string `json:"argv"`
				}
				if err := json.Unmarshal(line, &event); err != nil {
					t.Fatal(err)
				}
				if event.Name == "fetch" {
					found = true
				}
				for _, arg := range event.Argv {
					if arg == "fetch" {
						found = true
					}
				}
			}
			if found != want {
				t.Fatalf("lazy fetch observed=%v want=%v", found, want)
			}
		}
		assertFetch(false)
		// Positive fixture control: ordinary Git can retrieve that same tree from
		// the local temporary source, proving the no-fetch control was meaningful.
		runGraphGit(t, partial, "cat-file", "-e", tree)
		assertFetch(true)
	})
}

// fak-test:runtime medium est=1s lane=default
// fak-test:justify Checks canonical observation ordering against one bounded real Git fixture.
func TestRunIssueInventoryCanonicalObservationOrder(t *testing.T) {
	root, a, b, c := issueInventoryRepositoryFixture(t)
	t.Chdir(root)
	issues := writeInventoryFixture(t, root, "issues.json", []map[string]any{
		{"number": 1, "title": "first verified"}, {"number": 2, "title": "second verified"},
		{"number": 3, "title": "first refused"}, {"number": 4, "title": "second refused"},
	})
	observations := make([]issuepolicy.ErrorObservation, 4)
	for i := range observations {
		observations[i] = issuepolicy.ErrorObservation{Issue: i + 1,
			Observed: issuepolicy.ErrorEvidence{Status: "fail", Fingerprint: "fp-" + strconv.Itoa(i), Module: "internal/x", FailureClass: "panic", Commit: a, Witness: "observed"},
			Fix:      issuepolicy.ErrorEvidence{Status: "pass", Commit: b, Witness: "fix"},
			Current:  issuepolicy.ErrorEvidence{Status: "pass", Commit: c, Witness: "current"},
		}
	}
	observations[2].Observed.Commit = "not-a-commit"
	observations[3].Current.ModuleVersion = "not-a-module-version"
	var canonical []byte
	for i, rows := range [][]issuepolicy.ErrorObservation{observations, {observations[3], observations[2], observations[1], observations[0]}} {
		path := writeInventoryFixture(t, root, "observations-"+strconv.Itoa(i)+".json", issueInventoryInput{GeneratedAt: "2026-09-01T00:00:00Z", Observations: rows})
		var out, stderr bytes.Buffer
		if code := RunIssueInventory(&out, &stderr, []string{"--from-issues", issues, "--from-observations", path, "--json"}); code != 0 {
			t.Fatalf("code=%d stderr=%s", code, stderr.String())
		}
		var report issuepolicy.ErrorInventory
		if err := json.Unmarshal(out.Bytes(), &report); err != nil {
			t.Fatal(err)
		}
		if len(report.Issues) != 4 || report.Issues[0].Disposition != issuepolicy.DispositionFixPresentTrunk || report.Issues[1].Disposition != issuepolicy.DispositionFixPresentTrunk || report.Issues[2].Disposition != issuepolicy.DispositionReproRequired || report.Issues[3].Disposition != issuepolicy.DispositionReproRequired {
			t.Fatalf("fixture must resolve two verified and two refused observations: %+v", report.Issues)
		}
		if i == 0 {
			canonical = append([]byte(nil), out.Bytes()...)
		} else if !bytes.Equal(canonical, out.Bytes()) {
			t.Fatal("permuting observations changed the canonical report or snapshot digest")
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestIssueInventoryTrunkRefUsesBranchRole(t *testing.T) {
	t.Setenv("FLEET_BRANCH_PILOT", "")
	root := t.TempDir()
	if got, err := issueInventoryTrunkRef(root, ""); err != nil || got != "refs/heads/main" {
		t.Fatalf("unconfigured ref=%q err=%v", got, err)
	}
	path := filepath.Join(root, "dos.toml")
	if err := os.WriteFile(path, []byte("[branch_roles]\ndevelopment_branch = \"integration\"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if got, err := issueInventoryTrunkRef(root, ""); err != nil || got != "refs/heads/integration" {
		t.Fatalf("configured ref=%q err=%v", got, err)
	}
	if err := os.WriteFile(path, []byte("[branch_roles]\ndevelopment_branch = 123\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := issueInventoryTrunkRef(root, ""); err == nil {
		t.Fatal("malformed roles silently accepted")
	}
	if got, err := issueInventoryTrunkRef(root, "refs/heads/reviewed"); err != nil || got != "refs/heads/reviewed" {
		t.Fatalf("explicit ref=%q err=%v", got, err)
	}
}
