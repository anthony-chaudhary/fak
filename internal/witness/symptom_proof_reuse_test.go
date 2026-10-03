package witness

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const symptomProofReusedDetail = "reused confirmed selected symptom proof (parent failed; candidate passed)"

func writeSymptomProofFile(t *testing.T, root, name, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func symptomProofHead(t *testing.T, root string) string {
	t.Helper()
	cmd := exec.Command("git", "rev-parse", "HEAD")
	cleanGitEnv(cmd)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	return strings.TrimSpace(string(out))
}

func commitSymptomProofCandidate(t *testing.T, root, message string) string {
	t.Helper()
	writeSymptomProofFile(t, root, "sign.go", `package symptom

import dep "helper.test"

func Sign(n int) int {
	if n < 0 {
		return -1 + dep.Bias()
	}
	if n > 0 {
		return 1 + dep.Bias()
	}
	return dep.Bias()
}
`)
	writeSymptomProofFile(t, root, "sign_test.go", `package symptom

import "testing"

func TestSelectedNegative(t *testing.T) {
	if got := Sign(-3); got != -1 {
		t.Fatalf("Sign(-3) = %d, want -1", got)
	}
}

func TestAlternateNegative(t *testing.T) {
	if got := Sign(-7); got != -1 {
		t.Fatalf("Sign(-7) = %d, want -1", got)
	}
}
`)
	gitIn(t, root, "add", "sign.go", "sign_test.go")
	gitIn(t, root, "commit", "-q", "-m", message)
	return symptomProofHead(t, root)
}

func resolveSymptomProof(t *testing.T, root, ref string, selectors, tags []string) (abi.WitnessOutcome, string) {
	t.Helper()
	r := NewWithRunner(gitRunner, root).WithSymptomTests(selectors).WithSymptomTags(tags)
	return r.ResolveSymptomWithDetail(context.Background(), ref, true)
}

// fak-test:justify why=regression when=changed:internal/witness/** witness=internal/witness/symptom.go#func-Resolver.ResolveSymptomWithDetail
// fak-test:runtime slow est=220s lane=default
func TestSymptomProofReuseSurvivesUnrelatedChurnAndInvalidatesEveryExecutionInput(t *testing.T) {
	requireGoAndGit(t)
	t.Setenv(SymptomFlagEnv, "1")
	originalGoFlags := os.Getenv("GOFLAGS")

	root := newExecutionRepo(t)
	writeSymptomProofFile(t, root, "go.mod", `module example.test

go 1.22

require helper.test v0.0.0

replace helper.test => ./dep
`)
	writeSymptomProofFile(t, root, "dep/go.mod", "module helper.test\n\ngo 1.22\n")
	writeSymptomProofFile(t, root, "dep/value.go", "package value\n\nfunc Bias() int { return 0 }\n")
	writeSymptomProofFile(t, root, "sign.go", `package symptom

import dep "helper.test"

func Sign(n int) int {
	if n > 0 {
		return 1 + dep.Bias()
	}
	return dep.Bias()
}
`)
	gitIn(t, root, "add", ".")
	gitIn(t, root, "commit", "-q", "-m", "parent: negative sign bug")
	base := symptomProofHead(t, root)

	first := commitSymptomProofCandidate(t, root, "fix: selected symptom")
	selectors := []string{"^TestSelectedNegative$"}
	got, detail := resolveSymptomProof(t, root, first, selectors, nil)
	if got != abi.WitnessConfirmed || detail == symptomProofReusedDetail {
		t.Fatalf("first proof = %v %q, want freshly executed confirmation", got, detail)
	}

	gitIn(t, root, "checkout", "-q", "-b", "unrelated-parent", base)
	writeSymptomProofFile(t, root, "README.md", "unrelated parent churn\n")
	gitIn(t, root, "add", "README.md")
	gitIn(t, root, "commit", "-q", "-m", "docs: unrelated churn")
	unrelatedParent := symptomProofHead(t, root)
	second := commitSymptomProofCandidate(t, root, "fix: same selected symptom")
	got, detail = resolveSymptomProof(t, root, second, selectors, nil)
	if got != abi.WitnessConfirmed || detail != symptomProofReusedDetail {
		t.Fatalf("equivalent proof after unrelated churn = %v %q, want cache reuse", got, detail)
	}

	outer := NewWithRunner(gitRunner, root).WithSymptomTests(selectors)
	if got := outer.Resolve(context.Background(), nil, "symptom:"+second); got != abi.WitnessConfirmed {
		t.Fatalf("outer Resolve selected proof = %v, want CONFIRMED", got)
	}
	noMatch := NewWithRunner(gitRunner, root).WithSymptomTests([]string{"^TestMissing$"})
	if got := noMatch.Resolve(context.Background(), nil, "symptom:"+second); got != abi.WitnessAbstain {
		t.Fatalf("outer Resolve no-match after confirmed selection = %v, want ABSTAIN (no legacy cache collision)", got)
	}
	got, detail = resolveSymptomProof(t, root, second, []string{"^TestAlternateNegative$"}, nil)
	if got != abi.WitnessConfirmed || detail == symptomProofReusedDetail {
		t.Fatalf("changed selected test = %v %q, want fresh confirmation", got, detail)
	}
	got, detail = resolveSymptomProof(t, root, second, selectors, []string{"proof_dimension"})
	if got != abi.WitnessConfirmed || detail == symptomProofReusedDetail {
		t.Fatalf("changed normalized tags = %v %q, want fresh confirmation", got, detail)
	}
	t.Setenv("GOFLAGS", strings.TrimSpace(originalGoFlags+" -mod=mod"))
	got, detail = resolveSymptomProof(t, root, second, selectors, nil)
	if got != abi.WitnessConfirmed || detail == symptomProofReusedDetail {
		t.Fatalf("changed effective Go environment = %v %q, want fresh confirmation", got, detail)
	}
	t.Setenv("GOFLAGS", originalGoFlags)

	gitIn(t, root, "checkout", "-q", "-b", "changed-dependency", unrelatedParent)
	writeSymptomProofFile(t, root, "dep/value.go", "package value\n\nfunc Bias() int { value := 0; return value }\n")
	gitIn(t, root, "add", "dep/value.go")
	gitIn(t, root, "commit", "-q", "-m", "refactor dep.test dependency")
	third := commitSymptomProofCandidate(t, root, "fix: symptom with changed dependency")
	got, detail = resolveSymptomProof(t, root, third, selectors, nil)
	if got != abi.WitnessConfirmed || detail == symptomProofReusedDetail {
		t.Fatalf("changed dep.test source = %v %q, want fresh confirmation", got, detail)
	}
	assertRepoClean(t, root)
}
