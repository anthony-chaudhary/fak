package fixproof

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
)

type fixProofFixture struct {
	repository, companion string
	request               Request
}

func fixProofDigest(seed string) string {
	sum := sha256.Sum256([]byte(seed))
	return hex.EncodeToString(sum[:])
}

func fixProofRequestFixture() Request {
	return Request{
		Schema: RequestSchema, Contract: Contract,
		ParentCommit: strings.Repeat("1", 40), ParentTree: strings.Repeat("2", 40),
		CandidateCommit: strings.Repeat("3", 40), CandidateTree: strings.Repeat("4", 40),
		CompanionCommit: strings.Repeat("5", 40), CompanionTree: strings.Repeat("6", 40),
		Paths: []string{"pkg/calc/calc_test.go"}, TestOverlays: []Overlay{{Path: "pkg/calc/calc_test.go", Blob: strings.Repeat("7", 40)}},
		PackageDir: "pkg/calc", Selectors: []string{"^TestFix$"}, TimeoutMillis: 300_000,
		Context: ExecutionContext{
			GoExecutableDigest: fixProofDigest("go"), ToolchainDigest: fixProofDigest("toolchain"),
			TestEnvDigest: fixProofDigest("test-env"), WorkspaceDigest: fixProofDigest("workspace"), VerifierDigest: fixProofDigest("verifier"),
		},
	}
}

func cloneFixProofRequest(request Request) Request {
	request.Paths = append([]string(nil), request.Paths...)
	request.TestOverlays = append([]Overlay(nil), request.TestOverlays...)
	request.Selectors = append([]string(nil), request.Selectors...)
	request.BuildTags = append([]string(nil), request.BuildTags...)
	return request
}

func differentFixProofObject(object string) string {
	if strings.HasPrefix(object, "0") {
		return "1" + object[1:]
	}
	return "0" + object[1:]
}

func fixProofGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = dir
	cmd.Env = append(scrubFixProofGitEnv(os.Environ()),
		"GIT_AUTHOR_NAME=fixproof-test", "GIT_AUTHOR_EMAIL=fixproof@example.test",
		"GIT_COMMITTER_NAME=fixproof-test", "GIT_COMMITTER_EMAIL=fixproof@example.test",
		"GIT_TERMINAL_PROMPT=0")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, dir, err, out)
	}
	return strings.TrimSpace(string(out))
}

func scrubFixProofGitEnv(env []string) []string {
	out := make([]string, 0, len(env))
	for _, entry := range env {
		key, _, _ := strings.Cut(entry, "=")
		switch strings.ToUpper(key) {
		case "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES":
			continue
		}
		out = append(out, entry)
	}
	return out
}

func writeFixProofFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

const (
	fixProofParentImplementation    = "package calc\n\nfunc Abs(n int) int { return n }\n"
	fixProofCandidateImplementation = "package calc\n\nfunc Abs(n int) int { if n < 0 { return -n }; return n }\n"
	fixProofBrokenImplementation    = "package calc\n\nfunc Abs(n int) int { broken }\n"
	fixProofParentTestSource        = "package calc\n\n// TestFix is introduced by the candidate.\n"
)

func fixProofTestSource(marker string, fail bool) string {
	body := "package calc\n\nimport (\"os\"; \"testing\"; \"time\")\n\n"
	body += "func mark() { f, _ := os.OpenFile(" + strconv.Quote(marker) + ", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600); if f != nil { _, _ = f.WriteString(\"executed\\n\"); _ = f.Close() } }\n"
	body += "func assertFixed(t *testing.T) { t.Helper(); if got := Abs(-1); got != 1 { t.Fatalf(\"Abs(-1) = %d, want 1\", got) } }\n"
	body += "func TestFix(t *testing.T) { mark(); assertFixed(t)"
	if fail {
		body += "; t.Fatal(\"deliberate candidate failure\")"
	}
	body += " }\n"
	body += "func TestSlow(t *testing.T) { mark(); time.Sleep(2*time.Second); assertFixed(t)"
	if fail {
		body += "; t.Fatal(\"deliberate slow candidate failure\")"
	}
	body += " }\n"
	return body
}

func initFixProofSeed(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	fixProofGit(t, dir, "init", "-q", "-b", "main")
	fixProofGit(t, dir, "config", "user.name", "fixproof-test")
	fixProofGit(t, dir, "config", "user.email", "fixproof@example.test")
	fixProofGit(t, dir, "config", "commit.gpgsign", "false")
	fixProofGit(t, dir, "config", "core.hooksPath", "")
}

func cloneDetachedFixProofRepo(t *testing.T, source, destination, commit string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		t.Fatal(err)
	}
	fixProofGit(t, filepath.Dir(destination), "clone", "-q", "--no-local", source, destination)
	fixProofGit(t, destination, "config", "commit.gpgsign", "false")
	fixProofGit(t, destination, "config", "core.hooksPath", "")
	fixProofGit(t, destination, "checkout", "-q", "--detach", commit)
	if branch := fixProofGit(t, destination, "rev-parse", "--abbrev-ref", "HEAD"); branch != "HEAD" {
		t.Fatalf("sandbox clone remains branch-attached: %q", branch)
	}
	if status := fixProofGit(t, destination, "status", "--porcelain"); status != "" {
		t.Fatalf("sandbox clone is dirty: %q", status)
	}
	info, err := os.Stat(filepath.Join(destination, ".git"))
	if err != nil || !info.IsDir() {
		t.Fatalf("sandbox clone lacks an independent .git directory: info=%v err=%v", info, err)
	}
}

func fixProofCommonDir(t *testing.T, repository string) string {
	t.Helper()
	common := fixProofGit(t, repository, "rev-parse", "--path-format=absolute", "--git-common-dir")
	common, err := filepath.Abs(common)
	if err != nil {
		t.Fatal(err)
	}
	repository, err = filepath.Abs(repository)
	if err != nil {
		t.Fatal(err)
	}
	rel, err := filepath.Rel(repository, common)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		t.Fatalf("Git common directory %q escapes sandbox clone %q", common, repository)
	}
	return filepath.Clean(common)
}

func newFixProofFixture(t *testing.T, candidateTestSource, parentImplementation string) fixProofFixture {
	t.Helper()
	root := t.TempDir()
	seedRoot := filepath.Join(root, "seeds")
	repositorySeed := filepath.Join(seedRoot, "fak")
	companionSeed := filepath.Join(seedRoot, "fak-private")
	initFixProofSeed(t, repositorySeed)
	initFixProofSeed(t, companionSeed)

	writeFixProofFile(t, filepath.Join(companionSeed, "go.mod"), "module example.test/companion\n\ngo 1.23.0\n")
	writeFixProofFile(t, filepath.Join(companionSeed, "companion.go"), "package companion\n\nconst Ready = true\n")
	fixProofGit(t, companionSeed, "add", ".")
	fixProofGit(t, companionSeed, "commit", "-qm", "seed companion")
	companionCommit := fixProofGit(t, companionSeed, "rev-parse", "HEAD")
	companionTree := fixProofGit(t, companionSeed, "rev-parse", "HEAD^{tree}")

	writeFixProofFile(t, filepath.Join(repositorySeed, "go.mod"), "module example.test/fak\n\ngo 1.23.0\n")
	writeFixProofFile(t, filepath.Join(repositorySeed, "go.work"), "go 1.23.0\n\nuse (\n\t.\n\t../fak-private\n)\n")
	writeFixProofFile(t, filepath.Join(repositorySeed, "pkg", "calc", "calc.go"), parentImplementation)
	writeFixProofFile(t, filepath.Join(repositorySeed, "pkg", "calc", "calc_test.go"), fixProofParentTestSource)
	fixProofGit(t, repositorySeed, "add", ".")
	fixProofGit(t, repositorySeed, "commit", "-qm", "parent")
	parentCommit := fixProofGit(t, repositorySeed, "rev-parse", "HEAD")
	parentTree := fixProofGit(t, repositorySeed, "rev-parse", "HEAD^{tree}")

	writeFixProofFile(t, filepath.Join(repositorySeed, "pkg", "calc", "calc.go"), fixProofCandidateImplementation)
	writeFixProofFile(t, filepath.Join(repositorySeed, "pkg", "calc", "calc_test.go"), candidateTestSource)
	fixProofGit(t, repositorySeed, "add", "pkg/calc/calc_test.go")
	fixProofGit(t, repositorySeed, "add", "pkg/calc/calc.go")
	fixProofGit(t, repositorySeed, "commit", "-qm", "candidate")
	candidateCommit := fixProofGit(t, repositorySeed, "rev-parse", "HEAD")
	candidateTree := fixProofGit(t, repositorySeed, "rev-parse", "HEAD^{tree}")
	overlayBlob := fixProofGit(t, repositorySeed, "rev-parse", "HEAD:pkg/calc/calc_test.go")

	sandboxRoot := filepath.Join(root, "sandbox")
	repository := filepath.Join(sandboxRoot, "fak")
	companion := filepath.Join(sandboxRoot, "fak-private")
	cloneDetachedFixProofRepo(t, repositorySeed, repository, candidateCommit)
	cloneDetachedFixProofRepo(t, companionSeed, companion, companionCommit)
	if repositoryCommon, companionCommon := fixProofCommonDir(t, repository), fixProofCommonDir(t, companion); repositoryCommon == companionCommon {
		t.Fatalf("proof sandboxes share Git common directory %q", repositoryCommon)
	}

	request := fixProofRequestFixture()
	request.ParentCommit, request.ParentTree = parentCommit, parentTree
	request.CandidateCommit, request.CandidateTree = candidateCommit, candidateTree
	request.CompanionCommit, request.CompanionTree = companionCommit, companionTree
	request.Paths = []string{"pkg/calc/calc.go", "pkg/calc/calc_test.go"}
	request.TestOverlays[0].Blob = overlayBlob
	return fixProofFixture{repository: repository, companion: companion, request: request}
}

func fixProofRoots(f fixProofFixture) Roots {
	return Roots{RepositoryDir: f.repository, CompanionDir: f.companion}
}

// fak-test:runtime fast est=100ms lane=default
func TestFixProofRequestDigestBindsEveryIdentityAndContextField(t *testing.T) {
	request := fixProofRequestFixture()
	base, err := request.Digest()
	if err != nil {
		t.Fatalf("Digest(valid request): %v", err)
	}
	mutations := map[string]func(*Request){
		"parent commit":    func(r *Request) { r.ParentCommit = differentFixProofObject(r.ParentCommit) },
		"parent tree":      func(r *Request) { r.ParentTree = differentFixProofObject(r.ParentTree) },
		"candidate commit": func(r *Request) { r.CandidateCommit = differentFixProofObject(r.CandidateCommit) },
		"candidate tree":   func(r *Request) { r.CandidateTree = differentFixProofObject(r.CandidateTree) },
		"companion commit": func(r *Request) { r.CompanionCommit = differentFixProofObject(r.CompanionCommit) },
		"companion tree":   func(r *Request) { r.CompanionTree = differentFixProofObject(r.CompanionTree) },
		"paths":            func(r *Request) { r.Paths = []string{"pkg/calc/calc_test.go", "pkg/calc/extra.go"} },
		"overlay blob":     func(r *Request) { r.TestOverlays[0].Blob = differentFixProofObject(r.TestOverlays[0].Blob) },
		"package": func(r *Request) {
			r.PackageDir = "."
			r.Paths = []string{"calc_test.go"}
			r.TestOverlays = []Overlay{{Path: "calc_test.go", Blob: r.TestOverlays[0].Blob}}
		},
		"selector":      func(r *Request) { r.Selectors = []string{"^TestSlow$"} },
		"tags":          func(r *Request) { r.BuildTags = []string{"fixproof"} },
		"timeout":       func(r *Request) { r.TimeoutMillis++ },
		"go executable": func(r *Request) { r.Context.GoExecutableDigest = fixProofDigest("other-go") },
		"toolchain":     func(r *Request) { r.Context.ToolchainDigest = fixProofDigest("other-toolchain") },
		"test env":      func(r *Request) { r.Context.TestEnvDigest = fixProofDigest("other-test-env") },
		"workspace":     func(r *Request) { r.Context.WorkspaceDigest = fixProofDigest("other-workspace") },
		"verifier":      func(r *Request) { r.Context.VerifierDigest = fixProofDigest("other-verifier") },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			changed := cloneFixProofRequest(request)
			mutate(&changed)
			got, err := changed.Digest()
			if err != nil {
				t.Fatalf("mutated request became structurally invalid: %v", err)
			}
			if got == base {
				t.Fatalf("%s did not change request digest", name)
			}
		})
	}
	for _, tc := range []struct {
		name string
		edit func(*Request)
	}{
		{name: "noncanonical path", edit: func(r *Request) { r.Paths[0] = "../calc_test.go"; r.TestOverlays[0].Path = r.Paths[0] }},
		{name: "empty-matching selector", edit: func(r *Request) { r.Selectors[0] = ".*" }},
		{name: "uppercase object", edit: func(r *Request) { r.ParentCommit = "A" + r.ParentCommit[1:] }},
		{name: "zero timeout", edit: func(r *Request) { r.TimeoutMillis = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := cloneFixProofRequest(request)
			tc.edit(&bad)
			if err := bad.Validate(); !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("Validate error=%v, want ErrInvalidRequest", err)
			}
		})
	}
}

// fak-test:runtime slow est=160s lane=default
func TestFixProofExecuteConfirmsSelectedParentRedCandidateGreen(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "confirmed-marker.txt")
	f := newFixProofFixture(t, fixProofTestSource(marker, false), fixProofParentImplementation)
	result, err := Execute(context.Background(), fixProofRoots(f), f.request)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if result.Verdict != VerdictConfirmed || result.Schema != ResultSchema || result.Contract != Contract {
		t.Fatalf("confirmed result = %+v", result)
	}
	if err := result.Validate(f.request); err != nil {
		t.Fatalf("Result.Validate: %v", err)
	}
	raw, err := os.ReadFile(marker)
	if err != nil || string(raw) != "executed\nexecuted\n" {
		t.Fatalf("execution marker=%q err=%v", raw, err)
	}
}

// fak-test:runtime slow est=540s lane=default
func TestFixProofExecuteNeverConfirmsZeroMatchRedBuildFailureCancelOrTimeout(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "negative-marker.txt")
	f := newFixProofFixture(t, fixProofTestSource(marker, false), fixProofParentImplementation)
	for _, tc := range []struct {
		name        string
		ctx         func() context.Context
		edit        func(*Request)
		wantVerdict Verdict
		wantDetail  string
	}{
		{name: "zero selector match", ctx: context.Background, edit: func(r *Request) { r.Selectors = []string{"^TestMissing$"} }, wantVerdict: VerdictAbstained, wantDetail: "explicit symptom selector matched no changed top-level Test/Example"},
		{name: "timeout", ctx: context.Background, edit: func(r *Request) { r.Selectors = []string{"^TestSlow$"}; r.TimeoutMillis = 10 }, wantVerdict: VerdictAbstained},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := cloneFixProofRequest(f.request)
			tc.edit(&req)
			result, err := Execute(tc.ctx(), fixProofRoots(f), req)
			detailOK := result.Detail == tc.wantDetail
			if tc.name == "timeout" {
				detailOK = strings.Contains(result.Detail, "timed out before a trustworthy verdict") && strings.Contains(result.Detail, "context deadline exceeded")
			}
			if err != nil || result.Verdict != tc.wantVerdict || !detailOK {
				t.Fatalf("%s result=%+v err=%v, want verdict=%s detail=%q", tc.name, result, err, tc.wantVerdict, tc.wantDetail)
			}
		})
	}

	t.Run("cancelled during selected test", func(t *testing.T) {
		cancelMarker := filepath.Join(t.TempDir(), "cancel-marker.txt")
		cancelFixture := newFixProofFixture(t, fixProofTestSource(cancelMarker, false), fixProofParentImplementation)
		cancelFixture.request.Selectors = []string{"^TestSlow$"}
		cancelCtx, cancel := context.WithCancel(context.Background())
		cancelled := make(chan struct{})
		go func() {
			defer close(cancelled)
			for {
				if _, err := os.Stat(cancelMarker); err == nil {
					cancel()
					return
				}
				select {
				case <-cancelCtx.Done():
					return
				case <-time.After(10 * time.Millisecond):
				}
			}
		}()
		cancelResult, cancelErr := Execute(cancelCtx, fixProofRoots(cancelFixture), cancelFixture.request)
		cancel()
		<-cancelled
		if cancelErr != nil || cancelResult.Verdict != VerdictAbstained || !strings.Contains(cancelResult.Detail, "cancel") {
			t.Fatalf("cancelled selected test result=%+v err=%v, want bounded cancellation abstention", cancelResult, cancelErr)
		}
	})

	t.Run("candidate red", func(t *testing.T) {
		redMarker := filepath.Join(t.TempDir(), "candidate-red-marker.txt")
		red := newFixProofFixture(t, fixProofTestSource(redMarker, true), fixProofParentImplementation)
		if result, err := Execute(context.Background(), fixProofRoots(red), red.request); err != nil || result.Verdict != VerdictRefuted || result.Detail != "candidate selected symptom test failed" {
			t.Fatalf("candidate RED result=%+v err=%v", result, err)
		}
	})

	t.Run("parent production build failure", func(t *testing.T) {
		buildMarker := filepath.Join(t.TempDir(), "parent-build-marker.txt")
		buildFail := newFixProofFixture(t, fixProofTestSource(buildMarker, false), fixProofBrokenImplementation)
		if result, err := Execute(context.Background(), fixProofRoots(buildFail), buildFail.request); err != nil || result.Verdict != VerdictAbstained || result.Detail != "parent selected symptom test did not build" {
			t.Fatalf("parent build failure result=%+v err=%v", result, err)
		}
	})
}

// fak-test:runtime slow est=150s lane=default
func TestFixProofExecuteRejectsPinBindingAndEmptyOverlayBeforeTests(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "refusal-marker.txt")
	f := newFixProofFixture(t, fixProofTestSource(marker, false), fixProofParentImplementation)
	for _, tc := range []struct {
		name string
		edit func(*Request)
	}{
		{name: "parent commit", edit: func(r *Request) { r.ParentCommit = differentFixProofObject(r.ParentCommit) }},
		{name: "parent tree", edit: func(r *Request) { r.ParentTree = differentFixProofObject(r.ParentTree) }},
		{name: "candidate commit", edit: func(r *Request) { r.CandidateCommit = differentFixProofObject(r.CandidateCommit) }},
		{name: "candidate tree", edit: func(r *Request) { r.CandidateTree = differentFixProofObject(r.CandidateTree) }},
		{name: "companion commit", edit: func(r *Request) { r.CompanionCommit = differentFixProofObject(r.CompanionCommit) }},
		{name: "companion tree", edit: func(r *Request) { r.CompanionTree = differentFixProofObject(r.CompanionTree) }},
		{name: "overlay blob", edit: func(r *Request) { r.TestOverlays[0].Blob = differentFixProofObject(r.TestOverlays[0].Blob) }},
		{name: "path set", edit: func(r *Request) { r.Paths = []string{"pkg/calc/calc_test.go"} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := cloneFixProofRequest(f.request)
			tc.edit(&req)
			if result, err := Execute(context.Background(), fixProofRoots(f), req); err == nil || result.Verdict == VerdictConfirmed || !errors.Is(err, ErrIdentityMismatch) {
				t.Fatalf("identity mutation result=%+v err=%v, want ErrIdentityMismatch", result, err)
			}
		})
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("identity preflight executed tests: %v", err)
	}

	empty := cloneFixProofRequest(f.request)
	empty.TestOverlays = nil
	if result, err := Execute(context.Background(), fixProofRoots(f), empty); !errors.Is(err, ErrUnsupportedProofShape) || result.Verdict == VerdictConfirmed {
		t.Fatalf("empty overlay result=%+v err=%v, want ErrUnsupportedProofShape", result, err)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("empty-overlay refusal executed tests: %v", err)
	}

	validDigest, err := f.request.Digest()
	if err != nil {
		t.Fatal(err)
	}
	badResult := Result{Schema: ResultSchema, Contract: Contract, RequestDigest: fixProofDigest("wrong-request"), Verdict: VerdictConfirmed, Detail: "fabricated"}
	if err := badResult.Validate(f.request); err == nil || badResult.RequestDigest == validDigest {
		t.Fatalf("tampered result accepted: %v", err)
	}
	if reflect.TypeOf(Result{}).NumField() != 5 {
		t.Fatalf("Result grew beyond the closed non-counting facade: %+v", reflect.TypeOf(Result{}))
	}
}

// fak-test:runtime slow est=45s lane=default
func TestFixProofExecuteRefusesLinkedWorktreeBeforeTests(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "linked-worktree-marker.txt")
	f := newFixProofFixture(t, fixProofTestSource(marker, false), fixProofParentImplementation)
	linked := filepath.Join(t.TempDir(), "linked-fak")
	fixProofGit(t, f.repository, "worktree", "add", "-q", "--detach", linked, f.request.CandidateCommit)

	linkedGit, err := os.Stat(filepath.Join(linked, ".git"))
	if err != nil || !linkedGit.Mode().IsRegular() {
		t.Fatalf("fixture is not a linked worktree: info=%v err=%v", linkedGit, err)
	}
	linkedCommon := filepath.Clean(fixProofGit(t, linked, "rev-parse", "--path-format=absolute", "--git-common-dir"))
	if linkedCommon != fixProofCommonDir(t, f.repository) {
		t.Fatalf("fixture does not share Git common namespace: linked=%q repository=%q", linkedCommon, fixProofCommonDir(t, f.repository))
	}

	result, err := Execute(context.Background(), Roots{RepositoryDir: linked, CompanionDir: f.companion}, f.request)
	if err == nil || result.Verdict == VerdictConfirmed {
		t.Fatalf("linked worktree result=%+v err=%v, want refusal", result, err)
	}
	if _, statErr := os.Stat(marker); !os.IsNotExist(statErr) {
		t.Fatalf("linked-worktree refusal executed selected tests: %v", statErr)
	}
}
