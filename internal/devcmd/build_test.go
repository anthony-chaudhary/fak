package devcmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func withRunBuildSeams(t *testing.T) {
	t.Helper()
	origNow := buildNow
	origGather := buildGatherProvenance
	origPrepare := buildPrepareOutput
	origExecute := buildExecute
	origInspect := buildInspectArtifact
	origSmoke := buildSmoke
	origWrite := buildWriteReceipt
	t.Cleanup(func() {
		buildNow = origNow
		buildGatherProvenance = origGather
		buildPrepareOutput = origPrepare
		buildExecute = origExecute
		buildInspectArtifact = origInspect
		buildSmoke = origSmoke
		buildWriteReceipt = origWrite
	})

	tick := 0
	buildNow = func() time.Time {
		at := time.Date(2026, 8, 28, 12, 0, 0, tick*10_000_000, time.UTC)
		tick++
		return at
	}
	buildGatherProvenance = func(string) (buildProvenance, error) {
		return buildProvenance{
			Source: buildSource{Commit: "0123456789abcdef", CommittedTree: "tree012345", WorkingTreeSHA256: strings.Repeat("a", 64)},
			Toolchain: buildToolchain{
				GoVersion: "go version go1.26.6 test/arch", GOOS: "test", GOARCH: "arch",
				GOCache: "/cache", CGOEnabled: "0", GoToolchain: "auto", GoFlags: "",
			},
			PackageCount: 699,
		}, nil
	}
	buildInspectArtifact = func(path string) (buildArtifact, error) {
		return buildArtifact{Path: path, SizeBytes: 1234, SHA256: strings.Repeat("b", 64)}, nil
	}
	buildPrepareOutput = func(string) error { return nil }
	buildSmoke = func(path string) buildSmokeResult {
		return buildSmokeResult{Command: []string{path, "version", "--json"}, Outcome: "success", ExitCode: 0, Output: `{"version":"test"}`}
	}
}

func TestRunBuildSuccessRendersOrderedTimingsAndReceipt(t *testing.T) {
	withRunBuildSeams(t)
	var gotCommand buildCommand
	buildExecute = func(command buildCommand, stderr io.Writer) (int, error) {
		gotCommand = command
		_, _ = io.WriteString(stderr, "compiler child output\n")
		return 0, nil
	}
	var persisted buildReceipt
	var persistedPath string
	buildWriteReceipt = func(path string, receipt buildReceipt) error {
		persistedPath, persisted = path, receipt
		return nil
	}

	var stdout, stderr bytes.Buffer
	if code := RunBuild(&stdout, &stderr, nil); code != 0 {
		t.Fatalf("RunBuild code = %d, want 0; stderr=%s", code, stderr.String())
	}
	if !reflect.DeepEqual(gotCommand.Argv, []string{"sh", "scripts/build.sh"}) {
		t.Fatalf("command argv = %v, want canonical script invocation", gotCommand.Argv)
	}
	if gotCommand.Profile != "dev" || gotCommand.Environment["PROFILE"] != "dev" || gotCommand.Environment["OUT"] != gotCommand.Output {
		t.Fatalf("canonical command did not carry exact profile/output: %+v", gotCommand)
	}
	if gotCommand.Output != filepath.Join(repoRoot(), defaultBuildOutput()) || filepath.Dir(gotCommand.Output) != filepath.Join(repoRoot(), ".fak", "bin") {
		t.Fatalf("output = %q, want non-self repository artifact path", gotCommand.Output)
	}
	if !strings.HasSuffix(filepath.ToSlash(persistedPath), "/.fak/build-receipt.json") {
		t.Fatalf("default receipt path = %q", persistedPath)
	}
	if persisted.Schema != buildReceiptSchema || persisted.Outcome != "success" || persisted.ExitCode != 0 {
		t.Fatalf("terminal receipt = %+v", persisted)
	}
	if persisted.CacheState != "inherited_uncontrolled" || persisted.PackageCount != 699 {
		t.Fatalf("cache/package provenance missing: %+v", persisted)
	}
	var phaseNames []string
	for _, phase := range persisted.Phases {
		phaseNames = append(phaseNames, phase.Name)
	}
	if want := []string{"provenance", "output_prepare", "compile_link", "artifact", "smoke", "total"}; !reflect.DeepEqual(phaseNames, want) {
		t.Fatalf("phase order = %v, want %v", phaseNames, want)
	}
	if persisted.Artifact == nil || persisted.Artifact.SizeBytes != 1234 || persisted.Smoke == nil || persisted.Smoke.Outcome != "success" {
		t.Fatalf("artifact/smoke missing: %+v", persisted)
	}
	for _, want := range []string{"fak build: OK", "provenance", "compile_link", "artifact", gotCommand.Output, "smoke", "total", "receipt"} {
		if !strings.Contains(stdout.String(), want) {
			t.Errorf("human stdout missing %q:\n%s", want, stdout.String())
		}
	}
	if !strings.Contains(stderr.String(), "compiler child output") {
		t.Fatalf("child output did not stay on stderr: %q", stderr.String())
	}
}

func TestRunBuildDefaultOutputUsesExeForWindowsTarget(t *testing.T) {
	t.Setenv("GOOS", "windows")
	if got, want := defaultBuildOutput(), filepath.Join(".fak", "bin", "fak.exe"); got != want {
		t.Fatalf("defaultBuildOutput() = %q, want %q", got, want)
	}
}

func TestRunBuildJSONStdoutIsOneCleanObject(t *testing.T) {
	withRunBuildSeams(t)
	buildExecute = func(_ buildCommand, stderr io.Writer) (int, error) {
		_, _ = io.WriteString(stderr, "child-only\n")
		return 0, nil
	}
	buildWriteReceipt = func(string, buildReceipt) error { return nil }

	var stdout, stderr bytes.Buffer
	code := RunBuild(&stdout, &stderr, []string{"--json", "--profile", "release", "--out", "dist/fak", "--receipt", "receipts/build.json", "--version", "v1.2.3", "--tags", "cuda", "--gcflags", "all=-N -l"})
	if code != 0 {
		t.Fatalf("RunBuild code = %d; stderr=%s", code, stderr.String())
	}
	dec := json.NewDecoder(&stdout)
	var got buildReceipt
	if err := dec.Decode(&got); err != nil {
		t.Fatalf("decode stdout object: %v\n%s", err, stdout.String())
	}
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		t.Fatalf("stdout contains data after the receipt: err=%v output=%q", err, stdout.String())
	}
	if got.Command.Profile != "release" || !strings.HasSuffix(filepath.ToSlash(got.Command.Output), "/dist/fak") {
		t.Fatalf("explicit command fields not recorded: %+v", got.Command)
	}
	if got.Command.Environment["VERSION"] != "v1.2.3" || got.Command.Environment["TAGS"] != "cuda" || got.Command.Environment["GCFLAGS"] != "all=-N -l" {
		t.Fatalf("explicit build settings not recorded: %+v", got.Command.Environment)
	}
	if strings.Contains(stdout.String(), "child-only") || !strings.Contains(stderr.String(), "child-only") {
		t.Fatalf("child output leaked onto JSON stdout; stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
}

func TestRunBuildFailureWritesPartialTerminalReceipt(t *testing.T) {
	withRunBuildSeams(t)
	buildExecute = func(buildCommand, io.Writer) (int, error) { return 7, nil }
	inspectCalled := false
	buildInspectArtifact = func(string) (buildArtifact, error) {
		inspectCalled = true
		return buildArtifact{}, nil
	}
	var persisted buildReceipt
	buildWriteReceipt = func(_ string, receipt buildReceipt) error {
		persisted = receipt
		return nil
	}

	var stdout, stderr bytes.Buffer
	if code := RunBuild(&stdout, &stderr, []string{"--json"}); code != 7 {
		t.Fatalf("RunBuild code = %d, want child exit 7; stderr=%s", code, stderr.String())
	}
	if inspectCalled {
		t.Fatal("artifact inspection ran after compile/link failure")
	}
	if persisted.Outcome != "failed" || persisted.ExitCode != 7 || !strings.Contains(persisted.Error, "compile/link failed") {
		t.Fatalf("failure receipt is not terminal/actionable: %+v", persisted)
	}
	var names []string
	for _, phase := range persisted.Phases {
		names = append(names, phase.Name)
	}
	if want := []string{"provenance", "output_prepare", "compile_link", "total"}; !reflect.DeepEqual(names, want) {
		t.Fatalf("partial phase order = %v, want %v", names, want)
	}
	var emitted buildReceipt
	if err := json.Unmarshal(stdout.Bytes(), &emitted); err != nil {
		t.Fatalf("failure JSON is not one receipt object: %v: %s", err, stdout.String())
	}
}

func TestRunBuildReceiptWriteFailureIsReported(t *testing.T) {
	withRunBuildSeams(t)
	buildExecute = func(buildCommand, io.Writer) (int, error) { return 0, nil }
	buildWriteReceipt = func(string, buildReceipt) error { return errors.New("disk full") }

	var stdout, stderr bytes.Buffer
	if code := RunBuild(&stdout, &stderr, []string{"--json"}); code != 1 {
		t.Fatalf("RunBuild code = %d, want 1; stderr=%s", code, stderr.String())
	}
	var got buildReceipt
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode JSON: %v", err)
	}
	if got.Outcome != "failed" || !strings.Contains(got.Error, "disk full") {
		t.Fatalf("receipt-write failure not represented in result: %+v", got)
	}
}

func TestRunBuildOutputPrepareFailureWritesReceiptWithoutBuild(t *testing.T) {
	withRunBuildSeams(t)
	buildPrepareOutput = func(string) error { return errors.New("permission denied") }
	buildCalled := false
	buildExecute = func(buildCommand, io.Writer) (int, error) {
		buildCalled = true
		return 0, nil
	}
	var persisted buildReceipt
	buildWriteReceipt = func(_ string, receipt buildReceipt) error {
		persisted = receipt
		return nil
	}

	var stdout, stderr bytes.Buffer
	if code := RunBuild(&stdout, &stderr, []string{"--json"}); code != 1 {
		t.Fatalf("RunBuild code = %d, want 1; stderr=%s", code, stderr.String())
	}
	if buildCalled {
		t.Fatal("canonical build ran after output directory preparation failed")
	}
	if persisted.Outcome != "failed" || !strings.Contains(persisted.Error, "prepare output directory") || !strings.Contains(persisted.Error, "permission denied") {
		t.Fatalf("terminal receipt is not actionable: %+v", persisted)
	}
	var names []string
	for _, phase := range persisted.Phases {
		names = append(names, phase.Name)
	}
	if want := []string{"provenance", "output_prepare", "total"}; !reflect.DeepEqual(names, want) {
		t.Fatalf("partial phase order = %v, want %v", names, want)
	}
}

func TestRunBuildReceiptWriterAtomicallyReplacesCompleteJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "receipt.json")
	first := buildReceipt{Schema: buildReceiptSchema, Outcome: "failed", ExitCode: 1}
	second := buildReceipt{Schema: buildReceiptSchema, Outcome: "success", PackageCount: 699}
	if err := writeBuildReceiptAtomic(path, first); err != nil {
		t.Fatalf("first atomic write: %v", err)
	}
	if err := writeBuildReceiptAtomic(path, second); err != nil {
		t.Fatalf("replacement atomic write: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read receipt: %v", err)
	}
	var got buildReceipt
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("receipt is not complete JSON: %v: %q", err, raw)
	}
	if got.Outcome != "success" || got.PackageCount != 699 {
		t.Fatalf("receipt = %+v, want complete replacement", got)
	}
	temps, err := filepath.Glob(filepath.Join(filepath.Dir(path), ".build-receipt-*.tmp"))
	if err != nil || len(temps) != 0 {
		t.Fatalf("temporary receipt residue = %v, err=%v", temps, err)
	}
}

func TestRunBuildRejectsCollidingOutputAndReceiptBeforeMutation(t *testing.T) {
	withRunBuildSeams(t)
	called := false
	buildGatherProvenance = func(string) (buildProvenance, error) {
		called = true
		return buildProvenance{}, nil
	}
	buildExecute = func(buildCommand, io.Writer) (int, error) {
		called = true
		return 0, nil
	}
	buildWriteReceipt = func(string, buildReceipt) error {
		called = true
		return nil
	}

	var stdout, stderr bytes.Buffer
	code := RunBuild(&stdout, &stderr, []string{"--out", ".fak/../artifact", "--receipt", "artifact"})
	if code != 2 {
		t.Fatalf("RunBuild code = %d, want usage error 2; stderr=%s", code, stderr.String())
	}
	if called {
		t.Fatal("a provenance, build, or receipt mutation seam ran despite colliding paths")
	}
	if !strings.Contains(stderr.String(), "resolve to the same path") {
		t.Fatalf("collision error is not actionable: %q", stderr.String())
	}
}

func TestRunBuildProvenanceHashesCommittedTreeAndExactDeltaOnly(t *testing.T) {
	root := t.TempDir()
	newPath := filepath.Join(root, "new.txt")
	unchangedPath := filepath.Join(root, "unchanged.txt")
	if err := os.WriteFile(newPath, []byte("new-v1"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(unchangedPath, []byte("tracked-v1"), 0o644); err != nil {
		t.Fatal(err)
	}

	origCapture := buildCaptureOutput
	t.Cleanup(func() { buildCaptureOutput = origCapture })
	var calls []string
	buildCaptureOutput = func(_ string, name string, args ...string) ([]byte, error) {
		call := name + " " + strings.Join(args, " ")
		calls = append(calls, call)
		switch call {
		case "git rev-parse HEAD":
			return []byte("commit123\n"), nil
		case "git rev-parse HEAD^{tree}":
			return []byte("tree456\n"), nil
		case "git diff --binary --no-ext-diff HEAD --":
			return []byte("exact tracked binary delta\n"), nil
		case "git ls-files --others --exclude-standard -z":
			return []byte("new.txt\x00"), nil
		case "go version":
			return []byte("go version go1.26.6 test/arch\n"), nil
		case "go env GOOS GOARCH GOCACHE CGO_ENABLED GOTOOLCHAIN GOFLAGS":
			return []byte("test\narch\n/cache\n0\nauto\n\n"), nil
		case "go list -deps ./cmd/fak":
			return []byte("one\ntwo\n"), nil
		default:
			return nil, errors.New("unexpected command: " + call)
		}
	}

	first, err := gatherBuildProvenance(root)
	if err != nil {
		t.Fatalf("first provenance: %v", err)
	}
	if first.Source.Commit != "commit123" || first.Source.CommittedTree != "tree456" {
		t.Fatalf("source base identity = %+v", first.Source)
	}
	if err := os.WriteFile(unchangedPath, []byte("tracked-but-not-in-delta-input"), 0o644); err != nil {
		t.Fatal(err)
	}
	second, err := gatherBuildProvenance(root)
	if err != nil {
		t.Fatalf("second provenance: %v", err)
	}
	if second.Source.WorkingTreeSHA256 != first.Source.WorkingTreeSHA256 {
		t.Fatalf("unchanged tracked-file scan leaked into digest: first=%s second=%s", first.Source.WorkingTreeSHA256, second.Source.WorkingTreeSHA256)
	}
	if err := os.WriteFile(newPath, []byte("new-v2"), 0o644); err != nil {
		t.Fatal(err)
	}
	third, err := gatherBuildProvenance(root)
	if err != nil {
		t.Fatalf("third provenance: %v", err)
	}
	if third.Source.WorkingTreeSHA256 == first.Source.WorkingTreeSHA256 {
		t.Fatal("untracked content change did not change working-tree digest")
	}
	joined := strings.Join(calls, "\n")
	if strings.Contains(joined, "ls-files -co") || strings.Contains(joined, "git status") {
		t.Fatalf("provenance still scans every tracked path:\n%s", joined)
	}
	for _, want := range []string{"git rev-parse HEAD^{tree}", "git diff --binary --no-ext-diff HEAD --", "git ls-files --others --exclude-standard -z"} {
		if !strings.Contains(joined, want) {
			t.Errorf("provenance command log missing %q:\n%s", want, joined)
		}
	}
}
