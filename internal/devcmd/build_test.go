package devcmd

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
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
	var gotExecutionEnvironment map[string]string
	buildExecute = func(execution buildExecution, stderr io.Writer) (int, error) {
		gotCommand = execution.command
		gotExecutionEnvironment = execution.environment
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
	if gotCommand.Environment["PGO"] != "off" || gotExecutionEnvironment["PGO"] != "off" {
		t.Fatalf("default PGO was not recorded and executed as off: command=%q execution=%q", gotCommand.Environment["PGO"], gotExecutionEnvironment["PGO"])
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
	if persisted.PGO != (buildPGO{Mode: "off"}) {
		t.Fatalf("default PGO receipt = %+v, want mode off only", persisted.PGO)
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
	buildExecute = func(_ buildExecution, stderr io.Writer) (int, error) {
		_, _ = io.WriteString(stderr, "child-only\n")
		return 0, nil
	}
	buildWriteReceipt = func(string, buildReceipt) error { return nil }

	var stdout, stderr bytes.Buffer
	code := RunBuild(&stdout, &stderr, []string{"--json", "--profile", "release", "--out", "dist/fak", "--receipt", "receipts/build.json", "--version", "v1.2.3", "--tags", "cuda", "--gcflags", "all=-N -l"})
	if code != 0 {
		t.Fatalf("RunBuild code = %d; stderr=%s", code, stderr.String())
	}
	rawStdout := append([]byte(nil), stdout.Bytes()...)
	dec := json.NewDecoder(bytes.NewReader(rawStdout))
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
	if strings.Contains(string(rawStdout), "child-only") || !strings.Contains(stderr.String(), "child-only") {
		t.Fatalf("child output leaked onto JSON stdout; stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
	type legacyReceipt struct {
		Schema  string       `json:"schema"`
		Outcome string       `json:"outcome"`
		Command buildCommand `json:"command"`
	}
	var legacy legacyReceipt
	if err := json.Unmarshal(rawStdout, &legacy); err != nil {
		t.Fatalf("legacy receipt consumer rejected additive PGO field: %v", err)
	}
	if legacy.Schema != buildReceiptSchema || legacy.Outcome != "success" {
		t.Fatalf("legacy receipt fields changed: %+v", legacy)
	}
}

func TestRunBuildExplicitPGOUsesPrivateSnapshotAndScrubsReceipt(t *testing.T) {
	withRunBuildSeams(t)
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "customer-secret-profile-name.pprof")
	original := []byte("raw-private-profile-label:original-bytes")
	mutated := []byte("raw-private-profile-label:mutated-after-validation")
	if err := os.WriteFile(sourcePath, original, 0o600); err != nil {
		t.Fatal(err)
	}

	baseGather := buildGatherProvenance
	buildGatherProvenance = func(root string) (buildProvenance, error) {
		if err := os.WriteFile(sourcePath, mutated, 0o600); err != nil {
			return buildProvenance{}, err
		}
		return baseGather(root)
	}
	var snapshotPath string
	buildExecute = func(execution buildExecution, _ io.Writer) (int, error) {
		if got := execution.command.Environment["PGO"]; got != "profile" {
			t.Errorf("serialized command PGO = %q, want scrubbed profile mode", got)
		}
		snapshotPath = execution.environment["PGO"]
		if snapshotPath == "" || snapshotPath == sourcePath {
			t.Errorf("execution PGO = %q, want private snapshot distinct from source", snapshotPath)
			return 1, nil
		}
		got, err := os.ReadFile(snapshotPath)
		if err != nil {
			return 1, err
		}
		if !bytes.Equal(got, original) {
			t.Errorf("compiled PGO snapshot = %q, want pre-mutation bytes %q", got, original)
		}
		return 0, nil
	}
	var persisted buildReceipt
	buildWriteReceipt = func(_ string, receipt buildReceipt) error {
		persisted = receipt
		return nil
	}

	var stdout, stderr bytes.Buffer
	code := RunBuild(&stdout, &stderr, []string{
		"--json", "--profile", "release", "--pgo", sourcePath,
		"--out", filepath.Join(dir, "fak"), "--receipt", filepath.Join(dir, "receipt.json"),
	})
	if code != 0 {
		t.Fatalf("RunBuild code = %d; stderr=%s", code, stderr.String())
	}
	sum := sha256.Sum256(original)
	wantDigest := hex.EncodeToString(sum[:])
	wantPGO := buildPGO{Mode: "profile", Identity: "sha256:" + wantDigest, SHA256: wantDigest, SizeBytes: int64(len(original))}
	if persisted.PGO != wantPGO {
		t.Fatalf("persisted PGO = %+v, want %+v", persisted.PGO, wantPGO)
	}
	if snapshotPath == "" {
		t.Fatal("build did not receive a private PGO snapshot")
	}
	if _, err := os.Stat(snapshotPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("private PGO snapshot was not cleaned up: stat err=%v", err)
	}

	persistedJSON, err := json.Marshal(persisted)
	if err != nil {
		t.Fatal(err)
	}
	for label, raw := range map[string][]byte{"JSON stdout": stdout.Bytes(), "persisted receipt": persistedJSON} {
		for _, secret := range []string{sourcePath, filepath.Base(sourcePath), snapshotPath, string(original), string(mutated)} {
			if strings.Contains(string(raw), secret) {
				t.Errorf("%s disclosed private PGO source/snapshot data %q: %s", label, secret, raw)
			}
		}
	}
	dec := json.NewDecoder(bytes.NewReader(stdout.Bytes()))
	var emitted buildReceipt
	if err := dec.Decode(&emitted); err != nil {
		t.Fatalf("decode JSON stdout: %v", err)
	}
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		t.Fatalf("JSON stdout contains more than one object: err=%v output=%q", err, stdout.String())
	}
}

func TestRunBuildPGOIdentityDependsOnlyOnBytes(t *testing.T) {
	withRunBuildSeams(t)
	buildExecute = func(execution buildExecution, _ io.Writer) (int, error) {
		_, err := os.ReadFile(execution.environment["PGO"])
		return 0, err
	}
	var persisted buildReceipt
	buildWriteReceipt = func(_ string, receipt buildReceipt) error {
		persisted = receipt
		return nil
	}

	dir := t.TempDir()
	firstPath := filepath.Join(dir, "first-private-name.pprof")
	secondPath := filepath.Join(dir, "different-private-name.pprof")
	changedPath := filepath.Join(dir, "third-private-name.pprof")
	for path, contents := range map[string][]byte{
		firstPath:   []byte("same-profile-bytes"),
		secondPath:  []byte("same-profile-bytes"),
		changedPath: []byte("changed-profile-bytes"),
	} {
		if err := os.WriteFile(path, contents, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	run := func(source, suffix string) buildPGO {
		t.Helper()
		var stdout, stderr bytes.Buffer
		code := RunBuild(&stdout, &stderr, []string{
			"--json", "--profile", "release", "--pgo", source,
			"--out", filepath.Join(dir, "fak-"+suffix), "--receipt", filepath.Join(dir, "receipt-"+suffix+".json"),
		})
		if code != 0 {
			t.Fatalf("RunBuild(%s) code = %d; stderr=%s", suffix, code, stderr.String())
		}
		return persisted.PGO
	}

	first := run(firstPath, "first")
	second := run(secondPath, "second")
	changed := run(changedPath, "changed")
	if first != second {
		t.Fatalf("same bytes under different filenames changed PGO identity: first=%+v second=%+v", first, second)
	}
	if first.Identity == changed.Identity || first.SHA256 == changed.SHA256 {
		t.Fatalf("changed bytes did not change PGO identity: first=%+v changed=%+v", first, changed)
	}
}

func TestRunBuildRejectsInvalidPGOBeforeMutation(t *testing.T) {
	withRunBuildSeams(t)
	dir := t.TempDir()
	validPath := filepath.Join(dir, "private-valid-profile.pprof")
	emptyPath := filepath.Join(dir, "private-empty-profile.pprof")
	directoryPath := filepath.Join(dir, "private-profile-directory")
	missingPath := filepath.Join(dir, "private-missing-profile.pprof")
	if err := os.WriteFile(validPath, []byte("profile"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(emptyPath, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(directoryPath, 0o700); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		profile string
		pgo     string
	}{
		{name: "dev profile", profile: "dev", pgo: validPath},
		{name: "race profile", profile: "race", pgo: validPath},
		{name: "release missing", profile: "release", pgo: missingPath},
		{name: "release empty", profile: "release", pgo: emptyPath},
		{name: "release directory", profile: "release", pgo: directoryPath},
		{name: "release auto", profile: "release", pgo: "auto"},
		{name: "release explicit empty", profile: "release", pgo: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			called := false
			buildGatherProvenance = func(string) (buildProvenance, error) {
				called = true
				return buildProvenance{}, nil
			}
			buildPrepareOutput = func(string) error {
				called = true
				return nil
			}
			buildExecute = func(buildExecution, io.Writer) (int, error) {
				called = true
				return 0, nil
			}
			buildWriteReceipt = func(string, buildReceipt) error {
				called = true
				return nil
			}
			output := filepath.Join(dir, strings.ReplaceAll(tt.name, " ", "-")+"-output")
			receipt := filepath.Join(dir, strings.ReplaceAll(tt.name, " ", "-")+"-receipt.json")
			var stdout, stderr bytes.Buffer
			code := RunBuild(&stdout, &stderr, []string{
				"--json", "--profile", tt.profile, "--pgo", tt.pgo,
				"--out", output, "--receipt", receipt,
			})
			if code != 2 {
				t.Fatalf("RunBuild code = %d, want usage error 2; stderr=%s", code, stderr.String())
			}
			if called {
				t.Fatal("provenance, output preparation, build execution, or receipt write ran for invalid PGO")
			}
			if stdout.Len() != 0 {
				t.Fatalf("invalid PGO emitted JSON stdout: %q", stdout.String())
			}
			for _, path := range []string{output, receipt} {
				if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("invalid PGO mutated %s: stat err=%v", path, err)
				}
			}
			if tt.pgo != "" && tt.pgo != "auto" {
				for _, secret := range []string{tt.pgo, filepath.Base(tt.pgo)} {
					if strings.Contains(stderr.String(), secret) {
						t.Fatalf("invalid PGO error disclosed private path/name %q: %s", secret, stderr.String())
					}
				}
			}
		})
	}
}

func TestRunBuildRejectsPGOOutputOrReceiptCollisionBeforeMutation(t *testing.T) {
	withRunBuildSeams(t)
	for _, collision := range []string{"output", "receipt"} {
		t.Run(collision, func(t *testing.T) {
			dir := t.TempDir()
			profilePath := filepath.Join(dir, "private-collision-profile.pprof")
			profileBytes := []byte("must-not-be-mutated")
			if err := os.WriteFile(profilePath, profileBytes, 0o600); err != nil {
				t.Fatal(err)
			}
			output := filepath.Join(dir, "fak")
			receipt := filepath.Join(dir, "receipt.json")
			if collision == "output" {
				output = profilePath
			} else {
				receipt = profilePath
			}
			called := false
			buildGatherProvenance = func(string) (buildProvenance, error) { called = true; return buildProvenance{}, nil }
			buildPrepareOutput = func(string) error { called = true; return nil }
			buildExecute = func(buildExecution, io.Writer) (int, error) { called = true; return 0, nil }
			buildWriteReceipt = func(string, buildReceipt) error { called = true; return nil }

			var stdout, stderr bytes.Buffer
			code := RunBuild(&stdout, &stderr, []string{
				"--json", "--profile", "release", "--pgo", profilePath,
				"--out", output, "--receipt", receipt,
			})
			if code != 2 {
				t.Fatalf("RunBuild code = %d, want usage error 2; stderr=%s", code, stderr.String())
			}
			if called || stdout.Len() != 0 {
				t.Fatalf("collision reached a mutation seam or JSON output: called=%v stdout=%q", called, stdout.String())
			}
			got, err := os.ReadFile(profilePath)
			if err != nil || !bytes.Equal(got, profileBytes) {
				t.Fatalf("colliding PGO source was mutated: bytes=%q err=%v", got, err)
			}
			if strings.Contains(stderr.String(), profilePath) || strings.Contains(stderr.String(), filepath.Base(profilePath)) {
				t.Fatalf("collision error disclosed private PGO path/name: %s", stderr.String())
			}
		})
	}
}

func TestRunBuildFailureWritesPartialTerminalReceipt(t *testing.T) {
	withRunBuildSeams(t)
	buildExecute = func(buildExecution, io.Writer) (int, error) { return 7, nil }
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
	buildExecute = func(buildExecution, io.Writer) (int, error) { return 0, nil }
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
	buildExecute = func(buildExecution, io.Writer) (int, error) {
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
	buildExecute = func(buildExecution, io.Writer) (int, error) {
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
