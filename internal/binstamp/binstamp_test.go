package binstamp

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

func TestStampFromBuildInfo(t *testing.T) {
	if got := stampFrom(nil); got.HasVCS || got.Revision != "" {
		t.Fatalf("nil BuildInfo: got %+v, want zero stamp", got)
	}
	bi := &debug.BuildInfo{Settings: []debug.BuildSetting{
		{Key: "vcs.revision", Value: "abcdef1234567890"},
		{Key: "vcs.modified", Value: "true"},
	}}
	got := stampFrom(bi)
	if !got.HasVCS || got.Revision != "abcdef1234567890" || !got.Dirty {
		t.Fatalf("got %+v, want rev set + dirty + HasVCS", got)
	}
}

func TestCompareFreshnessRules(t *testing.T) {
	const head = "abcdef1234567890abcdef1234567890abcdef12"

	cases := []struct {
		name    string
		running Stamp
		head    string
		want    Freshness
	}{
		{"equal full rev => fresh",
			Stamp{Revision: head, HasVCS: true}, head, Fresh},
		{"short prefix matches => fresh",
			Stamp{Revision: "abcdef1234567", HasVCS: true}, head, Fresh},
		{"different clean rev => stale",
			Stamp{Revision: "ffffffffffffffff", HasVCS: true}, head, Stale},
		{"no embedded rev => unknown",
			Stamp{HasVCS: false}, head, Unknown},
		{"empty head => unknown",
			Stamp{Revision: head, HasVCS: true}, "", Unknown},
		{"dirty build => unknown (never restart)",
			Stamp{Revision: "ffffffffffffffff", HasVCS: true, Dirty: true}, head, Unknown},
		{"too-short prefix doesn't falsely match",
			Stamp{Revision: "abcd", HasVCS: true}, head, Stale},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Compare(c.running, c.head); got != c.want {
				t.Fatalf("Compare = %v, want %v", got, c.want)
			}
		})
	}
}

func TestExplainDistinguishesUnknownCauses(t *testing.T) {
	const head = "abcdef1234567890abcdef1234567890abcdef12"

	cases := []struct {
		name      string
		running   Stamp
		head      string
		wantFresh Freshness
		wantCause Cause
	}{
		{"fresh matches",
			Stamp{Revision: head, HasVCS: true}, head, Fresh, CauseMatched},
		{"clean different rev diverged",
			Stamp{Revision: "ffffffffffffffff", HasVCS: true}, head, Stale, CauseDiverged},
		{"no stamp is unstamped, not a silent unknown",
			Stamp{HasVCS: false}, head, Unknown, CauseUnstamped},
		{"unstamped dominates even with no head",
			Stamp{HasVCS: false}, "", Unknown, CauseUnstamped},
		{"stamped but no head",
			Stamp{Revision: head, HasVCS: true}, "", Unknown, CauseNoHead},
		{"dirty build is its own cause",
			Stamp{Revision: "ffffffffffffffff", HasVCS: true, Dirty: true}, head, Unknown, CauseDirty},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotFresh, gotCause := Explain(c.running, c.head)
			if gotFresh != c.wantFresh {
				t.Fatalf("Explain freshness = %v, want %v", gotFresh, c.wantFresh)
			}
			if gotCause != c.wantCause {
				t.Fatalf("Explain cause = %v, want %v", gotCause, c.wantCause)
			}
			// The invariant that lets Compare delegate: Explain's verdict must equal Compare's.
			if got := Compare(c.running, c.head); got != gotFresh {
				t.Fatalf("Compare = %v but Explain freshness = %v — they must never drift", got, gotFresh)
			}
		})
	}
}

func TestRevisionsMatch(t *testing.T) {
	if !revisionsMatch("ABCDEF1234567", "abcdef1234567890") {
		t.Fatal("case-insensitive prefix should match")
	}
	if revisionsMatch("abcde", "abcdef1234567890") {
		t.Fatal("a <7 char short rev must not match (too weak)")
	}
	if revisionsMatch("1234567", "abcdef1234567890") {
		t.Fatal("non-prefix must not match")
	}
}

func TestSelfUsesInjectedCommitWhenGoVCSStampMissing(t *testing.T) {
	old := appversion.BuildCommit
	appversion.BuildCommit = "0123456789abcdef0123456789abcdef01234567"
	t.Cleanup(func() { appversion.BuildCommit = old })

	// Self reads this test binary's normal build info. Go test binaries generally carry no
	// vcs.revision here, so the injected installer provenance must remain observable.
	got := Self()
	if !got.HasVCS || got.Revision != appversion.BuildCommit || got.Dirty {
		t.Fatalf("Self() = %+v, want clean injected commit %s", got, appversion.BuildCommit)
	}
}

func TestExecutableProvenanceObservesBuildMetadataAndOpenedBytes(t *testing.T) {
	const revision = "0123456789abcdef0123456789abcdef01234567"
	for _, dirty := range []bool{false, true} {
		t.Run(map[bool]string{false: "clean", true: "dirty"}[dirty], func(t *testing.T) {
			binary := []byte("device-free fak executable fixture")
			path := t.TempDir() + string(os.PathSeparator) + "fak-fixture"
			if err := os.WriteFile(path, binary, 0o600); err != nil {
				t.Fatal(err)
			}
			modified := "false"
			if dirty {
				modified = "true"
			}
			got, err := observeExecutableProvenance(
				buildInfoReader(revision, modified),
				func() (string, error) { return path, nil },
				func(path string) (provenanceFile, error) { return os.Open(path) },
			)
			if err != nil {
				t.Fatalf("observeExecutableProvenance: %v", err)
			}
			wantDigest := sha256.Sum256(binary)
			if got.Revision() != revision || got.Dirty() != dirty || got.BinaryBytes() != int64(len(binary)) || got.BinarySHA256() != hex.EncodeToString(wantDigest[:]) {
				t.Fatalf("observation = %+v", got)
			}
		})
	}

	// The production entry point has no request or expected-identity input. A
	// caller therefore cannot forge labels that are copied into the result.
	if typ := reflect.TypeOf(ObserveExecutableProvenance); typ.NumIn() != 0 {
		t.Fatalf("ObserveExecutableProvenance accepts %d caller inputs, want 0", typ.NumIn())
	}
	for i, typ := 0, reflect.TypeOf(ExecutableProvenance{}); i < typ.NumField(); i++ {
		if field := typ.Field(i); field.IsExported() {
			t.Fatalf("ExecutableProvenance field %q is exported; callers could forge it", field.Name)
		}
	}
}

func TestExecutableProvenanceRejectsMissingOrMalformedBuildIdentity(t *testing.T) {
	const revision = "0123456789abcdef0123456789abcdef01234567"
	path := t.TempDir() + string(os.PathSeparator) + "fak-fixture"
	if err := os.WriteFile(path, []byte("binary"), 0o600); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name string
		read func() (*debug.BuildInfo, bool)
	}{
		{"unavailable", func() (*debug.BuildInfo, bool) { return nil, false }},
		{"nil", func() (*debug.BuildInfo, bool) { return nil, true }},
		{"missing revision", buildInfoReader("", "false")},
		{"missing modified", func() (*debug.BuildInfo, bool) {
			return &debug.BuildInfo{Settings: []debug.BuildSetting{{Key: "vcs.revision", Value: revision}}}, true
		}},
		{"short revision", buildInfoReader("0123456", "false")},
		{"non hex revision", buildInfoReader("zz23456789abcdef0123456789abcdef01234567", "false")},
		{"untyped modified", buildInfoReader(revision, "yes")},
		{"duplicate revision", func() (*debug.BuildInfo, bool) {
			return &debug.BuildInfo{Settings: []debug.BuildSetting{
				{Key: "vcs.revision", Value: revision},
				{Key: "vcs.revision", Value: "ffffffffffffffffffffffffffffffffffffffff"},
				{Key: "vcs.modified", Value: "false"},
			}}, true
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := observeExecutableProvenance(
				tc.read,
				func() (string, error) { return path, nil },
				func(path string) (provenanceFile, error) { return os.Open(path) },
			)
			if err == nil || got != (ExecutableProvenance{}) {
				t.Fatalf("observation = %+v, err=%v; want zero plus error", got, err)
			}
		})
	}
}

func TestExecutableProvenanceRejectsMissingOrMismatchedExecutable(t *testing.T) {
	const revision = "0123456789abcdef0123456789abcdef01234567"
	validBuild := buildInfoReader(revision, "false")

	cases := []struct {
		name string
		path func() (string, error)
		open func(string) (provenanceFile, error)
	}{
		{
			name: "path error",
			path: func() (string, error) { return "", errors.New("unavailable") },
			open: func(string) (provenanceFile, error) { return nil, errors.New("must not open") },
		},
		{
			name: "empty path",
			path: func() (string, error) { return "", nil },
			open: func(string) (provenanceFile, error) { return nil, errors.New("must not open") },
		},
		{
			name: "open error",
			path: func() (string, error) { return "fak", nil },
			open: func(string) (provenanceFile, error) { return nil, errors.New("unreadable") },
		},
		{
			name: "empty file",
			path: func() (string, error) { return "fak", nil },
			open: func(string) (provenanceFile, error) {
				return &provenanceFixtureFile{Reader: bytes.NewReader(nil), info: provenanceFixtureInfo{}}, nil
			},
		},
		{
			name: "non regular",
			path: func() (string, error) { return "fak", nil },
			open: func(string) (provenanceFile, error) {
				return &provenanceFixtureFile{Reader: bytes.NewReader([]byte("x")), info: provenanceFixtureInfo{size: 1, mode: fs.ModeDir}}, nil
			},
		},
		{
			name: "byte count mismatch",
			path: func() (string, error) { return "fak", nil },
			open: func(string) (provenanceFile, error) {
				return &provenanceFixtureFile{Reader: bytes.NewReader([]byte("abc")), info: provenanceFixtureInfo{size: 4}}, nil
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := observeExecutableProvenance(validBuild, tc.path, tc.open)
			if err == nil || got != (ExecutableProvenance{}) {
				t.Fatalf("observation = %+v, err=%v; want zero plus error", got, err)
			}
		})
	}
}

func buildInfoReader(revision, modified string) func() (*debug.BuildInfo, bool) {
	return func() (*debug.BuildInfo, bool) {
		settings := make([]debug.BuildSetting, 0, 2)
		if revision != "" {
			settings = append(settings, debug.BuildSetting{Key: "vcs.revision", Value: revision})
		}
		if modified != "" {
			settings = append(settings, debug.BuildSetting{Key: "vcs.modified", Value: modified})
		}
		return &debug.BuildInfo{Settings: settings}, true
	}
}

type provenanceFixtureFile struct {
	*bytes.Reader
	info provenanceFixtureInfo
}

func (f *provenanceFixtureFile) Close() error               { return nil }
func (f *provenanceFixtureFile) Stat() (fs.FileInfo, error) { return f.info, nil }

type provenanceFixtureInfo struct {
	size int64
	mode fs.FileMode
}

func (i provenanceFixtureInfo) Name() string       { return "fak" }
func (i provenanceFixtureInfo) Size() int64        { return i.size }
func (i provenanceFixtureInfo) Mode() fs.FileMode  { return i.mode }
func (i provenanceFixtureInfo) ModTime() time.Time { return time.Unix(1, 0) }
func (i provenanceFixtureInfo) IsDir() bool        { return i.mode.IsDir() }
func (i provenanceFixtureInfo) Sys() any           { return nil }

func TestObserveExecutableProvenancePlatformBehavior(t *testing.T) {
	if runtime.GOOS != "linux" {
		got, err := ObserveExecutableProvenance()
		if err == nil {
			t.Fatalf("ObserveExecutableProvenance on %s succeeded: %+v, want fail-closed error", runtime.GOOS, got)
		}
		if !got.IsZero() {
			t.Fatalf("ObserveExecutableProvenance on %s returned non-zero observation: %+v", runtime.GOOS, got)
		}
		if got.Modified() != got.Dirty() {
			t.Fatalf("Modified (%v) != Dirty (%v)", got.Modified(), got.Dirty())
		}
	} else {
		path, err := runningExecutablePath()
		if err != nil || path != "/proc/self/exe" {
			t.Fatalf("runningExecutablePath() = (%q, %v), want (/proc/self/exe, nil)", path, err)
		}
	}
}

const helperEnvVar = "GO_WANT_BINSTAMP_HELPER_PROCESS"

func TestHelperProcessPathnameReplacement(t *testing.T) {
	if os.Getenv(helperEnvVar) != "1" {
		return
	}
	defer os.Exit(0)

	// Step 1: Signal READY to parent on stdout.
	if _, err := fmt.Println("READY"); err != nil {
		fmt.Fprintf(os.Stderr, "failed to signal ready: %v\n", err)
		os.Exit(1)
	}

	// Step 2: Wait for parent to signal CONTINUE on stdin.
	scanner := bufio.NewScanner(os.Stdin)
	if !scanner.Scan() {
		fmt.Fprintf(os.Stderr, "failed to read continue signal: %v\n", scanner.Err())
		os.Exit(1)
	}
	if text := strings.TrimSpace(scanner.Text()); text != "CONTINUE" {
		fmt.Fprintf(os.Stderr, "unexpected continue signal: %q\n", text)
		os.Exit(1)
	}

	// Step 3: Observe executable provenance using the production runningExecutablePath.
	const testRevision = "0123456789abcdef0123456789abcdef01234567"
	got, err := observeExecutableProvenance(
		buildInfoReader(testRevision, "false"),
		runningExecutablePath,
		func(path string) (provenanceFile, error) { return os.Open(path) },
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "observeExecutableProvenance failed: %v\n", err)
		os.Exit(1)
	}

	// Step 4: Output observed digest, byte count, and what os.Executable() would read.
	var exeDigest string
	if exePath, exeErr := os.Executable(); exeErr == nil && exePath != "" {
		if exeBytes, readErr := os.ReadFile(exePath); readErr == nil {
			sum := sha256.Sum256(exeBytes)
			exeDigest = hex.EncodeToString(sum[:])
		}
	}
	fmt.Printf("OBSERVED_DIGEST:%s OBSERVED_BYTES:%d EXE_DIGEST:%s\n",
		got.BinarySHA256(), got.BinaryBytes(), exeDigest)
	os.Exit(0)
}

func TestExecutableProvenanceBindsMappedImageAcrossPathnameReplacement(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mapped executable image observation requires linux (/proc/self/exe)")
	}

	testExe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	origBytes, err := os.ReadFile(testExe)
	if err != nil {
		t.Fatalf("read test binary %q: %v", testExe, err)
	}
	origSum := sha256.Sum256(origBytes)
	origSHA256 := hex.EncodeToString(origSum[:])

	tempDir := t.TempDir()
	launchPath := filepath.Join(tempDir, "fak-child-test")
	if err := os.WriteFile(launchPath, origBytes, 0o755); err != nil {
		t.Fatalf("write launch binary: %v", err)
	}

	replacementBytes := []byte("adversarial replacement binary image fixture bytes")
	repSum := sha256.Sum256(replacementBytes)
	replacementSHA256 := hex.EncodeToString(repSum[:])
	replacementPath := filepath.Join(tempDir, "fak-replacement")
	if err := os.WriteFile(replacementPath, replacementBytes, 0o755); err != nil {
		t.Fatalf("write replacement binary: %v", err)
	}

	cmd := exec.Command(launchPath, "-test.run=^TestHelperProcessPathnameReplacement$", "-test.v=false")
	cmd.Env = append(os.Environ(), helperEnvVar+"=1")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("stdin pipe: %v", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		t.Fatalf("start child process: %v", err)
	}
	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()

	scanner := bufio.NewScanner(stdout)
	if !scanner.Scan() {
		t.Fatalf("failed to read child ready: scan=%v, stderr=%s", scanner.Err(), stderr.String())
	}
	if line := strings.TrimSpace(scanner.Text()); line != "READY" {
		t.Fatalf("expected child READY, got %q (stderr: %s)", line, stderr.String())
	}

	// Replace the launch pathname while the child process is running.
	if err := os.Rename(replacementPath, launchPath); err != nil {
		t.Fatalf("failed to rename replacement over launch path: %v", err)
	}

	// Signal the child to observe executable provenance.
	if _, err := io.WriteString(stdin, "CONTINUE\n"); err != nil {
		t.Fatalf("failed to write continue to child: %v", err)
	}

	if !scanner.Scan() {
		t.Fatalf("failed to read child result: scan=%v, stderr=%s", scanner.Err(), stderr.String())
	}
	resultLine := strings.TrimSpace(scanner.Text())

	if err := cmd.Wait(); err != nil {
		t.Fatalf("child process failed: %v (stderr: %s)", err, stderr.String())
	}

	var observedDigest, exeDigest string
	var observedBytes int64
	n, err := fmt.Sscanf(resultLine, "OBSERVED_DIGEST:%s OBSERVED_BYTES:%d EXE_DIGEST:%s",
		&observedDigest, &observedBytes, &exeDigest)
	if err != nil || n != 3 {
		t.Fatalf("failed to parse child output %q: %v", resultLine, err)
	}

	// Verify that os.Executable() opened the replacement binary after rename.
	if exeDigest != replacementSHA256 {
		t.Fatalf("expected os.Executable to see replacement digest %q, got %q", replacementSHA256, exeDigest)
	}

	// Verify that the observer bound to the mapped image via /proc/self/exe, not the replacement.
	if observedDigest != origSHA256 {
		t.Fatalf("observed digest %q does not match original binary %q (replacement was %q)",
			observedDigest, origSHA256, replacementSHA256)
	}
	if observedBytes != int64(len(origBytes)) {
		t.Fatalf("observed bytes %d != expected %d", observedBytes, len(origBytes))
	}
	if observedDigest == replacementSHA256 {
		t.Fatalf("FATAL: observer accepted replacement binary digest instead of mapped image")
	}
}
