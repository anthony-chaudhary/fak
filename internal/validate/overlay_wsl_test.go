package validate

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// overlayWSLTestCmdLineLimit is the Windows CreateProcess command-line limit
// (in characters) that a single `bash -lc <script>` argument must not exceed.
const overlayWSLTestCmdLineLimit = 32767

// overlayWSLTestRootPrefix names every WSL scratch root these tests create, so
// cleanup can refuse to remove anything else.
const overlayWSLTestRootPrefix = "/tmp/fak-validate-overlaytest-"

var (
	overlayWSLTestGateOnce   sync.Once
	overlayWSLTestGateReason string
	overlayWSLTestExitCodeRe = regexp.MustCompile(`exit_code=(-?\d+)`)
)

// overlayWSLTestRequire skips unless this is a Windows host with a working WSL.
func overlayWSLTestRequire(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "windows" {
		t.Skip("WSL overlay requires a Windows host")
	}
	overlayWSLTestGateOnce.Do(func() {
		if _, err := exec.LookPath("wsl.exe"); err != nil {
			overlayWSLTestGateReason = "wsl.exe not on PATH: " + err.Error()
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, "wsl.exe", "bash", "-lc", "echo ok").Output()
		if err != nil {
			overlayWSLTestGateReason = fmt.Sprintf("wsl.exe bash -lc \"echo ok\" failed: %v", err)
			return
		}
		if overlayWSLTestLastLine(out) != "ok" {
			overlayWSLTestGateReason = fmt.Sprintf("wsl.exe bash -lc \"echo ok\" printed %q", out)
		}
	})
	if overlayWSLTestGateReason != "" {
		t.Skip(overlayWSLTestGateReason)
	}
}

// overlayWSLTestLastLine returns the last non-empty trimmed line of out.
func overlayWSLTestLastLine(out []byte) string {
	lines := strings.Split(strings.ReplaceAll(string(out), "\r\n", "\n"), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if s := strings.TrimSpace(lines[i]); s != "" {
			return s
		}
	}
	return ""
}

// overlayWSLTestBash runs script inside WSL via `wsl.exe bash -lc` and returns stdout.
func overlayWSLTestBash(ctx context.Context, t *testing.T, script string) []byte {
	t.Helper()
	cmd := exec.CommandContext(ctx, "wsl.exe", "bash", "-lc", script)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("wsl.exe bash -lc %q: %v (stdout=%q stderr=%q)", script, err, out, stderr.String())
	}
	return out
}

// overlayWSLTestQuote single-quotes s for a POSIX shell.
func overlayWSLTestQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'"'"'`) + "'"
}

// overlayWSLTestAssertNamedFailure pins R2: the failure names exit_code, paths,
// script_bytes, command_chars and carries non-empty detail. It returns the exit code.
func overlayWSLTestAssertNamedFailure(t *testing.T, err error, wantPaths int) int {
	t.Helper()
	if err == nil {
		t.Fatal("overlayMinePathsWSLWithin returned nil error, want a named failure")
	}
	msg := err.Error()
	t.Logf("overlay error: %s", msg)
	if strings.TrimSpace(msg) == "" {
		t.Fatal("overlay error message is empty")
	}
	const prefix = "overlay owned paths in WSL: "
	if !strings.HasPrefix(msg, prefix) {
		t.Errorf("error %q does not start with %q", msg, prefix)
	}
	m := overlayWSLTestExitCodeRe.FindStringSubmatch(msg)
	if m == nil {
		t.Fatalf("error %q does not name exit_code=<n>", msg)
	}
	code, convErr := strconv.Atoi(m[1])
	if convErr != nil {
		t.Fatalf("exit_code %q is not an integer: %v", m[1], convErr)
	}
	if want := fmt.Sprintf("paths=%d", wantPaths); !strings.Contains(msg, want) {
		t.Errorf("error %q does not name %s", msg, want)
	}
	for _, key := range []string{"script_bytes", "command_chars"} {
		re := regexp.MustCompile(key + `=(\d+)`)
		km := re.FindStringSubmatch(msg)
		if km == nil {
			t.Errorf("error %q does not name %s=<n>", msg, key)
			continue
		}
		if n, _ := strconv.Atoi(km[1]); n <= 0 {
			t.Errorf("error %q names %s=%s, want > 0", msg, key, km[1])
		}
	}
	idx := strings.LastIndex(msg, "): ")
	if idx < 0 {
		t.Fatalf("error %q has no \"): \" detail separator", msg)
	}
	if detail := strings.TrimSpace(msg[idx+len("): "):]); detail == "" {
		t.Errorf("error %q has empty detail after the final \"): \"", msg)
	}
	return code
}

func TestOverlayMinePathsWSLStartFailureIsNamed(t *testing.T) {
	srcRoot := t.TempDir()
	if err := os.MkdirAll(filepath.Join(srcRoot, "a"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(srcRoot, "a", "b.go"), []byte("package a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// An empty PATH directory: wsl.exe cannot be resolved on any OS.
	t.Setenv("PATH", t.TempDir())

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	calls := 0
	err := overlayMinePathsWSLWithin(ctx, srcRoot, "/tmp/fak-validate-never", []string{"a/b.go"}, func(string) { calls++ })

	code := overlayWSLTestAssertNamedFailure(t, err, 1)
	if code != -1 {
		t.Errorf("exit_code=%d, want -1 when wsl.exe could not be started (error %q)", code, err)
	}
	msg := err.Error()
	for _, zero := range []string{"script_bytes=0", "command_chars=0"} {
		if strings.Contains(msg, zero) {
			t.Errorf("error %q reports %s, want a positive size", msg, zero)
		}
	}
	if calls != 0 {
		t.Errorf("checked called %d times on failure, want 0", calls)
	}
}

func TestOverlayMinePathsWSLFailureNamesExitCode(t *testing.T) {
	overlayWSLTestRequire(t)
	srcRoot := t.TempDir()
	if err := os.MkdirAll(filepath.Join(srcRoot, "a"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(srcRoot, "a", "b.go"), []byte("package a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var nonce [6]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		t.Fatal(err)
	}
	// mkdir under /proc fails inside WSL, so the child runs and exits non-zero.
	wslRoot := "/proc/fak-overlay-denied-" + hex.EncodeToString(nonce[:])

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	calls := 0
	err := overlayMinePathsWSLWithin(ctx, srcRoot, wslRoot, []string{"a/b.go"}, func(string) { calls++ })

	code := overlayWSLTestAssertNamedFailure(t, err, 1)
	if code <= 0 {
		t.Errorf("exit_code=%d, want the positive exit status of the failed WSL child (error %q)", code, err)
	}
	if calls != 0 {
		t.Errorf("checked called %d times on failure, want 0", calls)
	}
}

func TestOverlayMinePathsWSLLargeOwnedSet(t *testing.T) {
	if testing.Short() {
		t.Skip("large WSL overlay skipped in -short mode")
	}
	overlayWSLTestRequire(t)

	// The Windows-side fixture is built before the deadline starts, so the
	// 5-minute budget covers only WSL work.
	const fileCount = 640
	srcRoot := t.TempDir()
	files := make([]string, 0, fileCount)
	contents := make(map[string][]byte, fileCount)
	for i := 0; i < fileCount; i++ {
		dir := fmt.Sprintf("pkg/d%02d/sub", i%16)
		if i%3 == 0 {
			dir += "/inner"
		}
		rel := fmt.Sprintf("%s/file_%04d.go", dir, i)
		body := []byte(fmt.Sprintf("package d%02d\n\n// overlay fixture %04d\nconst v%04d = %d\n", i%16, i, i, i*7919))
		abs := filepath.Join(srcRoot, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(abs, body, 0o644); err != nil {
			t.Fatal(err)
		}
		files = append(files, rel)
		contents[rel] = body
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// Fresh WSL scratch root, removed on cleanup.
	wslRoot := overlayWSLTestLastLine(overlayWSLTestBash(ctx, t, "mktemp -d "+overlayWSLTestRootPrefix+"XXXXXX"))
	if !strings.HasPrefix(wslRoot, overlayWSLTestRootPrefix) {
		t.Fatalf("mktemp returned %q, want prefix %q", wslRoot, overlayWSLTestRootPrefix)
	}
	t.Cleanup(func() {
		if !strings.HasPrefix(wslRoot, overlayWSLTestRootPrefix) {
			return
		}
		cctx, ccancel := context.WithTimeout(context.Background(), time.Minute)
		defer ccancel()
		if out, err := exec.CommandContext(cctx, "wsl.exe", "bash", "-lc", "rm -rf -- "+overlayWSLTestQuote(wslRoot)).CombinedOutput(); err != nil {
			t.Logf("cleanup of %s failed: %v: %s", wslRoot, err, out)
		}
	})

	// A pre-existing WSL-side file that the overlay must delete.
	const gone = "gone/old.go"
	overlayWSLTestBash(ctx, t, "mkdir -p -- "+overlayWSLTestQuote(wslRoot+"/gone")+
		" && printf 'package gone\\n' > "+overlayWSLTestQuote(wslRoot+"/"+gone)+
		" && test -f "+overlayWSLTestQuote(wslRoot+"/"+gone)+" && echo seeded")

	paths := make([]string, 0, fileCount+1)
	paths = append(paths, files[:fileCount/2]...)
	paths = append(paths, gone) // absent from srcRoot: a deletion
	paths = append(paths, files[fileCount/2:]...)

	// Pin the regime: one `bash -lc` argument carrying a mkdir+cp per path would
	// exceed the CreateProcess limit. The source prefix is counted as empty, so
	// this is a strict lower bound on that argument's length.
	lowerBound := 0
	for _, rel := range files {
		lowerBound += len("mkdir -p -- '") + len(wslRoot) + len("/") + len(path.Dir(rel)) + len("'; ") +
			len("cp -- '") + len("/") + len(rel) + len("' '") + len(wslRoot) + len("/") + len(rel) + len("'; ")
	}
	if lowerBound <= overlayWSLTestCmdLineLimit {
		t.Fatalf("fixture too small: single-argument lower bound %d <= %d; the large-owned-set regime is not reached", lowerBound, overlayWSLTestCmdLineLimit)
	}
	t.Logf("single-argument lower bound %d chars > CreateProcess limit %d (%d owned paths)", lowerBound, overlayWSLTestCmdLineLimit, len(paths))

	seen := make(map[string]int, len(paths))
	var mu sync.Mutex
	start := time.Now()
	err := overlayMinePathsWSLWithin(ctx, srcRoot, wslRoot, paths, func(rel string) {
		mu.Lock()
		seen[rel]++
		mu.Unlock()
	})
	t.Logf("overlay of %d paths took %s", len(paths), time.Since(start))
	if err != nil {
		t.Fatalf("overlayMinePathsWSLWithin with %d owned paths: %v", len(paths), err)
	}

	if len(seen) != len(paths) {
		t.Errorf("checked saw %d distinct paths, want %d", len(seen), len(paths))
	}
	for _, rel := range paths {
		if n := seen[rel]; n != 1 {
			t.Errorf("checked(%q) called %d times, want 1", rel, n)
		}
	}

	countOut := overlayWSLTestBash(ctx, t, "find "+overlayWSLTestQuote(wslRoot)+" -type f | wc -l")
	count, convErr := strconv.Atoi(overlayWSLTestLastLine(countOut))
	if convErr != nil {
		t.Fatalf("parse file count from %q: %v", countOut, convErr)
	}
	if count != fileCount {
		t.Errorf("WSL root holds %d regular files, want %d", count, fileCount)
	}

	goneOut := overlayWSLTestBash(ctx, t, "if test -e "+overlayWSLTestQuote(wslRoot+"/"+gone)+"; then echo present; else echo absent; fi")
	if got := overlayWSLTestLastLine(goneOut); got != "absent" {
		t.Errorf("%s after overlay: %q, want absent (owned deletion not applied)", gone, got)
	}

	for _, rel := range []string{files[0], files[fileCount/2], files[fileCount-1]} {
		got := overlayWSLTestBash(ctx, t, "cat -- "+overlayWSLTestQuote(wslRoot+"/"+rel))
		if string(got) != string(contents[rel]) {
			t.Errorf("WSL copy of %s = %q, want %q", rel, got, contents[rel])
		}
	}
}
