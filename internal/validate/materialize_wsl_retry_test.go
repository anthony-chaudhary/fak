package validate

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

// wslRetryTestTornOutput is the verbatim drvfs failure a peer lander's
// packed-refs rewrite produced mid-land under `fak validate --mine`.
const (
	wslRetryTestTornOutput = "fatal: unterminated line in /mnt/c/work/fak-private/.git/packed-refs: 99e5c6e06a66d0ae37afc3449ca95dcad07280e9 refs/tags/orphan-ops-up\n"
	wslRetryTestTip        = "0123456789abcdef0123456789abcdef01234567"
	// Bounds the real wsl.exe run on the parent commit, where the seam is bypassed.
	wslRetryTestTimeout = 30 * time.Second
)

// wslRetryTestFake records every validateWSLCommand call, split into the
// materialize script (`--cd <repo> bash -lc <script>`) and the scratch cleanup.
type wslRetryTestFake struct {
	mu          sync.Mutex
	materialize [][]string
	cleanup     [][]string
}

func (f *wslRetryTestFake) counts() (int, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.materialize), len(f.cleanup)
}

// installWSLRetryTestFake swaps the package-global seam; callers must not run in parallel.
func installWSLRetryTestFake(t *testing.T, onMaterialize func(n int) ([]byte, error)) *wslRetryTestFake {
	t.Helper()
	fake := &wslRetryTestFake{}
	prev := validateWSLCommand
	t.Cleanup(func() { validateWSLCommand = prev })
	validateWSLCommand = func(_ context.Context, args ...string) ([]byte, error) {
		args = append([]string(nil), args...)
		fake.mu.Lock()
		switch {
		case len(args) > 0 && args[0] == "--cd":
			fake.materialize = append(fake.materialize, args)
			n := len(fake.materialize)
			fake.mu.Unlock()
			return onMaterialize(n)
		case len(args) > 0 && args[0] == "bash" && strings.Contains(args[len(args)-1], "rm -rf"):
			fake.cleanup = append(fake.cleanup, args)
			fake.mu.Unlock()
			return nil, nil
		default:
			fake.mu.Unlock()
			t.Errorf("unexpected WSL command: %q", args)
			return nil, errors.New("unexpected WSL command")
		}
	}
	return fake
}

// wslRetryTestScripts asserts every materialize call has the contract shape and
// returns the scripts in call order.
func wslRetryTestScripts(t *testing.T, fake *wslRetryTestFake, repo string) []string {
	t.Helper()
	fake.mu.Lock()
	defer fake.mu.Unlock()
	scripts := make([]string, 0, len(fake.materialize))
	for i, args := range fake.materialize {
		if len(args) != 5 || args[1] != repo || args[2] != "bash" || args[3] != "-lc" {
			t.Fatalf("materialize call %d args = %q, want [--cd %s bash -lc <script>]", i+1, args, repo)
		}
		scripts = append(scripts, args[4])
	}
	return scripts
}

func wslRetryTestSameScript(t *testing.T, scripts []string) {
	t.Helper()
	for i := 1; i < len(scripts); i++ {
		if scripts[i] != scripts[0] {
			t.Fatalf("attempt %d ran a different script than attempt 1:\n%s\nvs\n%s", i+1, scripts[i], scripts[0])
		}
	}
}

// wslRetryTestCleanupTarget returns the single cleanup call's rm -rf target.
func wslRetryTestCleanupTarget(t *testing.T, fake *wslRetryTestFake) string {
	t.Helper()
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.cleanup) != 1 {
		t.Fatalf("cleanup calls = %d, want 1: %q", len(fake.cleanup), fake.cleanup)
	}
	args := fake.cleanup[0]
	_, target, ok := strings.Cut(args[len(args)-1], "rm -rf -- ")
	if !ok {
		t.Fatalf("cleanup script %q has no rm -rf -- target", args[len(args)-1])
	}
	target = strings.Trim(strings.TrimSpace(target), "'")
	if !strings.HasPrefix(target, "/tmp/fak-validate-") || target == "/tmp/fak-validate-cache" {
		t.Fatalf("cleanup target = %q, want the /tmp/fak-validate-* scratch dir (never the archive cache)", target)
	}
	return target
}

// TestExtractCommittedTipWSLRetriesTransientPackedRefsParse is the symptom
// witness: a torn packed-refs read through /mnt/c must not abort the land.
func TestExtractCommittedTipWSLRetriesTransientPackedRefsParse(t *testing.T) {
	repo := t.TempDir()
	fake := installWSLRetryTestFake(t, func(n int) ([]byte, error) {
		if n == 1 {
			return []byte(wslRetryTestTornOutput), errors.New("exit status 128")
		}
		return nil, nil
	})
	ctx, cancel := context.WithTimeout(context.Background(), wslRetryTestTimeout)
	defer cancel()

	dir, err := extractCommittedTipWSLWithin(ctx, repo, wslRetryTestTip)
	materialize, cleanup := fake.counts()
	if err != nil {
		t.Fatalf("extractCommittedTipWSLWithin: %v (materialize calls=%d)", err, materialize)
	}
	if !strings.HasPrefix(dir, "/tmp/fak-validate-") {
		t.Fatalf("dir = %q, want /tmp/fak-validate-* prefix", dir)
	}
	if materialize != 2 {
		t.Fatalf("materialize calls = %d, want 2 (torn read, then retry)", materialize)
	}
	scripts := wslRetryTestScripts(t, fake, repo)
	wslRetryTestSameScript(t, scripts)
	if !strings.Contains(scripts[0], dir) {
		t.Fatalf("returned dir %q is not the dir the script materialized:\n%s", dir, scripts[0])
	}
	if cleanup != 0 {
		t.Fatalf("cleanup calls = %d, want 0 on success", cleanup)
	}
}

func TestExtractCommittedTipWSLDoesNotRetryGenuineFailure(t *testing.T) {
	repo := t.TempDir()
	fake := installWSLRetryTestFake(t, func(int) ([]byte, error) {
		return []byte("fatal: not a valid object name: " + wslRetryTestTip + "\n"), errors.New("exit status 128")
	})
	ctx, cancel := context.WithTimeout(context.Background(), wslRetryTestTimeout)
	defer cancel()

	dir, err := extractCommittedTipWSLWithin(ctx, repo, wslRetryTestTip)
	if err == nil {
		t.Fatalf("extractCommittedTipWSLWithin succeeded with dir %q, want error", dir)
	}
	msg := err.Error()
	if !strings.Contains(msg, "materialize committed tip in WSL: ") || !strings.Contains(msg, "not a valid object name") {
		t.Fatalf("error = %q, want materialize prefix and the git detail", msg)
	}
	if dir != "" {
		t.Fatalf("dir = %q, want empty on failure", dir)
	}
	if materialize, _ := fake.counts(); materialize != 1 {
		t.Fatalf("materialize calls = %d, want 1 (non-transient failure must not retry)", materialize)
	}
	scripts := wslRetryTestScripts(t, fake, repo)
	// The cleanup must remove the scratch dir this attempt created.
	if target := wslRetryTestCleanupTarget(t, fake); !strings.Contains(scripts[0], target) {
		t.Fatalf("cleanup target %q is not the materialized dir:\n%s", target, scripts[0])
	}
}

func TestExtractCommittedTipWSLFailsClosedWhenPackedRefsStaysTorn(t *testing.T) {
	repo := t.TempDir()
	fake := installWSLRetryTestFake(t, func(int) ([]byte, error) {
		return []byte(wslRetryTestTornOutput), errors.New("exit status 128")
	})
	ctx, cancel := context.WithTimeout(context.Background(), wslRetryTestTimeout)
	defer cancel()

	// Sleeps through the full backoff (~1.75s) by design.
	dir, err := extractCommittedTipWSLWithin(ctx, repo, wslRetryTestTip)
	if err == nil {
		t.Fatalf("extractCommittedTipWSLWithin succeeded with dir %q, want fail-closed error", dir)
	}
	msg := err.Error()
	if !strings.HasPrefix(msg, "materialize committed tip in WSL: ") || !strings.Contains(msg, "packed-refs") || !strings.Contains(msg, "attempts") {
		t.Fatalf("error = %q, want materialize prefix, the last packed-refs output, and the attempt count", msg)
	}
	if dir != "" {
		t.Fatalf("dir = %q, want empty on failure", dir)
	}
	if materialize, _ := fake.counts(); materialize != 4 {
		t.Fatalf("materialize calls = %d, want 4 (bounded retries)", materialize)
	}
	wslRetryTestSameScript(t, wslRetryTestScripts(t, fake, repo))
	wslRetryTestCleanupTarget(t, fake)
}

func TestExtractCommittedTipWSLStopsOnContextCancel(t *testing.T) {
	repo := t.TempDir()
	base, cancelBase := context.WithTimeout(context.Background(), wslRetryTestTimeout)
	defer cancelBase()
	ctx, cancel := context.WithCancel(base)
	defer cancel()
	fake := installWSLRetryTestFake(t, func(n int) ([]byte, error) {
		if n == 1 {
			cancel() // operator abort lands while the torn read is being reported
		}
		return []byte(wslRetryTestTornOutput), errors.New("exit status 128")
	})

	start := time.Now()
	_, err := extractCommittedTipWSLWithin(ctx, repo, wslRetryTestTip)
	elapsed := time.Since(start)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if materialize, _ := fake.counts(); materialize != 1 {
		t.Fatalf("materialize calls = %d, want 1 (no retry after cancel)", materialize)
	}
	if elapsed >= time.Second {
		t.Fatalf("returned after %s, want < 1s (must not sleep out the backoff)", elapsed)
	}
}
