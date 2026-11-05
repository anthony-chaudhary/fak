package model

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

// dflash2CountingFileCloser counts Close calls AND closes the bound file, so
// a retained handle's retirement releases the descriptor on every host. The
// binding is swapped on reopen (one open per load) and Close is idempotent
// per bound file.
type dflash2CountingFileCloser struct {
	mu    sync.Mutex
	f     *os.File
	calls atomic.Int32
}

func (c *dflash2CountingFileCloser) Bind(f *os.File) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.f = f
}

func (c *dflash2CountingFileCloser) Close() error {
	c.calls.Add(1)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.f == nil {
		return nil
	}
	bound := c.f
	c.f = nil
	return bound.Close()
}

// dflash2RetentionFixture installs a tiny real sidecar file on disk and swaps
// the retention opener seam so the lifecycle runs against it instead of the
// 3.8 GiB stock artifact. The swap is restored on cleanup; tests that rely on
// it must not run parallel to each other.
func dflash2RetentionFixture(t *testing.T) (path string, closer *dflash2CountingFileCloser) {
	t.Helper()
	file, contract := dflash2TestFixture(t)
	dir := t.TempDir()
	path = filepath.Join(dir, "sidecar.safetensors")
	if err := os.WriteFile(path, file, 0o600); err != nil {
		t.Fatal(err)
	}
	closer = &dflash2CountingFileCloser{}
	opener := func(ctx context.Context, p string) (*dflash2Weights, error) {
		f, err := os.Open(p)
		if err != nil {
			return nil, err
		}
		st, err := f.Stat()
		if err != nil {
			return nil, errors.Join(err, f.Close())
		}
		if !st.Mode().IsRegular() {
			return nil, errors.Join(errors.New("fixture: sidecar must be a regular file"), f.Close())
		}
		closer.Bind(f)
		return newDFlash2Weights(ctx, f, st.Size(), closer, contract)
	}
	prior := dflash2RetainSidecarOpener
	dflash2RetainSidecarOpener = opener
	t.Cleanup(func() { dflash2RetainSidecarOpener = prior })
	return path, closer
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2SidecarRetentionLifecycle(t *testing.T) {
	path, closer := dflash2RetentionFixture(t)

	first, err := RetainQwen38DFlash2Sidecar(context.Background(), path)
	if err != nil {
		t.Fatalf("first retain: %v", err)
	}
	if got := first.Path(); got != filepath.Clean(path) {
		t.Fatalf("handle path=%q want %q", got, filepath.Clean(path))
	}
	if n := dflash2SidecarRefCount(path); n != 1 {
		t.Fatalf("refcount after first retain=%d want 1", n)
	}
	if closer.calls.Load() != 0 {
		t.Fatalf("artifact closed before any release: %d", closer.calls.Load())
	}

	second, err := RetainQwen38DFlash2Sidecar(context.Background(), path+"/")
	if err != nil {
		t.Fatalf("second retain: %v", err)
	}
	if n := dflash2SidecarRefCount(path); n != 2 {
		t.Fatalf("refcount after second retain=%d want 2", n)
	}
	a, errA := first.borrow()
	b, errB := second.borrow()
	if errA != nil || errB != nil || a != b {
		t.Fatalf("shared reader identity: a=%v b=%v errA=%v errB=%v", a, b, errA, errB)
	}
	dst := make([]float32, 2)
	if err := a.ReadBF16(context.Background(), "a", 0, dst); err != nil || dst[0] != 1 {
		t.Fatalf("retained reader span: %v %v", dst, err)
	}

	if err := first.Release(); err != nil {
		t.Fatalf("first release: %v", err)
	}
	if n := dflash2SidecarRefCount(path); n != 1 {
		t.Fatalf("refcount after first release=%d want 1", n)
	}
	if closer.calls.Load() != 0 {
		t.Fatalf("artifact closed while a reference lived: %d", closer.calls.Load())
	}
	if _, err := second.borrow(); err != nil {
		t.Fatalf("surviving reference lost its reader: %v", err)
	}

	if err := second.Release(); err != nil {
		t.Fatalf("last release: %v", err)
	}
	if n := dflash2SidecarRefCount(path); n != 0 {
		t.Fatalf("refcount after last release=%d want 0", n)
	}
	if closer.calls.Load() != 1 {
		t.Fatalf("close count=%d want 1", closer.calls.Load())
	}
	if err := second.Release(); !errors.Is(err, ErrDFlash2SidecarReleased) {
		t.Fatalf("double release err=%v want ErrDFlash2SidecarReleased", err)
	}

	// A fresh load after full retirement opens the artifact again.
	third, err := RetainQwen38DFlash2Sidecar(context.Background(), path)
	if err != nil {
		t.Fatalf("retain after retirement: %v", err)
	}
	t.Cleanup(func() { _ = third.Release() })
	if n := dflash2SidecarRefCount(path); n != 1 {
		t.Fatalf("refcount after re-retain=%d want 1", n)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2SidecarRetentionUnavailable(t *testing.T) {
	path, _ := dflash2RetentionFixture(t)

	if _, err := RetainQwen38DFlash2Sidecar(context.Background(), filepath.Join(t.TempDir(), "absent.safetensors")); !errors.Is(err, ErrDFlash2SidecarUnavailable) {
		t.Fatalf("absent artifact err=%v want ErrDFlash2SidecarUnavailable", err)
	}
	if _, err := RetainQwen38DFlash2Sidecar(context.Background(), "relative/sidecar.safetensors"); !errors.Is(err, ErrDFlash2SidecarUnavailable) {
		t.Fatalf("relative path err=%v want ErrDFlash2SidecarUnavailable", err)
	}
	if _, err := RetainQwen38DFlash2Sidecar(context.Background(), ""); !errors.Is(err, ErrDFlash2SidecarUnavailable) {
		t.Fatalf("empty path err=%v want ErrDFlash2SidecarUnavailable", err)
	}
	if n := dflash2SidecarRefCount(path); n != 0 {
		t.Fatalf("failed retains registered a refcount: %d", n)
	}

	// A corrupted payload must authenticate as unavailable, not open.
	corrupt, _ := dflash2TestFixture(t)
	corrupt[len(corrupt)-1] ^= 1
	badPath := filepath.Join(t.TempDir(), "corrupt.safetensors")
	if err := os.WriteFile(badPath, corrupt, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := RetainQwen38DFlash2Sidecar(context.Background(), badPath); !errors.Is(err, ErrDFlash2SidecarUnavailable) {
		t.Fatalf("corrupt artifact err=%v want ErrDFlash2SidecarUnavailable", err)
	}

	// Cancellation is a distinct class: it is not reported as unavailable.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := RetainQwen38DFlash2Sidecar(ctx, path); !errors.Is(err, context.Canceled) || errors.Is(err, ErrDFlash2SidecarUnavailable) {
		t.Fatalf("cancelled retain err=%v want context.Canceled only", err)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2SidecarConcurrentRetainsOpenOnce(t *testing.T) {
	path, closer := dflash2RetentionFixture(t)
	const workers = 8
	handles := make([]*Qwen38DFlash2Sidecar, workers)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := range handles {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			handles[i], errs[i] = RetainQwen38DFlash2Sidecar(context.Background(), path)
		}(i)
	}
	close(start)
	wg.Wait()
	for i := range handles {
		if errs[i] != nil {
			t.Fatalf("concurrent retain %d: %v", i, errs[i])
		}
	}
	n := dflash2SidecarRefCount(path)
	if n != workers {
		t.Fatalf("refcount=%d want %d", n, workers)
	}
	for i := range handles {
		if err := handles[i].Release(); err != nil {
			t.Fatalf("release %d: %v", i, err)
		}
	}
	if closer.calls.Load() != 1 {
		t.Fatalf("close count=%d want 1 (one open per load)", closer.calls.Load())
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2SidecarBorrowAfterReleaseRefused(t *testing.T) {
	path, _ := dflash2RetentionFixture(t)
	handle, err := RetainQwen38DFlash2Sidecar(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if err := handle.Release(); err != nil {
		t.Fatal(err)
	}
	if _, err := handle.borrow(); !errors.Is(err, ErrDFlash2SidecarReleased) {
		t.Fatalf("borrow after release err=%v want ErrDFlash2SidecarReleased", err)
	}

	// A released sidecar refuses request construction before any target state
	// moves: the caller retains the fresh session and the serial verify route
	// stays visible. No silent reopen, no host fallback.
	m := qwen35TargetFeatureTestModel(t)
	target := m.NewSession()
	t.Cleanup(target.Close)
	if _, err := NewQwen38DFlash2RequestWithSidecar(context.Background(), target, handle); !errors.Is(err, ErrDFlash2SidecarReleased) {
		t.Fatalf("request on released sidecar err=%v want ErrDFlash2SidecarReleased", err)
	}
	if _, err := NewQwen38DFlash2RequestWithOptionsWithSidecar(context.Background(), target, handle, Qwen38DFlash2RequestOptions{}); !errors.Is(err, ErrDFlash2SidecarReleased) {
		t.Fatalf("options request on released sidecar err=%v want ErrDFlash2SidecarReleased", err)
	}
	if target.halStep != 0 || target.qwen35TargetLayerCapture != nil {
		t.Fatal("refused construction moved target state")
	}
}

// dflash2SidecarRefCount exposes the registry depth for assertions. It is a
// test-only read of the package-private retention table.
func dflash2SidecarRefCount(path string) int {
	dflash2SidecarRetention.Lock()
	defer dflash2SidecarRetention.Unlock()
	if entry := dflash2SidecarRetention.entries[filepath.Clean(path)]; entry != nil {
		return entry.refs
	}
	return 0
}
