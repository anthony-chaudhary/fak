package modelroute

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// routeManifest builds a tiny manifest that routes tool "x" to the named model, with
// a fail-closed default. Distinct models let a test prove WHICH manifest is live.
func routeManifestFor(model string) Manifest {
	return Manifest{
		Version: Version,
		Default: Plan{Members: []Member{{Model: "default", Role: "primary"}}},
		Rules: []Rule{{
			Name:  "route-x",
			Match: Match{Aspect: AspectToolCall, Tool: "x"},
			Plan:  Plan{Members: []Member{{Model: model}}},
		}},
	}
}

func writeManifestFile(t *testing.T, path string, model string) {
	t.Helper()
	m := routeManifestFor(model)
	if err := os.WriteFile(path, m.JSON(), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
}

// routedModel returns the model tool "x" routes to under the live policy.
func routedModel(l *Live) string {
	return l.Route(Subject{Aspect: AspectToolCall, Tool: "x"}).Plan.Primary()
}

func newTestWatcher(t *testing.T, path string, model string) (*Live, *Watcher) {
	t.Helper()
	writeManifestFile(t, path, model)
	m, err := LoadManifest(path)
	if err != nil {
		t.Fatalf("load initial manifest: %v", err)
	}
	live := NewLive(&m)
	w := NewWatcher(path, live, time.Millisecond, nil)
	return live, w
}

// TestWatcherPicksUpChange: an edit to the installed manifest is reflected by the
// live policy after a reload — no restart, the swap is observable.
func TestWatcherPicksUpChange(t *testing.T) {
	path := filepath.Join(t.TempDir(), "route.json")
	live, w := newTestWatcher(t, path, "alpha")
	if got := routedModel(live); got != "alpha" {
		t.Fatalf("initial route = %q, want alpha", got)
	}

	writeManifestFile(t, path, "beta")
	ev := w.Reload()
	if !ev.Reloaded || ev.Err != nil {
		t.Fatalf("reload event = %+v, want Reloaded with no error", ev)
	}
	if got := routedModel(live); got != "beta" {
		t.Fatalf("post-reload route = %q, want beta", got)
	}
	if live.Reloads() != 1 {
		t.Fatalf("Reloads() = %d, want 1", live.Reloads())
	}
}

// TestWatcherDoesNotMissEditBetweenLoadAndWatch covers the startup race: the host
// loads the initial manifest, then constructs the watcher. If the file changes in
// that window, the watcher must not seed the NEW bytes as already-installed.
func TestWatcherDoesNotMissEditBetweenLoadAndWatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "route.json")
	writeManifestFile(t, path, "alpha")
	m, err := LoadManifest(path)
	if err != nil {
		t.Fatalf("load initial manifest: %v", err)
	}
	live := NewLive(&m)

	writeManifestFile(t, path, "beta")
	w := NewWatcher(path, live, time.Millisecond, nil)
	if got := routedModel(live); got != "alpha" {
		t.Fatalf("pre-reload route = %q, want alpha", got)
	}
	ev := w.Reload()
	if !ev.Reloaded || ev.Err != nil {
		t.Fatalf("reload event = %+v, want the beta edit applied", ev)
	}
	if got := routedModel(live); got != "beta" {
		t.Fatalf("post-reload route = %q, want beta", got)
	}
}

// TestWatcherIdenticalContentIsNoOp: re-reading the same bytes does not swap or count
// a reload, so a touch / no-op write is not a spurious policy change.
func TestWatcherIdenticalContentIsNoOp(t *testing.T) {
	path := filepath.Join(t.TempDir(), "route.json")
	live, w := newTestWatcher(t, path, "alpha")

	// Rewrite identical content (advances mtime, same bytes).
	writeManifestFile(t, path, "alpha")
	ev := w.Reload()
	if ev.Reloaded || ev.Changed {
		t.Fatalf("identical-content reload = %+v, want no-op", ev)
	}
	if live.Reloads() != 0 {
		t.Fatalf("Reloads() = %d after identical write, want 0", live.Reloads())
	}
}

// TestWatcherRejectsMalformedKeepsLastGood: a malformed edit is rejected, the
// last-good manifest stays installed, and a subsequent VALID edit still reloads
// (proving the rejection did not poison the change-detection baseline).
func TestWatcherRejectsMalformedKeepsLastGood(t *testing.T) {
	path := filepath.Join(t.TempDir(), "route.json")
	live, w := newTestWatcher(t, path, "alpha")

	// 1) A malformed edit: rejected, last-good (alpha) kept.
	if err := os.WriteFile(path, []byte("{ this is not valid json"), 0o600); err != nil {
		t.Fatalf("write malformed: %v", err)
	}
	ev := w.Reload()
	if ev.Err == nil {
		t.Fatalf("malformed reload = %+v, want a rejection error", ev)
	}
	if ev.Reloaded {
		t.Fatalf("malformed reload reported Reloaded=true, want false")
	}
	if got := routedModel(live); got != "alpha" {
		t.Fatalf("after malformed edit route = %q, want last-good alpha", got)
	}
	if live.Rejects() != 1 {
		t.Fatalf("Rejects() = %d, want 1", live.Rejects())
	}

	// 2) A manifest that PARSES but fails validation (empty default plan) is also
	//    rejected (the security-boundary fail-loud contract, not just JSON syntax).
	if err := os.WriteFile(path, []byte(`{"version":"fak-route/v1","default":{"members":[]}}`), 0o600); err != nil {
		t.Fatalf("write invalid: %v", err)
	}
	if ev := w.Reload(); ev.Err == nil {
		t.Fatalf("validation-failing reload = %+v, want rejection", ev)
	}
	if got := routedModel(live); got != "alpha" {
		t.Fatalf("after invalid manifest route = %q, want last-good alpha", got)
	}
	if live.Rejects() != 2 {
		t.Fatalf("Rejects() = %d, want 2", live.Rejects())
	}

	// 3) Recovery: a valid edit reloads despite the prior rejections.
	writeManifestFile(t, path, "gamma")
	ev = w.Reload()
	if !ev.Reloaded || ev.Err != nil {
		t.Fatalf("recovery reload = %+v, want Reloaded", ev)
	}
	if got := routedModel(live); got != "gamma" {
		t.Fatalf("post-recovery route = %q, want gamma", got)
	}
}

// TestLiveAtomicSwapUnderConcurrentRoute exercises the torn-read-free property
// directly: many readers classify while a writer swaps the whole manifest. Every
// observed route must be a WHOLE manifest's answer (alpha or beta), never a torn
// in-between. Run with -race to prove the pointer publish/consume is data-race-free.
func TestLiveAtomicSwapUnderConcurrentRoute(t *testing.T) {
	mAlpha := routeManifestFor("alpha")
	mBeta := routeManifestFor("beta")
	live := NewLive(&mAlpha)

	var wg sync.WaitGroup
	stop := make(chan struct{})

	// Writer: flip the whole policy as fast as it can.
	wg.Add(1)
	go func() {
		defer wg.Done()
		flip := true
		for {
			select {
			case <-stop:
				return
			default:
				if flip {
					live.Store(&mBeta)
				} else {
					live.Store(&mAlpha)
				}
				flip = !flip
			}
		}
	}()

	// Readers: classify continuously; assert each decision is a coherent whole.
	for r := 0; r < 8; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 20000; i++ {
				got := routedModel(live)
				if got != "alpha" && got != "beta" {
					t.Errorf("torn route observed: %q", got)
					return
				}
			}
		}()
	}

	// Let the readers finish their fixed iteration counts, then stop the writer.
	time.Sleep(20 * time.Millisecond)
	close(stop)
	wg.Wait()
}

// TestWatcherRunStopsOnContextCancel: the polling loop honors ctx cancellation (the
// serve-lifetime contract) and reloads a real edit while running.
func TestWatcherRunStopsOnContextCancel(t *testing.T) {
	path := filepath.Join(t.TempDir(), "route.json")
	live, w := newTestWatcher(t, path, "alpha")

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()

	// Edit the file; the running loop should pick it up within a few poll intervals.
	writeManifestFile(t, path, "beta")
	deadline := time.After(2 * time.Second)
	for routedModel(live) != "beta" {
		select {
		case <-deadline:
			cancel()
			t.Fatalf("running watcher did not pick up the edit; route still %q", routedModel(live))
		case <-time.After(time.Millisecond):
		}
	}

	cancel()
	select {
	case err := <-done:
		if err != context.Canceled {
			t.Fatalf("Run returned %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Run did not return after cancel")
	}
}
