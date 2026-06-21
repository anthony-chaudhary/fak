package enginecache

// Witness tests closing OPEN proof obligations for internal/enginecache.
// Discipline: fak/docs/proofs/00-METHOD.md.
//
// OPEN (1) [enginecache-end-to-end-not-served-stale]:
//   After Invalidate succeeds against a live serving engine, a subsequent
//   request for the invalidated prefix/span observes a cache MISS (is
//   recomputed), not the pre-invalidation cached value.
//   mechanism: enginecache.go:57 (Invalidate), enginecache.go:130 (SupportsExactSpan)
//
// enginecache hosts no cache of its own; it translates cachemeta directives
// into the documented engine control-plane calls (SGLang POST /flush_cache,
// vLLM POST /reset_prefix_cache). The "live serving engine" is the stateful
// peer. We therefore stand up a fake serving engine that faithfully models the
// only contract enginecache relies on: its prefix cache is CLEARED when (and
// only when) the documented reset endpoint is POSTed. We then exercise the real
// Invalidate against it and assert the end-to-end recompute property.
//
// The fake engine serves a prefix by caching the FIRST computed value for that
// prefix (a generation counter), so a stale serve is byte-detectable: a HIT
// returns the warmed value, a MISS recomputes a strictly newer value. This is a
// real metamorphic/round-trip assertion (warm -> invalidate -> re-serve must
// MISS), not a smoke test.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// fakeServingEngine models a serving engine's prefix cache plus its documented
// reset control endpoint. serve(prefix) returns (value, hit): on a cold prefix
// it "recomputes" the next generation value and caches it; on a warm prefix it
// returns the cached value (a HIT). resetPath POSTs clear the entire prefix
// cache, exactly as SGLang /flush_cache and vLLM /reset_prefix_cache do.
type fakeServingEngine struct {
	mu        sync.Mutex
	cache     map[string]int // prefix -> cached generation value
	nextGen   int            // monotonically increasing recompute counter
	resets    int            // number of honored whole-cache resets
	resetPath string         // documented control endpoint for this engine
}

func newFakeServingEngine(resetPath string) *fakeServingEngine {
	return &fakeServingEngine{cache: map[string]int{}, resetPath: resetPath}
}

// serve returns the value the engine would return for prefix, and whether it
// was a cache HIT. A MISS recomputes a strictly newer value and warms the cache.
func (f *fakeServingEngine) serve(prefix string) (val int, hit bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if v, ok := f.cache[prefix]; ok {
		return v, true
	}
	f.nextGen++
	f.cache[prefix] = f.nextGen
	return f.nextGen, false
}

func (f *fakeServingEngine) handler(t *testing.T) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("control-plane method = %s, want POST", r.Method)
			http.Error(w, "method", http.StatusMethodNotAllowed)
			return
		}
		if r.URL.Path != f.resetPath {
			t.Errorf("control-plane path = %q, want %q", r.URL.Path, f.resetPath)
			http.Error(w, "path", http.StatusNotFound)
			return
		}
		f.mu.Lock()
		f.cache = map[string]int{} // documented whole-prefix-cache reset
		f.resets++
		f.mu.Unlock()
		_, _ = w.Write([]byte(`{"success":true}`))
	}
}

// runEndToEndNotStale exercises the warm -> Invalidate -> re-serve cycle against
// a fake engine and asserts the recompute (anti-stale) property holds.
func runEndToEndNotStale(t *testing.T, engine Engine, resetPath string, idle time.Duration) {
	t.Helper()
	const prefix = "tokens:1,2,3|glm-5.2"

	eng := newFakeServingEngine(resetPath)
	ts := httptest.NewServer(eng.handler(t))
	defer ts.Close()

	// (a) Warm the engine cache: first serve is a MISS and computes a value.
	warm, hit := eng.serve(prefix)
	if hit {
		t.Fatalf("first serve should be a MISS on a cold engine, got HIT")
	}

	// (b) Confirm the cache is now genuinely warm: re-serving WITHOUT any
	//     invalidation returns the identical cached value (a HIT). Without
	//     this the anti-stale assertion in (d) would be vacuous.
	again, hit := eng.serve(prefix)
	if !hit {
		t.Fatalf("re-serve before invalidation should be a HIT (warm cache)")
	}
	if again != warm {
		t.Fatalf("warm cache changed without invalidation: %d -> %d", warm, again)
	}

	// (c) Invalidate the prefix through the REAL enginecache client against the
	//     live engine. This must succeed and drive exactly one whole-cache reset.
	client := Client{Engine: engine, BaseURL: ts.URL, IdleTimeout: idle}
	res, err := client.Invalidate(
		context.Background(),
		[]cachemeta.ExternalInvalidationDirective{sampleDirective(string(engine))},
	)
	if err != nil {
		t.Fatalf("Invalidate against live %s engine: %v", engine, err)
	}
	if res.StatusCode != http.StatusOK {
		t.Fatalf("Invalidate status = %d, want 200", res.StatusCode)
	}
	if eng.resets != 1 {
		t.Fatalf("engine resets = %d, want exactly 1 after a successful Invalidate", eng.resets)
	}

	// (d) THE PROPERTY: a subsequent request for the invalidated prefix must
	//     observe a cache MISS (recompute) and must NOT return the
	//     pre-invalidation cached value.
	post, hit := eng.serve(prefix)
	if hit {
		t.Fatalf("served STALE: re-serve after Invalidate was a HIT, want MISS (recompute)")
	}
	if post == warm {
		t.Fatalf("served STALE value %d after Invalidate; recompute must yield a fresh value", warm)
	}
	if post <= warm {
		t.Fatalf("recompute value %d not strictly newer than pre-invalidation %d", post, warm)
	}
}

// TestEndToEndNotServedStaleSGLang witnesses the anti-stale property for the
// SGLang control plane (POST /flush_cache), including the idle-timeout query.
func TestEndToEndNotServedStaleSGLang(t *testing.T) {
	runEndToEndNotStale(t, EngineSGLang, "/flush_cache", 30*time.Second)
}

// TestEndToEndNotServedStaleVLLM witnesses the anti-stale property for the vLLM
// control plane (POST /reset_prefix_cache).
func TestEndToEndNotServedStaleVLLM(t *testing.T) {
	runEndToEndNotStale(t, EngineVLLM, "/reset_prefix_cache", 0)
}

// TestNoInvalidateLeavesCacheStale is the metamorphic CONTRAST that proves the
// anti-stale tests above are non-vacuous: with NO Invalidate call the warmed
// value is still served (a HIT). The anti-stale property is a direct
// consequence of Invalidate, not of re-serving alone.
func TestNoInvalidateLeavesCacheStale(t *testing.T) {
	const prefix = "tokens:1,2,3|glm-5.2"
	eng := newFakeServingEngine("/flush_cache")
	ts := httptest.NewServer(eng.handler(t))
	defer ts.Close()

	warm, hit := eng.serve(prefix)
	if hit {
		t.Fatalf("first serve should MISS")
	}
	// No Invalidate. The engine keeps serving the warmed value.
	got, hit := eng.serve(prefix)
	if !hit {
		t.Fatalf("without Invalidate the warmed prefix must stay a HIT")
	}
	if got != warm {
		t.Fatalf("without Invalidate the served value must remain %d, got %d", warm, got)
	}
	if eng.resets != 0 {
		t.Fatalf("no reset should occur without Invalidate, got %d", eng.resets)
	}
}
