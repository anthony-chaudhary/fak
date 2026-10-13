package workerworktree

import (
	"context"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:runtime fast est=100ms
func TestDisambiguationCacheWaitHonorsCancellation(t *testing.T) {
	oldResolve, oldCacheRoot, oldRead := resolveDisambiguationTree, disambiguationCacheRoot, readDisambiguation
	t.Cleanup(func() {
		resolveDisambiguationTree, disambiguationCacheRoot, readDisambiguation = oldResolve, oldCacheRoot, oldRead
	})
	const tree = "shared-cache-tree"
	cacheRoot := t.TempDir()
	key := disambiguationCacheKey(tree)
	want := DisambiguationWitness{Fresh: true, SemanticValid: true, CriticalClean: true, Coverage: 100}
	if err := writeCachedDisambiguation(filepath.Join(cacheRoot, key+".json"), key, want); err != nil {
		t.Fatal(err)
	}
	resolveDisambiguationTree = func(context.Context, string, string) (string, error) { return tree, nil }

	holderAtCache, releaseHolder, holderExited := make(chan struct{}), make(chan struct{}), make(chan struct{})
	holderResult := make(chan DisambiguationWitness, 1)
	waiterAtLock, waiterExited, callerExited := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var releaseOnce, expireOnce, phaseOnce sync.Once
	manual := newManualDeadlineContext()
	var cacheCalls atomic.Int32
	disambiguationCacheRoot = func(string) (string, error) {
		if cacheCalls.Add(1) == 1 {
			close(holderAtCache)
			<-releaseHolder
		}
		return cacheRoot, nil
	}
	readDisambiguation = func(ctx context.Context, repo, tree string, phase func(string)) DisambiguationWitness {
		defer close(waiterExited)
		return readDisambiguationWitness(ctx, repo, tree, func(next string) {
			phase(next)
			if next == "cache-lock" {
				phaseOnce.Do(func() { close(waiterAtLock) })
			}
		})
	}
	waiterStarted := false
	t.Cleanup(func() {
		expireOnce.Do(manual.expire)
		releaseOnce.Do(func() { close(releaseHolder) })
		<-holderExited
		if waiterStarted {
			<-callerExited
			<-waiterExited
		}
	})
	go func() {
		defer close(holderExited)
		holderResult <- readDisambiguationWitness(context.Background(), "holder", tree, func(string) {})
	}()
	select {
	case <-holderAtCache:
	case <-time.After(5 * time.Second):
		t.Fatal("holder did not acquire the cache stripe")
	}
	type result struct {
		witness  DisambiguationWitness
		complete bool
	}
	out := make(chan result, 1)
	waiterStarted = true
	go func() {
		defer close(callerExited)
		witness, complete := readOneBounded(manual, "before", "waiter", tree, defaultDisambiguationTimeout)
		out <- result{witness, complete}
	}()
	select {
	case <-waiterAtLock:
	case <-time.After(time.Second):
		t.Error("contended reader did not attribute its wait to cache-lock")
	}
	expireOnce.Do(manual.expire)
	select {
	case got := <-out:
		d := got.witness.Diagnostic
		if got.complete || d == nil || d.Code != DisambiguationTimeoutCode || !d.Recoverable || d.Subphase != "cache-lock" || !strings.Contains(d.Recovery, DisambiguationTimeoutEnv) {
			t.Errorf("cache wait did not retain the typed recoverable deadline: complete=%t diagnostic=%+v", got.complete, d)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("bounded caller did not return after its deadline")
	}
	select {
	case <-waiterExited:
	case <-time.After(time.Second):
		t.Error("expired cache-stripe reader remains stranded behind the live holder")
	}
	if got := cacheCalls.Load(); got != 1 {
		t.Errorf("expired waiter entered cache work while holder was blocked: calls=%d", got)
	}
	select {
	case <-holderExited:
		t.Error("canceled waiter released another reader's stripe")
	default:
	}
	releaseOnce.Do(func() { close(releaseHolder) })
	<-holderExited
	<-waiterExited
	if got := <-holderResult; got.CacheState != "hit" || !got.SemanticValid {
		t.Fatalf("holder's cached verdict changed: %+v", got)
	}
	later := readDisambiguationWitness(context.Background(), "later", tree, func(string) {})
	if later.CacheState != "hit" || !later.SemanticValid || cacheCalls.Load() != 2 {
		t.Errorf("stripe was not reusable without canceled cache work: state=%q semantic_valid=%t cache_calls=%d", later.CacheState, later.SemanticValid, cacheCalls.Load())
	}
}
