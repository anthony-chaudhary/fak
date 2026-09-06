package codetools

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// TestSearchSingleflightGrep verifies that 5-10 concurrent Grep calls with identical
// query parameters coalesce into a single execution, where the leader receives coalesced: false
// and all joiners receive coalesced: true.
func TestSearchSingleflightGrep(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Seed files: 10 files, 5 with matches, 5 without
	const matchToken = "COALESCE_TEST_NEEDLE"
	for i := 0; i < 10; i++ {
		content := fmt.Sprintf("file %d content\n", i)
		if i%2 == 0 {
			content += fmt.Sprintf("hit: %s found here\n", matchToken)
		}
		mustWrite(t, filepath.Join(dir, fmt.Sprintf("sub/file_%02d.txt", i)), content)
	}

	const total = 8
	key := grepFlightKey(ts.root, matchToken, "", ts.limits.MaxMatches)

	// searchHook pauses the leader during its walk until all (total-1) joiners have entered Do.
	var hookOnce sync.Once
	ts.searchHook = func() {
		hookOnce.Do(func() {
			deadline := time.Now().Add(5 * time.Second)
			for ts.grepFlight.Waiters(key) < int32(total-1) {
				if time.Now().After(deadline) {
					t.Errorf("timed out waiting for %d joiners to enter Do (have %d)", total-1, ts.grepFlight.Waiters(key))
					return
				}
				time.Sleep(1 * time.Millisecond)
			}
		})
	}

	type result struct {
		data map[string]any
		err  error
	}
	results := make([]result, total)
	var wg sync.WaitGroup
	wg.Add(total)

	start := make(chan struct{})
	for i := 0; i < total; i++ {
		idx := i
		go func() {
			defer wg.Done()
			<-start
			out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: matchToken}))
			if isErr {
				results[idx].err = fmt.Errorf("grep failed: %s", string(out))
				return
			}
			results[idx].data = decodeResult(t, out)
		}()
	}

	close(start)
	wg.Wait()

	leaderCount := 0
	joinerCount := 0

	for i, r := range results {
		if r.err != nil {
			t.Fatalf("goroutine %d failed: %v", i, r.err)
		}
		if r.data["pattern"] != matchToken {
			t.Fatalf("goroutine %d pattern = %v, want %s", i, r.data["pattern"], matchToken)
		}
		if r.data["match_count"] != float64(5) {
			t.Fatalf("goroutine %d match_count = %v, want 5", i, r.data["match_count"])
		}
		matches, ok := r.data["matches"].([]any)
		if !ok || len(matches) != 5 {
			t.Fatalf("goroutine %d len(matches) = %d, want 5", i, len(matches))
		}

		coalesced, ok := r.data["coalesced"].(bool)
		if !ok {
			t.Fatalf("goroutine %d missing coalesced boolean field: %+v", i, r.data)
		}
		if coalesced {
			joinerCount++
		} else {
			leaderCount++
		}
	}

	if leaderCount != 1 {
		t.Fatalf("leaderCount = %d, want 1", leaderCount)
	}
	if joinerCount != total-1 {
		t.Fatalf("joinerCount = %d, want %d", joinerCount, total-1)
	}
	if coalesced := ts.GrepCoalesced(); coalesced != int64(total-1) {
		t.Fatalf("GrepCoalesced() = %d, want %d", coalesced, total-1)
	}
}

// TestSearchSingleflightGlob verifies that concurrent Glob calls coalesce into a single execution.
func TestSearchSingleflightGlob(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Seed files
	for i := 0; i < 8; i++ {
		mustWrite(t, filepath.Join(dir, fmt.Sprintf("src/mod_%02d.go", i)), "package main\n")
		mustWrite(t, filepath.Join(dir, fmt.Sprintf("docs/doc_%02d.md", i)), "# Title\n")
	}

	const total = 6
	key := globFlightKey(ts.root, "**/*.go")

	var hookOnce sync.Once
	ts.searchHook = func() {
		hookOnce.Do(func() {
			deadline := time.Now().Add(5 * time.Second)
			for ts.globFlight.Waiters(key) < int32(total-1) {
				if time.Now().After(deadline) {
					t.Errorf("timed out waiting for %d joiners to enter Do (have %d)", total-1, ts.globFlight.Waiters(key))
					return
				}
				time.Sleep(1 * time.Millisecond)
			}
		})
	}

	type result struct {
		data map[string]any
		err  error
	}
	results := make([]result, total)
	var wg sync.WaitGroup
	wg.Add(total)

	start := make(chan struct{})
	for i := 0; i < total; i++ {
		idx := i
		go func() {
			defer wg.Done()
			<-start
			out, isErr := ts.glob(context.Background(), argsOf(t, GlobArgs{Pattern: "**/*.go"}))
			if isErr {
				results[idx].err = fmt.Errorf("glob failed: %s", string(out))
				return
			}
			results[idx].data = decodeResult(t, out)
		}()
	}

	close(start)
	wg.Wait()

	leaderCount := 0
	joinerCount := 0

	for i, r := range results {
		if r.err != nil {
			t.Fatalf("goroutine %d failed: %v", i, r.err)
		}
		if r.data["pattern"] != "**/*.go" {
			t.Fatalf("goroutine %d pattern = %v, want **/*.go", i, r.data["pattern"])
		}
		if r.data["count"] != float64(8) {
			t.Fatalf("goroutine %d count = %v, want 8", i, r.data["count"])
		}
		coalesced, ok := r.data["coalesced"].(bool)
		if !ok {
			t.Fatalf("goroutine %d missing coalesced boolean field: %+v", i, r.data)
		}
		if coalesced {
			joinerCount++
		} else {
			leaderCount++
		}
	}

	if leaderCount != 1 {
		t.Fatalf("leaderCount = %d, want 1", leaderCount)
	}
	if joinerCount != total-1 {
		t.Fatalf("joinerCount = %d, want %d", joinerCount, total-1)
	}
	if coalesced := ts.GlobCoalesced(); coalesced != int64(total-1) {
		t.Fatalf("GlobCoalesced() = %d, want %d", coalesced, total-1)
	}
}

// TestSearchSingleflightSimultaneousBurst verifies that a burst of 10 simultaneous
// search goroutines all receive valid results and at least one joiner coalesces.
func TestSearchSingleflightSimultaneousBurst(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Write 50 files so the walk has enough work to provide a natural window
	const pattern = "BURST_NEEDLE"
	for i := 0; i < 50; i++ {
		mustWrite(t, filepath.Join(dir, fmt.Sprintf("deep/nested/path_%02d.txt", i)),
			fmt.Sprintf("some content line\n%s on line 2\nend\n", pattern))
	}

	const total = 10
	type result struct {
		data map[string]any
		err  error
	}
	results := make([]result, total)
	var wg sync.WaitGroup
	wg.Add(total)

	start := make(chan struct{})
	for i := 0; i < total; i++ {
		idx := i
		go func() {
			defer wg.Done()
			<-start
			out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: pattern}))
			if isErr {
				results[idx].err = fmt.Errorf("grep failed: %s", string(out))
				return
			}
			results[idx].data = decodeResult(t, out)
		}()
	}

	close(start)
	wg.Wait()

	leaders := 0
	joiners := 0
	for i, r := range results {
		if r.err != nil {
			t.Fatalf("goroutine %d failed: %v", i, r.err)
		}
		if r.data["match_count"] != float64(50) {
			t.Fatalf("goroutine %d match_count = %v, want 50", i, r.data["match_count"])
		}
		coalesced, ok := r.data["coalesced"].(bool)
		if !ok {
			t.Fatalf("goroutine %d missing coalesced field", i)
		}
		if coalesced {
			joiners++
		} else {
			leaders++
		}
	}

	if leaders < 1 {
		t.Fatalf("leaders = %d, want >= 1", leaders)
	}
	if joiners < 1 {
		t.Fatalf("joiners = %d, want >= 1 in concurrent burst", joiners)
	}
	if ts.GrepCoalesced() < 1 {
		t.Fatalf("GrepCoalesced = %d, want >= 1", ts.GrepCoalesced())
	}
}

// TestSearchSingleflightDistinctQueriesDoNotCoalesce verifies that distinct query parameters
// do not coalesce into each other.
func TestSearchSingleflightDistinctQueriesDoNotCoalesce(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	mustWrite(t, filepath.Join(dir, "f1.txt"), "target_alpha match\n")
	mustWrite(t, filepath.Join(dir, "f2.txt"), "target_beta match\n")

	var wg sync.WaitGroup
	wg.Add(2)

	var resAlpha, resBeta map[string]any
	start := make(chan struct{})

	go func() {
		defer wg.Done()
		<-start
		out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "target_alpha"}))
		if isErr {
			t.Errorf("grep alpha failed: %s", string(out))
			return
		}
		resAlpha = decodeResult(t, out)
	}()

	go func() {
		defer wg.Done()
		<-start
		out, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "target_beta"}))
		if isErr {
			t.Errorf("grep beta failed: %s", string(out))
			return
		}
		resBeta = decodeResult(t, out)
	}()

	close(start)
	wg.Wait()

	if resAlpha == nil || resBeta == nil {
		t.Fatal("expected both queries to complete")
	}

	if resAlpha["coalesced"] != false {
		t.Fatalf("alpha coalesced = %v, want false", resAlpha["coalesced"])
	}
	if resBeta["coalesced"] != false {
		t.Fatalf("beta coalesced = %v, want false", resBeta["coalesced"])
	}
	if resAlpha["match_count"] != float64(1) {
		t.Fatalf("alpha match_count = %v, want 1", resAlpha["match_count"])
	}
	if resBeta["match_count"] != float64(1) {
		t.Fatalf("beta match_count = %v, want 1", resBeta["match_count"])
	}
}

// TestSearchSingleflightLateArrivalStartsFresh verifies that a query arriving after
// an in-flight search has completed starts its own fresh execution rather than adopting
// a completed answer (anti-cache invariant).
func TestSearchSingleflightLateArrivalStartsFresh(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	mustWrite(t, filepath.Join(dir, "f1.txt"), "first match\n")

	// 1. Initial query finishes
	out1, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "first match"}))
	if isErr {
		t.Fatalf("first grep failed: %s", string(out1))
	}
	m1 := decodeResult(t, out1)
	if m1["coalesced"] != false {
		t.Fatalf("initial coalesced = %v, want false", m1["coalesced"])
	}

	// 2. Subsequent query for the exact same pattern after flight is completed
	out2, isErr := ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: "first match"}))
	if isErr {
		t.Fatalf("second grep failed: %s", string(out2))
	}
	m2 := decodeResult(t, out2)
	if m2["coalesced"] != false {
		t.Fatalf("late arrival coalesced = %v, want false (starts fresh)", m2["coalesced"])
	}
}

// TestSearchSingleflightJoinerTimeoutUnblocksImmediately verifies that a joiner with a
// short timeout context unblocks immediately with CANCELED when its context expires, while
// the leader continues and later finishes successfully.
func TestSearchSingleflightJoinerTimeoutUnblocksImmediately(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	const matchToken = "JOINER_TIMEOUT_NEEDLE"
	mustWrite(t, filepath.Join(dir, "f1.txt"), "hit: "+matchToken+" here\n")

	key := grepFlightKey(ts.root, matchToken, "", ts.limits.MaxMatches)

	leaderHold := make(chan struct{})
	leaderStarted := make(chan struct{})
	var hookOnce sync.Once

	ts.searchHook = func() {
		hookOnce.Do(func() {
			close(leaderStarted)
			<-leaderHold
		})
	}

	// 1. Start leader with long context; it pauses in searchHook.
	var leaderOut []byte
	var leaderErr bool
	leaderDone := make(chan struct{})
	go func() {
		defer close(leaderDone)
		leaderOut, leaderErr = ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: matchToken}))
	}()

	select {
	case <-leaderStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("leader did not start in time")
	}

	// 2. Start joiner with a 50ms timeout.
	ctxJoiner, cancelJoiner := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancelJoiner()

	start := time.Now()
	joinerOut, joinerErr := ts.grep(ctxJoiner, argsOf(t, GrepArgs{Pattern: matchToken}))
	elapsed := time.Since(start)

	if elapsed > 400*time.Millisecond {
		t.Fatalf("joiner hung for %v, want timeout in ~50ms", elapsed)
	}
	if !joinerErr {
		t.Fatalf("joiner succeeded, want canceled refusal: %s", string(joinerOut))
	}
	code := errCode(t, joinerOut)
	if code != CodeCanceled {
		t.Fatalf("joiner refusal code = %q, want %q", code, CodeCanceled)
	}

	// Waiters count must be 0 after joiner unblocks.
	if waiters := ts.grepFlight.Waiters(key); waiters != 0 {
		t.Fatalf("Waiters = %d, want 0 after joiner unblocked", waiters)
	}

	// Leader must still be held.
	select {
	case <-leaderDone:
		t.Fatal("leader finished prematurely while still held")
	default:
	}

	// 3. Release leader and verify it succeeds.
	close(leaderHold)

	select {
	case <-leaderDone:
	case <-time.After(2 * time.Second):
		t.Fatal("leader did not complete after release")
	}

	if leaderErr {
		t.Fatalf("leader failed: %s", string(leaderOut))
	}
	leaderData := decodeResult(t, leaderOut)
	if leaderData["match_count"] != float64(1) {
		t.Fatalf("leader match_count = %v, want 1", leaderData["match_count"])
	}
}

// TestSearchSingleflightLeaderCancellationJoinerRecovers verifies that when a leader's
// context is canceled, waiting joiners do not inherit the premature cancellation refusal,
// but instead recover and complete their search successfully.
func TestSearchSingleflightLeaderCancellationJoinerRecovers(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	const matchToken = "LEADER_CANCEL_NEEDLE"
	mustWrite(t, filepath.Join(dir, "match.txt"), "found: "+matchToken+"\n")

	key := grepFlightKey(ts.root, matchToken, "", ts.limits.MaxMatches)

	ctxLeader, cancelLeader := context.WithCancel(context.Background())
	defer cancelLeader()

	var hookOnce sync.Once
	ts.searchHook = func() {
		hookOnce.Do(func() {
			deadline := time.Now().Add(2 * time.Second)
			for ts.grepFlight.Waiters(key) < 1 {
				if time.Now().After(deadline) {
					t.Errorf("timed out waiting for joiner to enter Do")
					return
				}
				time.Sleep(1 * time.Millisecond)
			}
			cancelLeader()
		})
	}

	var leaderOut []byte
	var leaderErr bool
	var joinerOut []byte
	var joinerErr bool
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		leaderOut, leaderErr = ts.grep(ctxLeader, argsOf(t, GrepArgs{Pattern: matchToken}))
	}()

	go func() {
		defer wg.Done()
		joinerOut, joinerErr = ts.grep(context.Background(), argsOf(t, GrepArgs{Pattern: matchToken}))
	}()

	wg.Wait()

	if !leaderErr {
		t.Fatalf("leader unexpectedly succeeded: %s", string(leaderOut))
	}
	if code := errCode(t, leaderOut); code != CodeCanceled {
		t.Fatalf("leader error code = %q, want %q", code, CodeCanceled)
	}

	if joinerErr {
		t.Fatalf("joiner failed prematurely: %s", string(joinerOut))
	}
	joinerData := decodeResult(t, joinerOut)
	if joinerData["match_count"] != float64(1) {
		t.Fatalf("joiner match_count = %v, want 1", joinerData["match_count"])
	}
}

// TestSearchSingleflightGlobJoinerTimeoutUnblocksImmediately verifies that a glob joiner
// unblocks immediately when its context times out.
func TestSearchSingleflightGlobJoinerTimeoutUnblocksImmediately(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	mustWrite(t, filepath.Join(dir, "src/main.go"), "package main\n")
	key := ts.root + "\x00**/*.go"

	leaderHold := make(chan struct{})
	leaderStarted := make(chan struct{})
	var hookOnce sync.Once

	ts.searchHook = func() {
		hookOnce.Do(func() {
			close(leaderStarted)
			<-leaderHold
		})
	}

	var leaderOut []byte
	var leaderErr bool
	leaderDone := make(chan struct{})
	go func() {
		defer close(leaderDone)
		leaderOut, leaderErr = ts.glob(context.Background(), argsOf(t, GlobArgs{Pattern: "**/*.go"}))
	}()

	select {
	case <-leaderStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("leader did not start in time")
	}

	ctxJoiner, cancelJoiner := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancelJoiner()

	start := time.Now()
	joinerOut, joinerErr := ts.glob(ctxJoiner, argsOf(t, GlobArgs{Pattern: "**/*.go"}))
	elapsed := time.Since(start)

	if elapsed > 400*time.Millisecond {
		t.Fatalf("glob joiner hung for %v, want timeout in ~50ms", elapsed)
	}
	if !joinerErr {
		t.Fatalf("glob joiner succeeded, want canceled refusal: %s", string(joinerOut))
	}
	code := errCode(t, joinerOut)
	if code != CodeCanceled {
		t.Fatalf("glob joiner refusal code = %q, want %q", code, CodeCanceled)
	}

	if waiters := ts.globFlight.Waiters(key); waiters != 0 {
		t.Fatalf("Waiters = %d, want 0 after joiner unblocked", waiters)
	}

	close(leaderHold)
	select {
	case <-leaderDone:
	case <-time.After(2 * time.Second):
		t.Fatal("leader did not complete after release")
	}

	if leaderErr {
		t.Fatalf("leader failed: %s", string(leaderOut))
	}
}

// TestSearchSingleflightGlobLeaderCancellationJoinerRecovers verifies that when a glob leader's
// context is canceled, waiting joiners recover and complete without failing prematurely.
func TestSearchSingleflightGlobLeaderCancellationJoinerRecovers(t *testing.T) {
	dir := t.TempDir()
	ts, err := New(Config{Root: dir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	mustWrite(t, filepath.Join(dir, "src/lib.go"), "package lib\n")
	key := ts.root + "\x00**/*.go"

	ctxLeader, cancelLeader := context.WithCancel(context.Background())
	defer cancelLeader()

	var hookOnce sync.Once
	ts.searchHook = func() {
		hookOnce.Do(func() {
			deadline := time.Now().Add(2 * time.Second)
			for ts.globFlight.Waiters(key) < 1 {
				if time.Now().After(deadline) {
					t.Errorf("timed out waiting for joiner to enter Do")
					return
				}
				time.Sleep(1 * time.Millisecond)
			}
			cancelLeader()
		})
	}

	var leaderOut []byte
	var leaderErr bool
	var joinerOut []byte
	var joinerErr bool
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		leaderOut, leaderErr = ts.glob(ctxLeader, argsOf(t, GlobArgs{Pattern: "**/*.go"}))
	}()

	go func() {
		defer wg.Done()
		joinerOut, joinerErr = ts.glob(context.Background(), argsOf(t, GlobArgs{Pattern: "**/*.go"}))
	}()

	wg.Wait()

	if !leaderErr {
		t.Fatalf("leader unexpectedly succeeded: %s", string(leaderOut))
	}
	if code := errCode(t, leaderOut); code != CodeCanceled {
		t.Fatalf("leader error code = %q, want %q", code, CodeCanceled)
	}

	if joinerErr {
		t.Fatalf("joiner failed prematurely: %s", string(joinerOut))
	}
	joinerData := decodeResult(t, joinerOut)
	if joinerData["count"] != float64(1) {
		t.Fatalf("joiner count = %v, want 1", joinerData["count"])
	}
}
