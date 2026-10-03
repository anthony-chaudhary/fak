package tokencache_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/clonescan"
	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/tokencache"
)

const interruptedBudgetBytes int64 = 4096
const interruptedBudgetEntries = 4
const interruptedPuts = 64

// TestInterruptedRetentionContract asserts the documented per-instance retained-state
// envelope without caller-owned final maintenance. Allowances are one MaxBytes
// budget and max(1, min(64, MaxEntries/16)) entries per instance. This does not
// assert a global fleet bound or an in-flight temporary/physical-byte peak.
// Total-directory assertions apply only to these fresh/initially within-budget
// fixtures; existing excess and additional independent writers are not covered.
// fak-test:runtime medium est=1s lane=default
func TestInterruptedRetentionContract(t *testing.T) {
	t.Setenv(tokencache.FlagEnv, "on")
	t.Setenv(tokencache.MaxBytesEnv, strconv.FormatInt(interruptedBudgetBytes, 10))
	t.Setenv(tokencache.MaxEntriesEnv, strconv.Itoa(interruptedBudgetEntries))
	t.Setenv(tokencache.TempGraceEnv, "24h")

	t.Run("many_puts_without_final_maintain", func(t *testing.T) {
		dir := t.TempDir()
		c := tokencache.New(dir, "retention-witness-v1")
		checkInterruptedBurst(t, c, dir, interruptedBudgetEntries)
	})

	t.Run("byte_binding_burst_without_final_maintain", func(t *testing.T) {
		// Isolate the byte ceiling: every entry fits individually, and all 64
		// together fit below this entry-count ceiling. Count-only pruning fails.
		const entryCeiling = 1024
		t.Setenv(tokencache.MaxEntriesEnv, strconv.Itoa(entryCeiling))
		dir := t.TempDir()
		c := tokencache.New(dir, "retention-witness-v1")
		checkInterruptedBurst(t, c, dir, entryCeiling)
	})

	t.Run("individually_oversized_entry", func(t *testing.T) {
		dir := t.TempDir()
		c := tokencache.New(dir, "retention-witness-v1")
		src := "oversized-source"
		// Payload alone exceeds the budget, independent of JSON framing/version.
		keys, spans := []string{strings.Repeat("k", int(interruptedBudgetBytes)+1)}, [][2]int{{1, 2}}
		c.Put(src, keys, spans)
		gotKeys, gotSpans, admitted := c.Get(src)
		bytes, entries := interruptedStats(t, dir)
		t.Logf("budget_bytes=%d retained_bytes=%d entries=%d admitted=%t", interruptedBudgetBytes, bytes, entries, admitted)
		if admitted && (!reflect.DeepEqual(keys, gotKeys) || !reflect.DeepEqual(spans, gotSpans)) {
			t.Fatal("admitted oversized entry was corrupted")
		}
		if admitted || bytes != 0 || entries != 0 {
			t.Fatal("oversized entry must be bypassed instead of retained for final maintenance")
		}
	})

	t.Run("held_lock_caps_accumulated_near_budget_entry", func(t *testing.T) {
		// Make byte admission independently binding: two entries are below the
		// 64-entry allowance, while their combined bytes exceed one byte budget.
		t.Setenv(tokencache.MaxEntriesEnv, "1024")
		seedKeys, nearKeys := []string{"valid"}, []string{strings.Repeat("k", int(interruptedBudgetBytes)-192)}
		spans := [][2]int{{1, 2}}
		serializedSize := func(src string, keys []string) int64 {
			dir := t.TempDir()
			probe := tokencache.New(dir, "retention-witness-v1")
			probe.Put(src, keys, spans)
			bytes, entries := interruptedStats(t, dir)
			if _, _, ok := probe.Get(src); !ok || entries != 1 {
				t.Fatal("calibration requires one individually admissible entry")
			}
			return bytes
		}
		seedBytes := serializedSize("small-seed", seedKeys)
		nearBytes := serializedSize("near-budget", nearKeys)
		if seedBytes >= interruptedBudgetBytes/16 || nearBytes > interruptedBudgetBytes || seedBytes+nearBytes <= interruptedBudgetBytes {
			t.Fatalf("fixture must leave sub-threshold debt before a near-budget entry: seed=%d near=%d budget=%d", seedBytes, nearBytes, interruptedBudgetBytes)
		}
		dir := t.TempDir()
		c := tokencache.New(dir, "retention-witness-v1")
		lock, err := os.OpenFile(filepath.Join(dir, ".maintenance.lock"), os.O_CREATE|os.O_RDWR, 0o600)
		if err != nil {
			t.Fatal(err)
		}
		defer lock.Close()
		if err := flock.TryLock(lock); err != nil {
			t.Fatal(err)
		}
		defer flock.Unlock(lock)
		c.Put("small-seed", seedKeys, spans)
		c.Put("near-budget", nearKeys, spans)
		bytes, entries := interruptedStats(t, dir)
		t.Logf("held_lock=true seed_serialized_bytes=%d near_serialized_bytes=%d retained_bytes=%d entries=%d", seedBytes, nearBytes, bytes, entries)
		// No successful maintenance can reset this instance's admission debt:
		// all retained bytes came from it after construction, with the lock held.
		if bytes > interruptedBudgetBytes {
			t.Fatalf("one instance accumulated %d bytes without successful maintenance, allowance=%d", bytes, interruptedBudgetBytes)
		}
	})

	for _, failure := range []string{"maintenance_lock_busy", "maintenance_lock_open_error"} {
		t.Run(failure, func(t *testing.T) {
			dir := t.TempDir()
			c := tokencache.New(dir, "retention-witness-v1")
			c.Put("preserved", []string{"valid"}, [][2]int{{1, 2}})
			assertInterruptedSentinel(t, c)
			// New's lock placement is the existing implementation's filesystem
			// contract. No private method, injected fake remover, or new API is used.
			lockPath := filepath.Join(dir, ".maintenance.lock")
			if failure == "maintenance_lock_busy" {
				lock, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
				if err != nil {
					t.Fatal(err)
				}
				defer lock.Close()
				if err := flock.TryLock(lock); err != nil {
					t.Fatal(err)
				}
				defer flock.Unlock(lock)
			} else {
				if err := os.Remove(lockPath); err != nil && !errors.Is(err, fs.ErrNotExist) {
					t.Fatal(err)
				}
				if err := os.Mkdir(lockPath, 0o700); err != nil {
					t.Fatal(err)
				}
			}
			checkInterruptedBurst(t, c, dir, interruptedBudgetEntries)
			assertInterruptedSentinel(t, c)
			assertInterruptedTokenization(t, tokencache.New(dir, clonescan.TokenizerVersion()))
			assertInterruptedSentinel(t, c)
		})
	}

	t.Run("write_failure_preserves_valid_cache_and_source", func(t *testing.T) {
		root := t.TempDir()
		dir := filepath.Join(root, "cache")
		c := tokencache.New(dir, "retention-witness-v1")
		c.Put("preserved", []string{"valid"}, [][2]int{{1, 2}})
		assertInterruptedSentinel(t, c)
		// A regular file at the cache path deterministically fails MkdirAll even
		// as root. Preserve the old directory; do not depend on chmod behavior.
		preservedDir := filepath.Join(root, "preserved-cache")
		if err := os.Rename(dir, preservedDir); err != nil {
			t.Fatal(err)
		}
		blocker := []byte("not a directory; must remain unchanged\n")
		if err := os.WriteFile(dir, blocker, 0o600); err != nil {
			t.Fatal(err)
		}
		c.Put("failed-write", []string{"must-not-land"}, [][2]int{{1, 2}})
		if _, _, ok := c.Get("failed-write"); ok {
			t.Fatal("failed write unexpectedly became a hit")
		}
		assertInterruptedTokenization(t, tokencache.New(dir, clonescan.TokenizerVersion()))
		got, err := os.ReadFile(dir)
		if err != nil || !reflect.DeepEqual(got, blocker) {
			t.Fatalf("write failure changed blocker: %q, %v", got, err)
		}
		assertInterruptedSentinel(t, tokencache.New(preservedDir, "retention-witness-v1"))
	})
}

func checkInterruptedBurst(t *testing.T, c *tokencache.Cache, dir string, entryCeiling int) {
	t.Helper()
	var peakBytes int64
	var peakEntries int
	for i := 0; i < interruptedPuts; i++ {
		c.Put(fmt.Sprintf("unique-source-%03d", i), []string{strings.Repeat("k", 256)}, [][2]int{{1, 2}})
		bytes, entries := interruptedStats(t, dir)
		peakBytes = max(peakBytes, bytes)
		peakEntries = max(peakEntries, entries)
	}
	t.Logf("writers=1 puts=%d budget_bytes=%d budget_entries=%d peak_retained_bytes=%d peak_retained_entries=%d excess_bytes=%d excess_entries=%d final_maintain=false", interruptedPuts, interruptedBudgetBytes, entryCeiling, peakBytes, peakEntries, max(0, peakBytes-interruptedBudgetBytes), max(0, peakEntries-entryCeiling))
	entryAllowance := max(1, min(64, entryCeiling/16))
	if peakBytes > 2*interruptedBudgetBytes || peakEntries > entryCeiling+entryAllowance {
		t.Fatalf("retention exceeds documented ONE-instance allowance: bytes <= %d, entries <= %d", 2*interruptedBudgetBytes, entryCeiling+entryAllowance)
	}
}

func interruptedStats(t *testing.T, dir string) (int64, int) {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	var count int
	for _, de := range ents {
		if strings.HasPrefix(de.Name(), ".entry-") && strings.HasSuffix(de.Name(), ".tmp") {
			t.Fatalf("completed sequential Put leaked temporary %q", de.Name())
		}
		if de.IsDir() || !strings.HasSuffix(de.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, de.Name()))
		if err != nil || !json.Valid(b) {
			t.Fatalf("retained cache entry is unreadable/invalid JSON: %s: %v", de.Name(), err)
		}
		bytes += int64(len(b))
		count++
	}
	return bytes, count
}

func assertInterruptedSentinel(t *testing.T, c *tokencache.Cache) {
	t.Helper()
	keys, spans, ok := c.Get("preserved")
	if !ok || !reflect.DeepEqual(keys, []string{"valid"}) || !reflect.DeepEqual(spans, [][2]int{{1, 2}}) {
		t.Fatal("pre-existing valid cache entry was lost or changed")
	}
}

// Expose only the existing WindowCache methods, intentionally hiding final Maintain.
type interruptedCacheOnly struct{ clonescan.WindowCache }

func assertInterruptedTokenization(t *testing.T, cache clonescan.WindowCache) {
	t.Helper()
	source := "package p\nfunc score(x int) int {\n" + strings.Repeat("if x > 0 { x += 2; x *= 3 }\n", 12) + "return x\n}\n"
	tree := map[string]string{"a.go": source, "b.go": source + "// different bytes, same tokens\n"}
	root := t.TempDir()
	for name, src := range tree {
		if err := os.WriteFile(filepath.Join(root, name), []byte(src), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	wantKeys := clonescan.CandidateKeys(source)
	wantIndex := clonescan.BuildTreeIndex(tree)
	want := wantIndex.Query(wantKeys, "a.go", 0)
	if len(wantKeys) == 0 || len(want) != 1 {
		t.Fatal("parity fixture must exercise actual qualifying token windows")
	}
	gotIndex := clonescan.BuildTreeIndex(tree, interruptedCacheOnly{cache})
	if !reflect.DeepEqual(gotIndex, wantIndex) {
		t.Fatal("optional cache failure changed the complete token-window index")
	}
	got := gotIndex.Query(wantKeys, "a.go", 0)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("optional cache failure changed exact tokenization: got=%+v want=%+v", got, want)
	}
	for name, src := range tree {
		b, err := os.ReadFile(filepath.Join(root, name))
		if err != nil || string(b) != src {
			t.Fatalf("source %s changed: %v", name, err)
		}
	}
}

// TestInterruptedRetentionSingleInstanceConcurrentCalls adds the missing same-instance
// concurrency coverage. Observations are settled round boundaries, never an
// instantaneous/global peak. Cache misses from concurrent eviction are allowed;
// every returned hit must preserve exact keys and spans.
// fak-test:runtime medium est=500ms lane=default
func TestInterruptedRetentionSingleInstanceConcurrentCalls(t *testing.T) {
	t.Setenv(tokencache.FlagEnv, "on")
	t.Setenv(tokencache.MaxBytesEnv, strconv.FormatInt(interruptedBudgetBytes, 10))
	t.Setenv(tokencache.MaxEntriesEnv, strconv.Itoa(interruptedBudgetEntries))
	t.Setenv(tokencache.TempGraceEnv, "24h")
	dir := t.TempDir()
	c := tokencache.New(dir, "retention-witness-v1")
	const rounds, callers = 8, 8
	keys, spans := []string{strings.Repeat("k", 256)}, [][2]int{{1, 2}}
	var peakBytes int64
	var peakEntries int
	for round := 0; round < rounds; round++ {
		start := make(chan struct{})
		errors := make(chan string, callers)
		var wg sync.WaitGroup
		for caller := 0; caller < callers; caller++ {
			wg.Add(1)
			go func(source string) {
				defer wg.Done()
				<-start
				c.Put(source, keys, spans)
				gotKeys, gotSpans, ok := c.Get(source)
				// Concurrent eviction may cause a miss; a hit must stay exact.
				if ok && (!reflect.DeepEqual(gotKeys, keys) || !reflect.DeepEqual(gotSpans, spans)) {
					errors <- source
				}
			}(fmt.Sprintf("same-instance-round-%d-caller-%d", round, caller))
		}
		close(start)
		wg.Wait()
		close(errors)
		for source := range errors {
			t.Errorf("concurrent hit corrupted keys/spans for %q", source)
		}
		bytes, entries := interruptedStats(t, dir)
		peakBytes, peakEntries = max(peakBytes, bytes), max(peakEntries, entries)
		if bytes > 2*interruptedBudgetBytes || entries > interruptedBudgetEntries+1 {
			t.Errorf("round %d: one-instance retained bytes=%d entries=%d exceeds documented bound bytes<=%d entries<=%d", round, bytes, entries, 2*interruptedBudgetBytes, interruptedBudgetEntries+1)
		}
	}
	t.Logf("instances=1 concurrent_callers=%d rounds=%d total_puts=%d settled_peak_bytes=%d settled_peak_entries=%d final_maintain=false", callers, rounds, callers*rounds, peakBytes, peakEntries)
}
