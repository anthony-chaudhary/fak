package tokencache

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

func TestMaintenanceBoundsEntriesBytesAndStaleTemps(t *testing.T) {
	dir := t.TempDir()
	c := New(dir, "v1")
	now := time.Unix(1_700_000_000, 0)

	var sources []string
	for i := 0; i < 8; i++ {
		src := fmt.Sprintf("package p\nfunc f%d() string { return %q }\n", i, strings.Repeat(string(rune('a'+i)), 240))
		c.Put(src, []string{strings.Repeat(fmt.Sprintf("k%d", i), 80)}, [][2]int{{i + 1, i + 2}})
		path := c.path(c.digest(src))
		stamp := now.Add(time.Duration(i-20) * time.Minute)
		if err := os.Chtimes(path, stamp, stamp); err != nil {
			t.Fatal(err)
		}
		sources = append(sources, src)
	}

	malformed := filepath.Join(dir, strings.Repeat("0", 64)+".json")
	if err := os.WriteFile(malformed, []byte("{not-json"), 0o600); err != nil {
		t.Fatal(err)
	}
	oldest := now.Add(-48 * time.Hour)
	if err := os.Chtimes(malformed, oldest, oldest); err != nil {
		t.Fatal(err)
	}

	staleTemp := filepath.Join(dir, ".entry-stale.tmp")
	activeTemp := filepath.Join(dir, ".entry-active.tmp")
	for path, stamp := range map[string]time.Time{
		staleTemp:  now.Add(-25 * time.Hour),
		activeTemp: now.Add(-23 * time.Hour),
	} {
		if err := os.WriteFile(path, []byte("partial"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, stamp, stamp); err != nil {
			t.Fatal(err)
		}
	}

	newest := sources[len(sources)-4:]
	var byteBudget int64
	for _, src := range newest {
		st, err := os.Stat(c.path(c.digest(src)))
		if err != nil {
			t.Fatal(err)
		}
		byteBudget += st.Size()
	}
	beforeBytes, beforeEntries := jsonStats(t, dir)
	if beforeEntries != 9 || beforeBytes <= byteBudget {
		t.Fatalf("fixture did not cross both limits: entries=%d bytes=%d budget_entries=4 budget_bytes=%d", beforeEntries, beforeBytes, byteBudget)
	}

	receipt := c.maintain(
		MaintenanceOptions{MaxBytes: byteBudget, MaxEntries: 4, TempGrace: 24 * time.Hour},
		now,
		os.Remove,
	)
	if receipt.BeforeEntries != 9 || receipt.BeforeBytes != beforeBytes {
		t.Fatalf("before receipt = entries %d bytes %d, want 9/%d", receipt.BeforeEntries, receipt.BeforeBytes, beforeBytes)
	}
	if receipt.AfterEntries != 4 || receipt.AfterBytes != byteBudget {
		t.Fatalf("after receipt = entries %d bytes %d, want 4/%d", receipt.AfterEntries, receipt.AfterBytes, byteBudget)
	}
	if receipt.RemovedEntries != 5 || receipt.RemovedBytes != beforeBytes-byteBudget {
		t.Fatalf("removals = entries %d bytes %d, want 5/%d", receipt.RemovedEntries, receipt.RemovedBytes, beforeBytes-byteBudget)
	}
	if receipt.StaleTempsRemoved != 1 || receipt.SkippedLockedFiles != 0 || receipt.Verdict != VerdictPruned {
		t.Fatalf("receipt = %+v", receipt)
	}
	if _, err := os.Stat(staleTemp); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("stale temp remains: %v", err)
	}
	if _, err := os.Stat(activeTemp); err != nil {
		t.Fatalf("active temp was removed: %v", err)
	}
	for _, src := range newest {
		if _, _, ok := c.Get(src); !ok {
			t.Fatalf("recent valid entry was not preserved: %q", src)
		}
	}
	assertAllJSONValid(t, dir)
}

func TestMaintenanceReportsSkippedLockedFile(t *testing.T) {
	dir := t.TempDir()
	c := New(dir, "v1")
	c.Put("old", []string{"old"}, [][2]int{{1, 1}})
	c.Put("new", []string{"new"}, [][2]int{{1, 1}})
	oldPath := c.path(c.digest("old"))
	now := time.Unix(1_700_000_000, 0)
	if err := os.Chtimes(oldPath, now.Add(-time.Hour), now.Add(-time.Hour)); err != nil {
		t.Fatal(err)
	}
	newPath := c.path(c.digest("new"))
	if err := os.Chtimes(newPath, now, now); err != nil {
		t.Fatal(err)
	}

	receipt := c.maintain(MaintenanceOptions{MaxBytes: 1, MaxEntries: 1, TempGrace: time.Hour}, now, func(path string) error {
		if path == oldPath {
			return fs.ErrPermission
		}
		return os.Remove(path)
	})
	if receipt.SkippedLockedFiles != 1 || receipt.Verdict != VerdictPartial {
		t.Fatalf("locked-file receipt = %+v", receipt)
	}
	if _, err := os.Stat(oldPath); err != nil {
		t.Fatalf("locked entry should survive: %v", err)
	}
}

func TestMaintenanceLockBusyIsObservableAndNonblocking(t *testing.T) {
	dir := t.TempDir()
	lock, err := os.OpenFile(filepath.Join(dir, ".maintenance.lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	if err := flock.TryLock(lock); err != nil {
		t.Fatal(err)
	}
	defer flock.Unlock(lock)

	start := time.Now()
	for i := 0; i < 10; i++ {
		receipt := New(dir, "v1").maintain(MaintenanceOptions{MaxBytes: 1, MaxEntries: 1, TempGrace: time.Hour}, start, os.Remove)
		if receipt.Verdict != VerdictLockBusy {
			t.Fatalf("contended receipt %d = %+v", i, receipt)
		}
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("nonblocking maintenance waited %v", elapsed)
	}
	pending, err := filepath.Glob(filepath.Join(dir, ".maintenance.lock.pending*"))
	if err != nil || len(pending) != 1 {
		t.Fatalf("contended pending markers = %v, err=%v; want one bounded signal", pending, err)
	}
}

func TestConcurrentPutGetAndMaintenanceConverges(t *testing.T) {
	dir := t.TempDir()
	const writers = 32
	const maxEntries = 6
	now := time.Now()

	hot := "package hot\nfunc hot() string { return \"hot\" }\n"
	seed := New(dir, "v1")
	seed.Put(hot, []string{"hot"}, [][2]int{{1, 2}})
	hotPath := seed.path(seed.digest(hot))
	future := now.Add(time.Hour)
	if err := os.Chtimes(hotPath, future, future); err != nil {
		t.Fatal(err)
	}

	var maxEntryBytes int64
	for i := 0; i < writers; i++ {
		src := concurrentSource(i)
		b, err := json.Marshal(entry{Schema: entrySchema, Version: "v1", Digest: seed.digest(src), Keys: []string{strings.Repeat("k", 160)}, Spans: [][2]int{{1, 2}}})
		if err != nil {
			t.Fatal(err)
		}
		if int64(len(b)) > maxEntryBytes {
			maxEntryBytes = int64(len(b))
		}
	}
	byteBudget := maxEntryBytes * maxEntries
	t.Setenv(MaxBytesEnv, strconv.FormatInt(byteBudget, 10))
	t.Setenv(MaxEntriesEnv, strconv.Itoa(maxEntries))
	t.Setenv(TempGraceEnv, "24h")

	start := make(chan struct{})
	stopReaders := make(chan struct{})
	errCh := make(chan error, writers+1)
	var readers sync.WaitGroup
	readers.Add(1)
	go func() {
		defer readers.Done()
		for {
			select {
			case <-stopReaders:
				return
			default:
				if _, _, ok := seed.Get(hot); !ok {
					errCh <- errors.New("concurrent hot Get missed")
					return
				}
			}
		}
	}()

	var wg sync.WaitGroup
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			c := New(dir, "v1")
			c.Put(concurrentSource(i), []string{strings.Repeat("k", 160)}, [][2]int{{1, 2}})
			c.Maintain()
		}(i)
	}
	close(start)
	wg.Wait()
	close(stopReaders)
	readers.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}

	afterBytes, afterEntries := jsonStats(t, dir)
	if afterEntries > maxEntries+1 {
		t.Fatalf("concurrent maintenance left %d entries, want <= %d", afterEntries, maxEntries+1)
	}
	if afterBytes > byteBudget+maxEntryBytes {
		t.Fatalf("concurrent maintenance left %d bytes, want <= %d", afterBytes, byteBudget+maxEntryBytes)
	}
	if _, _, ok := seed.Get(hot); !ok {
		t.Fatal("recent hot entry was evicted")
	}
	if _, err := os.Stat(filepath.Join(dir, ".maintenance.lock.pending")); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("settled writers left a pending maintenance signal: %v", err)
	}
	assertAllJSONValid(t, dir)
}

func TestMaintenanceDefaultsAndOverrides(t *testing.T) {
	t.Setenv(MaxBytesEnv, "1024")
	t.Setenv(MaxEntriesEnv, "17")
	t.Setenv(TempGraceEnv, "90m")
	got := MaintenanceDefaults()
	if got.MaxBytes != 1024 || got.MaxEntries != 17 || got.TempGrace != 90*time.Minute {
		t.Fatalf("overrides = %+v", got)
	}

	t.Setenv(MaxBytesEnv, "invalid")
	t.Setenv(MaxEntriesEnv, "0")
	t.Setenv(TempGraceEnv, "-1h")
	got = MaintenanceDefaults()
	if got.MaxBytes != defaultMaxBytes || got.MaxEntries != defaultMaxEntries || got.TempGrace != defaultTempGrace {
		t.Fatalf("invalid overrides should fall back to defaults, got %+v", got)
	}
}

func TestOpenRunsStartupMaintenance(t *testing.T) {
	root := initGitRepo(t)
	dir := TokenCacheDir(filepath.Join(root, ".git"))
	c := New(dir, "v1")
	c.Put("abandoned", []string{strings.Repeat("k", 80)}, [][2]int{{1, 1}})
	staleTemp := filepath.Join(dir, ".entry-abandoned.tmp")
	if err := os.WriteFile(staleTemp, []byte("partial"), 0o600); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(staleTemp, old, old); err != nil {
		t.Fatal(err)
	}
	t.Setenv(MaxBytesEnv, "1")
	t.Setenv(MaxEntriesEnv, "1")
	t.Setenv(TempGraceEnv, "1h")

	if got := Open(root); got == nil {
		t.Fatal("Open returned nil")
	}
	if bytes, entries := jsonStats(t, dir); bytes != 0 || entries != 0 {
		t.Fatalf("startup maintenance left %d bytes / %d entries", bytes, entries)
	}
	if _, err := os.Stat(staleTemp); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("startup maintenance left stale temp: %v", err)
	}
}

func TestMaintainRefusesSymlinkEscape(t *testing.T) {
	root := initGitRepo(t)
	outside := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, ".git", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	cacheDir := TokenCacheDir(filepath.Join(root, ".git"))
	if err := os.Symlink(outside, cacheDir); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	sentinel := filepath.Join(outside, strings.Repeat("f", 64)+".json")
	if err := os.WriteFile(sentinel, []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}

	receipt := Maintain(root, MaintenanceOptions{MaxBytes: 1, MaxEntries: 1, TempGrace: time.Hour})
	if receipt.Verdict != VerdictUnsafePath {
		t.Fatalf("verdict = %q, want %q (%+v)", receipt.Verdict, VerdictUnsafePath, receipt)
	}
	if _, err := os.Stat(sentinel); err != nil {
		t.Fatalf("escape sentinel was touched: %v", err)
	}
}

func concurrentSource(i int) string {
	return fmt.Sprintf("package p\nfunc f%d() string { return %q }\n", i, strings.Repeat(fmt.Sprintf("v%d", i), 120))
}

func jsonStats(t *testing.T, dir string) (int64, int) {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	var count int
	for _, de := range ents {
		if de.IsDir() || !strings.HasSuffix(de.Name(), ".json") {
			continue
		}
		info, err := de.Info()
		if err != nil {
			t.Fatal(err)
		}
		bytes += info.Size()
		count++
	}
	return bytes, count
}

func assertAllJSONValid(t *testing.T, dir string) {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, de := range ents {
		if de.IsDir() || !strings.HasSuffix(de.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, de.Name()))
		if err != nil {
			t.Fatal(err)
		}
		var got entry
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("surviving entry %s is malformed: %v", de.Name(), err)
		}
	}
}
