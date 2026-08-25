package orchestration

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRunStateLocatorReopensAndResolvesSameStore(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 123, time.UTC)
	root := t.TempDir()
	first := testRunStateLocator(t, root, now)
	created, err := first.Create("run-1")
	if err != nil {
		t.Fatal(err)
	}
	duplicate, err := first.Create("run-1")
	if err != nil || duplicate != created {
		t.Fatalf("exact duplicate = %#v, %v; want %#v", duplicate, err, created)
	}
	store1, err := first.Resolve("run-1", time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	second := testRunStateLocator(t, root, now.Add(time.Minute))
	opened, err := second.Open("run-1")
	if err != nil || opened != created {
		t.Fatalf("reopened = %#v, %v; want %#v", opened, err, created)
	}
	store2, err := second.Resolve("run-1", time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if store1.dir != store2.dir || store1.dir != filepath.Join(second.root, admissionStoreDir, "run-1") {
		t.Fatalf("resolved dirs = %q, %q", store1.dir, store2.dir)
	}
}

func TestRunStateLocatorRejectsMissingMismatchStaleAndFuture(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	locator := testRunStateLocator(t, t.TempDir(), now)
	if _, err := locator.Open("missing"); err == nil {
		t.Fatal("Open accepted a missing manifest")
	}

	writeRunManifest(t, locator, "asked", RunStateManifest{runStateSchema, "different", canonicalAdmissionStore("different"), now.Format(time.RFC3339Nano), RunStateAuthorityVersion})
	if _, err := locator.Open("asked"); err == nil || !strings.Contains(err.Error(), "identity mismatch") {
		t.Fatalf("mismatched run error = %v", err)
	}
	writeRunManifest(t, locator, "stale", RunStateManifest{runStateSchema, "stale", canonicalAdmissionStore("stale"), now.Add(-2 * time.Hour).Format(time.RFC3339Nano), RunStateAuthorityVersion})
	if _, err := locator.Open("stale"); err == nil || !strings.Contains(err.Error(), "stale") {
		t.Fatalf("stale error = %v", err)
	}
	writeRunManifest(t, locator, "future", RunStateManifest{runStateSchema, "future", canonicalAdmissionStore("future"), now.Add(time.Nanosecond).Format(time.RFC3339Nano), RunStateAuthorityVersion})
	if _, err := locator.Open("future"); err == nil || !strings.Contains(err.Error(), "future") {
		t.Fatalf("future error = %v", err)
	}
}

func TestRunStateLocatorRejectsUnsafeLocatorsTamperedOnDisk(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	for _, value := range []string{"..", "../out", `..\out`, "/tmp/out", `C:\out`, `\\server\share`, "a/b", `a\b`, "."} {
		t.Run(strings.NewReplacer("/", "_", `\`, "_").Replace(value), func(t *testing.T) {
			locator := testRunStateLocator(t, t.TempDir(), now)
			writeRunManifest(t, locator, "run", RunStateManifest{runStateSchema, "run", value, now.Format(time.RFC3339Nano), RunStateAuthorityVersion})
			if _, err := locator.Resolve("run", time.Hour); err == nil {
				t.Fatalf("Resolve accepted unsafe locator %q", value)
			}
		})
	}
}

func TestRunStateLocatorRejectsUnsupportedAuthorityVersion(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	locator := testRunStateLocator(t, t.TempDir(), now)
	for _, authorityVersion := range []string{
		RunStateAuthorityVersion + ".operator",
		"fak.orchestration.authority/v2",
	} {
		writeRunManifest(t, locator, "run", RunStateManifest{runStateSchema, "run", canonicalAdmissionStore("run"), now.Format(time.RFC3339Nano), authorityVersion})
		if _, err := locator.Open("run"); err == nil || !strings.Contains(err.Error(), "unsupported authority version") {
			t.Fatalf("Open authority version %q error = %v", authorityVersion, err)
		}
	}
}

func TestRunStateLocatorRejectsCrossRunAdmissionStoreAlias(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	locator := testRunStateLocator(t, t.TempDir(), now)
	writeRunManifest(t, locator, "run-2", RunStateManifest{runStateSchema, "run-2", canonicalAdmissionStore("run-1"), now.Format(time.RFC3339Nano), RunStateAuthorityVersion})
	if _, err := locator.Resolve("run-2", time.Hour); err == nil || !strings.Contains(err.Error(), "does not match run ID") {
		t.Fatalf("Resolve cross-run alias error = %v", err)
	}
}

func TestRunStateLocatorRejectsMalformedUnknownAndPrivateFields(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	base := `{"schema":"fak.orchestration.run-state/v1","run_id":"run","admission_store":"admissions/run","created_at":"2026-08-25T12:00:00Z","authority_version":"fak.orchestration.authority/v1"}`
	cases := map[string]string{
		"malformed":         `{`,
		"unknown":           strings.TrimSuffix(base, "}") + `,"extra":true}`,
		"private unknown":   strings.TrimSuffix(base, "}") + `,"_raw":"secret"}`,
		"trailing":          base + `{}`,
		"noncanonical time": strings.Replace(base, "2026-08-25T12:00:00Z", "2026-08-25T12:00:00.000Z", 1),
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			locator := testRunStateLocator(t, t.TempDir(), now)
			if err := os.WriteFile(locator.manifestPath("run"), []byte(data), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := locator.Open("run"); err == nil {
				t.Fatal("Open accepted malformed manifest")
			}
		})
	}
}

func TestRunStateLocatorRejectsSymlinkAdmissionStore(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	root := t.TempDir()
	locator := testRunStateLocator(t, root, now)
	if _, err := locator.Create("run"); err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(root, admissionStoreDir)); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if _, err := locator.Resolve("run", time.Hour); err == nil {
		t.Fatal("Resolve followed a symlink admission store")
	}
}

func TestRunStateLocatorOwnerOnlyModes(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix permission bits are not enforced on Windows")
	}
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	locator := testRunStateLocator(t, t.TempDir(), now)
	if _, err := locator.Create("run"); err != nil {
		t.Fatal(err)
	}
	for path, want := range map[string]os.FileMode{
		locator.root:                             0o700,
		filepath.Join(locator.root, runStateDir): 0o700,
		locator.manifestPath("run"):              0o600,
	} {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if got := info.Mode().Perm(); got != want {
			t.Errorf("%s mode = %o, want %o", path, got, want)
		}
	}
}

func TestRunStateLocatorConcurrentCreateIsIdempotent(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	locator := testRunStateLocator(t, t.TempDir(), now)
	const creators = 20
	start := make(chan struct{})
	results := make(chan error, creators)
	var wg sync.WaitGroup
	for i := 0; i < creators; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, err := locator.Create("run")
			results <- err
		}()
	}
	close(start)
	wg.Wait()
	close(results)
	successes := 0
	for err := range results {
		if err == nil {
			successes++
		}
	}
	if successes != creators {
		t.Fatalf("successful identical creators = %d, want %d", successes, creators)
	}
	_, err := locator.Open("run")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := locator.Create("run"); err != nil {
		t.Fatalf("winning manifest was not immutable/idempotent: %v", err)
	}
}

func testRunStateLocator(t *testing.T, root string, now time.Time) *RunStateLocator {
	t.Helper()
	locator, err := newRunStateLocator(root, time.Hour, func() time.Time { return now })
	if err != nil {
		t.Fatal(err)
	}
	return locator
}

func writeRunManifest(t *testing.T, locator *RunStateLocator, name string, manifest RunStateManifest) {
	t.Helper()
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(locator.manifestPath(name), data, 0o600); err != nil {
		t.Fatal(err)
	}
}
