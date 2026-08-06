package launchshim

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
)

func TestRoundTripAndDirect(t *testing.T) {
	p := filepath.Join(t.TempDir(), "launch.json")
	t.Setenv("FAK_LAUNCH_CONFIG", p)
	in := Config{Default: "claude", Providers: map[string]Provider{"claude": {Command: "/real/claude"}}}
	if err := Save(in); err != nil {
		t.Fatal(err)
	}
	got, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if got.Default != "claude" || got.Providers["claude"].Command != "/real/claude" {
		t.Fatalf("got %+v", got)
	}
	t.Setenv("FAK_DIRECT", "1")
	if !EffectiveDirect(got, false) {
		t.Fatal("FAK_DIRECT must bypass fak")
	}
	_ = os.Remove(p)
}

func TestCanonicalCommandResolvesSymlinkAndRejectsDirectory(t *testing.T) {
	dir := t.TempDir()
	real := filepath.Join(dir, "provider")
	if err := os.WriteFile(real, []byte("provider"), 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := CanonicalCommand(real)
	if err != nil {
		t.Fatal(err)
	}
	if !SameCommand(got, real) {
		t.Fatalf("canonical=%q real=%q", got, real)
	}
	if runtime.GOOS != "windows" {
		link := filepath.Join(dir, "alias")
		if err := os.Symlink(real, link); err != nil {
			t.Fatal(err)
		}
		if !SameCommand(link, real) {
			t.Fatalf("symlink %q does not resolve to %q", link, real)
		}
	}
	if _, err := CanonicalCommand(dir); err == nil {
		t.Fatal("directory accepted as provider command")
	}
}

func TestSaveFailurePreservesLastKnownGood(t *testing.T) {
	path := filepath.Join(t.TempDir(), "launch.json")
	t.Setenv("FAK_LAUNCH_CONFIG", path)
	old := Config{Default: "claude", Providers: map[string]Provider{"claude": {Command: "old"}}}
	if err := Save(old); err != nil {
		t.Fatal(err)
	}
	prior := saveReplace
	saveReplace = func(_, _ string) error { return errors.New("injected rename failure") }
	t.Cleanup(func() { saveReplace = prior })
	if err := Save(Config{Default: "codex"}); err == nil {
		t.Fatal("Save unexpectedly succeeded")
	}
	got, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if got.Default != "claude" || got.Providers["claude"].Command != "old" {
		t.Fatalf("last-good lost: %+v", got)
	}
}
func TestConcurrentSaveNeverTearsConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "launch.json")
	t.Setenv("FAK_LAUNCH_CONFIG", path)
	const writers = 32
	var wg sync.WaitGroup
	errs := make(chan error, writers)
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			name := fmt.Sprintf("provider-%02d", i)
			errs <- Save(Config{Default: name, Providers: map[string]Provider{name: {Command: name}}})
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent save: %v", err)
		}
	}
	got, err := Load()
	if err != nil {
		t.Fatalf("torn final config: %v", err)
	}
	if got.Default == "" || got.Providers[got.Default].Command != got.Default {
		t.Fatalf("incoherent final config: %+v", got)
	}
	matches, err := filepath.Glob(path + ".tmp-*")
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 0 {
		t.Fatalf("temporary files leaked: %v", matches)
	}
}
