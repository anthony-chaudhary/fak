package main

import (
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchcache"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestDispatchRoutedBacklogCache(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[lanes]\ncmd = [\"cmd/fak/**\"]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	oldView := dispatchTickView
	dispatchTickView = ""
	t.Cleanup(func() { dispatchTickView = oldView })

	now := time.Unix(100, 0)
	oldCache := dispatchRoutedBacklogCache
	dispatchRoutedBacklogCache = dispatchcache.New[dispatchtick.RouterPayload](func() time.Time { return now })
	t.Cleanup(func() { dispatchRoutedBacklogCache = oldCache })

	calls := 0
	stubDispatchIssueFetches(t, nil, func(string, int) ([]dispatchtick.Issue, error) {
		calls++
		return []dispatchtick.Issue{dispatchViewTestIssue(4168)}, nil
	})
	if _, err := dispatchRoutedBeforePrereqHold(root, io.Discard); err != nil {
		t.Fatal(err)
	}
	if _, err := dispatchRoutedBeforePrereqHold(root, io.Discard); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("fetch calls inside TTL = %d, want 1", calls)
	}
	now = now.Add(dispatchRoutedBacklogTTL)
	if _, err := dispatchRoutedBeforePrereqHold(root, io.Discard); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("fetch calls after TTL = %d, want 2", calls)
	}
}
