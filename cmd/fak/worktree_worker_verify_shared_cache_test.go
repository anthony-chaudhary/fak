package main

import (
	"os"
	"path/filepath"
	"testing"
)

func noSharedGoCache([]string) (string, bool) { return "", false }

func notInCheckout(string) bool { return false }

// fak-test:runtime fast est=50ms lane=default
func TestGoBuildVerifyEnvReplacesPerLaneWorktreeCacheWithSharedCache(t *testing.T) {
	wt := t.TempDir()
	if err := os.WriteFile(filepath.Join(wt, ".git"), []byte("gitdir: elsewhere\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	laneCache := filepath.Join(wt, ".gocache")
	outside := t.TempDir()
	shared := t.TempDir()
	isolated := map[string]string{"GOCACHE": filepath.Join(wt, ".cand-gocache"), "GOTMPDIR": filepath.Join(wt, ".gotmp")}
	userCacheOK := func() (string, error) { return t.TempDir(), nil }
	sharedOK := func([]string) (string, bool) { return shared, true }

	tests := []struct {
		name      string
		base      []string
		resolve   func([]string) (string, bool)
		wantCache string
	}{
		{name: "worktree cache uses shared", base: []string{"GOCACHE=" + laneCache}, resolve: sharedOK, wantCache: shared},
		{name: "outside absolute cache kept", base: []string{"GOCACHE=" + outside}, resolve: sharedOK, wantCache: outside},
		{name: "unresolved shared keeps worktree cache", base: []string{"GOCACHE=" + laneCache}, resolve: noSharedGoCache, wantCache: laneCache},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			env := worktreeWorkerGoBuildVerifyEnv(tc.base, isolated, userCacheOK, tc.resolve, worktreeWorkerPathInCheckout)
			if got, _ := goBuildVerifyEnvValue(env, "GOCACHE"); got != tc.wantCache {
				t.Fatalf("GOCACHE = %q, want %q", got, tc.wantCache)
			}
		})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestSharedGoCachePrefersAbsoluteFakSharedGoCache(t *testing.T) {
	shared := t.TempDir()
	got, ok := worktreeWorkerSharedGoCache([]string{"GOCACHE=" + filepath.Join(t.TempDir(), ".gocache"), "FAK_SHARED_GOCACHE=" + shared})
	if !ok || got != shared {
		t.Fatalf("shared cache = %q, %v; want %q, true", got, ok, shared)
	}
}
