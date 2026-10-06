package main

import (
	"errors"
	"path/filepath"
	"slices"
	"testing"
)

func TestGoBuildVerifyArgsAreCacheReusableAcrossCandidateRoots(t *testing.T) {
	args := worktreeWorkerGoBuildVerifyArgs()
	for _, want := range []string{"build", "-trimpath", "-buildvcs=false"} {
		if !slices.Contains(args, want) {
			t.Fatalf("verify argv %q lacks %q", args, want)
		}
	}
	if args[len(args)-1] != "./..." {
		t.Fatalf("verify argv %q must still check every package", args)
	}
}

func TestGoBuildVerifyEnvSharesUsableCallerCache(t *testing.T) {
	wt := t.TempDir()
	shared := t.TempDir()
	isolated := map[string]string{"GOCACHE": filepath.Join(wt, ".gocache"), "GOTMPDIR": filepath.Join(wt, ".gotmp")}
	userCacheOK := func() (string, error) { return t.TempDir(), nil }
	userCacheMissing := func() (string, error) { return "", errors.New("no home") }

	tests := []struct {
		name      string
		base      []string
		userCache func() (string, error)
		wantCache string
	}{
		{name: "explicit shared cache", base: []string{"PATH=x", "GOCACHE=" + shared}, userCache: userCacheMissing, wantCache: shared},
		{name: "default user cache", base: []string{"PATH=x"}, userCache: userCacheOK, wantCache: ""},
		{name: "empty GOCACHE means default", base: []string{"GOCACHE="}, userCache: userCacheOK, wantCache: ""},
		{name: "last duplicate wins", base: []string{"GOCACHE=off", "GOCACHE=" + shared}, userCache: userCacheMissing, wantCache: shared},
		{name: "no usable default falls back", base: []string{"PATH=x"}, userCache: userCacheMissing, wantCache: isolated["GOCACHE"]},
		{name: "cache off falls back", base: []string{"GOCACHE=off"}, userCache: userCacheOK, wantCache: isolated["GOCACHE"]},
		{name: "relative cache falls back", base: []string{"GOCACHE=rel/cache"}, userCache: userCacheOK, wantCache: isolated["GOCACHE"]},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			env := worktreeWorkerGoBuildVerifyEnv(tc.base, isolated, tc.userCache, noSharedGoCache, notInCheckout)
			if got, _ := goBuildVerifyEnvValue(env, "GOTMPDIR"); got != isolated["GOTMPDIR"] {
				t.Fatalf("GOTMPDIR = %q, want the worktree-local %q", got, isolated["GOTMPDIR"])
			}
			if got, _ := goBuildVerifyEnvValue(env, "GOCACHE"); got != tc.wantCache {
				t.Fatalf("GOCACHE = %q, want %q", got, tc.wantCache)
			}
		})
	}
}
