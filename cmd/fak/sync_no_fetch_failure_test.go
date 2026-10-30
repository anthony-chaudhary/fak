package main

import (
	"bytes"
	"context"
	"github.com/anthony-chaudhary/fak/internal/safesync"
	"testing"
)

func TestRunSyncNoFetchOnFailure(t *testing.T) {
	oldCapture, oldPush := syncCaptureSource, syncSafePush
	defer func() { syncCaptureSource, syncSafePush = oldCapture, oldPush }()
	syncCaptureSource = func(string) (string, error) { return "563893c6e7df3d8e0ef5a17c06432670a327c231", nil }
	called := false
	syncSafePush = func(_ context.Context, o safesync.PushOptions) (safesync.PushResult, error) {
		called = true
		if !o.NoFetchOnFailure || o.SourceRef != "563893c6e7df3d8e0ef5a17c06432670a327c231" {
			t.Fatalf("lost exact option/source: %+v", o)
		}
		return safesync.PushResult{Reason: safesync.PushReasonFetchDeferred}, nil
	}
	var out, errb bytes.Buffer
	if c := runSync(&out, &errb, []string{"push", "--repo", t.TempDir(), "--branch", "main", "--no-fetch-on-failure", "--json"}); c != syncExitRefused || !called {
		t.Fatalf("code=%d called=%v stderr=%s", c, called, errb.String())
	}
	called = false
	if c := runSync(&out, &errb, []string{"check", "--no-fetch-on-failure"}); c != syncExitUsage || called {
		t.Fatalf("non-push accepted: %d", c)
	}
}
