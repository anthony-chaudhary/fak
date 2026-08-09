package gitdaily

import (
	"context"
	"encoding/json"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

// TestDeterminism proves that an identical repository snapshot and clock produce
// an identical scheduler result. Each run gets a distinct Git common directory so
// serializer/ledger files from the first invocation cannot influence the second.
func TestDeterminism(t *testing.T) {
	fixed := time.Date(2026, 8, 4, 3, 0, 0, 0, time.UTC)
	run := func(t *testing.T) Result {
		t.Helper()
		opts := fakeRepo(t)
		opts.Now = fixed
		var calls []string
		got := Run(context.Background(), recordingRunner(&calls), opts)
		// Absolute roots are environmental inputs, not scheduler output. Normalize
		// them before comparing otherwise-identical snapshots.
		got.RepoRoot = filepath.ToSlash("/repo")
		got.GitCommonDir = filepath.ToSlash("/repo/.git")
		return got
	}

	first := run(t)
	second := run(t)
	if !reflect.DeepEqual(first, second) {
		a, _ := json.MarshalIndent(first, "", "  ")
		b, _ := json.MarshalIndent(second, "", "  ")
		t.Fatalf("identical git-daily inputs produced different results:\nfirst=%s\nsecond=%s", a, b)
	}
}
