package workerworktree

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestHandoffOwnerReplacesEphemeralLauncherPID(t *testing.T) {
	parent := t.TempDir()
	wt := filepath.Join(parent, WorktreeMarker+"-handoff")
	if err := os.Mkdir(wt, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeOwnerStamp(wt, OwnerStamp{PID: 111, LeaseID: "lease-1", CreatedAt: time.Unix(1, 0).UTC()}); err != nil {
		t.Fatal(err)
	}
	if err := HandoffOwner(wt, 222); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(OwnerStampPath(wt))
	if err != nil {
		t.Fatal(err)
	}
	var stamp OwnerStamp
	if err := json.Unmarshal(raw, &stamp); err != nil {
		t.Fatal(err)
	}
	if stamp.PID != 222 || stamp.LeaseID != "lease-1" {
		t.Fatalf("handoff stamp = %+v, want child pid and preserved lease", stamp)
	}

	opts := normalizeGCOptions(GCOptions{
		Now: time.Now().Add(2 * time.Hour), MaxAge: time.Minute,
		ProcessAlive: func(pid int) bool { return pid == 222 },
		LeaseLive:    func(string) bool { return false },
		PathActive:   func(string) (bool, error) { return false, nil },
		AllowedRoots: []string{filepath.Dir(wt)},
	})
	entry, include, failure := gcEntry("C:/repo", wt, (&gcGit{status: map[string]string{filepath.Clean(wt): ""}, heads: map[string]string{filepath.Clean(wt): "abc"}, ancestors: map[string]int{"abc": 0}}).run, opts)
	if include || failure != "owner_process_live" || !entry.OwnerLive || entry.Eligible {
		t.Fatalf("gcEntry = %+v include=%v failure=%q, want live child to protect tree", entry, include, failure)
	}
}
