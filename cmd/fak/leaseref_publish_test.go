package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// fak-test:justify why=integration when=changed:cmd/fak/leaseref* witness=leaseref.go#runLeaserefPublish
// fak-test:runtime integration est=1s lane=default
func TestLeaserefPublishCLI(t *testing.T) {
	t.Setenv("GIT_CONFIG_GLOBAL", os.DevNull)
	t.Setenv("GIT_CONFIG_NOSYSTEM", "1")
	root := t.TempDir()
	local, remote := filepath.Join(root, "local"), filepath.Join(root, "remote.git")
	reservedNamespaceCLIGit(t, root, "", "init", "-q", local)
	reservedNamespaceCLIGit(t, root, "", "init", "-q", "--bare", remote)
	reservedNamespaceCLIGit(t, local, "", "remote", "add", "origin", remote)
	store := leaseref.NewInDir(local)
	rec, verdict, err := store.AcquireFenced(context.Background(), leaseref.Record{
		ID: "fixture", Holder: "holder", SessionID: "session", TTLSeconds: 3600,
	}, time.Now())
	if err != nil || !verdict.OK {
		t.Fatalf("acquire: verdict=%+v error=%v", verdict, err)
	}
	oid := reservedNamespaceCLIGit(t, local, "", "rev-parse", rec.Ref())
	args := []string{"publish", "--dir", local, "--remote", "origin", "--ref", rec.Ref(),
		"--oid", oid, "--expect-absent", "--holder", rec.Holder, "--session", rec.SessionID,
		"--generation", "1", "--trusted-direct-refs", "authorized-server-control:disposable fixture owned by this test"}
	var stdout, stderr bytes.Buffer
	if code := runLeaseref(&stdout, &stderr, args); code != 0 {
		t.Fatalf("publication exit=%d stderr=%s stdout=%s", code, &stderr, &stdout)
	}
	var result struct {
		Confirmed bool   `json:"confirmed_publication"`
		Attempted bool   `json:"transport_attempted"`
		Completed bool   `json:"transport_completed"`
		PinnedOID string `json:"pinned_oid"`
		RemoteOID string `json:"remote_readback_oid"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !result.Confirmed || !result.Attempted || !result.Completed || result.PinnedOID != oid || result.RemoteOID != oid {
		t.Fatalf("unconfirmed publication receipt: %+v", result)
	}
	if got := reservedNamespaceCLIGit(t, remote, "", "rev-parse", rec.Ref()); got != oid {
		t.Fatalf("remote=%s, pinned=%s", got, oid)
	}
	stdout.Reset()
	stderr.Reset()
	if code := runLeaseref(&stdout, &stderr, args); code == 0 {
		t.Fatal("explicit absence accepted an existing remote ref")
	}
	stdout.Reset()
	stderr.Reset()
	if code := runLeaseref(&stdout, &stderr, []string{"publish", "--help"}); code != 0 {
		t.Fatalf("publish help exit=%d", code)
	}
}
