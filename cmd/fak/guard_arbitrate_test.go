package main

import (
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func guardArbitrateTestRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	cmd := exec.Command("git", "init", "--quiet")
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(`[lanes]
exclusive = ["exclusive"]
[lanes.trees]
cmd = ["cmd/**"]
docs = ["docs/**"]
exclusive = ["internal/**"]
`), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func guardArbitrateSeedLease(t *testing.T, root, id string, tree []string) {
	t.Helper()
	store := leaseref.NewInDir(root)
	_, err := store.Acquire(context.Background(), leaseref.Record{
		ID: id, Holder: "peer", TreeGlobs: tree, AcquiredAt: time.Now().Unix(), TTLSeconds: 60,
	})
	if err != nil {
		t.Fatalf("seed lease: %v", err)
	}
}

func TestGuardArbitrateEnforceRefusesOverlap(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	guardArbitrateSeedLease(t, root, "peer-cmd", []string{"cmd/**"})
	lease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce, Root: root, Tree: []string{"cmd/fak/**"},
	})
	if lease != nil {
		lease.Close()
		t.Fatal("overlap unexpectedly acquired a lease")
	}
	if err == nil || !strings.Contains(err.Error(), "COLLISION_RISK") || !strings.Contains(err.Error(), "peer-cmd") {
		t.Fatalf("overlap error = %v, want COLLISION_RISK naming peer-cmd", err)
	}
}

func TestGuardArbitrateDisjointPublishesAndReleases(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	guardArbitrateSeedLease(t, root, "peer-docs", []string{"docs/**"})
	lease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce, Root: root, Tree: []string{"cmd/**"},
	})
	if err != nil {
		t.Fatalf("disjoint acquire: %v", err)
	}
	if lease == nil {
		t.Fatal("disjoint acquire returned no lease")
	}
	id := lease.record.ID
	if _, ok, err := lease.store.Get(context.Background(), id); err != nil || !ok {
		t.Fatalf("published lease read-back: ok=%v err=%v", ok, err)
	}
	lease.Close()
	if _, ok, err := lease.store.Get(context.Background(), id); err != nil || ok {
		t.Fatalf("released lease read-back: ok=%v err=%v", ok, err)
	}
}

func TestGuardArbitrateShadowLogsWithoutPublishing(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	guardArbitrateSeedLease(t, root, "peer-cmd", []string{"cmd/**"})
	var stderr strings.Builder
	lease, err := guardArbitrateAcquire(context.Background(), &stderr, guardArbitrateConfig{
		Mode: guardArbitrateModeShadow, Root: root, Tree: []string{"cmd/**"}, ShowShadowNotice: true,
	})
	if err != nil || lease != nil {
		t.Fatalf("shadow = lease %v err %v", lease, err)
	}
	if got := stderr.String(); !strings.Contains(got, "shadow would refuse") || !strings.Contains(got, "peer-cmd") {
		t.Fatalf("shadow log = %q", got)
	}
}

func TestGuardArbitrateCompactStartupSuppressesShadowCollision(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	guardArbitrateSeedLease(t, root, "peer-cmd", []string{"cmd/**"})
	var stderr strings.Builder
	lease, err := guardArbitrateAcquire(context.Background(), &stderr, guardArbitrateConfig{
		Mode: guardArbitrateModeShadow, Root: root, Tree: []string{"cmd/**"},
	})
	if err != nil || lease != nil {
		t.Fatalf("shadow = lease %v err %v", lease, err)
	}
	if got := stderr.String(); got != "" {
		t.Fatalf("shadow collision polluted compact startup: %q", got)
	}
}

func TestGuardArbitrateForceBypassesNonExclusiveOverlap(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	guardArbitrateSeedLease(t, root, "peer-cmd", []string{"cmd/**"})
	lease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce, Root: root, Tree: []string{"cmd/fak/**"}, Force: true,
	})
	if err != nil {
		t.Fatalf("force non-exclusive acquire: %v", err)
	}
	if lease == nil {
		t.Fatal("force non-exclusive acquire returned no lease")
	}
	lease.Close()
}

func TestGuardArbitrateForceStillHonorsExclusive(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	guardArbitrateSeedLease(t, root, "peer-exclusive", []string{"internal/**"})
	lease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce, Root: root, Tree: []string{"docs/**"}, Force: true,
	})
	if lease != nil {
		lease.Close()
		t.Fatal("force bypassed live exclusive lease")
	}
	if err == nil || !strings.Contains(err.Error(), "peer-exclusive") {
		t.Fatalf("exclusive force error = %v", err)
	}
}

func TestGuardArbitrateUnreadableWorkspaceFailsOpen(t *testing.T) {
	lease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce, Root: filepath.Join(t.TempDir(), "missing"), Tree: []string{"cmd/**"},
	})
	if err != nil || lease != nil {
		t.Fatalf("fail-open = lease %v err %v", lease, err)
	}
}

func TestGuardArbitrateFlagValueParsesLeaseProfile(t *testing.T) {
	cfg := guardArbitrateConfig{Mode: guardArbitrateModeShadow}
	value := guardArbitrateFlagValue{cfg: &cfg}
	if err := value.Set("mode=enforce,lane=gateway,tree=internal/gateway/**,tree=cmd/fak/**,force=true"); err != nil {
		t.Fatal(err)
	}
	if cfg.Mode != guardArbitrateModeEnforce || cfg.Lane != "gateway" || !cfg.Force {
		t.Fatalf("config = %+v", cfg)
	}
	if got, want := cfg.Tree, []string{"internal/gateway/**", "cmd/fak/**"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("tree = %v, want %v", got, want)
	}
}

func TestGuardArbitrateFlagValueRejectsUnknownField(t *testing.T) {
	cfg := guardArbitrateConfig{Mode: guardArbitrateModeShadow}
	if err := (guardArbitrateFlagValue{cfg: &cfg}).Set("branch=main"); err == nil || !strings.Contains(err.Error(), "unknown lease field") {
		t.Fatalf("error = %v", err)
	}
}
