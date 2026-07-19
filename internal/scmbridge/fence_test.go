package scmbridge

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/servicelease"
)

const wl = "FakGuardControl"

func labNode(t *testing.T, f *LaunchFence) servicelease.Incarnation {
	t.Helper()
	inc := servicelease.Incarnation{Node: "lab-1", BootID: "boot-a"}
	f.Table.RecordIncarnation(inc)
	return inc
}

func TestLaunchFenceCollapsesThreeLaunchersToOne(t *testing.T) {
	f := NewLaunchFence(30000)
	inc := labNode(t, f)
	g, err := f.Admit(wl, RoleMachine, inc, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if g.RequestID == "" || g.Token.LeaseSeq == 0 {
		t.Fatalf("grant = %+v", g)
	}
	// The watchdog and the broker race the same workload on the same
	// incarnation: both are refused while the SCM grant is live.
	if _, err := f.Admit(wl, RoleWatchdog, inc, 2000); !errors.Is(err, ErrDuplicateLaunch) {
		t.Fatalf("watchdog: err = %v", err)
	}
	if _, err := f.Admit(wl, RoleBroker, inc, 2000); !errors.Is(err, ErrDuplicateLaunch) {
		t.Fatalf("broker: err = %v", err)
	}
	if h := f.Holder(wl, 2000); h != RoleMachine {
		t.Fatalf("holder = %q", h)
	}
	// A classified stop releases the grant; the next launcher wins cleanly.
	if !f.Release(wl, RoleMachine, g.Token) {
		t.Fatal("release refused")
	}
	if _, err := f.Admit(wl, RoleWatchdog, inc, 3000); err != nil {
		t.Fatalf("watchdog after release: %v", err)
	}
}

func TestLaunchFenceSameRoleHeartbeatRefreshes(t *testing.T) {
	f := NewLaunchFence(30000)
	inc := labNode(t, f)
	g1, err := f.Admit(wl, RoleMachine, inc, 1000)
	if err != nil {
		t.Fatal(err)
	}
	g2, err := f.Admit(wl, RoleMachine, inc, 2000)
	if err != nil {
		t.Fatalf("heartbeat re-admit: %v", err)
	}
	if g2.Token.LeaseSeq <= g1.Token.LeaseSeq {
		t.Fatalf("heartbeat did not mint a newer token: %v vs %v", g2.Token, g1.Token)
	}
	// The stale token can no longer release the live grant.
	if f.Release(wl, RoleMachine, g1.Token) {
		t.Fatal("stale token released the live grant")
	}
}

func TestLaunchFenceGenerationBumpFencesTheGrant(t *testing.T) {
	f := NewLaunchFence(30000)
	inc := labNode(t, f)
	if _, err := f.Admit(wl, RoleMachine, inc, 1000); err != nil {
		t.Fatal(err)
	}
	f.Supersede(wl) // desired-state change: every prior token is stale
	if h := f.Holder(wl, 1500); h != "" {
		t.Fatalf("superseded holder = %q", h)
	}
	if _, err := f.Admit(wl, RoleBroker, inc, 2000); err != nil {
		t.Fatalf("broker after supersede: %v", err)
	}
}

func TestLaunchFenceExpiryFreesTheGrantWithoutRelease(t *testing.T) {
	f := NewLaunchFence(1000)
	inc := labNode(t, f)
	if _, err := f.Admit(wl, RoleMachine, inc, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Admit(wl, RoleWatchdog, inc, 500); !errors.Is(err, ErrDuplicateLaunch) {
		t.Fatalf("pre-expiry: %v", err)
	}
	if _, err := f.Admit(wl, RoleWatchdog, inc, 1500); err != nil {
		t.Fatalf("post-expiry: %v", err)
	}
}

func TestLaunchFenceRefusesAnotherNodeWhileLeaseValid(t *testing.T) {
	f := NewLaunchFence(30000)
	inc := labNode(t, f)
	other := servicelease.Incarnation{Node: "lab-2", BootID: "boot-x"}
	f.Table.RecordIncarnation(other)
	if _, err := f.Admit(wl, RoleMachine, inc, 1000); err != nil {
		t.Fatal(err)
	}
	// Same role, different node: not a role race — a second OWNER. The lease
	// layer refuses it.
	if _, err := f.Admit(wl, RoleMachine, other, 2000); !errors.Is(err, servicelease.ErrLeaseHeld) {
		t.Fatalf("cross-node: %v", err)
	}
}

func TestLaunchFenceIsDurablePlainJSON(t *testing.T) {
	f := NewLaunchFence(30000)
	inc := labNode(t, f)
	if _, err := f.Admit(wl, RoleMachine, inc, 1000); err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(f)
	if err != nil {
		t.Fatal(err)
	}
	var back LaunchFence
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	// The reloaded fence still refuses the duplicate: the grant survived the
	// process that minted it.
	if _, err := back.Admit(wl, RoleWatchdog, inc, 2000); !errors.Is(err, ErrDuplicateLaunch) {
		t.Fatalf("reloaded fence: %v", err)
	}
}
