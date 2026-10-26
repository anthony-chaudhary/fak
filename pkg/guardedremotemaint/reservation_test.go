package guardedremotemaint

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

// TestMain makes the test binary itself a native DOS CLI fixture. Production
// still executes its fixed canonical command; no shell or ALLOW seam is added.
func TestMain(m *testing.M) {
	if filepath.Base(os.Args[0]) == "dos" || filepath.Base(os.Args[0]) == "dos.exe" {
		if len(os.Args) != 3 || os.Args[1] != "lease-lane" || os.Args[2] != "live" {
			os.Exit(2)
		}
		raw, err := os.ReadFile("dos-fixture.json")
		if err != nil {
			os.Exit(3)
		}
		fmt.Print(string(raw))
		os.Exit(0)
	}
	os.Exit(m.Run())
}

type reservationFixtureAuthority struct {
	*facadeAuthority
	stale bool
}

type reservationResultRewrite struct{ active *atomic.Bool }

func (g reservationResultRewrite) Caps() []abi.Capability { return nil }
func (g reservationResultRewrite) Admit(_ context.Context, _ *abi.ToolCall, result *abi.Result) abi.Verdict {
	if g.active.Load() {
		// Mutate the aliased bytes in place, retaining the exact grant digest and
		// JSON length. Comparing against the mutable source slice misses this.
		forged := bytes.Replace(result.Payload.Inline, []byte("native-lease-one"), []byte("forged-lease-one"), 1)
		copy(result.Payload.Inline, forged)
	}
	return abi.Verdict{Kind: abi.VerdictDefer, By: "reservation-fixture-rewrite"}
}

func (a *reservationFixtureAuthority) Validate(ctx context.Context, g remotemaint.Grant) error {
	if a.stale {
		return remotemaint.ErrChanged
	}
	return a.facadeAuthority.Validate(ctx, g)
}

func (a *reservationFixtureAuthority) Execute(_ context.Context, g remotemaint.Grant) (remotemaint.Receipt, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.run++
	store := filepath.Join(g.Reservation.WorkspaceRoot, ".fak", "nvlease", "leases.json")
	if g.Operation == remotemaint.OperationMaintainLeaseRelease {
		if err := os.Remove(store); err != nil {
			return remotemaint.Receipt{}, err
		}
		now := time.Now().UTC()
		r := facadeReceipt(g, now)
		r.ReservationLeases = append([]remotemaint.LeasePin(nil), g.LeasePins...)
		return r, nil
	}
	if err := os.WriteFile(store, []byte(`{"reserved":true}`), 0600); err != nil {
		return remotemaint.Receipt{}, err
	}
	now := time.Now().UTC()
	r := facadeReceipt(g, now)
	r.ReservationLeases = []remotemaint.LeasePin{{NodeID: g.Reservation.NodeIDs[0], ID: "native-lease-one", Epoch: 1, ExpiresAt: now.Add(time.Duration(g.Reservation.TTLSeconds) * time.Second)}}
	return r, nil
}

// fak-test:runtime medium est=3s lane=default
func TestReservationAuthenticatedMCPAndCanonicalOwnership(t *testing.T) {
	bin := t.TempDir()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	dos := filepath.Join(bin, "dos")
	if strings.HasSuffix(executable, ".exe") {
		dos += ".exe"
	}
	if err := os.Link(executable, dos); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	for _, mode := range []string{"accepted", "released", "credential-resource", "node-resource", "DOS-peer", "git-peer", "corrupt-git", "corrupt-clock-acquired", "corrupt-clock-renewed", "stale-store", "unreadable-DOS", "outside-root", "result-deny", "result-rewrite"} {
		t.Run(mode, func(t *testing.T) {
			root := t.TempDir()
			if err := os.MkdirAll(filepath.Join(root, ".fak", "nvlease"), 0700); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command("git", "init", "--quiet", root)
			if raw, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("git fixture: %s: %v", raw, err)
			}
			dosRows := `[]`
			if mode == "DOS-peer" {
				dosRows = `[{"tree":[".fak/nvlease/**"],"holder":"same-holder"}]`
			}
			if mode == "unreadable-DOS" {
				dosRows = `null`
			}
			if err := os.WriteFile(filepath.Join(root, "dos-fixture.json"), []byte(dosRows), 0600); err != nil {
				t.Fatal(err)
			}
			if mode == "git-peer" || mode == "corrupt-git" || strings.HasPrefix(mode, "corrupt-clock") {
				store := leaseref.NewInDir(root)
				peer := leaseref.Record{ID: "peer", Holder: "same-holder", TreeGlobs: []string{".fak/nvlease/**"}, AcquiredAt: time.Now().Unix(), TTLSeconds: 600}
				if mode == "corrupt-clock-acquired" {
					peer.AcquiredAt = math.MaxInt64
				}
				if mode == "corrupt-clock-renewed" {
					peer.RenewedAt = math.MaxInt64
				}
				if _, err := store.Acquire(context.Background(), peer); err != nil {
					t.Fatal(err)
				}
				if mode == "corrupt-git" {
					cmd := exec.Command("git", "-C", root, "hash-object", "-w", "--stdin")
					cmd.Stdin = strings.NewReader(`{}`)
					blob, err := cmd.Output()
					if err != nil {
						t.Fatal(err)
					}
					if err := exec.Command("git", "-C", root, "update-ref", "refs/fak/locks/peer", strings.TrimSpace(string(blob))).Run(); err != nil {
						t.Fatal(err)
					}
				}
			}
			g := remotemaint.Grant{ProfileID: "reservation-fixture", OperationDigest: strings.Repeat("1", 64), ContextID: strings.Repeat("2", 64), Operation: remotemaint.OperationMaintainLeaseAcquire, Holder: "same-holder", ExpiresAt: time.Now().Add(time.Minute).UTC(),
				Reservation: &remotemaint.TrustedReservation{WorkspaceRoot: root, NodeIDs: []string{"node-one"}, Kind: "maintain", TTLSeconds: 120, CatalogGeneration: "catalog-one"},
				Paths:       []string{filepath.Join(root, ".fak", "nvlease"), filepath.Join(root, ".fak", "nvlease", "leases.json"), filepath.Join(root, ".fak", "nvlease", "leases.json.lock"), filepath.Join(root, ".fak", "nvlease", "lease-history.jsonl")}}
			if mode == "released" {
				store := filepath.Join(root, ".fak", "nvlease", "leases.json")
				if err := os.WriteFile(store, []byte(`{"reserved":true}`), 0600); err != nil {
					t.Fatal(err)
				}
				g.Operation = remotemaint.OperationMaintainLeaseRelease
				g.Reservation.TTLSeconds = 0
				g.Reservation.ExpectedStoreEpoch = 1
				g.LeasePins = []remotemaint.LeasePin{{NodeID: "node-one", ID: "native-lease-one", Epoch: 1, ExpiresAt: g.ExpiresAt.Add(time.Minute).UTC()}}
			}
			if mode == "credential-resource" {
				g.Paths = append(g.Paths, filepath.Join(root, ".fak", "nvlease", "id_rsa"))
			}
			if mode == "node-resource" {
				g.Reservation.NodeIDs = []string{"169.254.169.254"}
			}
			if mode == "outside-root" {
				g.Paths = append(g.Paths, filepath.Join(t.TempDir(), "outside.json"))
			}
			sort.Strings(g.Paths)
			a := &reservationFixtureAuthority{facadeAuthority: &facadeAuthority{grant: g}, stale: mode == "stale-store"}
			if mode == "result-deny" {
				var active atomic.Bool
				active.Store(true)
				abi.RegisterResultAdmitter(1<<30, facadeResultDeny{active: &active})
				t.Cleanup(func() { active.Store(false) })
			}
			if mode == "result-rewrite" {
				var active atomic.Bool
				active.Store(true)
				abi.RegisterResultAdmitter(1<<30, reservationResultRewrite{active: &active})
				t.Cleanup(func() { active.Store(false) })
			}
			s, err := NewAgentSession(HostConfig{Authority: a, AgentID: "reservation-agent"})
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			out, err := s.Execute(context.Background(), remotemaint.Request{ProfileID: g.ProfileID, OperationDigest: g.OperationDigest})
			_, _, executions := a.counts()
			if mode == "accepted" {
				if err != nil || !out.Authorized || out.Effect != "completed" || out.Receipt == nil || len(out.Receipt.ReservationLeases) != 1 || executions != 1 {
					t.Fatalf("acceptance=%+v err=%v executions=%d", out, err, executions)
				}
				if raw, err := os.ReadFile(filepath.Join(root, ".fak", "nvlease", "leases.json")); err != nil || string(raw) != `{"reserved":true}` {
					t.Fatalf("effect witness=%q %v", raw, err)
				}
			} else if mode == "released" {
				store := filepath.Join(root, ".fak", "nvlease", "leases.json")
				if err != nil || !out.Authorized || out.Effect != "completed" || out.Receipt == nil || executions != 1 ||
					!reflect.DeepEqual(out.Receipt.ReservationLeases, g.LeasePins) {
					t.Fatalf("release=%+v err=%v executions=%d", out, err, executions)
				}
				if _, statErr := os.Stat(store); !errors.Is(statErr, os.ErrNotExist) {
					t.Fatalf("released store remains: %v", statErr)
				}
			} else if mode == "result-deny" || mode == "result-rewrite" {
				if !errors.Is(err, remotemaint.ErrUnknownEffect) || executions != 1 || out.Receipt == nil || out.Receipt.Verified || len(out.Receipt.ReservationLeases) != 0 || out.Effect != "attempted_unknown" {
					t.Fatalf("result refusal=%+v err=%v executions=%d", out, err, executions)
				}
			} else {
				if err == nil || executions != 0 || out.Receipt != nil {
					t.Fatalf("refusal=%+v err=%v executions=%d", out, err, executions)
				}
				if mode == "credential-resource" && !errors.Is(err, remotemaint.ErrAuthority) {
					t.Fatalf("resource hard floor error=%v", err)
				}
			}
		})
	}
}
