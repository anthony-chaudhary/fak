//go:build linux || darwin

package researcharm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func durableTestPath(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "authority")
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	return filepath.Join(dir, "leases.json")
}
func durableAcquire(t *testing.T, c *Coordinator) *LeaseInfo {
	t.Helper()
	l, err := c.AcquireLease(LeaseRequest{ArmID: "owner", Mode: LeaseModeExclusive, TTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	return l
}
func durableAdmit(c *Coordinator, arm string) error {
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("X-Fak-Project-Arm", arm)
	l, err := c.Admit(context.Background(), r, "/v1/chat/completions", "store-contract")
	if l != nil {
		l.Done(0, nil)
	}
	return err
}
func durableRead(t *testing.T, path string) []LeaseInfo {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var stored struct {
		Schema string      `json:"schema"`
		Leases []LeaseInfo `json:"leases"`
	}
	if err := json.Unmarshal(b, &stored); err != nil {
		t.Fatal(err)
	}
	if stored.Schema != "fak.researcharm.lease-store/v1" {
		t.Fatalf("store schema=%q", stored.Schema)
	}
	return stored.Leases
}

func durableLeaseEqual(got, want LeaseInfo) bool {
	return got.ID == want.ID && got.ArmID == want.ArmID && got.HolderPID == want.HolderPID && got.Mode == want.Mode && got.Concurrency == want.Concurrency && got.Token == want.Token && got.CreatedAt.Equal(want.CreatedAt) && got.ExpiresAt.Equal(want.ExpiresAt)
}

// fak-test:runtime medium est=1s lane=default
func TestDurableCoordinatorContract(t *testing.T) {
	t.Run("fresh authority requires acknowledged exclusive bootstrap", func(t *testing.T) {
		path := durableTestPath(t)
		c, err := NewDurableCoordinator(2, path)
		if err != nil {
			t.Fatal(err)
		}
		ready := func() bool {
			b, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var v struct {
				Ready bool `json:"admission_ready"`
			}
			if err := json.Unmarshal(b, &v); err != nil {
				t.Fatal(err)
			}
			return v.Ready
		}
		if ready() {
			t.Fatal("fresh store claimed migration admission ready")
		}
		if err := durableAdmit(c, "peer"); err == nil {
			t.Fatal("fresh authority admitted before exclusive bootstrap")
		}
		for _, mode := range []LeaseMode{LeaseModeShared, ""} {
			if _, err := c.AcquireLease(LeaseRequest{ArmID: "owner", Mode: mode, TTL: time.Minute}); err == nil {
				t.Fatalf("first %q acquisition opened authority without quiescent bootstrap", mode)
			}
		}
		dir := filepath.Dir(path)
		moved := dir + "-bootstrap-moved"
		if err := os.Rename(dir, moved); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dir, []byte("blocks persistence"), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := c.AcquireLease(LeaseRequest{ArmID: "owner", Mode: LeaseModeExclusive, TTL: time.Minute}); err == nil {
			t.Fatal("failed exclusive bootstrap acknowledged")
		}
		if err := durableAdmit(c, "owner"); err == nil {
			t.Fatal("failed exclusive bootstrap enabled admission")
		}
		if err := os.Remove(dir); err != nil {
			t.Fatal(err)
		}
		if err := os.Rename(moved, dir); err != nil {
			t.Fatal(err)
		}
		if ready() {
			t.Fatal("failed bootstrap persisted admission ready")
		}
		lease := durableAcquire(t, c)
		if !ready() {
			t.Fatal("successful exclusive acknowledgement did not persist admission ready")
		}
		if err := c.ReleaseLease(lease.ID, lease.Token); err != nil {
			t.Fatal(err)
		}
		if !ready() {
			t.Fatal("release reset established authority")
		}
		if err := c.Close(); err != nil {
			t.Fatal(err)
		}
		next, err := NewDurableCoordinator(2, path)
		if err != nil {
			t.Fatal(err)
		}
		defer next.Close()
		if err := durableAdmit(next, "peer"); err != nil {
			t.Fatalf("restart lost established admission readiness: %v", err)
		}
	})

	t.Run("acknowledged identity survives restart and release", func(t *testing.T) {
		path := durableTestPath(t)
		c, err := NewDurableCoordinator(2, path)
		if err != nil {
			t.Fatal(err)
		}
		acquired := durableAcquire(t, c)
		if got := durableRead(t, path); len(got) != 1 || !durableLeaseEqual(got[0], *acquired) {
			t.Fatalf("acquire acknowledgement not persisted exactly: %v want %+v", got, acquired)
		}
		if err := c.Close(); err != nil {
			t.Fatal(err)
		}
		reopened, err := NewDurableCoordinator(2, path)
		if err != nil {
			t.Fatal(err)
		}
		if err := durableAdmit(reopened, "peer"); !errors.Is(err, ErrExclusiveLeaseHeld) {
			t.Fatalf("restart lost exclusive admission: %v", err)
		}
		if err := reopened.ReleaseLease(acquired.ID, acquired.Token); err != nil {
			t.Fatalf("restart lost original release credential: %v", err)
		}
		if got := durableRead(t, path); len(got) != 0 {
			t.Fatalf("release acknowledged without persistence: %v", got)
		}
		if err := reopened.Close(); err != nil {
			t.Fatal(err)
		}
		final, err := NewDurableCoordinator(2, path)
		if err != nil {
			t.Fatal(err)
		}
		defer final.Close()
		if err := durableAdmit(final, "peer"); err != nil {
			t.Fatalf("released lease resurrected: %v", err)
		}
	})
	t.Run("lost initialized snapshot poisons without regeneration", func(t *testing.T) {
		path := durableTestPath(t)
		c, err := NewDurableCoordinator(2, path)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		lease := durableAcquire(t, c)
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := c.ReleaseLease(lease.ID, lease.Token); err == nil {
			t.Fatal("missing authoritative snapshot acknowledged release")
		}
		if c.Snapshot().DurableStatus != "unavailable" {
			t.Fatal("lost snapshot reported available authority")
		}
		for _, arm := range []string{"owner", "peer"} {
			if err := durableAdmit(c, arm); err == nil {
				t.Fatalf("lost snapshot admitted %s", arm)
			}
		}
		if _, err := c.AcquireLease(LeaseRequest{ArmID: "owner", Mode: LeaseModeExclusive, TTL: time.Minute}); err == nil {
			t.Fatal("poisoned authority acquired")
		}
		if err := c.ReleaseLease(lease.ID, lease.Token); err == nil {
			t.Fatal("poisoned authority released")
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("lost snapshot regenerated: %v", err)
		}
		if err := c.Close(); err != nil {
			t.Fatal(err)
		}
		next, err := NewDurableCoordinator(2, path)
		if err == nil {
			next.Close()
			t.Fatal("reopen forgot initialized missing snapshot")
		}
	})
	t.Run("stable sidecar excludes another process", func(t *testing.T) {
		path := durableTestPath(t)
		c, err := NewDurableCoordinator(2, path)
		if err != nil {
			t.Fatal(err)
		}
		before, err := os.Stat(path + ".lock")
		if err != nil {
			t.Fatal(err)
		}
		durableAcquire(t, c)
		cmd := exec.Command(os.Args[0], "-test.run=^TestDurableCoordinatorProcessHelper$")
		cmd.Env = append(os.Environ(), "FAK_DURABLE_STORE_CHILD="+path)
		output, err := cmd.CombinedOutput()
		if err != nil || !bytes.Contains(output, []byte("LOCK_REFUSED")) {
			t.Fatalf("second process did not refuse ownership: err=%v output=%s", err, output)
		}
		if err := c.Close(); err != nil {
			t.Fatal(err)
		}
		next, err := NewDurableCoordinator(2, path)
		if err != nil {
			t.Fatal(err)
		}
		defer next.Close()
		after, err := os.Stat(path + ".lock")
		if err != nil {
			t.Fatal(err)
		}
		if !os.SameFile(before, after) {
			t.Fatal("sidecar inode replaced across snapshot writes or reopen")
		}
	})
	t.Run("precommit failure preserves acknowledged authority and memory admission", func(t *testing.T) {
		path := durableTestPath(t)
		c, err := NewDurableCoordinator(2, path)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		original := durableAcquire(t, c)
		dir := filepath.Dir(path)
		moved := dir + "-moved"
		if err := os.Rename(dir, moved); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dir, []byte("blocks directory writes"), 0600); err != nil {
			t.Fatal(err)
		}
		restore := func() {
			if err := os.Remove(dir); err != nil {
				t.Fatal(err)
			}
			if err := os.Rename(moved, dir); err != nil {
				t.Fatal(err)
			}
		}
		defer restore()
		if _, err := c.AcquireLease(LeaseRequest{ArmID: "owner", Mode: LeaseModeExclusive, TTL: time.Minute}); err == nil {
			t.Fatal("failed write acknowledged renewed lease")
		}
		if err := c.ReleaseLease(original.ID, original.Token); err == nil {
			t.Fatal("failed write acknowledged release")
		}
		if err := durableAdmit(c, "peer"); !errors.Is(err, ErrExclusiveLeaseHeld) {
			t.Fatalf("failed write lost prior exclusion: %v", err)
		}
		if err := durableAdmit(c, "owner"); err != nil {
			t.Fatalf("memory admission performed store I/O or discarded prior lease: %v", err)
		}
		got := durableRead(t, filepath.Join(moved, "leases.json"))
		if len(got) != 1 || !durableLeaseEqual(got[0], *original) {
			t.Fatalf("failed mutation changed acknowledged state: %v", got)
		}
	})
	t.Run("uncertain commit poisons operations until reopen", func(t *testing.T) {
		for _, stage := range []string{"rename", "directory-sync"} {
			t.Run(stage, func(t *testing.T) {
				path := durableTestPath(t)
				c, err := NewDurableCoordinator(2, path)
				if err != nil {
					t.Fatal(err)
				}
				l := durableAcquire(t, c)
				if stage == "rename" {
					c.store.renameFile = func(string, string) error { return errors.New("injected uncertain rename") }
				} else {
					c.store.syncDirectory = func(string) error { return errors.New("injected parent sync uncertainty") }
				}
				if err := c.ReleaseLease(l.ID, l.Token); err == nil {
					t.Fatal("uncertain commit acknowledged release")
				}
				if err := durableAdmit(c, "owner"); err == nil {
					t.Fatal("poisoned coordinator admitted request")
				}
				if _, err := c.AcquireLease(LeaseRequest{ArmID: "owner", Mode: LeaseModeExclusive, TTL: time.Minute}); err == nil {
					t.Fatal("poisoned coordinator acquired lease")
				}
				if err := c.ReleaseLease(l.ID, l.Token); err == nil {
					t.Fatal("poisoned coordinator released lease")
				}
				if err := c.Close(); err != nil {
					t.Fatal(err)
				}
				next, err := NewDurableCoordinator(2, path)
				if err != nil {
					t.Fatal(err)
				}
				defer next.Close()
				if stage == "rename" {
					if err := durableAdmit(next, "peer"); !errors.Is(err, ErrExclusiveLeaseHeld) {
						t.Fatalf("prior persisted lease lost: %v", err)
					}
				} else {
					if err := durableAdmit(next, "peer"); err != nil {
						t.Fatalf("reopen did not reconcile renamed snapshot: %v", err)
					}
				}
			})
		}
	})
	t.Run("expired persisted lease does not exclude", func(t *testing.T) {
		path := durableTestPath(t)
		c, err := NewDurableCoordinator(2, path)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := c.AcquireLease(LeaseRequest{ArmID: "owner", Mode: LeaseModeExclusive, TTL: 5 * time.Millisecond}); err != nil {
			t.Fatal(err)
		}
		if err := c.Close(); err != nil {
			t.Fatal(err)
		}
		time.Sleep(20 * time.Millisecond)
		next, err := NewDurableCoordinator(2, path)
		if err != nil {
			t.Fatal(err)
		}
		defer next.Close()
		if _, err := next.AcquireLease(LeaseRequest{ArmID: "peer", Mode: LeaseModeExclusive, TTL: time.Minute}); err != nil {
			t.Fatalf("expired lease excluded replacement: %v", err)
		}
	})
	t.Run("close denies every operation", func(t *testing.T) {
		path := durableTestPath(t)
		c, err := NewDurableCoordinator(2, path)
		if err != nil {
			t.Fatal(err)
		}
		l := durableAcquire(t, c)
		if err := c.Close(); err != nil {
			t.Fatal(err)
		}
		if err := durableAdmit(c, "owner"); err == nil {
			t.Fatal("closed coordinator admitted")
		}
		if _, err := c.AcquireLease(LeaseRequest{ArmID: "owner"}); err == nil {
			t.Fatal("closed coordinator acquired")
		}
		if err := c.ReleaseLease(l.ID, l.Token); err == nil {
			t.Fatal("closed coordinator released")
		}
	})
	t.Run("strict stored authority validation", func(t *testing.T) {
		seedPath := durableTestPath(t)
		seed, err := NewDurableCoordinator(2, seedPath)
		if err != nil {
			t.Fatal(err)
		}
		lease := *durableAcquire(t, seed)
		if err := seed.Close(); err != nil {
			t.Fatal(err)
		}
		envelope := func(ls []LeaseInfo) []byte {
			b, err := json.Marshal(map[string]any{"schema": "fak.researcharm.lease-store/v1", "leases": ls, "admission_ready": true})
			if err != nil {
				t.Fatal(err)
			}
			return b
		}
		invalidMode := lease
		invalidMode.Mode = "CPU"
		invalidDate := lease
		invalidDate.ExpiresAt = invalidDate.CreatedAt
		invalidToken := lease
		invalidToken.Token = ""
		duplicateArm := lease
		duplicateArm.ID = "another-id"
		duplicateID := lease
		duplicateID.ArmID = "another-arm"
		valid := envelope([]LeaseInfo{lease})
		for name, payload := range map[string][]byte{
			"uppercase schema":        bytes.Replace(valid, []byte(`"schema":`), []byte(`"Schema":`), 1),
			"readiness case override": bytes.Replace(valid, []byte(`"admission_ready":true`), []byte(`"admission_ready":false,"Admission_Ready":true`), 1),
			"lease id case alias":     bytes.Replace(valid, []byte(`"id":`), []byte(`"ID":`), 1),

			"corrupt": []byte("{"), "unknown schema": []byte(`{"schema":"unknown","leases":[],"admission_ready":true}`),
			"duplicate field": []byte(`{"schema":"fak.researcharm.lease-store/v1","schema":"fak.researcharm.lease-store/v1","leases":[],"admission_ready":true}`),
			"unknown field":   []byte(`{"schema":"fak.researcharm.lease-store/v1","leases":[],"admission_ready":true,"grant_all":true}`),
			"duplicate id":    envelope([]LeaseInfo{lease, duplicateID}), "duplicate arm": envelope([]LeaseInfo{lease, duplicateArm}),
			"invalid mode": envelope([]LeaseInfo{invalidMode}), "invalid dates": envelope([]LeaseInfo{invalidDate}), "missing token": envelope([]LeaseInfo{invalidToken}),
			"oversize": bytes.Repeat([]byte(" "), (1<<20)+1),
		} {
			t.Run(name, func(t *testing.T) {
				path := durableTestPath(t)
				if err := os.WriteFile(path, payload, 0600); err != nil {
					t.Fatal(err)
				}
				c, err := NewDurableCoordinator(2, path)
				if err == nil {
					c.Close()
					t.Fatal("invalid stored authority accepted")
				}
			})
		}
	})
	t.Run("unsafe paths refuse without modifying target", func(t *testing.T) {
		for _, kind := range []string{"symlink", "insecure", "unavailable", "orphan-lock"} {
			t.Run(kind, func(t *testing.T) {
				path := durableTestPath(t)
				switch kind {
				case "symlink":
					target := path + "-target"
					if err := os.WriteFile(target, []byte("protected original"), 0600); err != nil {
						t.Fatal(err)
					}
					if err := os.Symlink(target, path); err != nil {
						t.Fatal(err)
					}
					defer func() {
						b, err := os.ReadFile(target)
						if err != nil || string(b) != "protected original" {
							t.Fatal("symlink target modified")
						}
					}()
				case "insecure":
					if err := os.WriteFile(path, []byte(`{"schema":"fak.researcharm.lease-store/v1","leases":[],"admission_ready":true}`), 0644); err != nil {
						t.Fatal(err)
					}
				case "unavailable":
					parent := filepath.Dir(path)
					if err := os.Remove(parent); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(parent, []byte("not directory"), 0600); err != nil {
						t.Fatal(err)
					}
				case "orphan-lock":
					if err := os.WriteFile(path+".lock", nil, 0600); err != nil {
						t.Fatal(err)
					}
				}
				c, err := NewDurableCoordinator(2, path)
				if err == nil {
					c.Close()
					t.Fatal("unsafe store path accepted")
				}
			})
		}
	})
}

// fak-test:runtime fast est=1ms lane=default
func TestDurableCoordinatorProcessHelper(t *testing.T) {
	path := os.Getenv("FAK_DURABLE_STORE_CHILD")
	if path == "" {
		return
	}
	c, err := NewDurableCoordinator(2, path)
	if err == nil {
		c.Close()
		t.Fatal("child acquired active durable authority")
	}
	fmt.Println("LOCK_REFUSED")
}

// fak-test:runtime fast est=1ms lane=default
func TestDurableCoordinatorSnapshotReportsAuthorityState(t *testing.T) {
	path := durableTestPath(t)
	c, err := NewDurableCoordinator(2, path)
	if err != nil {
		t.Fatal(err)
	}
	if got := c.Snapshot().DurableStatus; got != "bootstrap" {
		t.Fatalf("new authority status=%q, want bootstrap", got)
	}
	l := durableAcquire(t, c)
	if got := c.Snapshot().DurableStatus; got != "ready" {
		t.Fatalf("acknowledged authority status=%q, want ready", got)
	}
	c.store.renameFile = func(string, string) error { return errors.New("injected uncertainty") }
	if err := c.ReleaseLease(l.ID, l.Token); err == nil {
		t.Fatal("uncertain release acknowledged")
	}
	if got := c.Snapshot().DurableStatus; got != "unavailable" {
		t.Fatalf("poisoned authority status=%q, want unavailable", got)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if got := c.Snapshot().DurableStatus; got != "closed" {
		t.Fatalf("closed authority status=%q, want closed", got)
	}
	if got := NewCoordinator(2).Snapshot().DurableStatus; got != "" {
		t.Fatalf("memory-only coordinator falsely claimed durable status=%q", got)
	}
}
