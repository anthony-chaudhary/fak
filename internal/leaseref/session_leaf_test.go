package leaseref

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionref"
)

// fak-test:runtime fast est=20ms lane=default
func TestSessionPublishingLeafCompatibility(t *testing.T) {
	// The legacy defined type retains its name/package identity, wire layout and value methods.
	leaf := sessionref.SessionDescriptor{ID: "worker", Host: "host", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 10, AgentUUID: "uuid"}
	legacy := SessionDescriptor(leaf)
	if typ := reflect.TypeOf(legacy); typ.PkgPath() != "github.com/anthony-chaudhary/fak/internal/leaseref" || typ.Name() != "SessionDescriptor" {
		t.Fatalf("legacy type identity changed: %v", typ)
	}
	legacyJSON, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	leafJSON, err := json.Marshal(leaf)
	if err != nil {
		t.Fatal(err)
	}
	if string(legacyJSON) != string(leafJSON) {
		t.Fatalf("descriptor JSON differs: %s / %s", legacyJSON, leafJSON)
	}
	if legacy.Ref() != "refs/fak/locks/session-worker" || legacy.Expired(time.Unix(109, 0)) || !legacy.Expired(time.Unix(110, 0)) {
		t.Fatal("ref or TTL boundary changed")
	}
	for _, ttl := range []int64{0, -1} {
		legacy.TTLSecs = ttl
		if legacy.Expired(time.Unix(1000, 0)) {
			t.Fatal("nonpositive TTL must not expire")
		}
	}
	type result struct {
		ref, err string
		calls    []string
		blobs    []string
	}
	for _, tc := range []struct {
		name, id, oid, fail string
		real                bool
	}{
		{name: "injected", id: "worker"},
		{name: "real", id: "worker", real: true},
		{name: "no-object-db", id: "worker", fail: "probe", real: true},
		{name: "missing-git", id: "worker", fail: "exec", real: true},
		{name: "injected-missing-git", id: "worker", fail: "exec"},
		{name: "hash-exit", id: "worker", fail: "hash"},
		{name: "update-exit", id: "worker", fail: "update"},
		{name: "zero-object", id: "worker", oid: strings.Repeat("0", 40)},
		{name: "sha256", id: "worker", oid: strings.Repeat("A", 64)},
		{name: "empty"}, {name: "double-prefix", id: "session-worker"},
		{name: "escape", id: "../main"}, {name: "ref-special", id: "bad@{id"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			exercise := func(useLeaf bool) result {
				var got result
				run := func(_ context.Context, dir string, args ...string) (string, int, error) {
					if dir != "fixture" {
						t.Fatalf("directory changed: %q", dir)
					}
					if args[0] == "hash-object" {
						if len(args) != 3 || args[1] != "-w" {
							t.Fatalf("hash argv: %q", args)
						}
						blob, err := os.ReadFile(args[2])
						if err != nil {
							t.Fatal(err)
						}
						got.blobs = append(got.blobs, string(blob))
						got.calls = append(got.calls, "hash-object -w <temp>")
					} else {
						got.calls = append(got.calls, strings.Join(args, " "))
					}
					if tc.fail == "exec" {
						return "", -1, errors.New("fixture unavailable")
					}
					switch args[0] {
					case "rev-parse":
						if tc.fail == "probe" {
							return "", 128, nil
						}
						return ".git", 0, nil
					case "hash-object":
						if tc.fail == "hash" {
							return "", 7, nil
						}
						oid := tc.oid
						if oid == "" {
							oid = strings.Repeat("a", 40)
						}
						return oid + "\n", 0, nil
					case "update-ref":
						if tc.fail == "update" {
							return "", 9, nil
						}
						return "", 0, nil
					default:
						t.Fatalf("unexpected git mutation: %q", args)
						return "", -1, nil
					}
				}
				store := NewWithRunner(run, "fixture")
				store.realGit = tc.real
				var once sync.Once
				var off bool
				d := leaf
				d.ID = tc.id
				// Repeated publication witnesses the cached real-Git probe, not a new probe per call.
				for range 2 {
					var err error
					if useLeaf {
						got.ref, err = sessionref.PublishSession(context.Background(), d, func(ctx context.Context, ref string, v any) (string, error) {
							disabled := sessionref.PublishingDisabled(ctx, run, "fixture", tc.real, &once, &off)
							return sessionref.PutBlobRef(ctx, run, "fixture", ref, v, disabled)
						})
					} else {
						got.ref, err = store.PublishSession(context.Background(), SessionDescriptor(d))
					}
					if err != nil {
						got.err = err.Error()
					}
				}
				return got
			}
			old, new := exercise(false), exercise(true)
			if !reflect.DeepEqual(old, new) {
				t.Fatalf("legacy=%+v leaf=%+v", old, new)
			}
			wantErr := ""
			switch {
			case !sessionref.ValidSessionID(tc.id):
				wantErr = fmt.Sprintf("leaseref: invalid session id %q (must be one safe ref segment)", tc.id)
			case tc.fail == "exec" && !tc.real:
				wantErr = "leaseref: git not executable: fixture unavailable"
			case tc.fail == "hash":
				wantErr = "leaseref: hash-object exited 7"
			case tc.fail == "update":
				wantErr = "leaseref: update-ref refs/fak/locks/session-worker exited 9"
			case tc.name == "zero-object":
				wantErr = fmt.Sprintf("leaseref: hash-object produced invalid object id %q", tc.oid)
			}
			if new.err != wantErr {
				t.Fatalf("error=%q want %q", new.err, wantErr)
			}
			if wantErr != "" && new.ref != "" {
				t.Fatalf("failed publication returned ref %q", new.ref)
			}
			if !sessionref.ValidSessionID(tc.id) {
				if len(new.calls) != 0 || new.err == "" || new.ref != "" {
					t.Fatalf("invalid ID reached Git: %+v", new)
				}
				return
			}
			if tc.real {
				probes := 0
				for _, call := range new.calls {
					if call == "rev-parse --git-dir" {
						probes++
					}
				}
				if probes != 1 {
					t.Fatalf("probe count=%d", probes)
				}
			}
			if tc.real && (tc.fail == "probe" || tc.fail == "exec") {
				if new.err != "" || len(new.blobs) != 0 || new.ref != leaf.Ref() {
					t.Fatalf("nonrepo policy changed: %+v", new)
				}
			}
			if tc.name == "injected" {
				want := `{"id":"worker","host":"host","pcb_state":"RUNNING","updated_at":100,"ttl_seconds":10,"agent_uuid":"uuid"}`
				if len(new.blobs) != 2 || new.blobs[0] != want || new.ref != leaf.Ref() || new.err != "" {
					t.Fatalf("wire or ref changed: %+v", new)
				}
			}
		})
	}
	// Unset timestamp is defaulted in the published value, not in the caller's copy.
	d := SessionDescriptor{ID: "worker"}
	before := time.Now().Unix()
	_, err = sessionref.PublishSession(context.Background(), sessionref.SessionDescriptor(d), func(_ context.Context, _ string, v any) (string, error) {
		stamp := v.(sessionref.SessionDescriptor).UpdatedAt
		if stamp < before || stamp > time.Now().Unix() {
			t.Fatalf("default timestamp=%d", stamp)
		}
		return d.Ref(), nil
	})
	if err != nil || d.UpdatedAt != 0 {
		t.Fatalf("timestamp default changed caller: %+v %v", d, err)
	}
}
