package gateway

// leaseplane_test.go drives the multi-node dev-server READ plane (#2297) end-to-end:
// the handlers behind GET /v1/leases and GET /v1/sessions, with providers built over a
// real leaseref.Store whose Runner is injected canned git evidence — no real git, no
// repo — asserting the served JSON shapes. The provider closures here mirror the
// cmd/fak wiring (leaseplane_endpoint.go) so the tested seam is the shipped seam.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// fakeLockRunner scripts the two git reads the lease/presence views perform:
// `for-each-ref --format=%(refname) refs/fak/locks/` (the namespace scan) and
// `cat-file blob <ref>` (one record read). refs maps full ref name -> blob JSON.
func fakeLockRunner(refs map[string]string) leaseref.Runner {
	return func(_ context.Context, _ string, args ...string) (string, int, error) {
		switch args[0] {
		case "for-each-ref":
			names := make([]string, 0, len(refs))
			for ref := range refs {
				names = append(names, ref)
			}
			sort.Strings(names)
			return strings.Join(names, "\n"), 0, nil
		case "cat-file":
			if blob, ok := refs[args[2]]; ok {
				return blob, 0, nil
			}
			return "", 1, nil
		case "rev-parse":
			if len(args) >= 3 && args[1] == "--verify" {
				if _, ok := refs[args[len(args)-1]]; ok {
					return "deadbeef", 0, nil
				}
				return "", 1, nil
			}
			return "", 1, nil
		}
		return "", 1, nil
	}
}

// installLeasePlane wires providers over the injected store — the same closures
// cmd/fak installs — and restores the package state when the test ends, so the
// package-level providers never leak between tests.
func installLeasePlane(t *testing.T, store *leaseref.Store) {
	t.Helper()
	SetLeasePlaneProviders(
		func(ctx context.Context) (LeasePlaneView, error) {
			now := time.Now()
			leases, err := store.LiveLeases(ctx, now)
			if err != nil {
				return LeasePlaneView{}, err
			}
			raw, err := json.Marshal(leases)
			if err != nil {
				return LeasePlaneView{}, err
			}
			return LeasePlaneView{ObservedUnix: now.Unix(), LiveLeases: raw}, nil
		},
		func(ctx context.Context) (LeasePresenceView, error) {
			now := time.Now()
			live, _, err := store.LiveSessions(ctx, now)
			if err != nil {
				return LeasePresenceView{}, err
			}
			if live == nil {
				live = []leaseref.SessionDescriptor{}
			}
			classified, err := store.ClassifyLive(ctx, "", now)
			if err != nil {
				return LeasePresenceView{}, err
			}
			rawSessions, err := json.Marshal(live)
			if err != nil {
				return LeasePresenceView{}, err
			}
			rawClassified, err := json.Marshal(classified)
			if err != nil {
				return LeasePresenceView{}, err
			}
			return LeasePresenceView{ObservedUnix: now.Unix(), Sessions: rawSessions, ClassifiedLeases: rawClassified}, nil
		},
	)
	t.Cleanup(func() { SetLeasePlaneProviders(nil, nil) })
}

func TestLeaseReadPlaneUnconfiguredIs404(t *testing.T) {
	SetLeasePlaneProviders(nil, nil)
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	for _, path := range []string{"/v1/leases", "/v1/sessions"} {
		r, err := http.Get(ts.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		r.Body.Close()
		if r.StatusCode != http.StatusNotFound {
			t.Errorf("GET %s unconfigured status = %d, want 404 (never a silent empty reading)", path, r.StatusCode)
		}
	}
}

func TestLeaseReadPlaneIsGetOnly(t *testing.T) {
	installLeasePlane(t, leaseref.NewWithRunner(fakeLockRunner(nil), ""))
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	for _, path := range []string{"/v1/leases", "/v1/sessions"} {
		r, err := http.Post(ts.URL+path, "application/json", strings.NewReader(`{}`))
		if err != nil {
			t.Fatal(err)
		}
		r.Body.Close()
		if r.StatusCode != http.StatusMethodNotAllowed {
			t.Errorf("POST %s status = %d, want 405 (read plane is read-only; the write side is #2299)", path, r.StatusCode)
		}
	}
}

func TestLeaseReadPlaneServesObservedLiveLeases(t *testing.T) {
	refs := map[string]string{
		"refs/fak/locks/gateway": `{"id":"gateway","tree_globs":["internal/gateway/**"],"holder":"nodeA/guard-1","acquired_unix":1000,"ttl_seconds":0,"generation":2,"session_id":"guard-1"}`,
		// An EXPIRED lease must be dropped from the live view (reapable, not blocking).
		"refs/fak/locks/docs": `{"id":"docs","tree_globs":["docs/**"],"holder":"nodeB/guard-9","acquired_unix":100,"ttl_seconds":10}`,
		// A session descriptor must never appear as a lock lease (the namespace split).
		"refs/fak/locks/session-guard-1": `{"id":"guard-1","host":"nodeA","pcb_state":"RUNNING","updated_at":1000,"ttl_seconds":0}`,
	}
	installLeasePlane(t, leaseref.NewWithRunner(fakeLockRunner(refs), ""))
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Get(ts.URL + "/v1/leases")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/leases status = %d, want 200", r.StatusCode)
	}
	var resp struct {
		ObservedUnix int64  `json:"observed_unix"`
		Source       string `json:"source"`
		LiveLeases   []struct {
			Lane     string   `json:"lane"`
			LaneKind string   `json:"lane_kind"`
			Tree     []string `json:"tree"`
		} `json:"live_leases"`
	}
	if err := json.NewDecoder(r.Body).Decode(&resp); err != nil {
		t.Fatalf("decode /v1/leases: %v", err)
	}
	if resp.ObservedUnix <= 0 {
		t.Errorf("observed_unix = %d, want a positive read instant", resp.ObservedUnix)
	}
	if !strings.Contains(strings.ToLower(resp.Source), "observed") {
		t.Errorf("source = %q, want the OBSERVED qualifier (values are read, not client-claimed)", resp.Source)
	}
	if len(resp.LiveLeases) != 1 {
		t.Fatalf("live_leases = %+v, want exactly the one live lock lease (expired dropped, session refs filtered)", resp.LiveLeases)
	}
	got := resp.LiveLeases[0]
	if got.Lane != "gateway" || got.LaneKind != "cluster" || len(got.Tree) != 1 || got.Tree[0] != "internal/gateway/**" {
		t.Errorf("live lease row = %+v, want {lane:gateway lane_kind:cluster tree:[internal/gateway/**]} (the dos_arbitrate live_leases shape)", got)
	}
}

func TestLeaseSessionsServePresenceAndLiveness(t *testing.T) {
	refs := map[string]string{
		// A lease whose owning session heartbeats (TTL 0 never expires) -> peer-live.
		"refs/fak/locks/gateway":         `{"id":"gateway","tree_globs":["internal/gateway/**"],"holder":"nodeA/guard-1","acquired_unix":1000,"ttl_seconds":0,"generation":2,"session_id":"guard-1"}`,
		"refs/fak/locks/session-guard-1": `{"id":"guard-1","host":"nodeA","pcb_state":"RUNNING","updated_at":1000,"ttl_seconds":0}`,
		// A lease whose owning session's heartbeat lapsed -> positively dead, reclaimable.
		"refs/fak/locks/docs":            `{"id":"docs","tree_globs":["docs/**"],"holder":"nodeB/guard-2","acquired_unix":1000,"ttl_seconds":0,"generation":1,"session_id":"guard-2"}`,
		"refs/fak/locks/session-guard-2": `{"id":"guard-2","host":"nodeB","pcb_state":"RUNNING","updated_at":100,"ttl_seconds":10}`,
	}
	installLeasePlane(t, leaseref.NewWithRunner(fakeLockRunner(refs), ""))
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Get(ts.URL + "/v1/sessions")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/sessions status = %d, want 200", r.StatusCode)
	}
	var resp struct {
		ObservedUnix int64  `json:"observed_unix"`
		Source       string `json:"source"`
		Sessions     []struct {
			ID   string `json:"id"`
			Host string `json:"host"`
		} `json:"sessions"`
		ClassifiedLeases []struct {
			ID          string `json:"id"`
			Liveness    string `json:"liveness"`
			Reclaimable bool   `json:"reclaimable"`
		} `json:"classified_leases"`
	}
	if err := json.NewDecoder(r.Body).Decode(&resp); err != nil {
		t.Fatalf("decode /v1/sessions: %v", err)
	}
	if !strings.Contains(strings.ToLower(resp.Source), "observed") {
		t.Errorf("source = %q, want the OBSERVED qualifier", resp.Source)
	}
	// Only guard-1 is live; guard-2's heartbeat lapsed (updated_at 100 + ttl 10 << now).
	if len(resp.Sessions) != 1 || resp.Sessions[0].ID != "guard-1" || resp.Sessions[0].Host != "nodeA" {
		t.Errorf("sessions = %+v, want exactly the heartbeating guard-1 on nodeA", resp.Sessions)
	}
	byID := map[string]struct {
		Liveness    string
		Reclaimable bool
	}{}
	for _, c := range resp.ClassifiedLeases {
		byID[c.ID] = struct {
			Liveness    string
			Reclaimable bool
		}{c.Liveness, c.Reclaimable}
	}
	if got := byID["gateway"]; got.Liveness != leaseref.LivenessPeerLive || got.Reclaimable {
		t.Errorf("gateway lease = %+v, want peer-live and never reclaimable", got)
	}
	if got := byID["docs"]; got.Liveness != leaseref.LivenessPeerDead || !got.Reclaimable {
		t.Errorf("docs lease = %+v, want peer-dead (lapsed heartbeat) and reclaimable", got)
	}
}

func TestLeaseReadPlaneEmptyViewIsArraysNotNull(t *testing.T) {
	installLeasePlane(t, leaseref.NewWithRunner(fakeLockRunner(nil), ""))
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	for path, keys := range map[string][]string{
		"/v1/leases":   {"live_leases"},
		"/v1/sessions": {"sessions", "classified_leases"},
	} {
		r, err := http.Get(ts.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		var raw map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
			t.Fatalf("decode %s: %v", path, err)
		}
		r.Body.Close()
		for _, k := range keys {
			if string(raw[k]) != "[]" {
				t.Errorf("%s %s = %s, want [] (an empty view is an empty array, never null)", path, k, raw[k])
			}
		}
	}
}
