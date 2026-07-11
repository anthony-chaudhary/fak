package gateway

import (
	"strings"
	"testing"
)

// replicaNames builds a proxy fleet from the given base URLs and returns each
// replica's identity in registry (round-robin) order. It fails the test if the
// planner is not the multi-replica router.
func replicaNames(t *testing.T, urls ...string) []string {
	t.Helper()
	p, err := newProxyPlanner(Config{Provider: "openai"}, "fleet-model", urls)
	if err != nil {
		t.Fatalf("newProxyPlanner(%v): %v", urls, err)
	}
	rr, ok := p.(*ReplicaRouter)
	if !ok {
		t.Fatalf("planner = %T, want *ReplicaRouter", p)
	}
	names := make([]string, 0, len(urls))
	for _, r := range rr.Replicas() {
		names = append(names, r.Name)
	}
	return names
}

// AC1: same URL set in different flag order yields identical replica names —
// identity attaches to the endpoint, not the flag position.
func TestReplicaNamesOrderIndependent(t *testing.T) {
	a := "http://10.0.0.1:8001/v1"
	b := "http://10.0.0.2:8002/v1"
	c := "http://10.0.0.3:8003/v1"

	fwd := replicaNames(t, a, b, c)
	rev := replicaNames(t, c, b, a)

	// Per-endpoint identity is stable across the reorder (fwd[i] pairs with rev[len-1-i]).
	if fwd[0] != rev[2] || fwd[1] != rev[1] || fwd[2] != rev[0] {
		t.Fatalf("names are not endpoint-stable across reorder: fwd=%v rev=%v", fwd, rev)
	}
	// Positional replica-N naming is gone: a reorder used to relabel every survivor.
	for _, n := range append(append([]string{}, fwd...), rev...) {
		if n == "replica-1" || n == "replica-2" || n == "replica-3" {
			t.Fatalf("got positional name %q, want endpoint-derived identity", n)
		}
	}
}

// AC2: removing one URL leaves the other replicas' names (and thus metric labels
// and residency slots) unchanged.
func TestReplicaNamesSurviveRemoval(t *testing.T) {
	a := "http://10.0.0.1:8001/v1"
	b := "http://10.0.0.2:8002/v1"
	c := "http://10.0.0.3:8003/v1"

	three := replicaNames(t, a, b, c) // a, b, c
	two := replicaNames(t, a, c)      // a, c — b dropped

	if three[0] != two[0] {
		t.Fatalf("replica a renamed by dropping a peer: %q -> %q", three[0], two[0])
	}
	if three[2] != two[1] {
		t.Fatalf("replica c renamed by dropping a peer: %q -> %q", three[2], two[1])
	}
}

// AC3: name=URL parses to the operator-chosen id; a bare URL keeps the derived name.
func TestReplicaNameExplicitAndDerived(t *testing.T) {
	names := replicaNames(t, "primary=http://10.0.0.1:8001/v1", "http://10.0.0.2:8002/v1")
	if names[0] != "primary" {
		t.Fatalf("operator-chosen id: got %q, want %q", names[0], "primary")
	}
	if want := deriveReplicaName("http://10.0.0.2:8002/v1"); names[1] != want {
		t.Fatalf("bare URL identity: got %q, want derived %q", names[1], want)
	}
	if !strings.HasPrefix(names[1], "replica-") {
		t.Fatalf("derived identity %q lacks replica- prefix", names[1])
	}
}

func TestParseReplicaEntry(t *testing.T) {
	cases := []struct {
		raw, wantName, wantURL string
	}{
		{"http://h:1/v1", "", "http://h:1/v1"},
		{"nm=http://h:1/v1", "nm", "http://h:1/v1"},
		// '=' inside a query string must not be read as a name split.
		{"http://h:1/v1?k=v", "", "http://h:1/v1?k=v"},
		{"  spaced = http://h:1/v1  ", "spaced", "http://h:1/v1"},
		// A leading '=' (empty name) is not a name; treat the whole thing as the URL.
		{"=http://h:1/v1", "", "=http://h:1/v1"},
	}
	for _, c := range cases {
		gotName, gotURL := parseReplicaEntry(c.raw)
		if gotName != c.wantName || gotURL != c.wantURL {
			t.Errorf("parseReplicaEntry(%q) = (%q, %q), want (%q, %q)", c.raw, gotName, gotURL, c.wantName, c.wantURL)
		}
	}
}

func TestDeriveReplicaNameEndpointScoped(t *testing.T) {
	n := deriveReplicaName("http://h:1/v1")
	// Deterministic.
	if n != deriveReplicaName("http://h:1/v1") {
		t.Fatal("deriveReplicaName is not deterministic")
	}
	// Endpoint-scoped: identity is scheme://host:port, path-independent.
	if n != deriveReplicaName("http://h:1/other") {
		t.Fatalf("identity should be path-independent: %q vs %q", n, deriveReplicaName("http://h:1/other"))
	}
	// A different endpoint gets a different identity.
	if n == deriveReplicaName("http://h:2/v1") {
		t.Fatal("distinct endpoints collided on one identity")
	}
	// Shape: replica- plus 6 hex chars.
	if !strings.HasPrefix(n, "replica-") || len(n) != len("replica-")+6 {
		t.Fatalf("derived name %q is not replica-<6 hex>", n)
	}
}
