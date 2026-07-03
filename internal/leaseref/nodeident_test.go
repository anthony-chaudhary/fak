package leaseref

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestNodeIdentity is the #2304 witness: the <node-id>/<session-id> holder
// convention mints, parses, tolerates every legacy free-form holder without
// erroring (classifying NodeUnknown), and binds a hostname to its stable
// hardware-catalog machine id.
func TestNodeIdentity(t *testing.T) {
	t.Run("mint parse round trip", func(t *testing.T) {
		h := MintHolder("desktop", "sess-abc123")
		if h != "desktop/sess-abc123" {
			t.Fatalf("MintHolder = %q, want desktop/sess-abc123", h)
		}
		id := ParseHolder(h)
		if !id.Structured() || id.Node != "desktop" || id.Session != "sess-abc123" || id.Raw != h {
			t.Fatalf("ParseHolder(%q) = %+v, want structured {desktop sess-abc123}", h, id)
		}
	})

	t.Run("mint sanitizes so it always parses back structured", func(t *testing.T) {
		h := MintHolder("AMD Ryzen Desktop!", "sess:42")
		id := ParseHolder(h)
		if !id.Structured() || id.Session == "" {
			t.Fatalf("minted holder %q does not parse back structured: %+v", h, id)
		}
	})

	t.Run("mint degrades empty components to placeholders, never fails", func(t *testing.T) {
		h := MintHolder("", "")
		if h != NodeUnknown+"/unknown" {
			t.Fatalf("MintHolder(\"\", \"\") = %q, want %s/unknown", h, NodeUnknown)
		}
		if id := ParseHolder(h); id.Structured() {
			t.Fatalf("placeholder mint %q must classify node-unknown, got %+v", h, id)
		}
	})

	t.Run("legacy free-form holders classify node-unknown and never error", func(t *testing.T) {
		for _, legacy := range []string{
			"",    // anonymous
			"A:1", // the historic host:pid shape
			"gone-host:dead-sess",
			"just-a-name",
			"a/b/c",        // too many segments
			"/leading",     // empty node segment
			"trailing/",    // empty session segment
			"bad seg/sess", // whitespace in a segment
		} {
			id := ParseHolder(legacy)
			if id.Structured() || id.Node != NodeUnknown || id.Session != "" {
				t.Fatalf("ParseHolder(%q) = %+v, want {node-unknown, no session}", legacy, id)
			}
			if id.Raw != legacy {
				t.Fatalf("ParseHolder(%q) lost the raw holder: %+v", legacy, id)
			}
		}
	})

	t.Run("record holder node convenience", func(t *testing.T) {
		if n := (Record{Holder: "desktop/sess-1"}).HolderNode(); n != "desktop" {
			t.Fatalf("HolderNode = %q, want desktop", n)
		}
		if n := (Record{Holder: "A:1"}).HolderNode(); n != NodeUnknown {
			t.Fatalf("legacy HolderNode = %q, want %s", n, NodeUnknown)
		}
	})

	t.Run("resolve node id against the catalog", func(t *testing.T) {
		catalog := map[string]string{"amd-ryzen-desktop": "desktop", "cpu-server": "cpu-server-a"}
		if got := ResolveNodeID("AMD-RYZEN-DESKTOP", catalog); got != "desktop" {
			t.Fatalf("cataloged hostname resolved %q, want desktop (case-insensitive)", got)
		}
		if got := ResolveNodeID("uncataloged-box", catalog); got != "uncataloged-box" {
			t.Fatalf("uncataloged hostname resolved %q, want the sanitized hostname itself", got)
		}
		if got := ResolveNodeID("", catalog); got != NodeUnknown {
			t.Fatalf("blank hostname resolved %q, want %s", got, NodeUnknown)
		}
	})

	t.Run("load catalog node ids from the machines table", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "catalog.json")
		blob := `{"machines":{"desktop":{"hostname":"amd-ryzen-desktop","gpu":"AMD"},"a100":{"hostname":"Datacenter-GPU-Server"},"blank":{"hostname":""}}}`
		if err := os.WriteFile(path, []byte(blob), 0o644); err != nil {
			t.Fatalf("write catalog fixture: %v", err)
		}
		m, err := LoadCatalogNodeIDs(path)
		if err != nil {
			t.Fatalf("LoadCatalogNodeIDs: %v", err)
		}
		if m["amd-ryzen-desktop"] != "desktop" || m["datacenter-gpu-server"] != "a100" {
			t.Fatalf("catalog map = %v, want hostname(lowercased)->machine-id", m)
		}
		if _, ok := m[""]; ok {
			t.Fatalf("catalog map admitted a blank hostname: %v", m)
		}
		if got := ResolveNodeID("amd-ryzen-desktop", m); got != "desktop" {
			t.Fatalf("catalog-bound resolve = %q, want desktop", got)
		}
	})

	t.Run("local node id is best-effort and never empty", func(t *testing.T) {
		// No catalog under a temp root: falls back to the sanitized hostname
		// (or NodeUnknown on a host with no name) — never "" and never an error.
		if got := LocalNodeID(t.TempDir()); got == "" {
			t.Fatalf("LocalNodeID returned empty; want sanitized hostname or %s", NodeUnknown)
		}
	})
}

// TestNodeIdentitySurfacedInLiveness proves the classified (liveness) projection
// carries the node component: a conventional holder surfaces its machine id and a
// legacy holder surfaces NodeUnknown — the `fak leaseref liveness` acceptance
// rung of #2304, driven on the fake git seam.
func TestNodeIdentitySurfacedInLiveness(t *testing.T) {
	fake := newFakeGit()
	s := NewWithRunner(fake.run, "")
	ctx := context.Background()
	now := time.Unix(10_000, 0)

	for _, r := range []Record{
		{ID: "lane-node", TreeGlobs: []string{"a/**"}, Holder: "desktop/sess-1", SessionID: "sess-1", AcquiredAt: 9_000, TTLSeconds: 3600},
		{ID: "lane-legacy", TreeGlobs: []string{"b/**"}, Holder: "B:2", AcquiredAt: 9_000, TTLSeconds: 3600},
	} {
		if _, err := s.Acquire(ctx, r); err != nil {
			t.Fatalf("Acquire %s: %v", r.ID, err)
		}
	}
	rows, err := s.ClassifyLive(ctx, "", now)
	if err != nil {
		t.Fatalf("ClassifyLive: %v", err)
	}
	got := map[string]ClassifiedLease{}
	for _, r := range rows {
		got[r.ID] = r
	}
	if got["lane-node"].Node != "desktop" {
		t.Fatalf("conventional holder surfaced node %q, want desktop", got["lane-node"].Node)
	}
	if got["lane-legacy"].Node != NodeUnknown {
		t.Fatalf("legacy holder surfaced node %q, want %s", got["lane-legacy"].Node, NodeUnknown)
	}
}
