package supportgraph

import (
	"os"
	"testing"
	"time"
)

func TestSupportStatesAndAuthority(t *testing.T) {
	raw, _ := os.ReadFile("testdata/awq.json")
	g, e := Parse(raw)
	if e != nil {
		t.Fatal(e)
	}
	at := time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC)
	a := Query(g, g.Edges[0].Tuple, at)
	if a.State != Supported || a.Decisive[0].Tier != Witnessed {
		t.Fatalf("allow=%+v", a)
	}
	u := Query(g, g.Edges[1].Tuple, at)
	if u.State != Unsupported || u.Decisive[0].ID != "device-refusal" || u.Fallback == "" {
		t.Fatalf("unsupported=%+v", u)
	}
	s := Query(g, g.Edges[2].Tuple, at)
	if s.State != Stale {
		t.Fatalf("stale=%+v", s)
	}
	q := g.Edges[0].Tuple
	q.Layout = "unknown"
	if got := Query(g, q, at); got.State != Unknown {
		t.Fatalf("unknown=%+v", got)
	}
}
func TestEqualTierConflict(t *testing.T) {
	e := Evidence{ID: "a", State: Supported, Tier: Witnessed, Authority: "a", Source: "a"}
	f := e
	f.ID = "b"
	f.State = Unsupported
	g := Graph{Schema: Schema, Edges: []Edge{{Tuple: Tuple{Artifact: "a"}, Evidence: []Evidence{e, f}}}}
	if got := Query(g, g.Edges[0].Tuple, time.Now()); got.State != Conflict {
		t.Fatalf("%+v", got)
	}
}
func TestArtifactRuntimeBridge(t *testing.T) {
	q := ArtifactRuntimeRequest{Artifact: "a", Quant: "q", Hardware: "h"}
	if q.Tuple().Quant != "q" || q.Tuple().Hardware != "h" {
		t.Fatal(q.Tuple())
	}
}
