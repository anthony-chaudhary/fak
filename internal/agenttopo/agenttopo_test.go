package agenttopo

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/comm"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestReady(t *testing.T) {
	if !Ready() {
		t.Fatal("Ready() should report true for the generated skeleton")
	}
}

func TestDeclareValidatesEndpoints(t *testing.T) {
	g := testGroup(t, "a", "b")
	if _, err := Declare("bad", g, []Edge{{From: "a", To: "z"}}); !errors.Is(err, ErrNoMember) {
		t.Fatalf("Declare unknown endpoint err=%v, want ErrNoMember", err)
	}
}

func TestDeclareRejectsCycles(t *testing.T) {
	g := testGroup(t, "a", "b", "c")
	_, err := Declare("cycle", g, []Edge{
		{From: "a", To: "b"},
		{From: "b", To: "c"},
		{From: "c", To: "a"},
	})
	if !errors.Is(err, ErrCycle) {
		t.Fatalf("Declare cycle err=%v, want ErrCycle", err)
	}
}

func TestNeighborExchangePreservesDeclarationOrderIntoCombine(t *testing.T) {
	g := testGroup(t, "a", "b", "c", "sink")
	topo, err := Declare("explicit", g, []Edge{
		{From: "b", To: "sink"},
		{From: "a", To: "sink"},
		{From: "c", To: "sink"},
	})
	if err != nil {
		t.Fatal(err)
	}
	ex, err := topo.NeighborExchange("sink")
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(ex.In); !reflect.DeepEqual(got, []string{"b", "a", "c"}) {
		t.Fatalf("in-neighbor order=%v, want declaration order [b a c]", got)
	}

	res, err := topo.CombineIn("sink", map[string]string{
		"a": "from-a",
		"b": "from-b",
		"c": "from-c",
	}, modelroute.ReduceConcat)
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "from-b\nfrom-a\nfrom-c" {
		t.Fatalf("CombineIn output=%q, want declaration-ordered concat", res.Output)
	}
}

func TestLinearAndStarConstructors(t *testing.T) {
	g := testGroup(t, "a", "b", "c")
	linear, err := Linear("line", g)
	if err != nil {
		t.Fatal(err)
	}
	if got := linear.Edges(); !reflect.DeepEqual(got, []Edge{{From: "a", To: "b"}, {From: "b", To: "c"}}) {
		t.Fatalf("linear edges=%v", got)
	}

	star, err := Star("star", g, "b")
	if err != nil {
		t.Fatal(err)
	}
	if got := star.Edges(); !reflect.DeepEqual(got, []Edge{{From: "b", To: "a"}, {From: "b", To: "c"}}) {
		t.Fatalf("star edges=%v", got)
	}
	if _, err := Star("star", g, "z"); !errors.Is(err, ErrNoMember) {
		t.Fatalf("star unknown root err=%v, want ErrNoMember", err)
	}
}

func TestLaneCount(t *testing.T) {
	g, err := comm.New("w", "", []comm.Member{
		{ID: "a", Lane: "l1"},
		{ID: "b", Lane: "l2"},
		{ID: "c", Lane: "l1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	topo, err := Declare("lanes", g, []Edge{{From: "a", To: "b"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := topo.LaneCount(); got != 2 {
		t.Fatalf("LaneCount=%d, want 2", got)
	}
}

func testGroup(t *testing.T, ids ...string) *comm.Group {
	t.Helper()
	ms := make([]comm.Member, len(ids))
	for i, id := range ids {
		ms[i] = comm.Member{ID: id}
	}
	g, err := comm.New("wave", "trace", ms)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func ids(ms []comm.Member) []string {
	out := make([]string, len(ms))
	for i, m := range ms {
		out[i] = m.ID
	}
	return out
}
