package agenttopo

import (
	"bytes"
	"encoding/json"
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

func TestAdmitExpansionBoundaries(t *testing.T) {
	baseGroup, err := comm.New("wave", "parent", []comm.Member{
		{ID: "root", Lane: "control", Weight: 2},
		{ID: "child", Lane: "worker", Weight: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	base, err := Declare("adaptive", baseGroup, []Edge{{From: "root", To: "child"}})
	if err != nil {
		t.Fatal(err)
	}
	baseBefore := topologyBytes(t, base)

	limits := ExpansionLimits{MaxDepth: 3, MaxWidth: 2, MaxTotalNodes: 4}
	proposal := ExpansionProposal{
		Nodes: []comm.Member{{ID: "beta", Lane: "qa"}, {ID: "alpha", Lane: "docs"}},
		Edges: []Edge{{From: "child", To: "beta"}, {From: "child", To: "alpha"}},
	}
	got, err := AdmitExpansion(base, proposal, limits)
	if err != nil {
		t.Fatalf("AdmitExpansion(valid) error = %v", err)
	}
	if got == base {
		t.Fatal("AdmitExpansion returned the mutable base pointer")
	}
	wantNodes := []comm.Member{
		{ID: "alpha", Lane: "docs"},
		{ID: "beta", Lane: "qa"},
		{ID: "child", Lane: "worker", Weight: 1},
		{ID: "root", Lane: "control", Weight: 2},
	}
	if nodes := got.Nodes(); !reflect.DeepEqual(nodes, wantNodes) {
		t.Fatalf("expanded nodes = %v, want %v", nodes, wantNodes)
	}
	wantEdges := []Edge{
		{From: "root", To: "child"},
		{From: "child", To: "alpha"},
		{From: "child", To: "beta"},
	}
	if edges := got.Edges(); !reflect.DeepEqual(edges, wantEdges) {
		t.Fatalf("expanded edges = %v, want %v", edges, wantEdges)
	}
	first, err := got.group.Membership(0)
	if err != nil || got.group.WaveID() != "wave" || first.ParentTraceID != "parent" {
		t.Fatalf("expanded group metadata = wave %q parent %q err %v", got.group.WaveID(), first.ParentTraceID, err)
	}
	permuted, err := AdmitExpansion(base, ExpansionProposal{
		Nodes: []comm.Member{{ID: "alpha", Lane: "docs"}, {ID: "beta", Lane: "qa"}},
		Edges: []Edge{{From: "child", To: "alpha"}, {From: "child", To: "beta"}},
	}, limits)
	if err != nil {
		t.Fatalf("AdmitExpansion(permuted) error = %v", err)
	}
	if a, b := topologyBytes(t, got), topologyBytes(t, permuted); !bytes.Equal(a, b) {
		t.Fatalf("permuted proposal changed output bytes:\n%s\n%s", a, b)
	}
	if after := topologyBytes(t, base); !bytes.Equal(after, baseBefore) {
		t.Fatalf("successful admission mutated base:\nbefore=%s\nafter=%s", baseBefore, after)
	}

	tests := []struct {
		name     string
		proposal ExpansionProposal
		limits   ExpansionLimits
		want     error
		limit    ExpansionLimit
	}{
		{
			name:     "existing identity rewrite",
			proposal: ExpansionProposal{Nodes: []comm.Member{{ID: "child", Lane: "rewritten"}}},
			limits:   limits,
			want:     ErrIdentityRewrite,
		},
		{
			name: "duplicate new identity",
			proposal: ExpansionProposal{Nodes: []comm.Member{
				{ID: "new", Lane: "one"},
				{ID: "new", Lane: "two"},
			}},
			limits: limits,
			want:   ErrDuplicateNode,
		},
		{
			name:     "empty new identity",
			proposal: ExpansionProposal{Nodes: []comm.Member{{ID: ""}}},
			limits:   limits,
			want:     ErrInvalidNodeID,
		},
		{
			name: "unknown dependency endpoint",
			proposal: ExpansionProposal{
				Nodes: []comm.Member{{ID: "new"}},
				Edges: []Edge{{From: "missing", To: "new"}},
			},
			limits: limits,
			want:   ErrNoMember,
		},
		{
			name: "cycle",
			proposal: ExpansionProposal{
				Nodes: []comm.Member{{ID: "new"}},
				Edges: []Edge{{From: "child", To: "new"}, {From: "new", To: "root"}},
			},
			limits: ExpansionLimits{MaxDepth: 8, MaxWidth: 8, MaxTotalNodes: 8},
			want:   ErrCycle,
		},
		{
			name: "depth limit",
			proposal: ExpansionProposal{
				Nodes: []comm.Member{{ID: "new"}},
				Edges: []Edge{{From: "child", To: "new"}},
			},
			limits: ExpansionLimits{MaxDepth: 2, MaxWidth: 4, MaxTotalNodes: 4},
			want:   ErrDepthLimit,
			limit:  LimitDepth,
		},
		{
			name:     "width limit",
			proposal: proposal,
			limits:   ExpansionLimits{MaxDepth: 4, MaxWidth: 1, MaxTotalNodes: 4},
			want:     ErrWidthLimit,
			limit:    LimitWidth,
		},
		{
			name:     "total node limit",
			proposal: proposal,
			limits:   ExpansionLimits{MaxDepth: 4, MaxWidth: 4, MaxTotalNodes: 3},
			want:     ErrTotalNodesLimit,
			limit:    LimitTotalNodes,
		},
		{
			name:     "undeclared limit",
			proposal: ExpansionProposal{Nodes: []comm.Member{{ID: "new"}}},
			limits:   ExpansionLimits{MaxDepth: 0, MaxWidth: 4, MaxTotalNodes: 4},
			want:     ErrInvalidLimits,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := AdmitExpansion(base, tt.proposal, tt.limits)
			if !errors.Is(err, tt.want) {
				t.Fatalf("AdmitExpansion error = %v, want %v", err, tt.want)
			}
			if tt.limit != "" {
				var limitErr *ExpansionLimitError
				if !errors.As(err, &limitErr) || limitErr.Limit != tt.limit {
					t.Fatalf("AdmitExpansion limit error = %#v, want %q", limitErr, tt.limit)
				}
			}
			if after := topologyBytes(t, base); !bytes.Equal(after, baseBefore) {
				t.Fatalf("refused admission mutated base:\nbefore=%s\nafter=%s", baseBefore, after)
			}
		})
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

func topologyBytes(t *testing.T, topo *Topology) []byte {
	t.Helper()
	b, err := json.Marshal(struct {
		Name  string        `json:"name"`
		Nodes []comm.Member `json:"nodes"`
		Edges []Edge        `json:"edges"`
	}{
		Name:  topo.Name(),
		Nodes: topo.Nodes(),
		Edges: topo.Edges(),
	})
	if err != nil {
		t.Fatal(err)
	}
	return b
}
