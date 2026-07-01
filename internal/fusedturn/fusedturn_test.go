package fusedturn_test

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/fusedturn"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

func TestClassifyReadsDeclaration(t *testing.T) {
	cases := []struct {
		name string
		call *abi.ToolCall
		want fusedturn.OpClass
	}{
		{"classical constructor", fusedturn.Classical("git_commit", abi.Ref{}), fusedturn.ClassClassical},
		{"weight constructor", fusedturn.Weight("glm-5.2", "chat_completion", abi.Ref{}), fusedturn.ClassWeight},
		{"undeclared fails closed", &abi.ToolCall{Tool: "bash"}, fusedturn.ClassUnknown},
		{"empty meta fails closed", &abi.ToolCall{Tool: "bash", Meta: map[string]string{}}, fusedturn.ClassUnknown},
		{"unrecognized token fails closed", &abi.ToolCall{Tool: "x", Meta: map[string]string{fusedturn.MetaClassKey: "mystery"}}, fusedturn.ClassUnknown},
		{"nil call", nil, fusedturn.ClassUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := fusedturn.Classify(tc.call); got != tc.want {
				t.Fatalf("Classify = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestConstructorsStampTheDeclaration(t *testing.T) {
	c := fusedturn.Classical("git_commit", abi.Ref{})
	if c.Meta[fusedturn.MetaClassKey] != "classical" {
		t.Errorf("Classical did not stamp the meta tag: %v", c.Meta)
	}
	if c.Engine != "" {
		t.Errorf("Classical set an engine route %q, want the kernel default (empty)", c.Engine)
	}
	w := fusedturn.Weight("glm-5.2", "chat_completion", abi.Ref{})
	if w.Meta[fusedturn.MetaClassKey] != "weight" {
		t.Errorf("Weight did not stamp the meta tag: %v", w.Meta)
	}
	if w.Engine != "glm-5.2" {
		t.Errorf("Weight engine = %q, want glm-5.2", w.Engine)
	}
	// Tag(ClassUnknown) clears the declaration (fail-closed round-trip).
	fusedturn.Tag(w, fusedturn.ClassUnknown)
	if _, ok := w.Meta[fusedturn.MetaClassKey]; ok {
		t.Errorf("Tag(ClassUnknown) did not clear the declaration: %v", w.Meta)
	}
	if got := fusedturn.Classify(w); got != fusedturn.ClassUnknown {
		t.Errorf("cleared call classifies %v, want ClassUnknown", got)
	}
}

func TestFusedRequiresBothFamilies(t *testing.T) {
	classical := fusedturn.Classical("git_commit", abi.Ref{})
	weight := fusedturn.Weight("glm-5.2", "chat_completion", abi.Ref{})
	unknown := &abi.ToolCall{Tool: "bash"}

	cases := []struct {
		name      string
		calls     []*abi.ToolCall
		wantFused bool
		wantC     int
		wantW     int
		wantU     int
	}{
		{"classical only is a normal turn", []*abi.ToolCall{classical, classical}, false, 2, 0, 0},
		{"weight only is a normal turn", []*abi.ToolCall{weight}, false, 0, 1, 0},
		{"mixed is a FUSED turn", []*abi.ToolCall{classical, weight}, true, 1, 1, 0},
		{"unknown never makes a turn fused", []*abi.ToolCall{classical, unknown}, false, 1, 0, 1},
		{"unknown alongside both still fused", []*abi.ToolCall{classical, weight, unknown}, true, 1, 1, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ft := fusedturn.Fuse(tc.calls)
			if ft.Fused() != tc.wantFused {
				t.Errorf("Fused = %v, want %v", ft.Fused(), tc.wantFused)
			}
			if ft.Classical() != tc.wantC || ft.Weight() != tc.wantW || ft.Unknown() != tc.wantU {
				t.Errorf("counts classical=%d weight=%d unknown=%d, want %d/%d/%d",
					ft.Classical(), ft.Weight(), ft.Unknown(), tc.wantC, tc.wantW, tc.wantU)
			}
		})
	}
}

func TestFusePreservesSubmissionOrder(t *testing.T) {
	// A fused turn is a real interleaving of the two families, not two sorted piles.
	calls := []*abi.ToolCall{
		fusedturn.Weight("glm-5.2", "reason", abi.Ref{}),
		fusedturn.Classical("read_file", abi.Ref{}),
		fusedturn.Weight("glm-5.2", "chat", abi.Ref{}),
	}
	ft := fusedturn.Fuse(calls)
	wantOrder := []fusedturn.OpClass{fusedturn.ClassWeight, fusedturn.ClassClassical, fusedturn.ClassWeight}
	if len(ft.Ops) != len(wantOrder) {
		t.Fatalf("len(Ops) = %d, want %d", len(ft.Ops), len(wantOrder))
	}
	for i, want := range wantOrder {
		if ft.Ops[i].Class != want {
			t.Errorf("op %d class = %v, want %v", i, ft.Ops[i].Class, want)
		}
	}
}

func TestFuseSkipsNilCalls(t *testing.T) {
	ft := fusedturn.Fuse([]*abi.ToolCall{fusedturn.Classical("x", abi.Ref{}), nil, fusedturn.Weight("e", "y", abi.Ref{})})
	if len(ft.Ops) != 2 {
		t.Fatalf("len(Ops) = %d, want 2 (nil skipped)", len(ft.Ops))
	}
}

func TestSummary(t *testing.T) {
	ft := fusedturn.Fuse([]*abi.ToolCall{
		fusedturn.Classical("git_commit", abi.Ref{}),
		fusedturn.Weight("glm-5.2", "chat", abi.Ref{}),
		{Tool: "bash"},
	})
	s := ft.Summary()
	if s.Schema != fusedturn.Schema {
		t.Errorf("schema = %q, want %q", s.Schema, fusedturn.Schema)
	}
	if s.Ops != 3 || s.Classical != 1 || s.Weight != 1 || s.Unknown != 1 || !s.Fused {
		t.Errorf("summary = %+v, want ops=3 classical=1 weight=1 unknown=1 fused=true", s)
	}
}

// TestAdjudicateCrossesOneFloor is the load-bearing witness: a genuinely FUSED
// turn — a classical op and a weight-based op in the same turn — is governed by
// ONE kernel. It runs the mixed batch through a real *kernel.Kernel whose only
// policy allows read_file (classical) and chat_completion (weight) and denies a
// destructive classical op. That the SAME policy allows one family's benign op,
// allows the other family's benign op, AND denies the destructive op proves both
// concept-families cross the same default-deny floor — the fusion is real, not two
// side-paths that could diverge on policy.
func TestAdjudicateCrossesOneFloor(t *testing.T) {
	floor := adjudicator.New(adjudicator.Policy{Allow: map[string]bool{
		"read_file":       true, // a benign classical op
		"chat_completion": true, // a benign weight-based op
	}})
	k := kernel.New("", kernel.WithAdjudicators([]abi.Adjudicator{floor}))

	ft := fusedturn.Fuse([]*abi.ToolCall{
		fusedturn.Classical("read_file", abi.Ref{}),               // classical, allowed
		fusedturn.Weight("glm-5.2", "chat_completion", abi.Ref{}), // weight, allowed
		fusedturn.Classical("rm_rf", abi.Ref{}),                   // classical, NOT allowed -> denied
	})
	if !ft.Fused() {
		t.Fatalf("turn should be fused (has both families)")
	}

	rows := ft.Adjudicate(context.Background(), k)
	if len(rows) != 3 {
		t.Fatalf("len(rows) = %d, want 3 (one verdict per op)", len(rows))
	}

	// Both concept-families were governed by the one floor.
	gov := fusedturn.GovernedFamilies(rows)
	if len(gov) != 2 || gov[0] != fusedturn.ClassClassical || gov[1] != fusedturn.ClassWeight {
		t.Fatalf("GovernedFamilies = %v, want [classical weight] (both crossed one floor)", gov)
	}

	// The same policy that ALLOWS one family's benign op ALLOWS the other's, and
	// DENIES the destructive classical op — uniform governance across families.
	byTool := map[string]fusedturn.AdjudicatedOp{}
	for _, r := range rows {
		byTool[r.Tool] = r
	}
	if got := byTool["read_file"]; got.Kind != "allow" {
		t.Errorf("classical read_file verdict = %q, want allow", got.Kind)
	}
	if got := byTool["chat_completion"]; got.Kind != "allow" {
		t.Errorf("weight chat_completion verdict = %q, want allow", got.Kind)
	}
	if got := byTool["rm_rf"]; got.Kind != "deny" {
		t.Errorf("classical rm_rf verdict = %q, want deny (same floor refuses it)", got.Kind)
	}
}

// ftEngine is a fake abi.EngineDriver that self-declares its concept-family via the
// optional WeightBearing seam — the stand-in for a real model-forward engine vs a
// deterministic tool engine, so ClassifyResolved can be witnessed without a model.
type ftEngine struct{ weight bool }

func (ftEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	return &abi.Result{Status: abi.StatusOK}, nil
}
func (ftEngine) Caps() []abi.Capability { return nil }
func (e ftEngine) WeightBearing() bool  { return e.weight }

// ftPlainEngine implements EngineDriver but NOT WeightBearing — the "cannot tell"
// case that must stay ClassUnknown (fail-closed).
type ftPlainEngine struct{}

func (ftPlainEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	return &abi.Result{Status: abi.StatusOK}, nil
}
func (ftPlainEngine) Caps() []abi.Capability { return nil }

func TestClassifyResolvedReadsEngineSelfDeclaration(t *testing.T) {
	abi.RegisterEngine("ft-weighty", ftEngine{weight: true})
	abi.RegisterEngine("ft-tooly", ftEngine{weight: false})
	abi.RegisterEngine("ft-plain", ftPlainEngine{})

	cases := []struct {
		name string
		call *abi.ToolCall
		want fusedturn.OpClass
	}{
		{"weight-bearing engine => weight", &abi.ToolCall{Tool: "chat", Engine: "ft-weighty"}, fusedturn.ClassWeight},
		{"non-weight engine => classical", &abi.ToolCall{Tool: "bash", Engine: "ft-tooly"}, fusedturn.ClassClassical},
		{"engine without the seam => unknown", &abi.ToolCall{Tool: "x", Engine: "ft-plain"}, fusedturn.ClassUnknown},
		{"unregistered route => unknown", &abi.ToolCall{Tool: "x", Engine: "ft-nope"}, fusedturn.ClassUnknown},
		{"no engine => unknown", &abi.ToolCall{Tool: "x"}, fusedturn.ClassUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := fusedturn.ClassifyResolved(tc.call); got != tc.want {
				t.Fatalf("ClassifyResolved = %v, want %v", got, tc.want)
			}
		})
	}

	// An explicit declaration ALWAYS wins over the engine's self-declaration.
	c := fusedturn.Classical("x", abi.Ref{})
	c.Engine = "ft-weighty" // engine says weight, declaration says classical
	if got := fusedturn.ClassifyResolved(c); got != fusedturn.ClassClassical {
		t.Fatalf("declaration must win: ClassifyResolved = %v, want ClassClassical", got)
	}
}

func TestAdjudicateNilDeciderYieldsNil(t *testing.T) {
	ft := fusedturn.Fuse([]*abi.ToolCall{fusedturn.Classical("x", abi.Ref{})})
	if rows := ft.Adjudicate(context.Background(), nil); rows != nil {
		t.Fatalf("Adjudicate(nil decider) = %v, want nil (no floor, no witness)", rows)
	}
}
