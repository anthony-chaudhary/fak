package contextq

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestTreeExplore(t *testing.T) {
	body := []byte(`{"z":"sentinel-secret-subtree-that-is-intentionally-much-larger-than-any-selected-observation-and-must-never-appear-in-descriptors","a":{"deep":[10,{"leaf":"chosen"}]},"wide":{"c":3,"a":1,"b":2}}`)
	resolver, source := testSource(body)
	limits := TreeLimits{MaxSourceBytes: 4096, MaxOutputBytes: 2048, MaxDepth: 8, MaxWidth: 8, MaxPaths: 4, MaxNodes: 32, MaxWorkUnits: 8192, MaxLeafBytes: 128}

	keys, err := ExploreTree(context.Background(), resolver, source, TreePlan{Operation: TreeKeys, Path: "/wide", Offset: 0, Limit: 2}, limits)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(keys.Keys, []string{"a", "b"}) || keys.Total != 3 || keys.NextOffset != 2 || !keys.Truncated {
		t.Fatalf("unexpected keys view: %+v", keys)
	}
	replay, err := ExploreTree(context.Background(), resolver, source, keys.Plan, limits)
	if err != nil {
		t.Fatal(err)
	}
	if replay.OutputDigest != keys.OutputDigest || replay.PlanDigest != keys.PlanDigest {
		t.Fatal("replay hashes changed")
	}

	children, err := ExploreTree(context.Background(), resolver, source, TreePlan{Operation: TreeChildren, Path: "/a", Offset: 0, Limit: 5}, limits)
	if err != nil {
		t.Fatal(err)
	}
	if len(children.Children) != 1 || children.Children[0].Path != "/a/deep" || children.Children[0].Type != "array" || children.Children[0].ChildCount != 2 {
		t.Fatalf("unexpected children: %+v", children)
	}
	encoded, _ := json.Marshal(children)
	if strings.Contains(string(encoded), "chosen") || strings.Contains(string(encoded), "sentinel") {
		t.Fatalf("descriptor leaked payload: %s", encoded)
	}

	got, err := ExploreTree(context.Background(), resolver, source, TreePlan{Operation: TreeGet, Paths: []string{"/a/deep/1/leaf", "/wide/c"}}, limits)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Values) != 2 || string(got.Values[0].Value) != `"chosen"` || string(got.Values[1].Value) != `3` {
		t.Fatalf("unexpected values: %+v", got.Values)
	}
	if got.Source.Taint != abi.TaintTainted || got.Source.Scope != abi.ScopeAgent {
		t.Fatal("lineage labels changed")
	}
	if got.Accounting.OutputBytes >= got.Accounting.SourceBytes {
		t.Fatalf("visible bytes not bounded: %+v", got.Accounting)
	}
}

func TestTreeExploreRefusals(t *testing.T) {
	body := []byte(`{"a":{"b":{"c":1}},"wide":{"a":1,"b":2},"large":"0123456789"}`)
	resolver, source := testSource(body)
	base := TreeLimits{MaxSourceBytes: 4096, MaxOutputBytes: 2048, MaxDepth: 8, MaxWidth: 8, MaxPaths: 4, MaxNodes: 32, MaxWorkUnits: 8192, MaxLeafBytes: 128}
	cases := []struct {
		name   string
		plan   TreePlan
		mutate func(*TreeLimits, *abi.Ref)
		reason string
	}{
		{"invalid path", TreePlan{Operation: TreeKeys, Path: "a", Limit: 1}, nil, string(TreeReasonPathInvalid)},
		{"missing", TreePlan{Operation: TreeGet, Paths: []string{"/nope"}}, nil, string(TreeReasonPathMissing)},
		{"non leaf", TreePlan{Operation: TreeGet, Paths: []string{"/a"}}, nil, string(TreeReasonNonLeaf)},
		{"depth", TreePlan{Operation: TreeGet, Paths: []string{"/a/b/c"}}, func(l *TreeLimits, _ *abi.Ref) { l.MaxDepth = 2 }, string(TreeReasonDepthLimit)},
		{"width", TreePlan{Operation: TreeKeys, Path: "/wide", Limit: 1}, func(l *TreeLimits, _ *abi.Ref) { l.MaxWidth = 1 }, string(TreeReasonWidthLimit)},
		{"paths", TreePlan{Operation: TreeGet, Paths: []string{"/large", "/wide/a"}}, func(l *TreeLimits, _ *abi.Ref) { l.MaxPaths = 1 }, string(TreeReasonPathLimit)},
		{"leaf", TreePlan{Operation: TreeGet, Paths: []string{"/large"}}, func(l *TreeLimits, _ *abi.Ref) { l.MaxLeafBytes = 2 }, string(TreeReasonLeafLimit)},
		{"output", TreePlan{Operation: TreeGet, Paths: []string{"/large"}}, func(l *TreeLimits, _ *abi.Ref) { l.MaxOutputBytes = 2 }, string(TreeReasonOutputLimit)},
		{"work", TreePlan{Operation: TreeGet, Paths: []string{"/large"}}, func(l *TreeLimits, _ *abi.Ref) { l.MaxWorkUnits = 1 }, string(TreeReasonWorkLimit)},
		{"digest", TreePlan{Operation: TreeGet, Paths: []string{"/large"}}, func(_ *TreeLimits, r *abi.Ref) { r.Digest = strings.Repeat("0", 64) }, string(DeriveReasonDigestMismatch)},
		{"quarantine", TreePlan{Operation: TreeGet, Paths: []string{"/large"}}, func(_ *TreeLimits, r *abi.Ref) { r.Taint = abi.TaintQuarantined }, string(DeriveReasonSourceQuarantined)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			l := base
			r := source
			if tc.mutate != nil {
				tc.mutate(&l, &r)
			}
			_, err := ExploreTree(context.Background(), resolver, r, tc.plan, l)
			if err == nil || !strings.Contains(err.Error(), tc.reason) {
				t.Fatalf("want %s, got %v", tc.reason, err)
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := ExploreTree(ctx, resolver, source, TreePlan{Operation: TreeKeys, Path: "/", Limit: 1}, base)
	if err == nil || !strings.Contains(err.Error(), "canceled") {
		t.Fatalf("want cancellation, got %v", err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestTreeExploreLargePageLimit(t *testing.T) {
	const body = `{"c":3,"a":1,"b":2}`
	maxInt := int(^uint(0) >> 1)
	limits := TreeLimits{MaxSourceBytes: 128, MaxOutputBytes: 2048, MaxDepth: 2, MaxWidth: 3, MaxPaths: 1, MaxNodes: 4, MaxWorkUnits: 128, MaxLeafBytes: 8}
	for _, op := range []TreeOperation{TreeKeys, TreeChildren} {
		for _, tc := range []struct {
			name   string
			offset int
		}{
			{"start", 0},
			{"middle", 1},
			{"end", 3},
		} {
			t.Run(string(op)+"/"+tc.name, func(t *testing.T) {
				resolver, source := testSource([]byte(body))
				plan := TreePlan{Operation: op, Path: "/", Offset: tc.offset, Limit: 3}
				want, err := ExploreTree(context.Background(), resolver, source, plan, limits)
				if err != nil {
					t.Fatal(err)
				}
				plan.Limit = maxInt
				got, err := ExploreTree(context.Background(), resolver, source, plan, limits)
				if err != nil {
					t.Fatal(err)
				}
				// The requested limit remains part of the plan identity; only the
				// returned page is bounded by the three available entries.
				want.Plan = plan
				canonical, err := json.Marshal(plan)
				if err != nil {
					t.Fatal(err)
				}
				want.PlanDigest = digestBytes(canonical)
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("large limit changed the bounded page: got %+v, want %+v", got, want)
				}
			})
		}
	}
}
