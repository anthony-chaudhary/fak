package workflow

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"testing"
)

// dispatch builds a Runner that routes a task to a func keyed by its Op.
func dispatch(ops map[string]func(RunInput) (string, error)) Runner {
	return RunnerFunc(func(_ context.Context, in RunInput) (string, error) {
		f, ok := ops[in.Node.Op]
		if !ok {
			return "", fmt.Errorf("unknown op %q", in.Node.Op)
		}
		return f(in)
	})
}

// joinDeps concatenates a task's dependency outputs in sorted-key order — a stable fold
// so a reduce/join result is deterministic regardless of wave goroutine scheduling.
func joinDeps(in RunInput, sep string) string {
	keys := make([]string, 0, len(in.Deps))
	for k := range in.Deps {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	vals := make([]string, 0, len(keys))
	for _, k := range keys {
		vals = append(vals, in.Deps[k])
	}
	return strings.Join(vals, sep)
}

func mustRun(t *testing.T, g *Graph, r Runner, opt Options) Result {
	t.Helper()
	res := Execute(context.Background(), g, r, opt)
	return res
}

// orderIndex returns a map of task ID -> its position in Result.Order.
func orderIndex(res Result) map[string]int {
	idx := make(map[string]int, len(res.Order))
	for i, id := range res.Order {
		idx[id] = i
	}
	return idx
}

// --- Acceptance bullet 1 + 4: define a workflow in JSON, honor DAG dependencies. ---

func TestJSONDAGDependencyOrder(t *testing.T) {
	doc := []byte(`{
		"name": "etl",
		"tasks": [
			{"id": "extract",   "op": "emit",  "payload": "raw"},
			{"id": "transform", "op": "upper", "needs": ["extract"]},
			{"id": "load",      "op": "sink",  "needs": ["transform"]}
		]
	}`)
	g, err := CompileJSON(doc)
	if err != nil {
		t.Fatalf("CompileJSON: %v", err)
	}
	if g.Name != "etl" || len(g.Nodes) != 3 {
		t.Fatalf("graph = %+v", g)
	}
	r := dispatch(map[string]func(RunInput) (string, error){
		"emit":  func(in RunInput) (string, error) { return in.Node.Payload, nil },
		"upper": func(in RunInput) (string, error) { return strings.ToUpper(joinDeps(in, "")), nil },
		"sink":  func(in RunInput) (string, error) { return "loaded:" + joinDeps(in, ""), nil },
	})
	res := mustRun(t, g, r, Options{})
	if res.Failed {
		t.Fatalf("unexpected failure: %+v", res.Nodes)
	}
	for _, id := range []string{"extract", "transform", "load"} {
		if res.Nodes[id].Status != StatusSucceeded {
			t.Errorf("%s status = %s, want succeeded", id, res.Nodes[id].Status)
		}
	}
	if got := res.Nodes["load"].Output; got != "loaded:RAW" {
		t.Errorf("load output = %q, want %q", got, "loaded:RAW")
	}
	// Result.Order must respect every dependency edge.
	idx := orderIndex(res)
	for _, n := range g.Nodes {
		for _, d := range n.Needs {
			if idx[d] >= idx[n.ID] {
				t.Errorf("dependency order violated: %s (at %d) before its need %s (at %d)", n.ID, idx[n.ID], d, idx[d])
			}
		}
	}
}

// --- Acceptance bullet 2: map-reduce pattern. ---

func TestMapReducePattern(t *testing.T) {
	g, err := MapReduce("wordcount", "mark", "gather", []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("MapReduce: %v", err)
	}
	// 3 map tasks + 1 reduce.
	if len(g.Nodes) != 4 {
		t.Fatalf("nodes = %d, want 4", len(g.Nodes))
	}
	var reduce *Node
	for i := range g.Nodes {
		if g.Nodes[i].ID == "reduce" {
			reduce = &g.Nodes[i]
		}
	}
	if reduce == nil || len(reduce.Needs) != 3 {
		t.Fatalf("reduce node = %+v", reduce)
	}
	r := dispatch(map[string]func(RunInput) (string, error){
		"mark":   func(in RunInput) (string, error) { return in.Node.Payload + "!", nil },
		"gather": func(in RunInput) (string, error) { return joinDeps(in, ","), nil },
	})
	res := mustRun(t, g, r, Options{})
	if res.Failed {
		t.Fatalf("unexpected failure: %+v", res.Nodes)
	}
	if got := res.Nodes["reduce"].Output; got != "a!,b!,c!" {
		t.Errorf("reduce output = %q, want %q", got, "a!,b!,c!")
	}
}

// --- Acceptance bullet 3: fan-out pattern. ---

func TestFanOutPattern(t *testing.T) {
	g, err := FanOut("scatter", "seed", []string{"w1", "w2", "w3"}, "merge")
	if err != nil {
		t.Fatalf("FanOut: %v", err)
	}
	// source + 3 branches + join.
	if len(g.Nodes) != 5 {
		t.Fatalf("nodes = %d, want 5", len(g.Nodes))
	}
	// Every branch depends on source; join depends on every branch.
	branchRan := map[string]bool{}
	r := dispatch(map[string]func(RunInput) (string, error){
		"seed": func(in RunInput) (string, error) { return "S", nil },
		"w1":   func(in RunInput) (string, error) { return "w1(" + in.Deps["source"] + ")", nil },
		"w2":   func(in RunInput) (string, error) { return "w2(" + in.Deps["source"] + ")", nil },
		"w3":   func(in RunInput) (string, error) { return "w3(" + in.Deps["source"] + ")", nil },
		"merge": func(in RunInput) (string, error) {
			for k := range in.Deps {
				branchRan[k] = true
			}
			return joinDeps(in, "|"), nil
		},
	})
	res := mustRun(t, g, r, Options{})
	if res.Failed {
		t.Fatalf("unexpected failure: %+v", res.Nodes)
	}
	if got := res.Nodes["join"].Output; got != "w1(S)|w2(S)|w3(S)" {
		t.Errorf("join output = %q, want %q", got, "w1(S)|w2(S)|w3(S)")
	}
	for _, id := range []string{"branch:0", "branch:1", "branch:2"} {
		if !branchRan[id] {
			t.Errorf("join did not see branch %s", id)
		}
	}
	// Pure fan-out (no join) drops the join node.
	g2, err := FanOut("scatter", "seed", []string{"w1", "w2"}, "")
	if err != nil {
		t.Fatalf("FanOut no-join: %v", err)
	}
	for _, n := range g2.Nodes {
		if n.ID == "join" {
			t.Errorf("pure fan-out should have no join node")
		}
	}
}

// --- DAG validation: cycles, unknown deps, self-deps, duplicate IDs. ---

func TestCompileRejectsBadGraphs(t *testing.T) {
	cases := []struct {
		name string
		spec Spec
		want string
	}{
		{"cycle", Spec{Tasks: []TaskSpec{
			{ID: "a", Needs: []string{"b"}},
			{ID: "b", Needs: []string{"a"}},
		}}, "cycle"},
		{"unknown-dep", Spec{Tasks: []TaskSpec{
			{ID: "a", Needs: []string{"ghost"}},
		}}, "unknown task"},
		{"self-dep", Spec{Tasks: []TaskSpec{
			{ID: "a", Needs: []string{"a"}},
		}}, "depends on itself"},
		{"dup-id", Spec{Tasks: []TaskSpec{
			{ID: "a"}, {ID: "a"},
		}}, "duplicate task id"},
		{"empty", Spec{Name: "x"}, "empty"},
		{"ambiguous", Spec{
			Tasks:     []TaskSpec{{ID: "a"}},
			MapReduce: &MapReduceSpec{Map: "m", Reduce: "r", Items: []string{"x"}},
		}, "exactly one"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Compile(tc.spec)
			if err == nil {
				t.Fatalf("want error containing %q, got nil", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %q, want substring %q", err.Error(), tc.want)
			}
		})
	}
}

func TestParseSpecRejectsUnknownField(t *testing.T) {
	_, err := ParseSpec([]byte(`{"name":"x","taskz":[]}`))
	if err == nil {
		t.Fatal("want error on unknown field, got nil")
	}
}

// --- Fault tolerance: retries, fail-fast, continue-on-error. ---

func TestRetriesEventuallySucceed(t *testing.T) {
	var mu sync.Mutex
	calls := 0
	r := dispatch(map[string]func(RunInput) (string, error){
		"flaky": func(in RunInput) (string, error) {
			mu.Lock()
			defer mu.Unlock()
			calls++
			if calls < 3 {
				return "", fmt.Errorf("transient %d", calls)
			}
			return "ok", nil
		},
	})
	g, err := Compile(Spec{Tasks: []TaskSpec{{ID: "t", Op: "flaky", Retries: 2}}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	res := mustRun(t, g, r, Options{})
	if res.Failed {
		t.Fatalf("want success after retries, got %+v", res.Nodes["t"])
	}
	if got := res.Nodes["t"].Attempts; got != 3 {
		t.Errorf("attempts = %d, want 3", got)
	}
}

func TestRetriesExhaustedFails(t *testing.T) {
	r := dispatch(map[string]func(RunInput) (string, error){
		"boom": func(in RunInput) (string, error) { return "", fmt.Errorf("always") },
	})
	g, _ := Compile(Spec{Tasks: []TaskSpec{{ID: "t", Op: "boom", Retries: 1}}})
	res := mustRun(t, g, r, Options{})
	if !res.Failed || res.Nodes["t"].Status != StatusFailed {
		t.Fatalf("want failed, got %+v", res.Nodes["t"])
	}
	if res.Nodes["t"].Attempts != 2 {
		t.Errorf("attempts = %d, want 2 (1+1 retry)", res.Nodes["t"].Attempts)
	}
}

// failChain is a→bad (independent), a→x→y. Under fail-fast the x/y chain is skipped;
// under continue-on-error it completes because it does not depend on the failure.
func failChain() Spec {
	return Spec{Tasks: []TaskSpec{
		{ID: "a", Op: "emit", Payload: "A"},
		{ID: "bad", Op: "boom"},
		{ID: "x", Op: "emit", Payload: "X", Needs: []string{"a"}},
		{ID: "y", Op: "emit", Payload: "Y", Needs: []string{"x"}},
	}}
}

func faultRunner() Runner {
	return dispatch(map[string]func(RunInput) (string, error){
		"emit": func(in RunInput) (string, error) { return in.Node.Payload, nil },
		"boom": func(in RunInput) (string, error) { return "", fmt.Errorf("kaboom") },
	})
}

func TestFailFastSkipsRemaining(t *testing.T) {
	g, err := Compile(failChain())
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	res := mustRun(t, g, faultRunner(), Options{}) // fail-fast is the default
	if !res.Failed {
		t.Fatal("want Failed=true")
	}
	want := map[string]Status{
		"a": StatusSucceeded, "bad": StatusFailed, "x": StatusSkipped, "y": StatusSkipped,
	}
	for id, st := range want {
		if res.Nodes[id].Status != st {
			t.Errorf("%s = %s, want %s", id, res.Nodes[id].Status, st)
		}
	}
}

func TestContinueOnErrorIsolatesFailure(t *testing.T) {
	g, _ := Compile(failChain())
	res := mustRun(t, g, faultRunner(), Options{ContinueOnError: true})
	if !res.Failed {
		t.Fatal("want Failed=true (a task did fail)")
	}
	want := map[string]Status{
		"a": StatusSucceeded, "bad": StatusFailed, "x": StatusSucceeded, "y": StatusSucceeded,
	}
	for id, st := range want {
		if res.Nodes[id].Status != st {
			t.Errorf("%s = %s, want %s", id, res.Nodes[id].Status, st)
		}
	}
}

// TestSkipPropagatesToDescendants: a failure skips a whole downstream subtree, not just
// the immediate child, under continue-on-error.
func TestSkipPropagatesToDescendants(t *testing.T) {
	g, _ := Compile(Spec{Tasks: []TaskSpec{
		{ID: "root", Op: "boom"},
		{ID: "c1", Op: "emit", Needs: []string{"root"}},
		{ID: "c2", Op: "emit", Needs: []string{"c1"}},
		{ID: "c3", Op: "emit", Needs: []string{"c2"}},
	}})
	res := mustRun(t, g, faultRunner(), Options{ContinueOnError: true})
	for _, id := range []string{"c1", "c2", "c3"} {
		if res.Nodes[id].Status != StatusSkipped {
			t.Errorf("%s = %s, want skipped", id, res.Nodes[id].Status)
		}
	}
}

func ExampleCompileJSON() {
	doc := []byte(`{"name":"mr","map_reduce":{"map":"mark","reduce":"gather","items":["x","y"]}}`)
	g, err := CompileJSON(doc)
	if err != nil {
		panic(err)
	}
	r := dispatch(map[string]func(RunInput) (string, error){
		"mark":   func(in RunInput) (string, error) { return "[" + in.Node.Payload + "]", nil },
		"gather": func(in RunInput) (string, error) { return joinDeps(in, ""), nil },
	})
	res := Execute(context.Background(), g, r, Options{})
	fmt.Println(res.Nodes["reduce"].Output)
	// Output: [x][y]
}
