package ciyaml

import (
	"errors"
	"reflect"
	"testing"
)

func TestResolveDAGNeedsSingle(t *testing.T) {
	wf := &Workflow{
		Jobs: map[string]Job{
			"build": {ID: "build"},
			"test":  {ID: "test", Needs: []string{"build"}},
		},
	}

	order, err := wf.ResolveDAG()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []string{"build", "test"}
	if !reflect.DeepEqual(order, expected) {
		t.Fatalf("expected order %v, got %v", expected, order)
	}
}

func TestResolveDAGNeedsList(t *testing.T) {
	wf := &Workflow{
		Jobs: map[string]Job{
			"build":  {ID: "build"},
			"lint":   {ID: "lint"},
			"test":   {ID: "test", Needs: []string{"build", "lint"}},
			"deploy": {ID: "deploy", Needs: []string{"test"}},
		},
	}

	order, err := wf.ResolveDAG()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []string{"build", "lint", "test", "deploy"}
	if !reflect.DeepEqual(order, expected) {
		t.Fatalf("expected order %v, got %v", expected, order)
	}
}

func TestResolveDAGDependsOnSingle(t *testing.T) {
	wf := &Workflow{
		Jobs: map[string]Job{
			"prep": {ID: "prep"},
			"run":  {ID: "run", DependsOn: []string{"prep"}},
		},
	}

	order, err := wf.ResolveDAG()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []string{"prep", "run"}
	if !reflect.DeepEqual(order, expected) {
		t.Fatalf("expected order %v, got %v", expected, order)
	}
}

func TestResolveDAGDependsOnList(t *testing.T) {
	wf := &Workflow{
		Jobs: map[string]Job{
			"infra": {ID: "infra"},
			"auth":  {ID: "auth"},
			"api":   {ID: "api", DependsOn: []string{"infra", "auth"}},
		},
	}

	order, err := wf.ResolveDAG()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []string{"auth", "infra", "api"}
	if !reflect.DeepEqual(order, expected) {
		t.Fatalf("expected order %v, got %v", expected, order)
	}
}

func TestResolveDAGCombinedNeedsAndDependsOn(t *testing.T) {
	wf := &Workflow{
		Jobs: map[string]Job{
			"compile": {ID: "compile"},
			"schema":  {ID: "schema"},
			"service": {ID: "service", Needs: []string{"compile"}, DependsOn: []string{"schema"}},
		},
	}

	order, err := wf.ResolveDAG()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []string{"compile", "schema", "service"}
	if !reflect.DeepEqual(order, expected) {
		t.Fatalf("expected order %v, got %v", expected, order)
	}
}

func TestResolveDAGDiamondDependencies(t *testing.T) {
	wf := &Workflow{
		Jobs: map[string]Job{
			"A": {ID: "A"},
			"B": {ID: "B", Needs: []string{"A"}},
			"C": {ID: "C", Needs: []string{"A"}},
			"D": {ID: "D", Needs: []string{"B", "C"}},
		},
	}

	order, err := wf.ResolveDAG()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []string{"A", "B", "C", "D"}
	if !reflect.DeepEqual(order, expected) {
		t.Fatalf("expected order %v, got %v", expected, order)
	}
}

func TestResolveDAGIndependentParallelJobs(t *testing.T) {
	wf := &Workflow{
		Jobs: map[string]Job{
			"zulu":  {ID: "zulu"},
			"alpha": {ID: "alpha"},
			"bravo": {ID: "bravo"},
		},
	}

	order, err := wf.ResolveDAG()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Alphabetical order for deterministic execution
	expected := []string{"alpha", "bravo", "zulu"}
	if !reflect.DeepEqual(order, expected) {
		t.Fatalf("expected order %v, got %v", expected, order)
	}
}

func TestResolveDAGSelfDependencyCycle(t *testing.T) {
	wf := &Workflow{
		Jobs: map[string]Job{
			"loop": {ID: "loop", Needs: []string{"loop"}},
		},
	}

	order, err := wf.ResolveDAG()
	if err == nil {
		t.Fatalf("expected cycle error, got order: %v", order)
	}

	if !errors.Is(err, ErrCycleDetected) {
		t.Errorf("expected ErrCycleDetected, got %v", err)
	}

	var cycleErr *CycleError
	if errors.As(err, &cycleErr) {
		if len(cycleErr.Cycle) != 2 || cycleErr.Cycle[0] != "loop" || cycleErr.Cycle[1] != "loop" {
			t.Errorf("unexpected cycle path: %v", cycleErr.Cycle)
		}
	} else {
		t.Errorf("expected *CycleError type assertion, got %T", err)
	}
}

func TestResolveDAGDirectCycle(t *testing.T) {
	wf := &Workflow{
		Jobs: map[string]Job{
			"jobA": {ID: "jobA", Needs: []string{"jobB"}},
			"jobB": {ID: "jobB", Needs: []string{"jobA"}},
		},
	}

	order, err := wf.ResolveDAG()
	if err == nil {
		t.Fatalf("expected cycle error, got order: %v", order)
	}

	if !errors.Is(err, ErrCycleDetected) {
		t.Errorf("expected ErrCycleDetected, got %v", err)
	}

	var cycleErr *CycleError
	if errors.As(err, &cycleErr) {
		if len(cycleErr.Cycle) < 3 {
			t.Errorf("expected cycle path with at least 3 elements, got %v", cycleErr.Cycle)
		}
	} else {
		t.Errorf("expected *CycleError type assertion, got %T", err)
	}
}

func TestResolveDAGIndirectCycle(t *testing.T) {
	wf := &Workflow{
		Jobs: map[string]Job{
			"A": {ID: "A", Needs: []string{"B"}},
			"B": {ID: "B", Needs: []string{"C"}},
			"C": {ID: "C", Needs: []string{"A"}},
		},
	}

	order, err := wf.ResolveDAG()
	if err == nil {
		t.Fatalf("expected cycle error, got order: %v", order)
	}

	if !errors.Is(err, ErrCycleDetected) {
		t.Errorf("expected ErrCycleDetected, got %v", err)
	}

	var cycleErr *CycleError
	if errors.As(err, &cycleErr) {
		if len(cycleErr.Cycle) != 4 {
			t.Errorf("expected 4 elements in cycle A->B->C->A, got %v", cycleErr.Cycle)
		}
	} else {
		t.Errorf("expected *CycleError type assertion, got %T", err)
	}
}

func TestResolveDAGDisconnectedCycle(t *testing.T) {
	wf := &Workflow{
		Jobs: map[string]Job{
			"ok1": {ID: "ok1"},
			"ok2": {ID: "ok2", Needs: []string{"ok1"}},
			"c1":  {ID: "c1", Needs: []string{"c2"}},
			"c2":  {ID: "c2", Needs: []string{"c1"}},
		},
	}

	order, err := wf.ResolveDAG()
	if err == nil {
		t.Fatalf("expected cycle error on disconnected subgraph, got order: %v", order)
	}

	if !errors.Is(err, ErrCycleDetected) {
		t.Errorf("expected ErrCycleDetected, got %v", err)
	}
}

func TestResolveDAGMissingDependency(t *testing.T) {
	wf := &Workflow{
		Jobs: map[string]Job{
			"build":  {ID: "build"},
			"deploy": {ID: "deploy", Needs: []string{"build", "ghost-job"}},
		},
	}

	order, err := wf.ResolveDAG()
	if err == nil {
		t.Fatalf("expected missing dependency error, got order: %v", order)
	}

	if !errors.Is(err, ErrMissingDependency) {
		t.Errorf("expected ErrMissingDependency, got %v", err)
	}

	var missingErr *MissingDependencyError
	if errors.As(err, &missingErr) {
		if missingErr.Job != "deploy" || missingErr.Dependency != "ghost-job" {
			t.Errorf("unexpected missing dependency fields: %+v", missingErr)
		}
	} else {
		t.Errorf("expected *MissingDependencyError, got %T", err)
	}
}

func TestValidateDAG(t *testing.T) {
	validWF := &Workflow{
		Jobs: map[string]Job{
			"a": {ID: "a"},
			"b": {ID: "b", Needs: []string{"a"}},
		},
	}
	if err := validWF.ValidateDAG(); err != nil {
		t.Errorf("expected valid DAG to pass ValidateDAG, got: %v", err)
	}

	invalidWF := &Workflow{
		Jobs: map[string]Job{
			"a": {ID: "a", Needs: []string{"b"}},
			"b": {ID: "b", Needs: []string{"a"}},
		},
	}
	if err := invalidWF.ValidateDAG(); err == nil {
		t.Errorf("expected invalid cyclic DAG to fail ValidateDAG, got nil")
	}
}

func TestParseAndResolveDAGEndToEnd(t *testing.T) {
	content := `name: EndToEnd
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: golangci-lint
  build:
    runs-on: ubuntu-latest
    steps:
      - run: go build ./...
  test:
    runs-on: ubuntu-latest
    needs: [build, lint]
    steps:
      - run: go test ./...
  deploy:
    runs-on: ubuntu-latest
    depends_on: [test]
    steps:
      - run: ./deploy.sh
`

	wf, err := ParseWorkflow(content)
	if err != nil {
		t.Fatalf("ParseWorkflow failed: %v", err)
	}

	order, err := wf.ResolveDAG()
	if err != nil {
		t.Fatalf("ResolveDAG failed: %v", err)
	}

	expected := []string{"build", "lint", "test", "deploy"}
	if !reflect.DeepEqual(order, expected) {
		t.Fatalf("expected topological order %v, got %v", expected, order)
	}

	// Missing dependency in parsed YAML
	badContent := `jobs:
  test:
    needs: [missing-setup]
`
	badWF, err := ParseWorkflow(badContent)
	if err != nil {
		t.Fatalf("ParseWorkflow failed: %v", err)
	}
	_, err = badWF.ResolveDAG()
	if err == nil || !errors.Is(err, ErrMissingDependency) {
		t.Fatalf("expected ErrMissingDependency, got: %v", err)
	}

	// Cycle in parsed YAML
	cycleContent := `jobs:
  a:
    needs: [b]
  b:
    needs: [a]
`
	cycleWF, err := ParseWorkflow(cycleContent)
	if err != nil {
		t.Fatalf("ParseWorkflow failed: %v", err)
	}
	_, err = cycleWF.ResolveDAG()
	if err == nil || !errors.Is(err, ErrCycleDetected) {
		t.Fatalf("expected ErrCycleDetected, got: %v", err)
	}
}
