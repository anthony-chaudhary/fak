package valuechain

import (
	"reflect"
	"sync"
	"testing"
)

func TestVerticalValueChainDeterminism(t *testing.T) {
	costA, costB := 1.0, 0.5
	manifest := Manifest{
		Schema: Schema,
		Name:   "chain",
		Stages: []Stage{
			{ID: "prepare", Kind: "preparation"},
			{ID: "run", Kind: "outcome", DependsOn: []string{"prepare"}},
		},
		Arms:     []Arm{{ID: "baseline", Default: true}, {ID: "optimized"}},
		Outcomes: []Outcome{{ID: "tasks", Unit: "task"}},
	}
	input := Input{Schema: Schema, Observations: []Observation{
		{ID: "baseline-prepare", TraceID: "baseline-trace", StageID: "prepare", Arm: "baseline", Turns: 2, CostUSD: &costA, Provenance: "test-meter"},
		{ID: "baseline-run", TraceID: "baseline-trace", PairID: "pair-1", StageID: "run", Arm: "baseline", Turns: 2, CostUSD: &costA, Outcomes: map[string]float64{"tasks": 1}, Provenance: "test-meter"},
		{ID: "optimized-prepare", TraceID: "optimized-trace", StageID: "prepare", Arm: "optimized", Turns: 2, CostUSD: &costB, Provenance: "test-meter"},
		{ID: "optimized-run", TraceID: "optimized-trace", PairID: "pair-1", StageID: "run", Arm: "optimized", Turns: 2, CostUSD: &costB, Outcomes: map[string]float64{"tasks": 1}, Provenance: "test-meter"},
	}}
	baseline, err := Audit(manifest, input)
	if err != nil {
		t.Fatal(err)
	}
	const runs = 100
	errCh := make(chan error, runs)
	var wg sync.WaitGroup
	for i := 0; i < runs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, gotErr := Audit(manifest, input)
			if gotErr != nil {
				errCh <- gotErr
				return
			}
			if !reflect.DeepEqual(got, baseline) {
				errCh <- errNondeterministic{}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for gotErr := range errCh {
		t.Error(gotErr)
	}
}

type errNondeterministic struct{}

func (errNondeterministic) Error() string { return "value-chain report changed across identical runs" }
