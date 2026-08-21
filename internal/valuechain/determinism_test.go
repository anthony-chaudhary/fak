package valuechain

import (
	"reflect"
	"sync"
	"testing"
)

func TestVerticalValueChainDeterminism(t *testing.T) {
	costA, costB := 1.0, 0.5
	manifest := Manifest{Schema: "fak-value-chain/1", ChainID: "chain", Outcome: Outcome{Unit: "tasks"}, Stages: []Stage{{ID: "prepare", Name: "Prepare"}, {ID: "run", Name: "Run", Consumes: []string{"prepare"}}}, Arms: []Arm{{ID: "baseline", Label: "Baseline"}, {ID: "optimized", Label: "Optimized"}}}
	input := Input{Observations: []Observation{
		{StageID: "prepare", ArmID: "baseline", Runs: 2, CostUSD: &costA, InputTokens: 10, OutputTokens: 2, LatencyMS: 5, OutcomeUnits: 1},
		{StageID: "run", ArmID: "baseline", Runs: 2, CostUSD: &costA, InputTokens: 10, OutputTokens: 2, LatencyMS: 5, OutcomeUnits: 1},
		{StageID: "prepare", ArmID: "optimized", Runs: 2, CostUSD: &costB, InputTokens: 5, OutputTokens: 2, LatencyMS: 3, OutcomeUnits: 1},
		{StageID: "run", ArmID: "optimized", Runs: 2, CostUSD: &costB, InputTokens: 5, OutputTokens: 2, LatencyMS: 3, OutcomeUnits: 1},
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
