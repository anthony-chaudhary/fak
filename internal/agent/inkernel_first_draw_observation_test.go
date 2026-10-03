package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// firstDrawWire reads the existing reporter through JSON so the parent compiles
// before the diagnostic field exists. Expectations come from generated output.
func firstDrawWire(t *testing.T, p *InKernelPlanner, trace string) map[string]any {
	t.Helper()
	observation, ok := p.NativePhaseObservation(trace)
	if !ok {
		t.Fatalf("no native phase for actual decode trace %q", trace)
	}
	raw, err := json.Marshal(observation)
	if err != nil {
		t.Fatal(err)
	}
	var phase map[string]any
	if err := json.Unmarshal(raw, &phase); err != nil {
		t.Fatal(err)
	}
	draw, ok := phase["first_draw"].(map[string]any)
	if !ok {
		t.Fatalf("actual decode observation missing first_draw: %s", raw)
	}
	if len(draw) > 4 {
		t.Fatalf("unbounded first_draw fields: %s", raw)
	}
	return draw
}

func firstDrawGenerate(t *testing.T, p *InKernelPlanner, trace string, ceiling int, stops map[int]bool, bias model.LogitBias) ([]int, bool) {
	t.Helper()
	var emitted []int
	ctx := context.WithValue(context.Background(), "trace_id", trace)
	gen, _, _, _, _, _, _, stopped, err := p.generateReusedContextWithBias(ctx, []int{1, 2, 3, 4}, ceiling, 0, 0, 0, bias, 0, 0, stops, func(id int) bool { emitted = append(emitted, id); return false })
	if err != nil {
		t.Fatal(err)
	}
	if gen != len(emitted) {
		t.Fatalf("generated=%d emitted=%v", gen, emitted)
	}
	return emitted, stopped
}

// fak-test:runtime fast est=1s
func TestInKernelFirstDrawObservationMatchesActualDecode(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	makePlanner := func() *InKernelPlanner {
		return &InKernelPlanner{m: model.NewSynthetic(cfg), modelID: "first-draw-software"}
	}
	oracle := makePlanner()
	tokens, stopped := firstDrawGenerate(t, oracle, "oracle", 1, map[int]bool{}, nil)
	if stopped || len(tokens) != 1 {
		t.Fatalf("independent non-stop oracle = %v/%v", tokens, stopped)
	}
	first := tokens[0]
	cases := []struct {
		name         string
		ceiling      int
		stops        map[int]bool
		bias         model.LogitBias
		wantToken    int
		wantObserved bool
		class        string
		count        int
		stopped      bool
	}{
		{"first_non_stop", 1, map[int]bool{}, nil, first, true, "non_stop_token", 1, false},
		{"first_stop_before_emission", 1, map[int]bool{first: true}, nil, first, true, "stop_token", 0, true},
		{"zero_ceiling_no_draw", 0, map[int]bool{}, nil, 0, false, "not_drawn", 0, false},
		{"token_zero_is_observed", 1, map[int]bool{}, model.LogitBias{0: 1e9}, 0, true, "non_stop_token", 1, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			planner := makePlanner()
			emitted, hit := firstDrawGenerate(t, planner, "case", tc.ceiling, tc.stops, tc.bias)
			if len(emitted) != tc.count || hit != tc.stopped {
				t.Fatalf("output/stopping changed: emitted=%v stopped=%v", emitted, hit)
			}
			if tc.count > 0 && emitted[0] != tc.wantToken {
				t.Fatalf("actual first output=%d want=%d", emitted[0], tc.wantToken)
			}
			draw := firstDrawWire(t, planner, "case")
			if draw["resolved_output_ceiling"] != float64(tc.ceiling) || draw["observed"] != tc.wantObserved || draw["classification"] != tc.class {
				t.Fatalf("draw=%v want ceiling=%d observed=%v class=%s", draw, tc.ceiling, tc.wantObserved, tc.class)
			}
			token, present := draw["token_id"]
			if present != tc.wantObserved || (present && token != float64(tc.wantToken)) {
				t.Fatalf("token presence/value=%v/%v want observed=%v token=%d", present, token, tc.wantObserved, tc.wantToken)
			}
		})
	}
}

// fak-test:runtime slow est=2s
func TestInKernelFirstDrawObservationResetsAndRemainsBounded(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	p := &InKernelPlanner{m: model.NewSynthetic(cfg), modelID: "first-draw-software"}
	firstDrawGenerate(t, p, "same-trace", 1, map[int]bool{}, model.LogitBias{0: 1e9})
	firstDrawGenerate(t, p, "same-trace", 0, map[int]bool{}, nil)
	draw := firstDrawWire(t, p, "same-trace")
	if draw["observed"] != false || draw["resolved_output_ceiling"] != float64(0) || draw["classification"] != "not_drawn" {
		t.Fatalf("reused trace retained stale sample: %v", draw)
	}
	if _, ok := draw["token_id"]; ok {
		t.Fatalf("reused no-draw trace retained token: %v", draw)
	}
	for i := 0; i < 65; i++ {
		firstDrawGenerate(t, p, fmt.Sprintf("bounded-%d", i), 0, map[int]bool{}, nil)
	}
	if _, ok := p.NativePhaseObservation("bounded-0"); ok {
		t.Fatal("oldest of65 traces survived64-entry bound")
	}
	for i := 1; i < 65; i++ {
		if _, ok := p.NativePhaseObservation(fmt.Sprintf("bounded-%d", i)); !ok {
			t.Fatalf("live trace%d unexpectedly evicted", i)
		}
	}
}
