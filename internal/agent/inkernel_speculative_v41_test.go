package agent

import (
	"context"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime medium est=5s lane=default
func TestPlannerV41SpeculativeEmissionContinuation(t *testing.T) {
	// The existing public-loader fixture executes real native V4.1 arithmetic.
	// No private model state or replacement forward supplies the expected result.
	m := v41CompleteModel(t)
	t.Cleanup(func() { _ = m.CloseWeights() })
	prompt := []int{1, 2, 3}
	const depth = 4
	draft := speculativeAcceptedDraft(t, m, prompt, depth)
	for _, stopAfter := range []int{1, depth + 1} {
		name := "accepted-prefix-callback"
		if stopAfter == depth+1 {
			name = "bonus-callback"
		}
		t.Run(name, func(t *testing.T) {
			oracle := m.NewSession()
			defer oracle.Close()
			logits := oracle.Prefill(prompt)
			want := make([]int, stopAfter)
			for i := range want {
				want[i] = model.ArgmaxF32(logits)
				logits = oracle.Step(want[i])
			}

			probe := &speculativeRequestProbe{tokens: draft}
			p := NewInKernelPlanner(m, nil, "v41-spec-emission", false, nil, false)
			p.quant, p.tree, p.scopedTree = false, nil, nil
			p.EnableSpeculativeDecoding(probe, depth)
			var emitted []int
			observed := false
			p.specSessionCloseHook = func(target *model.Session) {
				observed = true
				// Tokens is the complete V4.1 snapshot's history authority, not
				// the generic host Cache length, which omits temporal state.
				snapshot, err := target.PrefixSnapshot()
				if err != nil {
					t.Fatal(err)
				}
				defer snapshot.Close()
				if snapshot.Tokens != len(prompt)+len(emitted) {
					t.Fatalf("V4.1 retained tokens=%d, want prompt+emitted=%d", snapshot.Tokens, len(prompt)+len(emitted))
				}
				// Continuing the actual serving target detects stale ring,
				// compressor/publication or hidden continuation state too.
				for _, token := range []int{7, 11} {
					if got, wantRow := target.Step(token), oracle.Step(token); !reflect.DeepEqual(got, wantRow) {
						t.Fatalf("continued V4.1 logits differ after callback stop at token %d", token)
					}
				}
			}
			res, err := p.generateReusedRecovering(context.Background(), prompt, depth+2, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
				emitted = append(emitted, token)
				return len(emitted) == stopAfter
			})
			if err != nil {
				t.Fatal(err)
			}
			if !observed || probe.calls != 1 || res.gen != stopAfter || !reflect.DeepEqual(emitted, want) {
				t.Fatalf("consumer observed=%t proposals=%d generated=%d emitted=%v want=%v", observed, probe.calls, res.gen, emitted, want)
			}
		})
	}
}
