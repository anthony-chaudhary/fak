package modelroute

import (
	"math"
	"testing"
)

// TestMFWinRateRouter verifies the matrix-factorization win-rate router
// selects the correct top route for known embeddings/scores, asserting against
// hand-computed source outputs (the source's forward() logit and
// pred_win_rate() sigmoid).
func TestMFWinRateRouter(t *testing.T) {
	// dim = 2. Learned model-embedding table P and classifier W.
	r := &MFWinRateRouter{
		ModelEmbed: map[string][]float64{
			"strong": {3, 4}, // 2-norm 5
			"weak":   {1, 0}, // 2-norm 1
			"mid":    {0, 2}, // 2-norm 2
		},
		Classifier: []float64{1, 1},
	}
	prompt := []float64{1, 1}

	// Hand-computed logits: logit = sum_i W[i] * (emb[i]/norm) * prompt[i].
	//   strong: [0.6,0.8] . [1,1] = 1.4
	//   weak:   [1,0]     . [1,1] = 1.0
	//   mid:    [0,1]     . [1,1] = 1.0
	for model, want := range map[string]float64{"strong": 1.4, "weak": 1.0, "mid": 1.0} {
		got, err := r.Score(model, prompt)
		if err != nil {
			t.Fatalf("Score(%q): unexpected error: %v", model, err)
		}
		if math.Abs(got-want) > 1e-12 {
			t.Errorf("Score(%q) = %v, want %v", model, got, want)
		}
	}

	// Top route: strong (1.4) beats weak and mid (both 1.0).
	got, err := r.Select([]string{"weak", "mid", "strong"}, prompt)
	if err != nil {
		t.Fatalf("Select: unexpected error: %v", err)
	}
	if got != "strong" {
		t.Errorf("Select = %q, want %q", got, "strong")
	}

	// WinRate(strong, weak) = sigmoid(1.4 - 1.0) = sigmoid(0.4).
	wantWR := 1.0 / (1.0 + math.Exp(-0.4)) // ≈ 0.5986877560135582
	gotWR, err := r.WinRate("strong", "weak", prompt)
	if err != nil {
		t.Fatalf("WinRate(strong,weak): unexpected error: %v", err)
	}
	if math.Abs(gotWR-wantWR) > 1e-12 {
		t.Errorf("WinRate(strong,weak) = %v, want %v", gotWR, wantWR)
	}

	// Symmetry: WinRate(weak, strong) = 1 - WinRate(strong, weak).
	gotWR2, err := r.WinRate("weak", "strong", prompt)
	if err != nil {
		t.Fatalf("WinRate(weak,strong): unexpected error: %v", err)
	}
	if math.Abs(gotWR2-(1-wantWR)) > 1e-12 {
		t.Errorf("WinRate(weak,strong) = %v, want %v", gotWR2, 1-wantWR)
	}
}

// TestMFWinRateRouterErrors pins the fail-loud surface.
func TestMFWinRateRouterErrors(t *testing.T) {
	r := &MFWinRateRouter{
		ModelEmbed: map[string][]float64{
			"a": {1, 0},
			"z": {0, 0}, // zero norm
		},
		Classifier: []float64{1, 1},
	}

	if _, err := r.Select(nil, []float64{1, 1}); err == nil {
		t.Error("Select(nil): expected error, got nil")
	}
	if _, err := r.Score("missing", []float64{1, 1}); err == nil {
		t.Error("Score(unknown): expected error, got nil")
	}
	if _, err := r.Score("a", []float64{1, 1, 1}); err == nil {
		t.Error("Score(dim mismatch): expected error, got nil")
	}
	if _, err := r.Score("z", []float64{1, 1}); err == nil {
		t.Error("Score(zero norm): expected error, got nil")
	}
}
