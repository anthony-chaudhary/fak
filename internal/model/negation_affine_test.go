package model

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

func loadNegationAffineLayers(t *testing.T) []NegationAffineLayer {
	t.Helper()
	data, err := os.ReadFile("testdata/negation_affine_pairs.json")
	if err != nil {
		t.Fatal(err)
	}
	var layers []NegationAffineLayer
	if err := json.Unmarshal(data, &layers); err != nil {
		t.Fatal(err)
	}
	return layers
}

func affineLayer(t *testing.T, layer int) NegationAffineLayer {
	t.Helper()
	for _, candidate := range loadNegationAffineLayers(t) {
		if candidate.Layer == layer {
			return candidate
		}
	}
	t.Fatalf("missing layer %d", layer)
	return NegationAffineLayer{}
}

func TestNegationAffineFit(t *testing.T) {
	candidate := affineLayer(t, 2)
	op, err := FitNegationAffine(candidate.Layer, candidate.Pairs)
	if err != nil {
		t.Fatal(err)
	}
	if len(op.Steering) != 2 || len(op.Matrix) != 2 {
		t.Fatalf("operator shape=%+v", op)
	}
	report, err := SweepNegationAffine(loadNegationAffineLayers(t))
	if err != nil {
		t.Fatal(err)
	}
	if report.BestLayer != 2 {
		t.Fatalf("best layer=%d report=%+v", report.BestLayer, report)
	}
	var best NegationAffineLayerResult
	for _, row := range report.Layers {
		if row.Layer == report.BestLayer {
			best = row
		}
	}
	if best.AffineRMSE > 1e-6 {
		t.Fatalf("affine fit did not converge: %+v", best)
	}
	if best.AffineRMSE >= best.SteeringRMSE {
		t.Fatalf("affine did not beat steering: %+v", best)
	}
	t.Logf("best_layer=L%d affine_rmse=%.9f steering_rmse=%.6f effect=%.6f", report.BestLayer, best.AffineRMSE, best.SteeringRMSE, report.BestEffect)
}

func TestNegationActivationPatch(t *testing.T) {
	candidate := affineLayer(t, 2)
	op, err := FitNegationAffine(candidate.Layer, candidate.Pairs)
	if err != nil {
		t.Fatal(err)
	}
	// Run a fresh forward pass through the real gated residual seam. The operator
	// must remain quiet at other layers and mutate only its fitted install point.
	m := &Model{Cfg: Config{EnableResidualHook: true}}
	m.SetResidualHook(op.Hook())
	hidden := []float32{0.25, 0.5}
	composeBlockAtLayer(0, PreNorm, hidden, identityNorm(), identityNorm(), 1e-5, m.Cfg, zeroSublayer, zeroSublayer)
	if !reflect.DeepEqual(hidden, []float32{0.25, 0.5}) {
		t.Fatalf("operator fired at control layer: %v", hidden)
	}
	composeBlockAtLayer(candidate.Layer, PreNorm, hidden, identityNorm(), identityNorm(), 1e-5, m.Cfg, zeroSublayer, zeroSublayer)
	want := []float32{1.75, -0.5}
	if !reflect.DeepEqual(hidden, want) {
		t.Fatalf("fresh-pass patched=%v want=%v", hidden, want)
	}

	// The generic capture/inject harness must carry the same fitted state.
	patch, err := NewActivationPatch(candidate.Layer)
	if err != nil {
		t.Fatal(err)
	}
	viaPatch := []float32{0.25, 0.5}
	if err := op.PatchActivation(patch, viaPatch); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(viaPatch, want) {
		t.Fatalf("activation-patch state=%v want=%v", viaPatch, want)
	}
	report, err := SweepNegationAffine(loadNegationAffineLayers(t))
	if err != nil {
		t.Fatal(err)
	}
	var best NegationAffineLayerResult
	for _, row := range report.Layers {
		if row.Layer == report.BestLayer {
			best = row
		}
	}
	const margin = 0.35
	if best.PatchEffect-best.RandomEffect < margin {
		t.Fatalf("patch does not beat random control by %.2f: %+v", margin, best)
	}
	if best.PatchedTargetScore < .999999 {
		t.Fatalf("behavioral target did not flip: %+v", best)
	}
	t.Logf("L%d target_score unpatched=%.6f patched=%.6f random=%.6f margin=%.6f", best.Layer, best.UnpatchedTargetScore, best.PatchedTargetScore, best.RandomTargetScore, best.PatchEffect-best.RandomEffect)
}

func TestNegationAffineLayerSweepWitness(t *testing.T) {
	report, err := SweepNegationAffine(loadNegationAffineLayers(t))
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("layer-sweep witness:\n%s", encoded)
	data, err := os.ReadFile("testdata/negation_affine_layer_sweep.json")
	if err != nil {
		t.Fatal(err)
	}
	var want NegationAffineSweepReport
	if err := json.Unmarshal(data, &want); err != nil {
		t.Fatal(err)
	}
	const numericalTolerance = 1e-12
	closeFloat := func(got, want float64) bool {
		return math.Abs(got-want) <= numericalTolerance
	}
	if report.BestLayer != want.BestLayer || !closeFloat(report.BestEffect, want.BestEffect) || len(report.Layers) != len(want.Layers) {
		t.Fatalf("layer-sweep witness structure drift\ngot=%+v\nwant=%+v", report, want)
	}
	for i, got := range report.Layers {
		want := want.Layers[i]
		if got.Layer != want.Layer {
			t.Fatalf("layer-sweep witness row %d changed layer: got=%d want=%d", i, got.Layer, want.Layer)
		}
		for _, field := range []struct {
			name      string
			got, want float64
		}{
			{name: "steering_rmse", got: got.SteeringRMSE, want: want.SteeringRMSE},
			{name: "affine_rmse", got: got.AffineRMSE, want: want.AffineRMSE},
			{name: "unpatched_target_score", got: got.UnpatchedTargetScore, want: want.UnpatchedTargetScore},
			{name: "patched_target_score", got: got.PatchedTargetScore, want: want.PatchedTargetScore},
			{name: "random_target_score", got: got.RandomTargetScore, want: want.RandomTargetScore},
			{name: "patch_effect", got: got.PatchEffect, want: want.PatchEffect},
			{name: "random_effect", got: got.RandomEffect, want: want.RandomEffect},
		} {
			if !closeFloat(field.got, field.want) {
				t.Fatalf("layer-sweep witness L%d %s drift: got=%.17g want=%.17g tolerance=%g", got.Layer, field.name, field.got, field.want, numericalTolerance)
			}
		}
	}
}
