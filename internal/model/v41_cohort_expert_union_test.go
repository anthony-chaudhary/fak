package model

import "testing"

func setV41CohortExpertUnionModeForTest(t *testing.T, mode v41CohortExpertUnionMode) {
	t.Helper()
	prev := v41CohortExpertUnionSetting
	v41CohortExpertUnionSetting = func() v41CohortExpertUnionMode { return mode }
	t.Cleanup(func() { v41CohortExpertUnionSetting = prev })
}

// fak-test:runtime fast est=1ms lane=default
func TestV41CohortExpertUnionMode(t *testing.T) {
	for value, want := range map[string]v41CohortExpertUnionMode{
		"":       v41CohortExpertUnionOn,
		"1":      v41CohortExpertUnionOn,
		"0":      v41CohortExpertUnionOff,
		" 0 ":    v41CohortExpertUnionOff,
		"auto":   v41CohortExpertUnionAuto,
		"AUTO":   v41CohortExpertUnionAuto,
		"always": v41CohortExpertUnionOn,
	} {
		if got := parseV41CohortExpertUnionMode(value); got != want {
			t.Fatalf("parse %q = %d, want %d", value, got, want)
		}
	}
	for _, test := range []struct {
		mode   v41CohortExpertUnionMode
		shared int
		want   bool
	}{
		{v41CohortExpertUnionOn, 0, true},
		{v41CohortExpertUnionAuto, 0, false},
		{v41CohortExpertUnionAuto, V41RouterTopK - 1, false},
		{v41CohortExpertUnionAuto, V41RouterTopK, true},
		{v41CohortExpertUnionAuto, 3 * V41RouterTopK, true},
	} {
		if got := test.mode.admitsSharedRows(test.shared); got != test.want {
			t.Fatalf("mode %d shared %d admits=%v, want %v", test.mode, test.shared, got, test.want)
		}
	}
}
