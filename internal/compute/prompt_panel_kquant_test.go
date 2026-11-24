package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestResolvePromptPanelKQuantMode(t *testing.T) {
	for _, tc := range []struct {
		mode, legacy string
		want         PromptPanelKQuantMode
		err          bool
	}{
		{"", "", PromptPanelKQuantBatched, false},
		{"", "1", PromptPanelKQuantRowExact, false},
		{"", "0", PromptPanelKQuantBatched, false},
		{"batched", "", PromptPanelKQuantBatched, false},
		{" Batched ", "", PromptPanelKQuantBatched, false},
		{"batched", "1", PromptPanelKQuantBatched, false}, // explicit mode wins over the legacy opt-in
		{"rowexact", "", PromptPanelKQuantRowExact, false},
		{"row-exact", "", PromptPanelKQuantRowExact, false},
		{"off", "1", PromptPanelKQuantOff, false},
		{"0", "", PromptPanelKQuantOff, false},
		{"fast", "", PromptPanelKQuantOff, true},
	} {
		got, err := ResolvePromptPanelKQuantMode(tc.mode, tc.legacy)
		if got != tc.want || (err != nil) != tc.err {
			t.Fatalf("Resolve(%q,%q) = %v,%v want %v err=%v", tc.mode, tc.legacy, got, err, tc.want, tc.err)
		}
		if tc.err && !errors.Is(err, ErrPromptPanelKQuantModeUnknown) {
			t.Fatalf("Resolve(%q) error %v does not wrap ErrPromptPanelKQuantModeUnknown", tc.mode, err)
		}
	}
}

// TestPromptPanelKQuantDtypeAdmission is the CUDA panel GEMM admission table: off restores
// the pre-batched admission (Q8_0 only), and no dtype is ever both row-exact and batched.
//
// fak-test:runtime fast est=5ms lane=default
func TestPromptPanelKQuantDtypeAdmission(t *testing.T) {
	type admit struct{ rowExact, batched bool }
	table := map[PromptPanelKQuantMode]map[Dtype]admit{
		PromptPanelKQuantBatched:  {Q8_0: {true, false}, Q4_K: {false, true}, Q6_K: {false, true}},
		PromptPanelKQuantRowExact: {Q8_0: {true, false}, Q4_K: {true, false}, Q6_K: {true, false}},
		PromptPanelKQuantOff:      {Q8_0: {true, false}},
	}
	for mode, rows := range table {
		for _, dt := range []Dtype{F32, F16, Q8_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K} {
			want := rows[dt]
			got := admit{promptPanelRowExactFor(mode, dt), promptPanelBatchedFor(mode, dt)}
			if got != want {
				t.Fatalf("%v/%v: admit %+v want %+v", mode, dt, got, want)
			}
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestPromptPanelBatchRowsBoundsNonMMQScratch(t *testing.T) {
	for _, tc := range []struct {
		dt          Dtype
		tokens      int
		mmq         bool
		want        int
		description string
	}{
		{Q4_K, 512, true, 512, "mmq takes the whole chunk"},
		{Q4_K, 512, false, DequantGEMMScratchRows - 1, "non-mmq stays below the dequant arm"},
		{Q4_K, DequantGEMMScratchRows - 1, false, DequantGEMMScratchRows - 1, "short chunk is one launch"},
		{Q4_K, 5, false, 5, "tile panel"},
		{Q6_K, 512, false, 512, "Q6_K tile panel allocates no weight image"},
	} {
		if got := promptPanelBatchRows(tc.dt, tc.tokens, tc.mmq); got != tc.want {
			t.Fatalf("%s: rows=%d want %d", tc.description, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestSetPromptPanelKQuantModeRestores(t *testing.T) {
	old := SetPromptPanelKQuantMode(PromptPanelKQuantOff)
	defer SetPromptPanelKQuantMode(old)
	if got := CurrentPromptPanelKQuantMode(); got != PromptPanelKQuantOff {
		t.Fatalf("mode=%v want off", got)
	}
	if prev := SetPromptPanelKQuantMode(PromptPanelKQuantRowExact); prev != PromptPanelKQuantOff {
		t.Fatalf("previous=%v want off", prev)
	}
}
