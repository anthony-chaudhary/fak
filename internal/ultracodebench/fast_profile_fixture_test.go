package ultracodebench

import (
	"encoding/json"
	"os"
	"testing"
)

func TestFastProfileReplayBundleHeterogeneousBindings(t *testing.T) {
	raw, err := os.ReadFile("testdata/issue8963-fast-profile-replay.json")
	if err != nil {
		t.Fatal(err)
	}
	var b FastProfileBundle
	if err := json.Unmarshal(raw, &b); err != nil {
		t.Fatal(err)
	}
	r := EvaluateFastProfile(b)
	if len(r.Comparisons) != 2 {
		t.Fatalf("comparisons=%d", len(r.Comparisons))
	}
	if r.Comparisons[0].Binding.Provider == r.Comparisons[1].Binding.Provider || r.Comparisons[0].Binding.Harness == r.Comparisons[1].Binding.Harness {
		t.Fatal("bindings are not heterogeneous")
	}
	if r.Comparisons[0].Verdict != "GAIN" || r.Comparisons[1].Verdict != "NO_GAIN" {
		t.Fatalf("verdicts=%s,%s reasons=%v", r.Comparisons[0].Verdict, r.Comparisons[1].Verdict, r.Reasons)
	}
}
