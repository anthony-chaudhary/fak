package qwen38quant

import (
	"strings"
	"testing"
)

func TestSelfcheck(t *testing.T) {
	if err := Selfcheck(); err != nil {
		t.Fatal(err)
	}
}
func TestRequiredTenArms(t *testing.T) {
	if len(RequiredArms) != 10 {
		t.Fatalf("arms=%d", len(RequiredArms))
	}
}
func TestCorpusDigestStable(t *testing.T) {
	got := CorpusDigest(DefaultCorpus())
	if len(got) != 64 {
		t.Fatal(got)
	}
}
func TestFailedTrialRetained(t *testing.T) {
	c := DefaultCorpus()
	r := validFixture(c)
	r.Trials[0].Quality = "FAIL"
	r.Trials[0].Failure = ""
	r.Verdict = "HOLD"
	if err := Validate(r, c); err == nil || !strings.Contains(err.Error(), "retained") {
		t.Fatalf("err=%v", err)
	}
}
func TestAcceptanceOnlyNeverCampaignReady(t *testing.T) {
	c := DefaultCorpus()
	for _, a := range []string{"fp8", "q4_k_m"} {
		r := LegacyAcceptance(a, c.ID, "Qwen3.8-27B", []float64{1, 2, 3})
		if err := Validate(r, c); err == nil {
			t.Fatal("accepted", a)
		}
	}
}
