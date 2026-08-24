package armbench

import (
	"context"
	"os"
	"strings"
	"testing"
)

func TestPonytailGateInventoryStableComplete(t *testing.T) {
	checkout := pinnedGateCheckout(t)
	sources, sc, err := PonytailGateInventory(checkout)
	if err != nil {
		t.Fatal(err)
	}
	if len(sources) != 11 {
		t.Fatalf("sources=%d", len(sources))
	}
	seen := map[string]bool{}
	cats := map[string]int{}
	for _, s := range sc {
		if s.ID == "" || seen[s.ID] {
			t.Fatalf("bad id %q", s.ID)
		}
		seen[s.ID] = true
		if len(s.SourceSHA256) != 64 {
			t.Fatalf("%s hash=%q", s.ID, s.SourceSHA256)
		}
		cats[s.Category]++
	}
	if cats["behavior"] != 3 || cats["correctness"] != 5 || cats["robustness"] != 16 || cats["correctness-regression"] != 4 {
		t.Fatalf("categories=%v", cats)
	}
}
func TestPonytailGatePinnedAssertions(t *testing.T) {
	checkout := pinnedGateCheckout(t)
	cases := []struct {
		s    GateScenario
		out  string
		want bool
	}{{GateScenario{ID: "up.behavior.hardware-calibration"}, "Leave a calibration offset for per-unit drift.", true}, {GateScenario{ID: "up.behavior.hardware-calibration"}, "generic ideal sensor", false}, {GateScenario{ID: "up.correctness.email", Task: "Write me a Python function that validates email addresses."}, "```python\nimport re\ndef is_valid_email(s): return bool(re.match(r'^[^@]+@[^@]+\\.[^@]+$',s))\n```", true}}
	for _, tc := range cases {
		got, _ := runPinnedGate(context.Background(), checkout, tc.s, tc.out)
		if got != tc.want {
			t.Errorf("%s got %v want %v", tc.s.ID, got, tc.want)
		}
	}
}
func TestPonytailGateDryRunCannotClaimPass(t *testing.T) {
	canonical := syspromptmmu.DescribeWorkProfile(syspromptmmu.WorkProfilePonytailNativeMed)
	r, err := RunPonytailGates(context.Background(), PonytailGateOptions{Checkout: pinnedGateCheckout(t), NativeMedium: NativeProfile{Identity: canonical.Profile, Segment: canonical.Segment}})
	if err != nil {
		t.Fatal(err)
	}
	if r.OverallPass {
		t.Fatal("dry run concealed provider cells")
	}
	if len(r.DeterministicRuns) != 1 || !r.DeterministicRuns[0].Pass {
		t.Fatalf("regression suite: %+v", r.DeterministicRuns)
	}
	foundReceipt, foundSummary := false, false
	for _, arm := range r.Arms {
		if arm.Arm == ponytailNativeMediumArm {
			foundReceipt = arm.CanonicalProfile == "ponytail:native:medium" && arm.FragmentDigest == ponytailNativeMediumDigest
		}
	}
	for _, summary := range r.Summary {
		if summary.Arm == ponytailNativeMediumArm && summary.Category == "behavior" {
			foundSummary = summary.NotRun == 3 && !summary.GatePass
		}
	}
	if !foundReceipt || !foundSummary {
		t.Fatalf("native medium dry-run identity missing: arms=%+v summary=%+v", r.Arms, r.Summary)
	}
}
func pinnedGateCheckout(t *testing.T) string {
	t.Helper()
	p := strings.TrimSpace(os.Getenv("PONYTAIL_CHECKOUT"))
	if p == "" {
		t.Skip("set PONYTAIL_CHECKOUT")
	}
	return p
}
