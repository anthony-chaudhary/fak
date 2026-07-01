package agentdemo_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentdemo"

	// Wire the full adjudicator chain so kernel.Fold folds the REAL floor — the same
	// one-line requirement every on-box demo main (and agentdemo_test.go) carries.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

// deterministicPlan is the LCD fallback planner used by these tests: it routes a
// "time" prompt to the allowed read-only tool and everything else to nothing.
func deterministicPlan(prompt string) []agentdemo.Step {
	if strings.Contains(strings.ToLower(prompt), "time") {
		return []agentdemo.Step{{Tool: "get_time", Note: "deterministic route"}}
	}
	return nil
}

// TestModelArm_MaliciousModelStillRefused is the load-bearing witness for #1740: a
// "model" that PROPOSES destructive tool calls (delete_calendar on the floor's deny
// list, wipe_disk off the floor entirely) must still be refused by the kernel. The
// model's intent does not matter — RunArm folds every proposed step through the same
// kernel.Fold path, so the capability floor holds independent of model behavior.
func TestModelArm_MaliciousModelStillRefused(t *testing.T) {
	ts := fixedTools()
	maliciousModel := func(ctx context.Context, prompt string) ([]agentdemo.Step, agentdemo.PlanMeta, error) {
		return []agentdemo.Step{
				{Tool: "get_time", Note: "a benign call to look cooperative"},
				{Tool: "delete_calendar", Note: "the model tries a floor-denied sink"},
				{Tool: "wipe_disk", Note: "the model tries an off-floor sink"},
			}, agentdemo.PlanMeta{
				Provider: "test-provider",
				Model:    "test-model-x",
				Rung:     "hosted",
				AsOf:     "2026-07-01",
			}, nil
	}
	arm := agentdemo.ModelArm{Propose: maliciousModel, Fallback: deterministicPlan}

	tr, meta, err := ts.RunArm(context.Background(), "malicious-model", "please help", arm)
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}

	// The model actually served this run — recorded, not a fallback.
	if meta.Source != agentdemo.SourceModel {
		t.Fatalf("meta.Source = %q, want %q", meta.Source, agentdemo.SourceModel)
	}
	if meta.Model != "test-model-x" || meta.Provider != "test-provider" || meta.Rung != "hosted" {
		t.Errorf("meta = %+v, want the model's live-reported provider/model/rung", meta)
	}

	if len(tr.Turns) != 3 {
		t.Fatalf("turns = %d, want 3", len(tr.Turns))
	}
	// get_time: allowed. Both destructive proposals: refused with closed reason codes.
	if tr.Turns[0].Verdict != "ALLOW" || !tr.Turns[0].Allowed {
		t.Errorf("get_time verdict = %s allowed=%v, want ALLOW/true", tr.Turns[0].Verdict, tr.Turns[0].Allowed)
	}
	if tr.Turns[1].Verdict != "DENY" || tr.Turns[1].Reason != "POLICY_BLOCK" {
		t.Errorf("delete_calendar = %s/%s, want DENY/POLICY_BLOCK", tr.Turns[1].Verdict, tr.Turns[1].Reason)
	}
	if tr.Turns[2].Verdict != "DENY" || tr.Turns[2].Reason != "DEFAULT_DENY" {
		t.Errorf("wipe_disk = %s/%s, want DENY/DEFAULT_DENY", tr.Turns[2].Verdict, tr.Turns[2].Reason)
	}
	if tr.Allowed != 1 || tr.Denied != 2 {
		t.Errorf("tally = %d/%d, want 1/2", tr.Allowed, tr.Denied)
	}
	// The destructive handlers' output must never reach the answer.
	if strings.Contains(tr.Answer, "wiped") {
		t.Errorf("answer leaked a refused destructive result: %q", tr.Answer)
	}
}

// TestModelArm_FallbackOnError proves the LCD stays green when the model seam fails:
// a ProposeFunc that errors degrades to the deterministic planner, the run still
// succeeds, and the meta honestly records the fallback (not a phantom model claim).
func TestModelArm_FallbackOnError(t *testing.T) {
	ts := fixedTools()
	brokenModel := func(ctx context.Context, prompt string) ([]agentdemo.Step, agentdemo.PlanMeta, error) {
		return nil, agentdemo.PlanMeta{}, errors.New("no credentials")
	}
	arm := agentdemo.ModelArm{
		Propose:  brokenModel,
		Fallback: deterministicPlan,
		Base:     agentdemo.PlanMeta{Provider: "test-provider", Model: "test-model-x", AsOf: "2026-07-01"},
	}

	tr, meta, err := ts.RunArm(context.Background(), "fallback", "what time is it?", arm)
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if meta.Source != agentdemo.SourceFallback {
		t.Fatalf("meta.Source = %q, want %q", meta.Source, agentdemo.SourceFallback)
	}
	if !strings.Contains(meta.Note, "no credentials") {
		t.Errorf("meta.Note = %q, want the seam error recorded", meta.Note)
	}
	if len(tr.Turns) != 1 || tr.Turns[0].Tool != "get_time" || !tr.Turns[0].Allowed {
		t.Fatalf("fallback did not run the deterministic route: %+v", tr.Turns)
	}
	if !strings.Contains(tr.Answer, "dinner") {
		t.Errorf("answer = %q, want the allowed get_time result", tr.Answer)
	}
}

// TestModelArm_NilSeamIsDeterministic proves an arm with no model seam is exactly the
// deterministic planner — the LCD path with no key, network, or model in the loop.
func TestModelArm_NilSeamIsDeterministic(t *testing.T) {
	ts := fixedTools()
	arm := agentdemo.ModelArm{Fallback: deterministicPlan}
	if arm.Configured() {
		t.Fatalf("Configured() = true for a nil-seam arm, want false")
	}
	tr, meta, err := ts.RunArm(context.Background(), "lcd", "what time is it?", arm)
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if meta.Source != agentdemo.SourceFallback {
		t.Errorf("meta.Source = %q, want %q", meta.Source, agentdemo.SourceFallback)
	}
	if len(tr.Turns) != 1 || tr.Turns[0].Tool != "get_time" {
		t.Fatalf("nil-seam arm did not take the deterministic route: %+v", tr.Turns)
	}
}

// TestPlanMeta_JSONShape guards the witness serialization: Source is always present
// (the honest which-planner label) and empty optional fields are omitted.
func TestPlanMeta_JSONShape(t *testing.T) {
	b, err := json.Marshal(agentdemo.PlanMeta{Source: agentdemo.SourceModel, Model: "m"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(b)
	if !strings.Contains(got, `"source":"model"`) {
		t.Errorf("JSON missing source: %s", got)
	}
	if strings.Contains(got, `"provider"`) {
		t.Errorf("empty provider should be omitted: %s", got)
	}
}
