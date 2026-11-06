package adjudicator

// The cross-package injection-quarantine composition formerly asserted here is
// witnessed by internal/normgate's own suite (TestQuarantinesObfuscatedInjection
// and peers assert Quarantine/PROMPT_INJECTION through normgate.New().Admit);
// this file keeps the adjudicator-side research-egress admission pin.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // registers the blob PageOut/Resolver backend
)

// TestResearchEgressAllowlistedFetchAdmitted pins the adjudicator's egress
// rung: an allowlisted research WebFetch is admitted by the adjudicator's
// egress rung.
func TestResearchEgressAllowlistedFetchAdmitted(t *testing.T) {
	ctx := context.Background()
	call := inlineCall("WebFetch", `{"url":"https://arxiv.org/abs/1706.03762"}`)
	a := New(Policy{ResearchEgressAllowHosts: []string{"arxiv.org"}})
	if v := a.Adjudicate(ctx, call); v.Kind != abi.VerdictAllow {
		t.Fatalf("allowlisted research fetch got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}
