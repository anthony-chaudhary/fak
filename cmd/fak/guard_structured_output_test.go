package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/policy"
)

// StructuredOutput is the harness's structured-output RETURN CHANNEL: Claude Code
// models "return the final response as structured JSON" as a tool named
// StructuredOutput — prompt-hook evaluators (the /goal Stop-condition judge) and
// schema'd subagents (Agent/Workflow structured outputs) rely on the model calling it.
// It executes nothing: the harness intercepts the call client-side and reads its args
// as the result, so admitting it grants no capability — the same reasoning that put
// Task*/SendMessage/ToolSearch on the floor (see the guardDefaultPolicyJSON comment).
//
// If the shipped floor default-denies it, two independent failures fire on every
// structured-output sidechannel call through a guarded gateway: the inbound tool-def
// compactor prunes its DEFINITION from the passthrough tools[] (NeverAdmits gates the
// drop set), and a call that still arrives is DEFAULT_DENY'd — either way the caller's
// structured verdict never materializes. This is the same out-of-band-call failure
// class as the repetition-steer/stop-hook fix in internal/gateway (9e94e4ca).
func TestGuardFloorAdmitsStructuredOutputReturnChannel(t *testing.T) {
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatalf("embedded guard floor is not a valid manifest: %v", err)
	}
	if rt.Adjudicator.NeverAdmits("StructuredOutput") {
		t.Fatal("the shipped guard floor must admit StructuredOutput (the harness's structured-output return channel); NeverAdmits=true prunes its definition from every Anthropic passthrough request and DEFAULT_DENYs any proposed call, silently breaking prompt-hook evaluators and schema'd subagents")
	}
}
