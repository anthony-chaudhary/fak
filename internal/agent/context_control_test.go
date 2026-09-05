package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

func TestContextControlInspect(t *testing.T) {
	DisarmContextControl()
	t.Cleanup(DisarmContextControl)

	catalog, err := ArmContextControl()
	if err != nil {
		t.Fatalf("ArmContextControl: %v", err)
	}
	if len(catalog) != 1 {
		t.Fatalf("catalog len = %d, want 1", len(catalog))
	}

	st := GetActiveContextControlState()
	if st == nil {
		t.Fatalf("GetActiveContextControlState() is nil")
	}

	ctx := context.Background()
	receipt := st.Execute(ctx, ContextControlRequest{
		Action: ActionInspect,
	})

	if receipt.Status != StatusAccepted {
		t.Fatalf("receipt.Status = %q, want %q", receipt.Status, StatusAccepted)
	}
	if receipt.Action != ActionInspect {
		t.Fatalf("receipt.Action = %q, want %q", receipt.Action, ActionInspect)
	}
	if receipt.Idempotent {
		t.Fatalf("receipt.Idempotent = true, want false")
	}

	details := receipt.Details
	if details == nil {
		t.Fatalf("receipt.Details is nil")
	}
	if _, ok := details["mandatory_policy"]; !ok {
		t.Errorf("expected mandatory_policy in details")
	}
	if _, ok := details["advisory_settings"]; !ok {
		t.Errorf("expected advisory_settings in details")
	}
	if _, ok := details["active_pins"]; !ok {
		t.Errorf("expected active_pins in details")
	}
	if _, ok := details["active_releases"]; !ok {
		t.Errorf("expected active_releases in details")
	}
}

func TestContextControlAcceptedPin(t *testing.T) {
	DisarmContextControl()
	t.Cleanup(DisarmContextControl)

	_, err := ArmContextControl()
	if err != nil {
		t.Fatalf("ArmContextControl: %v", err)
	}
	st := GetActiveContextControlState()

	ctx := context.Background()
	req := ContextControlRequest{
		Action:     ActionPin,
		SpanIDs:    []string{"span-user-task", "span-code-reference"},
		Digest:     "sha256:0123456789abcdef",
		Provenance: "user_turn_1",
		TTLSeconds: 3600,
		Horizon:    3,
	}

	receipt := st.Execute(ctx, req)
	if receipt.Status != StatusAccepted {
		t.Fatalf("pin receipt.Status = %q, want %q, reason = %s", receipt.Status, StatusAccepted, receipt.Reason)
	}
	if receipt.Action != ActionPin {
		t.Fatalf("pin receipt.Action = %q, want %q", receipt.Action, ActionPin)
	}

	snap := st.Snapshot()
	if len(snap.ActivePins) != 2 {
		t.Fatalf("snapshot ActivePins len = %d, want 2", len(snap.ActivePins))
	}
	if snap.ActivePins[0] != "span-code-reference" || snap.ActivePins[1] != "span-user-task" {
		t.Errorf("unexpected ActivePins: %v", snap.ActivePins)
	}

	// Verify inspect reflects pinned spans
	inspectReceipt := st.Execute(ctx, ContextControlRequest{Action: ActionInspect})
	if inspectReceipt.Status != StatusAccepted {
		t.Fatalf("inspect receipt.Status = %q, want %q", inspectReceipt.Status, StatusAccepted)
	}
	pins, ok := inspectReceipt.Details["active_pins"].([]string)
	if !ok || len(pins) != 2 {
		t.Fatalf("inspect active_pins = %v, want 2 items", inspectReceipt.Details["active_pins"])
	}
}

func TestContextControlAcceptedRelease(t *testing.T) {
	DisarmContextControl()
	t.Cleanup(DisarmContextControl)

	_, err := ArmContextControl()
	if err != nil {
		t.Fatalf("ArmContextControl: %v", err)
	}
	st := GetActiveContextControlState()
	ctx := context.Background()

	// First pin two spans
	st.Execute(ctx, ContextControlRequest{
		Action:  ActionPin,
		SpanIDs: []string{"span-to-keep", "span-to-release"},
	})

	// Now release one span
	releaseReq := ContextControlRequest{
		Action:  ActionRelease,
		SpanIDs: []string{"span-to-release"},
	}
	receipt := st.Execute(ctx, releaseReq)
	if receipt.Status != StatusAccepted {
		t.Fatalf("release receipt.Status = %q, want %q, reason: %s", receipt.Status, StatusAccepted, receipt.Reason)
	}
	if receipt.Action != ActionRelease {
		t.Fatalf("release receipt.Action = %q, want %q", receipt.Action, ActionRelease)
	}

	snap := st.Snapshot()
	if len(snap.ActivePins) != 1 || snap.ActivePins[0] != "span-to-keep" {
		t.Errorf("unexpected ActivePins after release: %v", snap.ActivePins)
	}
	if len(snap.ActiveReleases) != 1 || snap.ActiveReleases[0] != "span-to-release" {
		t.Errorf("unexpected ActiveReleases: %v", snap.ActiveReleases)
	}
}

func TestContextControlRefusedMutations(t *testing.T) {
	DisarmContextControl()
	t.Cleanup(DisarmContextControl)

	_, err := ArmContextControl()
	if err != nil {
		t.Fatalf("ArmContextControl: %v", err)
	}
	st := GetActiveContextControlState()
	ctx := context.Background()

	// 1. Unknown action -> MISROUTE
	t.Run("UnknownAction", func(t *testing.T) {
		receipt := st.Execute(ctx, ContextControlRequest{
			Action: "purge_all_history",
		})
		if receipt.Status != StatusRefused {
			t.Fatalf("status = %q, want %q", receipt.Status, StatusRefused)
		}
		if receipt.Reason != abi.ReasonName(abi.ReasonMisroute) {
			t.Errorf("reason = %q, want %q", receipt.Reason, abi.ReasonName(abi.ReasonMisroute))
		}
	})

	// 2. Over-budget budget preference -> OVERSIZE
	t.Run("OverBudgetPreference", func(t *testing.T) {
		huge := 9999999
		receipt := st.Execute(ctx, ContextControlRequest{
			Action: ActionBudgetPreference,
			Budget: &huge,
		})
		if receipt.Status != StatusRefused {
			t.Fatalf("status = %q, want %q", receipt.Status, StatusRefused)
		}
		if receipt.Reason != abi.ReasonName(abi.ReasonOversize) {
			t.Errorf("reason = %q, want %q", receipt.Reason, abi.ReasonName(abi.ReasonOversize))
		}
	})

	// 3. Release mandatory protected span -> POLICY_BLOCK
	t.Run("ReleaseProtectedSpan", func(t *testing.T) {
		receipt := st.Execute(ctx, ContextControlRequest{
			Action:  ActionRelease,
			SpanIDs: []string{"sys:prompt"},
		})
		if receipt.Status != StatusRefused {
			t.Fatalf("status = %q, want %q", receipt.Status, StatusRefused)
		}
		if receipt.Reason != abi.ReasonName(abi.ReasonPolicyBlock) {
			t.Errorf("reason = %q, want %q", receipt.Reason, abi.ReasonName(abi.ReasonPolicyBlock))
		}
	})

	// 4. Release mandatory prefix span -> POLICY_BLOCK
	t.Run("ReleaseProtectedPrefixSpan", func(t *testing.T) {
		receipt := st.Execute(ctx, ContextControlRequest{
			Action:  ActionRelease,
			SpanIDs: []string{"safety:guard_rule_1"},
		})
		if receipt.Status != StatusRefused {
			t.Fatalf("status = %q, want %q", receipt.Status, StatusRefused)
		}
		if receipt.Reason != abi.ReasonName(abi.ReasonPolicyBlock) {
			t.Errorf("reason = %q, want %q", receipt.Reason, abi.ReasonName(abi.ReasonPolicyBlock))
		}
	})

	// 5. Missing required params for pin -> MALFORMED
	t.Run("MissingSpanIDsForPin", func(t *testing.T) {
		receipt := st.Execute(ctx, ContextControlRequest{
			Action:  ActionPin,
			SpanIDs: nil,
		})
		if receipt.Status != StatusRefused {
			t.Fatalf("status = %q, want %q", receipt.Status, StatusRefused)
		}
		if receipt.Reason != abi.ReasonName(abi.ReasonMalformed) {
			t.Errorf("reason = %q, want %q", receipt.Reason, abi.ReasonName(abi.ReasonMalformed))
		}
	})

	// 6. Negative budget -> MALFORMED
	t.Run("NegativeBudget", func(t *testing.T) {
		neg := -100
		receipt := st.Execute(ctx, ContextControlRequest{
			Action: ActionBudgetPreference,
			Budget: &neg,
		})
		if receipt.Status != StatusRefused {
			t.Fatalf("status = %q, want %q", receipt.Status, StatusRefused)
		}
		if receipt.Reason != abi.ReasonName(abi.ReasonMalformed) {
			t.Errorf("reason = %q, want %q", receipt.Reason, abi.ReasonName(abi.ReasonMalformed))
		}
	})

	// 7. Budget below minimum floor -> MALFORMED
	t.Run("BudgetBelowFloor", func(t *testing.T) {
		tiny := 10
		receipt := st.Execute(ctx, ContextControlRequest{
			Action: ActionBudgetPreference,
			Budget: &tiny,
		})
		if receipt.Status != StatusRefused {
			t.Fatalf("status = %q, want %q", receipt.Status, StatusRefused)
		}
		if receipt.Reason != abi.ReasonName(abi.ReasonMalformed) {
			t.Errorf("reason = %q, want %q", receipt.Reason, abi.ReasonName(abi.ReasonMalformed))
		}
	})

	// 8. Invalid retrieval scope -> POLICY_BLOCK
	t.Run("InvalidScope", func(t *testing.T) {
		receipt := st.Execute(ctx, ContextControlRequest{
			Action: ActionRetrievalScope,
			Scope:  "unauthorized_external_scope",
		})
		if receipt.Status != StatusRefused {
			t.Fatalf("status = %q, want %q", receipt.Status, StatusRefused)
		}
		if receipt.Reason != abi.ReasonName(abi.ReasonPolicyBlock) {
			t.Errorf("reason = %q, want %q", receipt.Reason, abi.ReasonName(abi.ReasonPolicyBlock))
		}
	})

	// 9. Invalid digest format -> MALFORMED
	t.Run("InvalidDigestFormat", func(t *testing.T) {
		receipt := st.Execute(ctx, ContextControlRequest{
			Action:  ActionPin,
			SpanIDs: []string{"span-1"},
			Digest:  "bad sha",
		})
		if receipt.Status != StatusRefused {
			t.Fatalf("status = %q, want %q", receipt.Status, StatusRefused)
		}
		if receipt.Reason != abi.ReasonName(abi.ReasonMalformed) {
			t.Errorf("reason = %q, want %q", receipt.Reason, abi.ReasonName(abi.ReasonMalformed))
		}
	})

	// 10. Over-budget pins ceiling -> OVERSIZE
	t.Run("OverBudgetPins", func(t *testing.T) {
		manySpans := make([]string, 600)
		for i := 0; i < 600; i++ {
			manySpans[i] = fmt.Sprintf("span-overflow-%d", i)
		}
		receipt := st.Execute(ctx, ContextControlRequest{
			Action:  ActionPin,
			SpanIDs: manySpans,
		})
		if receipt.Status != StatusRefused {
			t.Fatalf("status = %q, want %q", receipt.Status, StatusRefused)
		}
		if receipt.Reason != abi.ReasonName(abi.ReasonOversize) {
			t.Errorf("reason = %q, want %q", receipt.Reason, abi.ReasonName(abi.ReasonOversize))
		}
	})
}

func TestContextControlRestore(t *testing.T) {
	DisarmContextControl()
	t.Cleanup(DisarmContextControl)

	_, err := ArmContextControl()
	if err != nil {
		t.Fatalf("ArmContextControl: %v", err)
	}
	st := GetActiveContextControlState()
	ctx := context.Background()

	// 1. Pin and then release a span
	const testDigest = "sha256:fedcba9876543210"
	st.Execute(ctx, ContextControlRequest{
		Action:  ActionPin,
		SpanIDs: []string{"span-recoverable"},
		Digest:  testDigest,
	})
	st.Execute(ctx, ContextControlRequest{
		Action:  ActionRelease,
		SpanIDs: []string{"span-recoverable"},
	})

	snap := st.Snapshot()
	if len(snap.ActiveReleases) != 1 || snap.ActiveReleases[0] != "span-recoverable" {
		t.Fatalf("expected span-recoverable in ActiveReleases, got %v", snap.ActiveReleases)
	}

	// 2. Restore by SpanIDs
	receipt := st.Execute(ctx, ContextControlRequest{
		Action:  ActionRestore,
		SpanIDs: []string{"span-recoverable"},
	})
	if receipt.Status != StatusAccepted {
		t.Fatalf("restore receipt.Status = %q, want %q, reason: %s", receipt.Status, StatusAccepted, receipt.Reason)
	}
	if receipt.Action != ActionRestore {
		t.Fatalf("restore receipt.Action = %q, want %q", receipt.Action, ActionRestore)
	}

	snapAfter := st.Snapshot()
	if len(snapAfter.ActiveReleases) != 0 {
		t.Errorf("expected 0 active releases after restore, got %d", len(snapAfter.ActiveReleases))
	}
	if len(snapAfter.ActivePins) != 1 || snapAfter.ActivePins[0] != "span-recoverable" {
		t.Errorf("expected span-recoverable in ActivePins, got %v", snapAfter.ActivePins)
	}

	// 3. Release and restore by Digest
	st.Execute(ctx, ContextControlRequest{
		Action:  ActionRelease,
		SpanIDs: []string{"span-recoverable"},
	})
	receiptDigest := st.Execute(ctx, ContextControlRequest{
		Action: ActionRestore,
		Digest: testDigest,
	})
	if receiptDigest.Status != StatusAccepted {
		t.Fatalf("restore by digest status = %q, want %q, reason: %s", receiptDigest.Status, StatusAccepted, receiptDigest.Reason)
	}

	// 4. Restore with unknown digest -> UNWITNESSED
	receiptUnwitnessed := st.Execute(ctx, ContextControlRequest{
		Action: ActionRestore,
		Digest: "sha256:unwitnessed9999999",
	})
	if receiptUnwitnessed.Status != StatusRefused {
		t.Fatalf("status = %q, want %q", receiptUnwitnessed.Status, StatusRefused)
	}
	if receiptUnwitnessed.Reason != abi.ReasonName(abi.ReasonUnwitnessed) {
		t.Errorf("reason = %q, want %q", receiptUnwitnessed.Reason, abi.ReasonName(abi.ReasonUnwitnessed))
	}

	// 5. Restore with empty params -> MALFORMED
	receiptEmpty := st.Execute(ctx, ContextControlRequest{
		Action: ActionRestore,
	})
	if receiptEmpty.Status != StatusRefused {
		t.Fatalf("status = %q, want %q", receiptEmpty.Status, StatusRefused)
	}
	if receiptEmpty.Reason != abi.ReasonName(abi.ReasonMalformed) {
		t.Errorf("reason = %q, want %q", receiptEmpty.Reason, abi.ReasonName(abi.ReasonMalformed))
	}
}

func TestContextControlQuery(t *testing.T) {
	DisarmContextControl()
	t.Cleanup(DisarmContextControl)

	_, err := ArmContextControl()
	if err != nil {
		t.Fatalf("ArmContextControl: %v", err)
	}
	st := GetActiveContextControlState()
	ctx := context.Background()

	st.Execute(ctx, ContextControlRequest{
		Action:     ActionPin,
		SpanIDs:    []string{"span-auth-rotation", "span-database-pool"},
		Provenance: "turn_query_prep",
	})

	// Query for auth
	receipt := st.Execute(ctx, ContextControlRequest{
		Action: ActionQuery,
		Query:  "auth",
	})

	if receipt.Status != StatusAccepted {
		t.Fatalf("query status = %q, want %q, reason: %s", receipt.Status, StatusAccepted, receipt.Reason)
	}
	if receipt.Action != ActionQuery {
		t.Fatalf("query action = %q, want %q", receipt.Action, ActionQuery)
	}

	matched, ok := receipt.Details["matched_spans"].([]string)
	if !ok {
		t.Fatalf("expected matched_spans slice in details")
	}
	if len(matched) != 1 || matched[0] != "span-auth-rotation" {
		t.Errorf("unexpected matched_spans: %v", matched)
	}
}

func TestContextControlIdempotency(t *testing.T) {
	DisarmContextControl()
	t.Cleanup(DisarmContextControl)

	_, err := ArmContextControl()
	if err != nil {
		t.Fatalf("ArmContextControl: %v", err)
	}
	st := GetActiveContextControlState()
	ctx := context.Background()

	const idempKey = "client-key-idem-8583"
	req := ContextControlRequest{
		Action:         ActionPin,
		SpanIDs:        []string{"span-idemp-1"},
		IdempotencyKey: idempKey,
	}

	// First execution: processed cleanly
	r1 := st.Execute(ctx, req)
	if r1.Status != StatusAccepted {
		t.Fatalf("first r1.Status = %q, want %q", r1.Status, StatusAccepted)
	}
	if r1.Idempotent {
		t.Fatalf("first r1.Idempotent = true, want false")
	}

	pinsBefore := len(st.Snapshot().ActivePins)
	if pinsBefore != 1 {
		t.Fatalf("pins before replay = %d, want 1", pinsBefore)
	}

	// Replay exact request: returns cached receipt with Idempotent=true
	r2 := st.Execute(ctx, req)
	if r2.Status != StatusAccepted {
		t.Fatalf("replayed r2.Status = %q, want %q", r2.Status, StatusAccepted)
	}
	if !r2.Idempotent {
		t.Fatalf("replayed r2.Idempotent = false, want true")
	}

	pinsAfter := len(st.Snapshot().ActivePins)
	if pinsAfter != pinsBefore {
		t.Fatalf("pinsAfter = %d, want %d (idempotency must not duplicate state)", pinsAfter, pinsBefore)
	}

	// Replaying with conflicting request parameters for the same key -> refused MALFORMED
	conflictReq := ContextControlRequest{
		Action:         ActionPin,
		SpanIDs:        []string{"span-conflicting-different"},
		IdempotencyKey: idempKey,
	}
	r3 := st.Execute(ctx, conflictReq)
	if r3.Status != StatusRefused {
		t.Fatalf("conflict r3.Status = %q, want %q", r3.Status, StatusRefused)
	}
	if r3.Reason != abi.ReasonName(abi.ReasonMalformed) {
		t.Errorf("conflict r3.Reason = %q, want %q", r3.Reason, abi.ReasonName(abi.ReasonMalformed))
	}
}

func TestContextControlRetrievalScopeAndBudgetPreference(t *testing.T) {
	DisarmContextControl()
	t.Cleanup(DisarmContextControl)

	_, err := ArmContextControl()
	if err != nil {
		t.Fatalf("ArmContextControl: %v", err)
	}
	st := GetActiveContextControlState()
	ctx := context.Background()

	// Set retrieval scope to "repo"
	scopeReceipt := st.Execute(ctx, ContextControlRequest{
		Action: ActionRetrievalScope,
		Scope:  "repo",
	})
	if scopeReceipt.Status != StatusAccepted {
		t.Fatalf("scope status = %q, want %q", scopeReceipt.Status, StatusAccepted)
	}
	if st.Snapshot().RetrievalScope != "repo" {
		t.Errorf("retrieval scope = %q, want %q", st.Snapshot().RetrievalScope, "repo")
	}

	// Set budget preference
	prefBudget := 16000
	budgetReceipt := st.Execute(ctx, ContextControlRequest{
		Action: ActionBudgetPreference,
		Budget: &prefBudget,
	})
	if budgetReceipt.Status != StatusAccepted {
		t.Fatalf("budget status = %q, want %q", budgetReceipt.Status, StatusAccepted)
	}
	snap := st.Snapshot()
	if snap.PreferredBudget == nil || *snap.PreferredBudget != 16000 {
		t.Errorf("preferred budget = %v, want 16000", snap.PreferredBudget)
	}
}

func TestTargetOperatingEnvelope(t *testing.T) {
	DisarmContextControl()
	t.Cleanup(DisarmContextControl)

	_, err := ArmContextControl()
	if err != nil {
		t.Fatalf("ArmContextControl: %v", err)
	}

	// Target operating envelope verified:
	// - typed_control_kinds >= 4
	// - native_harness_adapters >= 1
	// - malformed_request_classes >= 2

	typedKinds := CountTypedControlKinds()
	if typedKinds < 4 {
		t.Fatalf("typed_control_kinds = %d, want >= 4", typedKinds)
	}
	if len(SupportedControlKinds) < 4 {
		t.Fatalf("SupportedControlKinds len = %d, want >= 4", len(SupportedControlKinds))
	}

	adapters := CountNativeHarnessAdapters()
	if adapters < 1 {
		t.Fatalf("native_harness_adapters = %d, want >= 1", adapters)
	}

	malformedClasses := CountMalformedRequestClasses()
	if malformedClasses < 2 {
		t.Fatalf("malformed_request_classes = %d, want >= 2", malformedClasses)
	}
	if len(MalformedRequestClasses) < 2 {
		t.Fatalf("MalformedRequestClasses len = %d, want >= 2", len(MalformedRequestClasses))
	}
}

func TestContextControlEngineComplete(t *testing.T) {
	DisarmContextControl()
	t.Cleanup(DisarmContextControl)

	_, err := ArmContextControl()
	if err != nil {
		t.Fatalf("ArmContextControl: %v", err)
	}

	ctx := context.Background()

	// Successful complete call
	argsPayload, _ := json.Marshal(ContextControlRequest{
		Action: ActionInspect,
	})
	call := &abi.ToolCall{
		Tool: ToolContextControl,
		Args: putBytes(ctx, argsPayload),
	}

	result, err := activeContextControlEngine.Complete(ctx, call)
	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}
	if result.Status != abi.StatusOK {
		t.Fatalf("result.Status = %v, want %v", result.Status, abi.StatusOK)
	}

	var receipt ContextControlReceipt
	rawRes := refutil.Bytes(ctx, result.Payload)
	if err := json.Unmarshal(rawRes, &receipt); err != nil {
		t.Fatalf("json.Unmarshal result: %v", err)
	}
	if receipt.Status != StatusAccepted || receipt.Action != ActionInspect {
		t.Errorf("unexpected receipt: %+v", receipt)
	}

	// Complete with malformed JSON args
	badCall := &abi.ToolCall{
		Tool: ToolContextControl,
		Args: putBytes(ctx, []byte("{invalid-json")),
	}
	badRes, err := activeContextControlEngine.Complete(ctx, badCall)
	if err != nil {
		t.Fatalf("Complete badCall error: %v", err)
	}
	var badReceipt ContextControlReceipt
	if err := json.Unmarshal(refutil.Bytes(ctx, badRes.Payload), &badReceipt); err != nil {
		t.Fatalf("json.Unmarshal bad receipt: %v", err)
	}
	if badReceipt.Status != StatusRefused || badReceipt.Reason != abi.ReasonName(abi.ReasonMalformed) {
		t.Errorf("unexpected bad receipt: %+v", badReceipt)
	}
}

func TestContextControlWiringAndCatalog(t *testing.T) {
	DisarmContextControl()
	t.Cleanup(DisarmContextControl)

	// Verify catalog unarmed
	if cat := ContextControlCatalog(); cat != nil {
		t.Fatalf("unarmed ContextControlCatalog() = %v, want nil", cat)
	}
	if names := contextControlAllow(); names != nil {
		t.Fatalf("unarmed contextControlAllow() = %v, want nil", names)
	}
	if _, ok := contextControlMeta(ToolContextControl); ok {
		t.Fatalf("unarmed contextControlMeta() ok = true, want false")
	}

	// Arm via ArmContextControl
	defs, err := ArmContextControl()
	if err != nil {
		t.Fatalf("ArmContextControl: %v", err)
	}
	if len(defs) != 1 {
		t.Fatalf("defs len = %d, want 1", len(defs))
	}
	fn := defs[0].Function
	if fn.Name != ToolContextControl {
		t.Fatalf("fn.Name = %q, want %q", fn.Name, ToolContextControl)
	}

	// Check schema properties
	schemaStr := string(fn.Parameters)
	requiredProps := []string{"action", "span_ids", "digest", "budget", "query", "idempotency_key"}
	for _, prop := range requiredProps {
		if !strings.Contains(schemaStr, fmt.Sprintf("%q", prop)) {
			t.Errorf("schema missing property %q", prop)
		}
	}

	// Verify allow and meta
	if names := contextControlAllow(); len(names) != 1 || names[0] != ToolContextControl {
		t.Errorf("contextControlAllow() = %v, want [%s]", names, ToolContextControl)
	}
	meta, ok := contextControlMeta(ToolContextControl)
	if !ok || meta["consistency"] != "STRICT" {
		t.Errorf("contextControlMeta = %v, ok = %v", meta, ok)
	}

	// Test wiring with ArmCodeToolsWithOptions
	DisarmCodeTools()
	codeDefs, err := ArmCodeToolsWithOptions(CodeToolsOptions{
		EnableContextControl: true,
	})
	if err != nil {
		t.Fatalf("ArmCodeToolsWithOptions with context control: %v", err)
	}
	hasCC := false
	for _, d := range codeDefs {
		if d.Function.Name == ToolContextControl {
			hasCC = true
			break
		}
	}
	if !hasCC {
		t.Fatalf("expected context_control in codeDefs: %v", codeDefs)
	}

	// Test RunOption WithContextControl
	cfg := runConfig{}
	opt := WithContextControl()
	opt(&cfg)
	if !cfg.contextControl {
		t.Fatalf("WithContextControl did not set cfg.contextControl = true")
	}
	seedTools := cfg.seedTools()
	hasCCSeed := false
	for _, d := range seedTools {
		if d.Function.Name == ToolContextControl {
			hasCCSeed = true
			break
		}
	}
	if !hasCCSeed {
		t.Fatalf("expected context_control in cfg.seedTools()")
	}
}

func TestNativeHarnessAdapterIntegration(t *testing.T) {
	DisarmContextControl()
	t.Cleanup(DisarmContextControl)

	_, err := ArmContextControl()
	if err != nil {
		t.Fatalf("ArmContextControl: %v", err)
	}
	st := GetActiveContextControlState()

	// Set advisory preferred budget
	bVal := 12000
	st.Execute(context.Background(), ContextControlRequest{
		Action: ActionBudgetPreference,
		Budget: &bVal,
	})

	planner := &CtxViewPlanner{
		Enabled: true,
		Budget:  DefaultCtxViewBudget,
	}

	adapter := &nativeTurnLoopAdapter{state: st}
	if adapter.Name() != "native_turn_loop" {
		t.Errorf("adapter.Name() = %q, want %q", adapter.Name(), "native_turn_loop")
	}

	if err := adapter.ApplyToTurn(context.Background(), 1, planner); err != nil {
		t.Fatalf("ApplyToTurn failed: %v", err)
	}
	if planner.Budget != 12000 {
		t.Errorf("planner.Budget = %d, want 12000", planner.Budget)
	}

	snap := adapter.Snapshot()
	if snap.PreferredBudget == nil || *snap.PreferredBudget != 12000 {
		t.Errorf("snapshot PreferredBudget = %v, want 12000", snap.PreferredBudget)
	}
}
