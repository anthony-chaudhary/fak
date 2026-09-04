package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

func TestTodoStateValidation(t *testing.T) {
	st := NewTodoState()

	// 1. Clean update with single in_progress item.
	validItems := []TodoItem{
		{Content: "First task", Status: StatusCompleted, Priority: PriorityHigh},
		{Content: "Second task", Status: StatusInProgress, Priority: PriorityMedium},
		{Content: "Third task", Status: StatusPending, Priority: PriorityLow},
	}
	receipt, err := st.SetTodos(validItems)
	if err != nil {
		t.Fatalf("SetTodos failed on valid items: %v", err)
	}
	if receipt.Status != "ok" || receipt.Total != 3 || receipt.InProgress != 1 || receipt.Completed != 1 || receipt.Pending != 1 {
		t.Fatalf("unexpected receipt: %+v", receipt)
	}

	todos := st.GetTodos()
	if len(todos) != 3 {
		t.Fatalf("expected 3 todos, got %d", len(todos))
	}
	if todos[0].ID == "" || todos[1].ID == "" || todos[2].ID == "" {
		t.Errorf("expected synthesized task IDs, got: %+v", todos)
	}

	// 2. Reject multiple in_progress items.
	invalidMultipleInProgress := []TodoItem{
		{Content: "Task A", Status: StatusInProgress},
		{Content: "Task B", Status: StatusInProgress},
	}
	receipt, err = st.SetTodos(invalidMultipleInProgress)
	if err == nil {
		t.Fatalf("expected error for multiple in_progress items, got nil")
	}
	if receipt.Status != "error" {
		t.Fatalf("expected error status in receipt: %+v", receipt)
	}

	// 3. Reject empty content.
	invalidEmptyContent := []TodoItem{
		{Content: "   ", Status: StatusPending},
	}
	_, err = st.SetTodos(invalidEmptyContent)
	if err == nil {
		t.Fatalf("expected error for empty content, got nil")
	}

	// 4. Reject invalid status.
	invalidStatus := []TodoItem{
		{Content: "Task X", Status: "unknown_status"},
	}
	_, err = st.SetTodos(invalidStatus)
	if err == nil {
		t.Fatalf("expected error for invalid status, got nil")
	}

	// 5. Reject invalid priority.
	invalidPriority := []TodoItem{
		{Content: "Task Y", Status: StatusPending, Priority: "super_urgent"},
	}
	_, err = st.SetTodos(invalidPriority)
	if err == nil {
		t.Fatalf("expected error for invalid priority, got nil")
	}

	// 6. Reject exceeding MaxTodoItems bound.
	overflow := make([]TodoItem, MaxTodoItems+1)
	for i := range overflow {
		overflow[i] = TodoItem{Content: "task", Status: StatusPending}
	}
	_, err = st.SetTodos(overflow)
	if err == nil {
		t.Fatalf("expected error for exceeding MaxTodoItems, got nil")
	}
}

func TestTodoStateNonMutationOnFailure(t *testing.T) {
	st := NewTodoState()

	initial := []TodoItem{
		{Content: "Initial task", Status: StatusPending, Priority: PriorityHigh},
	}
	receipt, err := st.SetTodos(initial)
	if err != nil || receipt.Status != "ok" {
		t.Fatalf("initial SetTodos failed: %v", err)
	}

	// Attempt invalid update: multiple in_progress tasks.
	invalid := []TodoItem{
		{Content: "Task 1", Status: StatusInProgress},
		{Content: "Task 2", Status: StatusInProgress},
	}
	_, err = st.SetTodos(invalid)
	if err == nil {
		t.Fatalf("expected error on multiple in_progress")
	}

	current := st.GetTodos()
	if len(current) != 1 || current[0].Content != "Initial task" {
		t.Fatalf("state was corrupted after failed SetTodos: %+v", current)
	}
}

func TestTodoStateDefaultsAndTrimming(t *testing.T) {
	st := NewTodoState()

	items := []TodoItem{
		{Content: "  Trimmed task  ", Status: " PENDING ", Priority: ""},
		{Content: "Second task", Status: "IN_PROGRESS", Priority: "HIGH"},
	}
	receipt, err := st.SetTodos(items)
	if err != nil {
		t.Fatalf("SetTodos failed: %v", err)
	}
	if receipt.Status != "ok" || receipt.Pending != 1 || receipt.InProgress != 1 {
		t.Fatalf("unexpected receipt: %+v", receipt)
	}

	todos := st.GetTodos()
	if len(todos) != 2 {
		t.Fatalf("expected 2 todos, got %d", len(todos))
	}
	if todos[0].Content != "Trimmed task" {
		t.Errorf("content not trimmed: %q", todos[0].Content)
	}
	if todos[0].Status != StatusPending {
		t.Errorf("status not normalized: %q", todos[0].Status)
	}
	if todos[0].Priority != PriorityMedium {
		t.Errorf("expected default priority 'medium', got %q", todos[0].Priority)
	}
	if todos[1].Status != StatusInProgress || todos[1].Priority != PriorityHigh {
		t.Errorf("second task not normalized: %+v", todos[1])
	}

	// Empty list clears todos and has 0 in_progress.
	receipt, err = st.SetTodos([]TodoItem{})
	if err != nil || receipt.Status != "ok" || receipt.Total != 0 {
		t.Fatalf("clearing todos failed: receipt=%+v, err=%v", receipt, err)
	}
	if len(st.GetTodos()) != 0 {
		t.Fatalf("expected 0 todos after clear, got %d", len(st.GetTodos()))
	}
}

func TestTodoCatalogArmDisarm(t *testing.T) {
	DisarmTodoTools()
	if cat := TodoToolCatalog(); cat != nil {
		t.Fatalf("expected nil TodoToolCatalog when unarmed, got %v", cat)
	}

	cat, err := ArmTodoTools()
	if err != nil {
		t.Fatalf("ArmTodoTools failed: %v", err)
	}
	defer DisarmTodoTools()

	if len(cat) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(cat))
	}
	if cat[0].Function.Name != ToolTodoWrite || cat[1].Function.Name != ToolTodoRead {
		t.Fatalf("unexpected tool names in catalog: %+v", cat)
	}

	DisarmTodoTools()
	if cat := TodoToolCatalog(); cat != nil {
		t.Fatalf("expected nil TodoToolCatalog after DisarmTodoTools, got %v", cat)
	}
}

func TestTodoEngineExecution(t *testing.T) {
	cat, err := ArmTodoTools()
	if err != nil {
		t.Fatalf("ArmTodoTools failed: %v", err)
	}
	t.Cleanup(DisarmTodoTools)

	if len(cat) != 2 {
		t.Fatalf("expected 2 tools in TodoToolCatalog, got %d", len(cat))
	}

	metaRead, okRead := todoToolMeta(ToolTodoRead)
	if !okRead || metaRead["consistency"] != "BEST_EFFORT" || metaRead["readOnlyHint"] != "true" || metaRead["idempotentHint"] != "false" {
		t.Fatalf("unexpected todoToolMeta for todoread: %v, ok=%v", metaRead, okRead)
	}

	metaWrite, okWrite := todoToolMeta(ToolTodoWrite)
	if !okWrite || metaWrite["consistency"] != "BEST_EFFORT" || metaWrite["readOnlyHint"] != "false" {
		t.Fatalf("unexpected todoToolMeta for todowrite: %v, ok=%v", metaWrite, okWrite)
	}

	ctx := context.Background()

	// Write todos.
	payload := `{"todos":[{"content":"Plan item 1","status":"completed","priority":"high"},{"content":"Plan item 2","status":"in_progress","priority":"medium"}]}`
	ref := putBytes(ctx, []byte(payload))
	writeCall := &abi.ToolCall{
		Tool: ToolTodoWrite,
		Args: ref,
	}

	writeRes, err := activeTodoEngine.Complete(ctx, writeCall)
	if err != nil {
		t.Fatalf("todowrite Complete failed: %v", err)
	}
	if writeRes.Status != abi.StatusOK {
		t.Fatalf("todowrite status = %v, want OK", writeRes.Status)
	}

	var writeReceipt TodoReceipt
	bodyBytes := refutil.Bytes(ctx, writeRes.Payload)
	if err := json.Unmarshal(bodyBytes, &writeReceipt); err != nil {
		t.Fatalf("failed to decode write receipt: %v", err)
	}
	if writeReceipt.Status != "ok" || writeReceipt.Total != 2 || writeReceipt.Completed != 1 || writeReceipt.InProgress != 1 {
		t.Fatalf("unexpected write receipt: %+v", writeReceipt)
	}

	// Read todos.
	readCall := &abi.ToolCall{
		Tool: ToolTodoRead,
		Args: putBytes(ctx, []byte("{}")),
	}
	readRes, err := activeTodoEngine.Complete(ctx, readCall)
	if err != nil {
		t.Fatalf("todoread Complete failed: %v", err)
	}
	if readRes.Status != abi.StatusOK {
		t.Fatalf("todoread status = %v, want OK", readRes.Status)
	}

	var readPayload TodoReadResponse
	if err := json.Unmarshal(refutil.Bytes(ctx, readRes.Payload), &readPayload); err != nil {
		t.Fatalf("failed to decode read payload: %v", err)
	}
	if readPayload.Total != 2 || len(readPayload.Todos) != 2 {
		t.Fatalf("unexpected read payload: %+v", readPayload)
	}
	if readPayload.InProgress == nil || readPayload.InProgress.Content != "Plan item 2" {
		t.Fatalf("expected in_progress task 'Plan item 2', got: %+v", readPayload.InProgress)
	}
}

func TestTodoGateAdjudication(t *testing.T) {
	DisarmTodoTools()
	gate := todoGate{}

	// When unarmed, defers.
	c := &abi.ToolCall{Tool: ToolTodoWrite}
	v := gate.Adjudicate(context.Background(), c)
	if v.Kind != abi.VerdictDefer {
		t.Fatalf("expected VerdictDefer when unarmed, got %v", v.Kind)
	}

	// When armed, allows and pins engine.
	_, err := ArmTodoTools()
	if err != nil {
		t.Fatalf("ArmTodoTools: %v", err)
	}
	t.Cleanup(DisarmTodoTools)

	cWrite := &abi.ToolCall{Tool: ToolTodoWrite}
	vWrite := gate.Adjudicate(context.Background(), cWrite)
	if vWrite.Kind != abi.VerdictAllow || cWrite.Engine != EngineTodoWrite {
		t.Fatalf("todowrite verdict = %v, engine = %s", vWrite.Kind, cWrite.Engine)
	}

	cRead := &abi.ToolCall{Tool: ToolTodoRead}
	vRead := gate.Adjudicate(context.Background(), cRead)
	if vRead.Kind != abi.VerdictAllow || cRead.Engine != EngineTodoRead {
		t.Fatalf("todoread verdict = %v, engine = %s", vRead.Kind, cRead.Engine)
	}

	cOther := &abi.ToolCall{Tool: "some_other_tool"}
	vOther := gate.Adjudicate(context.Background(), cOther)
	if vOther.Kind != abi.VerdictDefer {
		t.Fatalf("expected defer for unknown tool, got %v", vOther.Kind)
	}
}

func TestRunArmWithTodoTools(t *testing.T) {
	_, err := ArmTodoTools()
	if err != nil {
		t.Fatalf("ArmTodoTools: %v", err)
	}
	t.Cleanup(DisarmTodoTools)

	turns := []*Completion{
		toolCallTurn(ToolTodoWrite, `{"todos":[{"content":"Audit architecture","status":"in_progress","priority":"high"},{"content":"Implement tests","status":"pending","priority":"medium"}]}`),
		toolCallTurn(ToolTodoRead, `{}`),
		{Message: Message{Content: "Finished plan setup."}},
	}

	var log []traceEvent
	planner := &recordingSysPlanner{turns: turns}

	metrics, err := RunArm(context.Background(), planner, "organize plan", true, len(turns)+1, &log, WithTodoTools())
	if err != nil {
		t.Fatalf("RunArm with WithTodoTools failed: %v", err)
	}

	if metrics.EngineCalls < 2 {
		t.Errorf("expected at least 2 engine calls, got %d", metrics.EngineCalls)
	}

	foundWrite, foundRead := false, false
	for _, ev := range log {
		if ev.Tool == ToolTodoWrite {
			foundWrite = true
			if ev.Verdict != "ALLOW" {
				t.Errorf("todowrite verdict = %q, want ALLOW", ev.Verdict)
			}
			if ev.By != RungNameTodo {
				t.Errorf("todowrite by = %q, want %q", ev.By, RungNameTodo)
			}
		}
		if ev.Tool == ToolTodoRead {
			foundRead = true
			if ev.Verdict != "ALLOW" {
				t.Errorf("todoread verdict = %q, want ALLOW", ev.Verdict)
			}
			if ev.By != RungNameTodo {
				t.Errorf("todoread by = %q, want %q", ev.By, RungNameTodo)
			}
		}
	}

	if !foundWrite {
		t.Errorf("todowrite not found in execution log")
	}
	if !foundRead {
		t.Errorf("todoread not found in execution log")
	}

	st := GetActiveTodoState()
	if st == nil {
		t.Fatalf("expected active TodoState, got nil")
	}
	items := st.GetTodos()
	if len(items) != 2 || items[0].Content != "Audit architecture" || items[0].Status != StatusInProgress {
		t.Fatalf("unexpected items in active TodoState: %+v", items)
	}
}

func TestTodoEngineErrorPaths(t *testing.T) {
	_, err := ArmTodoTools()
	if err != nil {
		t.Fatalf("ArmTodoTools: %v", err)
	}
	t.Cleanup(DisarmTodoTools)

	ctx := context.Background()

	// 1. Invalid JSON args to todowrite.
	badJSONCall := &abi.ToolCall{
		Tool: ToolTodoWrite,
		Args: putBytes(ctx, []byte(`{invalid-json`)),
	}
	res, err := activeTodoEngine.Complete(ctx, badJSONCall)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Status != abi.StatusError {
		t.Fatalf("expected StatusError for invalid JSON, got %v", res.Status)
	}

	// 2. Validation error inside todowrite (e.g. 2 in_progress items).
	badPlanCall := &abi.ToolCall{
		Tool: ToolTodoWrite,
		Args: putBytes(ctx, []byte(`{"todos":[{"content":"t1","status":"in_progress"},{"content":"t2","status":"in_progress"}]}`)),
	}
	res, err = activeTodoEngine.Complete(ctx, badPlanCall)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Status != abi.StatusError {
		t.Fatalf("expected StatusError for multiple in_progress items, got %v", res.Status)
	}
	var receipt TodoReceipt
	_ = json.Unmarshal(refutil.Bytes(ctx, res.Payload), &receipt)
	if receipt.Status != "error" || receipt.Error == "" {
		t.Fatalf("expected error receipt, got %+v", receipt)
	}

	// 3. Unknown tool call.
	unknownCall := &abi.ToolCall{
		Tool: "unknown_todo_op",
		Args: putBytes(ctx, []byte(`{}`)),
	}
	res, err = activeTodoEngine.Complete(ctx, unknownCall)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Status != abi.StatusError {
		t.Fatalf("expected StatusError for unknown tool, got %v", res.Status)
	}

	// 4. When unarmed.
	DisarmTodoTools()
	unarmedCall := &abi.ToolCall{
		Tool: ToolTodoRead,
		Args: putBytes(ctx, []byte(`{}`)),
	}
	res, err = activeTodoEngine.Complete(ctx, unarmedCall)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Status != abi.StatusError {
		t.Fatalf("expected StatusError when unarmed, got %v", res.Status)
	}
}

func TestRunArmDeduplicateTodoTools(t *testing.T) {
	cat, err := ArmTodoTools()
	if err != nil {
		t.Fatalf("ArmTodoTools: %v", err)
	}
	t.Cleanup(DisarmTodoTools)

	turns := []*Completion{
		toolCallTurn(ToolTodoRead, `{}`),
		{Message: Message{Content: "Done."}},
	}
	planner := &recordingSysPlanner{turns: turns}

	// Pass WithToolCatalog that already contains TodoToolCatalog(), plus WithTodoTools().
	// seedTools must not duplicate todowrite or todoread.
	var log []traceEvent
	metrics, err := RunArm(context.Background(), planner, "dedup test", true, 3, &log,
		WithToolCatalog(cat),
		WithTodoTools(),
	)
	if err != nil {
		t.Fatalf("RunArm failed: %v", err)
	}
	if metrics.EngineCalls+metrics.VDSOHits < 1 {
		t.Errorf("expected engine call or vDSO hit, got EngineCalls=%d, VDSOHits=%d", metrics.EngineCalls, metrics.VDSOHits)
	}
}
