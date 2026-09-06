package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

func TestTaskCatalogArmDisarm(t *testing.T) {
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)

	// 1. When unarmed, catalog, allowlist, and metadata must be empty/false.
	if cat := TaskToolCatalog(); cat != nil {
		t.Fatalf("expected nil TaskToolCatalog when unarmed, got %v", cat)
	}
	if allowed := taskToolAllow(); allowed != nil {
		t.Fatalf("expected nil taskToolAllow when unarmed, got %v", allowed)
	}
	if st := GetActiveTaskState(); st != nil {
		t.Fatalf("expected nil GetActiveTaskState when unarmed, got %v", st)
	}
	for _, tool := range []string{ToolTaskSpawn, ToolTaskWait, ToolTaskStatus, ToolTaskCancel} {
		if meta, ok := taskToolMeta(tool); ok || meta != nil {
			t.Fatalf("expected taskToolMeta(%q) ok=false when unarmed, got ok=%v, meta=%v", tool, ok, meta)
		}
	}

	// 2. Arm task tools and verify the returned catalog.
	cat, err := ArmTaskTools()
	if err != nil {
		t.Fatalf("ArmTaskTools failed: %v", err)
	}

	if len(cat) != 4 {
		t.Fatalf("expected 4 tool definitions in catalog, got %d", len(cat))
	}

	expectedTools := map[string]bool{
		ToolTaskSpawn:  false,
		ToolTaskWait:   false,
		ToolTaskStatus: false,
		ToolTaskCancel: false,
	}

	for _, d := range cat {
		if d.Type != "function" {
			t.Errorf("expected tool type 'function', got %q", d.Type)
		}
		name := d.Function.Name
		if _, ok := expectedTools[name]; !ok {
			t.Errorf("unexpected tool in catalog: %q", name)
		} else {
			expectedTools[name] = true
		}

		// Ensure parameters JSON is non-empty and valid schema
		if len(d.Function.Parameters) == 0 {
			t.Errorf("tool %q has empty parameters schema", name)
		}
		var schemaMap map[string]any
		if err := json.Unmarshal(d.Function.Parameters, &schemaMap); err != nil {
			t.Errorf("tool %q has invalid parameters JSON schema: %v", name, err)
		}
	}

	for name, found := range expectedTools {
		if !found {
			t.Errorf("tool %q not found in armed catalog", name)
		}
	}

	// 3. Verify specific schema requirements per specification.
	for _, d := range cat {
		schemaStr := string(d.Function.Parameters)
		switch d.Function.Name {
		case ToolTaskSpawn:
			for _, prop := range []string{"prompt", "description", "task_id", "subagent_type", "read_only", "idempotency_key"} {
				if !strings.Contains(schemaStr, fmt.Sprintf("%q", prop)) {
					t.Errorf("task_spawn schema missing property %q", prop)
				}
			}
			if !strings.Contains(schemaStr, `"required"`) || !strings.Contains(schemaStr, `"prompt"`) {
				t.Errorf("task_spawn schema missing required ['prompt']")
			}

		case ToolTaskWait:
			for _, prop := range []string{"task_ids", "task_id", "timeout_ms", "wait_all"} {
				if !strings.Contains(schemaStr, fmt.Sprintf("%q", prop)) {
					t.Errorf("task_wait schema missing property %q", prop)
				}
			}

		case ToolTaskStatus:
			for _, prop := range []string{"task_id", "limit"} {
				if !strings.Contains(schemaStr, fmt.Sprintf("%q", prop)) {
					t.Errorf("task_status schema missing property %q", prop)
				}
			}

		case ToolTaskCancel:
			for _, prop := range []string{"task_id", "reason"} {
				if !strings.Contains(schemaStr, fmt.Sprintf("%q", prop)) {
					t.Errorf("task_cancel schema missing property %q", prop)
				}
			}
			if !strings.Contains(schemaStr, `"required"`) || !strings.Contains(schemaStr, `"task_id"`) {
				t.Errorf("task_cancel schema missing required ['task_id']")
			}
		}
	}

	// 4. Verify taskToolAllow.
	allowed := taskToolAllow()
	if len(allowed) != 4 {
		t.Fatalf("expected 4 allowed tool names, got %d: %v", len(allowed), allowed)
	}

	// 5. Verify taskToolMeta consistency and read-only hints.
	spawnMeta, ok := taskToolMeta(ToolTaskSpawn)
	if !ok || spawnMeta["consistency"] != "BEST_EFFORT" || spawnMeta["readOnlyHint"] != "false" {
		t.Errorf("unexpected taskToolMeta for task_spawn: ok=%v, meta=%v", ok, spawnMeta)
	}

	waitMeta, ok := taskToolMeta(ToolTaskWait)
	if !ok || waitMeta["consistency"] != "BEST_EFFORT" || waitMeta["readOnlyHint"] != "true" {
		t.Errorf("unexpected taskToolMeta for task_wait: ok=%v, meta=%v", ok, waitMeta)
	}

	statusMeta, ok := taskToolMeta(ToolTaskStatus)
	if !ok || statusMeta["consistency"] != "BEST_EFFORT" || statusMeta["readOnlyHint"] != "true" || statusMeta["idempotentHint"] != "true" {
		t.Errorf("unexpected taskToolMeta for task_status: ok=%v, meta=%v", ok, statusMeta)
	}

	cancelMeta, ok := taskToolMeta(ToolTaskCancel)
	if !ok || cancelMeta["consistency"] != "BEST_EFFORT" || cancelMeta["destructive"] != "true" {
		t.Errorf("unexpected taskToolMeta for task_cancel: ok=%v, meta=%v", ok, cancelMeta)
	}

	// 6. Disarm and verify restored unarmed state.
	DisarmTaskTools()
	if cat := TaskToolCatalog(); cat != nil {
		t.Fatalf("expected nil TaskToolCatalog after DisarmTaskTools, got %v", cat)
	}
	if allowed := taskToolAllow(); allowed != nil {
		t.Fatalf("expected nil taskToolAllow after DisarmTaskTools, got %v", allowed)
	}
}

func TestTaskGateAdjudicationAndEngineAssignment(t *testing.T) {
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)

	gate := taskToolGate{}

	// 1. Verify capabilities when gate is created.
	if caps := gate.Caps(); caps != nil {
		t.Errorf("expected gate.Caps() to be nil, got %v", caps)
	}

	// 2. Unarmed deferral: all task tools and other tools must defer with RungNameTask ("tasktools").
	testTools := []string{ToolTaskSpawn, ToolTaskWait, ToolTaskStatus, ToolTaskCancel, "other_tool"}
	for _, tool := range testTools {
		c := &abi.ToolCall{Tool: tool}
		v := gate.Adjudicate(context.Background(), c)
		if v.Kind != abi.VerdictDefer {
			t.Errorf("unarmed gate.Adjudicate(%q).Kind = %v, want VerdictDefer", tool, v.Kind)
		}
		if v.By != RungNameTask {
			t.Errorf("unarmed gate.Adjudicate(%q).By = %q, want %q", tool, v.By, RungNameTask)
		}
		if c.Engine != "" {
			t.Errorf("unarmed toolcall Engine modified to %q, want empty", c.Engine)
		}
	}

	// 3. Nil ToolCall safety.
	vNil := gate.Adjudicate(context.Background(), nil)
	if vNil.Kind != abi.VerdictDefer || vNil.By != RungNameTask {
		t.Errorf("nil toolcall adjudication = %+v, want VerdictDefer by %q", vNil, RungNameTask)
	}

	// 4. Arm task tools and verify rank 23 adjudication and engine assignment.
	_, err := ArmTaskTools()
	if err != nil {
		t.Fatalf("ArmTaskTools failed: %v", err)
	}

	// Confirm rank constant
	if taskToolRank != 23 {
		t.Fatalf("taskToolRank = %d, want 23", taskToolRank)
	}
	if RungNameTask != "tasktools" {
		t.Fatalf("RungNameTask = %q, want 'tasktools'", RungNameTask)
	}

	cases := []struct {
		tool       string
		wantEngine string
		wantKind   abi.VerdictKind
	}{
		{ToolTaskSpawn, EngineTaskSpawn, abi.VerdictAllow},
		{ToolTaskWait, EngineTaskWait, abi.VerdictAllow},
		{ToolTaskStatus, EngineTaskStatus, abi.VerdictAllow},
		{ToolTaskCancel, EngineTaskCancel, abi.VerdictAllow},
		{"unrelated_tool", "", abi.VerdictDefer},
	}

	for _, tc := range cases {
		c := &abi.ToolCall{Tool: tc.tool}
		v := gate.Adjudicate(context.Background(), c)
		if v.Kind != tc.wantKind {
			t.Errorf("armed gate.Adjudicate(%q).Kind = %v, want %v", tc.tool, v.Kind, tc.wantKind)
		}
		if v.By != RungNameTask {
			t.Errorf("armed gate.Adjudicate(%q).By = %q, want %q", tc.tool, v.By, RungNameTask)
		}
		if c.Engine != tc.wantEngine {
			t.Errorf("armed gate.Adjudicate(%q).Engine = %q, want %q", tc.tool, c.Engine, tc.wantEngine)
		}
	}

	// 5. Verify taskToolGate is present in global abi.Adjudicators() registry.
	adjudicators := abi.Adjudicators()
	foundGate := false
	for _, adj := range adjudicators {
		if _, ok := adj.(taskToolGate); ok {
			foundGate = true
			break
		}
	}
	if !foundGate {
		t.Errorf("taskToolGate not found in abi.Adjudicators()")
	}
}

func TestTaskEngineExecution(t *testing.T) {
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)

	_, err := ArmTaskTools()
	if err != nil {
		t.Fatalf("ArmTaskTools failed: %v", err)
	}

	ctx := context.Background()

	// 1. Verify engine capabilities and weight-bearing contract.
	if caps := activeTaskEngine.Caps(); caps != nil {
		t.Errorf("expected engine Caps() to be nil, got %v", caps)
	}
	if activeTaskEngine.WeightBearing() {
		t.Errorf("expected engine WeightBearing() to be false")
	}

	// 2. Test task_spawn execution.
	spawnPayload := `{"prompt":"Analyze memory consumption across workers","description":"Memory audit","subagent_type":"researcher","read_only":true,"idempotency_key":"spawn-key-1"}`
	spawnCall := &abi.ToolCall{
		Tool: ToolTaskSpawn,
		Args: putBytes(ctx, []byte(spawnPayload)),
	}

	spawnRes, err := activeTaskEngine.Complete(ctx, spawnCall)
	if err != nil {
		t.Fatalf("task_spawn Complete failed: %v", err)
	}
	if spawnRes.Status != abi.StatusOK {
		t.Fatalf("task_spawn Status = %v, want OK", spawnRes.Status)
	}

	var spawnReceipt TaskSpawnReceipt
	if err := json.Unmarshal(refutil.Bytes(ctx, spawnRes.Payload), &spawnReceipt); err != nil {
		t.Fatalf("failed to decode spawn receipt: %v", err)
	}
	if spawnReceipt.Status != "accepted" || spawnReceipt.TaskID == "" {
		t.Fatalf("unexpected spawn receipt: %+v", spawnReceipt)
	}
	if spawnReceipt.SubagentType != "researcher" || !spawnReceipt.ReadOnly {
		t.Fatalf("spawn receipt fields mismatch: %+v", spawnReceipt)
	}
	assignedTaskID := spawnReceipt.TaskID

	// Test idempotency replay
	replayCall := &abi.ToolCall{
		Tool: ToolTaskSpawn,
		Args: putBytes(ctx, []byte(spawnPayload)),
	}
	replayRes, err := activeTaskEngine.Complete(ctx, replayCall)
	if err != nil {
		t.Fatalf("task_spawn replay Complete failed: %v", err)
	}
	var replayReceipt TaskSpawnReceipt
	if err := json.Unmarshal(refutil.Bytes(ctx, replayRes.Payload), &replayReceipt); err != nil {
		t.Fatalf("failed to decode replay receipt: %v", err)
	}
	if !replayReceipt.Idempotent || replayReceipt.TaskID != assignedTaskID {
		t.Fatalf("expected idempotent replay with task ID %q, got: %+v", assignedTaskID, replayReceipt)
	}

	// 3. Test task_status execution.
	// 3a. Single task status
	statusCall := &abi.ToolCall{
		Tool: ToolTaskStatus,
		Args: putBytes(ctx, []byte(fmt.Sprintf(`{"task_id":%q}`, assignedTaskID))),
	}
	statusRes, err := activeTaskEngine.Complete(ctx, statusCall)
	if err != nil {
		t.Fatalf("task_status Complete failed: %v", err)
	}
	var singleStatus TaskStatusReceipt
	if err := json.Unmarshal(refutil.Bytes(ctx, statusRes.Payload), &singleStatus); err != nil {
		t.Fatalf("failed to decode task_status: %v", err)
	}
	if singleStatus.Total != 1 || len(singleStatus.Tasks) != 1 || singleStatus.Tasks[0].ID != assignedTaskID {
		t.Fatalf("unexpected single status: %+v", singleStatus)
	}

	// 3b. All tasks status
	allStatusCall := &abi.ToolCall{
		Tool: ToolTaskStatus,
		Args: putBytes(ctx, []byte(`{}`)),
	}
	allStatusRes, err := activeTaskEngine.Complete(ctx, allStatusCall)
	if err != nil {
		t.Fatalf("task_status all Complete failed: %v", err)
	}
	var allStatus TaskStatusReceipt
	if err := json.Unmarshal(refutil.Bytes(ctx, allStatusRes.Payload), &allStatus); err != nil {
		t.Fatalf("failed to decode all status: %v", err)
	}
	if allStatus.Total < 1 {
		t.Fatalf("expected Total >= 1, got %d", allStatus.Total)
	}

	// 4. Test task_cancel execution.
	cancelCall := &abi.ToolCall{
		Tool: ToolTaskCancel,
		Args: putBytes(ctx, []byte(fmt.Sprintf(`{"task_id":%q,"reason":"superseded by coordinator"}`, assignedTaskID))),
	}
	cancelRes, err := activeTaskEngine.Complete(ctx, cancelCall)
	if err != nil {
		t.Fatalf("task_cancel Complete failed: %v", err)
	}
	var cancelReceipt TaskCancelReceipt
	if err := json.Unmarshal(refutil.Bytes(ctx, cancelRes.Payload), &cancelReceipt); err != nil {
		t.Fatalf("failed to decode cancel receipt: %v", err)
	}
	if cancelReceipt.Status != "cancelled" || !cancelReceipt.Cancelled || cancelReceipt.TaskID != assignedTaskID {
		t.Fatalf("unexpected cancel receipt: %+v", cancelReceipt)
	}

	// 5. Test task_wait on cancelled task (terminal).
	waitCall := &abi.ToolCall{
		Tool: ToolTaskWait,
		Args: putBytes(ctx, []byte(fmt.Sprintf(`{"task_id":%q,"timeout_ms":100}`, assignedTaskID))),
	}
	waitRes, err := activeTaskEngine.Complete(ctx, waitCall)
	if err != nil {
		t.Fatalf("task_wait Complete failed: %v", err)
	}
	var waitReceipt TaskWaitReceipt
	if err := json.Unmarshal(refutil.Bytes(ctx, waitRes.Payload), &waitReceipt); err != nil {
		t.Fatalf("failed to decode wait receipt: %v", err)
	}
	if waitReceipt.Cancelled != 1 || waitReceipt.Status != "completed" {
		t.Fatalf("unexpected wait receipt for cancelled task: %+v", waitReceipt)
	}

	// 6. Test spawn and completion with CompleteTask.
	st := GetActiveTaskState()
	spawn2, err := st.Spawn(TaskSpawnRequest{
		Prompt:       "Run background tests",
		SubagentType: "tester",
	})
	if err != nil {
		t.Fatalf("st.Spawn failed: %v", err)
	}

	err = st.CompleteTask(spawn2.TaskID, "all tests passed", nil)
	if err != nil {
		t.Fatalf("st.CompleteTask failed: %v", err)
	}

	wait2Call := &abi.ToolCall{
		Tool: ToolTaskWait,
		Args: putBytes(ctx, []byte(fmt.Sprintf(`{"task_ids":[%q]}`, spawn2.TaskID))),
	}
	wait2Res, err := activeTaskEngine.Complete(ctx, wait2Call)
	if err != nil {
		t.Fatalf("task_wait 2 Complete failed: %v", err)
	}
	var wait2Receipt TaskWaitReceipt
	if err := json.Unmarshal(refutil.Bytes(ctx, wait2Res.Payload), &wait2Receipt); err != nil {
		t.Fatalf("failed to decode wait 2 receipt: %v", err)
	}
	if wait2Receipt.Completed != 1 || wait2Receipt.Tasks[spawn2.TaskID].State != TaskStateCompleted {
		t.Fatalf("unexpected wait 2 receipt: %+v", wait2Receipt)
	}
}

func TestTaskEngineErrorHandling(t *testing.T) {
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)

	ctx := context.Background()

	// 1. Unarmed Complete returns error.
	call := &abi.ToolCall{Tool: ToolTaskSpawn, Args: putBytes(ctx, []byte(`{"prompt":"test"}`))}
	res, err := activeTaskEngine.Complete(ctx, call)
	if err != nil {
		t.Fatalf("unexpected error when unarmed: %v", err)
	}
	if res.Status != abi.StatusError {
		t.Fatalf("expected StatusError when unarmed, got %v", res.Status)
	}

	// 2. Armed, test invalid JSON arguments.
	_, err = ArmTaskTools()
	if err != nil {
		t.Fatalf("ArmTaskTools: %v", err)
	}

	badJSONCall := &abi.ToolCall{
		Tool: ToolTaskSpawn,
		Args: putBytes(ctx, []byte(`{invalid-json`)),
	}
	badRes, err := activeTaskEngine.Complete(ctx, badJSONCall)
	if err != nil {
		t.Fatalf("unexpected error on bad json: %v", err)
	}
	if badRes.Status != abi.StatusError {
		t.Fatalf("expected StatusError on bad json, got %v", badRes.Status)
	}

	// 3. Validation failure: empty prompt on spawn.
	emptyPromptCall := &abi.ToolCall{
		Tool: ToolTaskSpawn,
		Args: putBytes(ctx, []byte(`{"prompt":"   "}`)),
	}
	emptyRes, err := activeTaskEngine.Complete(ctx, emptyPromptCall)
	if err != nil {
		t.Fatalf("unexpected error on empty prompt: %v", err)
	}
	if emptyRes.Status != abi.StatusError {
		t.Fatalf("expected StatusError on empty prompt, got %v", emptyRes.Status)
	}
	var emptyReceipt TaskSpawnReceipt
	_ = json.Unmarshal(refutil.Bytes(ctx, emptyRes.Payload), &emptyReceipt)
	if emptyReceipt.Status != "error" || emptyReceipt.Error == "" {
		t.Fatalf("expected error in receipt, got %+v", emptyReceipt)
	}

	// 4. Cancel non-existent task.
	cancelMissingCall := &abi.ToolCall{
		Tool: ToolTaskCancel,
		Args: putBytes(ctx, []byte(`{"task_id":"nonexistent-task"}`)),
	}
	cmRes, err := activeTaskEngine.Complete(ctx, cancelMissingCall)
	if err != nil {
		t.Fatalf("unexpected error on cancel missing: %v", err)
	}
	if cmRes.Status != abi.StatusError {
		t.Fatalf("expected StatusError on cancel missing, got %v", cmRes.Status)
	}

	// 5. Unknown tool.
	unknownCall := &abi.ToolCall{
		Tool: "unknown_task_tool",
		Args: putBytes(ctx, []byte(`{}`)),
	}
	unkRes, err := activeTaskEngine.Complete(ctx, unknownCall)
	if err != nil {
		t.Fatalf("unexpected error on unknown tool: %v", err)
	}
	if unkRes.Status != abi.StatusError {
		t.Fatalf("expected StatusError on unknown tool, got %v", unkRes.Status)
	}
}

func TestTaskStateUnitAndCapacity(t *testing.T) {
	st := NewTaskState()

	// 1. Empty prompt rejected.
	_, err := st.Spawn(TaskSpawnRequest{Prompt: "  "})
	if err == nil {
		t.Fatalf("expected error on empty prompt")
	}

	// 2. Duplicate task ID rejected.
	_, err = st.Spawn(TaskSpawnRequest{Prompt: "Task 1", TaskID: "custom-id"})
	if err != nil {
		t.Fatalf("first spawn with custom-id failed: %v", err)
	}
	_, err = st.Spawn(TaskSpawnRequest{Prompt: "Task 2", TaskID: "custom-id"})
	if err == nil {
		t.Fatalf("expected error on duplicate task ID")
	}

	// 3. Status queries.
	status, err := st.Status(TaskStatusRequest{TaskID: "custom-id"})
	if err != nil || len(status.Tasks) != 1 {
		t.Fatalf("failed to query status by custom-id: %v, %+v", err, status)
	}
	_, err = st.Status(TaskStatusRequest{TaskID: "non-existent"})
	if err == nil {
		t.Fatalf("expected error for non-existent task status")
	}

	// 4. CompleteTask non-existent.
	err = st.CompleteTask("non-existent", nil, nil)
	if err == nil {
		t.Fatalf("expected error for CompleteTask on non-existent")
	}

	// 5. Test capacity rejection.
	stSmall := NewTaskState()
	stSmall.maxActive = 2
	stSmall.maxBacklog = 1

	for i := 0; i < 3; i++ {
		_, err := stSmall.Spawn(TaskSpawnRequest{Prompt: fmt.Sprintf("Task %d", i)})
		if err != nil {
			t.Fatalf("spawn %d failed within capacity: %v", i, err)
		}
	}
	// 4th should exceed active+backlog (2+1 = 3)
	_, err = stSmall.Spawn(TaskSpawnRequest{Prompt: "Task 4 (overflow)"})
	if err == nil {
		t.Fatalf("expected capacity rejection on 4th spawn, got nil")
	}

	// 6. Test async Wait timeout.
	waitReceipt, err := stSmall.Wait(context.Background(), TaskWaitRequest{
		TimeoutMs: 10,
	})
	if err != nil {
		t.Fatalf("Wait failed: %v", err)
	}
	if waitReceipt.Status != "timed_out" {
		t.Fatalf("expected status 'timed_out', got %q", waitReceipt.Status)
	}
	if waitReceipt.Running != 3 {
		t.Fatalf("expected 3 running/pending tasks, got %d", waitReceipt.Running)
	}

	// 7. Test async Wait wake on completion.
	go func() {
		time.Sleep(10 * time.Millisecond)
		_ = stSmall.CompleteTask("task-1", "done", nil)
		_ = stSmall.CompleteTask("task-2", "done", nil)
		_ = stSmall.CompleteTask("task-3", "done", nil)
	}()

	waitDoneReceipt, err := stSmall.Wait(context.Background(), TaskWaitRequest{
		TimeoutMs: 1000,
	})
	if err != nil {
		t.Fatalf("Wait after background complete failed: %v", err)
	}
	if waitDoneReceipt.Status != "completed" || waitDoneReceipt.Completed != 3 {
		t.Fatalf("expected all 3 completed, got: %+v", waitDoneReceipt)
	}
}

func TestRunArmWithTaskTools(t *testing.T) {
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)

	_, err := ArmTaskTools()
	if err != nil {
		t.Fatalf("ArmTaskTools: %v", err)
	}

	turns := []*Completion{
		toolCallTurn(ToolTaskSpawn, `{"prompt":"Explore architecture boundaries","subagent_type":"explore"}`),
		toolCallTurn(ToolTaskStatus, `{}`),
		{Message: Message{Content: "Tasks spawned and monitored."}},
	}

	var log []traceEvent
	planner := &recordingSysPlanner{turns: turns}

	metrics, err := RunArm(context.Background(), planner, "fan out tasks", true, len(turns)+1, &log, WithTaskTools())
	if err != nil {
		t.Fatalf("RunArm with WithTaskTools failed: %v", err)
	}

	if metrics.EngineCalls < 2 {
		t.Errorf("expected at least 2 engine calls, got %d", metrics.EngineCalls)
	}

	foundSpawn, foundStatus := false, false
	for _, ev := range log {
		if ev.Tool == ToolTaskSpawn {
			foundSpawn = true
			if ev.Verdict != "ALLOW" {
				t.Errorf("task_spawn verdict = %q, want ALLOW", ev.Verdict)
			}
			if ev.By != RungNameTask {
				t.Errorf("task_spawn by = %q, want %q", ev.By, RungNameTask)
			}
		}
		if ev.Tool == ToolTaskStatus {
			foundStatus = true
			if ev.Verdict != "ALLOW" {
				t.Errorf("task_status verdict = %q, want ALLOW", ev.Verdict)
			}
			if ev.By != RungNameTask {
				t.Errorf("task_status by = %q, want %q", ev.By, RungNameTask)
			}
		}
	}

	if !foundSpawn {
		t.Errorf("task_spawn not found in execution log")
	}
	if !foundStatus {
		t.Errorf("task_status not found in execution log")
	}

	st := GetActiveTaskState()
	if st == nil {
		t.Fatalf("expected active TaskState, got nil")
	}
	tasks := st.GetTasks()
	if len(tasks) != 1 || tasks[0].Prompt != "Explore architecture boundaries" {
		t.Fatalf("unexpected tasks in active TaskState: %+v", tasks)
	}
}
