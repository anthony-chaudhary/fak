package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

func TestResultLivelockObservesExecPollingAndResetsOnProgress(t *testing.T) {
	s := &Server{}
	mk := func(tool, digest string) []ResultAdmission {
		return []ResultAdmission{{Tool: tool, ResultDigest: digest, Verdict: WireVerdict{Kind: "ALLOW"}}}
	}
	var got []ResultAdmission
	for i := 0; i < 9; i++ {
		got = mk("write_stdin", "same-poll")
		s.annotateResultLivelock("codex-poll", got)
	}
	if got[0].Livelock == nil || !got[0].Livelock.Escalate {
		t.Fatalf("repeated poll did not reach abort threshold: %+v", got[0])
	}
	progress := mk("search_kb", "new-output")
	s.annotateResultLivelock("codex-poll", progress)
	again := mk("write_stdin", "same-poll")
	s.annotateResultLivelock("codex-poll", again)
	if again[0].Livelock != nil {
		t.Fatalf("progress did not reset polling run: %+v", again[0])
	}

	// Verify todowrite low-info receipt livelock detection
	for i := 0; i < 9; i++ {
		got = mk("todowrite", string(guardrsi.ArgsDigest("Plan updated")))
		s.annotateResultLivelock("todowrite-poll", got)
	}
	if got[0].Livelock == nil || !got[0].Livelock.Escalate {
		t.Fatalf("repeated todowrite low-info receipt did not reach abort threshold: %+v", got[0])
	}
	todowriteProgress := mk("todowrite", "non-empty-receipt-progress")
	s.annotateResultLivelock("todowrite-poll", todowriteProgress)
	againTodowrite := mk("todowrite", string(guardrsi.ArgsDigest("Plan updated")))
	s.annotateResultLivelock("todowrite-poll", againTodowrite)
	if againTodowrite[0].Livelock != nil {
		t.Fatalf("progress did not reset todowrite livelock run: %+v", againTodowrite[0])
	}
}

func TestIsUpdatePlanTool(t *testing.T) {
	planTools := []string{
		"update_plan",
		"functions.update_plan",
		"todowrite",
		"agent.todowrite",
		"functions.todowrite",
		"TodoWrite",
		" agent.todowrite ",
		"FUNCTIONS.UPDATE_PLAN",
	}
	for _, tool := range planTools {
		if !isUpdatePlanTool(tool) {
			t.Errorf("isUpdatePlanTool(%q) = false, want true", tool)
		}
	}

	nonPlanTools := []string{
		"todoread",
		"exec_command",
		"bash",
		"write_file",
	}
	for _, tool := range nonPlanTools {
		if isUpdatePlanTool(tool) {
			t.Errorf("isUpdatePlanTool(%q) = true, want false", tool)
		}
	}
}
