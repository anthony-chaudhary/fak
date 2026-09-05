package trajectory_test

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func TestEpisodeDerive_ClassifyToolIntent_ClosedVocabulary(t *testing.T) {
	cases := []struct {
		name    string
		tool    string
		input   string
		isError bool
		want    ctxmmu.EpisodeType
	}{
		// Explore tools: Read, Grep, Glob
		{"read file", "read", "main.go", false, ctxmmu.EpisodeExplore},
		{"read uppercase", "Read", "internal/mmu.go", false, ctxmmu.EpisodeExplore},
		{"fak read", "fak_read", "file.go", false, ctxmmu.EpisodeExplore},
		{"grep codebase", "grep", "func Test", false, ctxmmu.EpisodeExplore},
		{"glob pattern", "glob", "*.go", false, ctxmmu.EpisodeExplore},
		{"file search", "file_search", "pattern", false, ctxmmu.EpisodeExplore},
		{"web fetch", "webfetch", "https://example.com", false, ctxmmu.EpisodeExplore},

		// Mutate tools: Edit, Write
		{"edit file", "edit", "main.go", false, ctxmmu.EpisodeMutate},
		{"edit uppercase", "Edit", "main.go", false, ctxmmu.EpisodeMutate},
		{"write file", "write", "new_file.go", false, ctxmmu.EpisodeMutate},
		{"write uppercase", "Write", "new_file.go", false, ctxmmu.EpisodeMutate},
		{"patch diff", "apply_patch", "diff text", false, ctxmmu.EpisodeMutate},
		{"create file", "create_file", "path.go", false, ctxmmu.EpisodeMutate},

		// Verify tools: Test, build, lint
		{"go test", "test", "go test ./...", false, ctxmmu.EpisodeVerify},
		{"unit test", "unit_test", "run", false, ctxmmu.EpisodeVerify},
		{"build binary", "build", "go build ./cmd/fak", false, ctxmmu.EpisodeVerify},
		{"compile step", "compile", "target", false, ctxmmu.EpisodeVerify},
		{"lint code", "lint", "golangci-lint run", false, ctxmmu.EpisodeVerify},
		{"vet code", "vet", "go vet ./...", false, ctxmmu.EpisodeVerify},
		{"validate verb", "validate", "fak validate --mine", false, ctxmmu.EpisodeVerify},

		// Recovery tools: Error, rollback, undo
		{"explicit error tool", "error", "panic stack", false, ctxmmu.EpisodeRecovery},
		{"rollback tool", "rollback", "revert step", false, ctxmmu.EpisodeRecovery},
		{"undo tool", "undo", "last edit", false, ctxmmu.EpisodeRecovery},
		{"revert tool", "revert", "git revert HEAD", false, ctxmmu.EpisodeRecovery},
		{"git reset tool", "git_reset", "HEAD~1", false, ctxmmu.EpisodeRecovery},
		{"git checkout restore", "git_checkout", "file.go", false, ctxmmu.EpisodeRecovery},

		// Error flag forces recovery regardless of tool
		{"read with error", "read", "main.go", true, ctxmmu.EpisodeRecovery},
		{"edit with error", "edit", "main.go", true, ctxmmu.EpisodeRecovery},
		{"test with error", "test", "go test", true, ctxmmu.EpisodeRecovery},

		// Shell command inspection
		{"bash test", "bash", "go test -v ./internal/ctxmmu/...", false, ctxmmu.EpisodeVerify},
		{"bash build", "bash", "go build ./cmd/fak", false, ctxmmu.EpisodeVerify},
		{"bash lint", "bash", "golangci-lint run", false, ctxmmu.EpisodeVerify},
		{"bash restore", "bash", "git checkout -- main.go", false, ctxmmu.EpisodeRecovery},
		{"bash reset", "bash", "git reset --hard HEAD", false, ctxmmu.EpisodeRecovery},
		{"bash recover", "bash", "fak recover LOCK_BUSY", false, ctxmmu.EpisodeRecovery},
		{"bash diff", "bash", "git diff", false, ctxmmu.EpisodeExplore},
		{"bash status", "bash", "git status", false, ctxmmu.EpisodeExplore},
		{"bash echo redirect", "bash", "echo 'package foo' > foo.go", false, ctxmmu.EpisodeMutate},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := trajectory.ClassifyToolIntent(tc.tool, tc.input, tc.isError)
			if got != tc.want {
				t.Errorf("ClassifyToolIntent(%q, %q, %v) = %s; want %s",
					tc.tool, tc.input, tc.isError, got, tc.want)
			}
		})
	}
}

func TestEpisodeDerive_SequenceClassifier_StepTransitions(t *testing.T) {
	classifier := trajectory.NewEpisodeSequenceClassifier()

	steps := []struct {
		turnSeq        int
		tool           string
		input          string
		isError        bool
		wantEpisode    ctxmmu.EpisodeType
		wantTransition bool
		from           ctxmmu.EpisodeType
		to             ctxmmu.EpisodeType
	}{
		{1, "read", "main.go", false, ctxmmu.EpisodeExplore, false, "", ""},
		{2, "grep", "func Run", false, ctxmmu.EpisodeExplore, false, "", ""},
		{3, "glob", "*.go", false, ctxmmu.EpisodeExplore, false, "", ""},
		{4, "edit", "main.go", false, ctxmmu.EpisodeMutate, true, ctxmmu.EpisodeExplore, ctxmmu.EpisodeMutate},
		{5, "write", "util.go", false, ctxmmu.EpisodeMutate, false, "", ""},
		{6, "test", "go test ./...", false, ctxmmu.EpisodeVerify, true, ctxmmu.EpisodeMutate, ctxmmu.EpisodeVerify},
		{7, "bash", "go test ./...", true, ctxmmu.EpisodeRecovery, true, ctxmmu.EpisodeVerify, ctxmmu.EpisodeRecovery},
		{8, "rollback", "git checkout -- main.go", false, ctxmmu.EpisodeRecovery, false, "", ""},
		{9, "edit", "main.go", false, ctxmmu.EpisodeMutate, true, ctxmmu.EpisodeRecovery, ctxmmu.EpisodeMutate},
		{10, "test", "go test ./...", false, ctxmmu.EpisodeVerify, true, ctxmmu.EpisodeMutate, ctxmmu.EpisodeVerify},
	}

	for _, s := range steps {
		ep, trans := classifier.Step(s.tool, s.input, s.isError, s.turnSeq)
		if ep != s.wantEpisode {
			t.Fatalf("turn %d: got episode %s, want %s", s.turnSeq, ep, s.wantEpisode)
		}
		if s.wantTransition {
			if trans == nil {
				t.Fatalf("turn %d: expected transition from %s to %s, got nil", s.turnSeq, s.from, s.to)
			}
			if trans.FromEpisode != s.from || trans.ToEpisode != s.to {
				t.Errorf("turn %d: transition %s -> %s; want %s -> %s",
					s.turnSeq, trans.FromEpisode, trans.ToEpisode, s.from, s.to)
			}
			if trans.TurnSeq != s.turnSeq {
				t.Errorf("turn %d: transition turnSeq %d; want %d", s.turnSeq, trans.TurnSeq, s.turnSeq)
			}
		} else {
			if trans != nil {
				t.Fatalf("turn %d: unexpected transition: %+v", s.turnSeq, trans)
			}
		}
	}

	transitions := classifier.Transitions()
	if len(transitions) != 5 {
		t.Fatalf("expected 5 transitions, got %d", len(transitions))
	}
}

func TestEpisodeDerive_TransitionsAndSpans(t *testing.T) {
	turns := []trajectory.Turn{
		{Seq: 1, Tool: "read", Query: "main.go"},
		{Seq: 2, Tool: "grep", Query: "target"},
		{Seq: 3, Tool: "edit", Query: "main.go"},
		{Seq: 4, Tool: "write", Query: "helper.go"},
		{Seq: 5, Tool: "test", Query: "go test"},
		{Seq: 6, Tool: "test", Query: "go test", Verdict: "DENY", Reason: "compilation error"},
		{Seq: 7, Tool: "undo", Query: "revert"},
	}

	transitions := trajectory.DeriveEpisodeTransitions(turns)
	if len(transitions) != 3 {
		t.Fatalf("expected 3 transitions, got %d: %+v", len(transitions), transitions)
	}

	expectedTransitions := []struct {
		from ctxmmu.EpisodeType
		to   ctxmmu.EpisodeType
		seq  int
	}{
		{ctxmmu.EpisodeExplore, ctxmmu.EpisodeMutate, 3},
		{ctxmmu.EpisodeMutate, ctxmmu.EpisodeVerify, 5},
		{ctxmmu.EpisodeVerify, ctxmmu.EpisodeRecovery, 6},
	}

	for i, exp := range expectedTransitions {
		if transitions[i].FromEpisode != exp.from || transitions[i].ToEpisode != exp.to || transitions[i].TurnSeq != exp.seq {
			t.Errorf("transition %d: got %s->%s at %d; want %s->%s at %d",
				i, transitions[i].FromEpisode, transitions[i].ToEpisode, transitions[i].TurnSeq, exp.from, exp.to, exp.seq)
		}
	}

	spans := trajectory.DeriveEpisodeSpans(turns)
	if len(spans) != 4 {
		t.Fatalf("expected 4 spans, got %d: %+v", len(spans), spans)
	}

	expectedSpans := []struct {
		ep    ctxmmu.EpisodeType
		start int
		end   int
		count int
	}{
		{ctxmmu.EpisodeExplore, 1, 2, 2},
		{ctxmmu.EpisodeMutate, 3, 4, 2},
		{ctxmmu.EpisodeVerify, 5, 5, 1},
		{ctxmmu.EpisodeRecovery, 6, 7, 2},
	}

	for i, exp := range expectedSpans {
		if spans[i].EpisodeType != exp.ep || spans[i].StartSeq != exp.start || spans[i].EndSeq != exp.end || spans[i].TurnCount != exp.count {
			t.Errorf("span %d: got %s (%d..%d, %d); want %s (%d..%d, %d)",
				i, spans[i].EpisodeType, spans[i].StartSeq, spans[i].EndSeq, spans[i].TurnCount,
				exp.ep, exp.start, exp.end, exp.count)
		}
	}
}

func TestEpisodeDerive_FromEvents(t *testing.T) {
	now := time.Now().UTC()
	events := []trajectory.Event{
		{
			Schema:         trajectory.EventSchema,
			ID:             "evt-1",
			ConversationID: "conv-1",
			Kind:           trajectory.EventTool,
			Action:         "called",
			Timestamp:      now,
			Sequence:       1,
			Visibility:     trajectory.VisibilityPublic,
			Source:         trajectory.EventSource{Type: "test", Adapter: "test", AdapterVersion: "1"},
			Payload:        json.RawMessage(`{"tool":"read","query":"config.json"}`),
		},
		{
			Schema:         trajectory.EventSchema,
			ID:             "evt-2",
			ConversationID: "conv-1",
			Kind:           trajectory.EventTool,
			Action:         "called",
			Timestamp:      now.Add(time.Second),
			Sequence:       2,
			Visibility:     trajectory.VisibilityPublic,
			Source:         trajectory.EventSource{Type: "test", Adapter: "test", AdapterVersion: "1"},
			Payload:        json.RawMessage(`{"tool":"edit","query":"config.json"}`),
		},
		{
			Schema:         trajectory.EventSchema,
			ID:             "evt-3",
			ConversationID: "conv-1",
			Kind:           trajectory.EventTool,
			Action:         "called",
			Timestamp:      now.Add(2 * time.Second),
			Sequence:       3,
			Visibility:     trajectory.VisibilityPublic,
			Source:         trajectory.EventSource{Type: "test", Adapter: "test", AdapterVersion: "1"},
			Payload:        json.RawMessage(`{"tool":"test","query":"go test"}`),
		},
	}

	spans, trans := trajectory.DeriveEpisodesFromEvents(events)
	if len(spans) != 3 {
		t.Fatalf("expected 3 spans, got %d", len(spans))
	}
	if len(trans) != 2 {
		t.Fatalf("expected 2 transitions, got %d", len(trans))
	}
}

func TestEpisodeDerive_CompileFromTurns(t *testing.T) {
	turns := []trajectory.Turn{
		{Seq: 1, Tool: "read", Query: "file1.go", TokenEstimate: 100},
		{Seq: 2, Tool: "read", Query: "file2.go", TokenEstimate: 150},
		{Seq: 3, Tool: "edit", Query: "file1.go", TokenEstimate: 200},
		{Seq: 4, Tool: "test", Query: "go test", TokenEstimate: 300},
	}

	tracker := ctxmmu.NewEpisodeTracker(nil)
	digests, err := trajectory.CompileEpisodesFromTurns(turns, tracker)
	if err != nil {
		t.Fatalf("CompileEpisodesFromTurns failed: %v", err)
	}

	// Transitions: Explore -> Mutate (at turn 3), Mutate -> Verify (at turn 4)
	if len(digests) != 2 {
		t.Fatalf("expected 2 digests, got %d", len(digests))
	}

	if digests[0].Type != ctxmmu.EpisodeExplore {
		t.Errorf("digest 0 type = %s; want explore", digests[0].Type)
	}
	if digests[0].ToolCallCount != 2 {
		t.Errorf("digest 0 tool calls = %d; want 2", digests[0].ToolCallCount)
	}
	if digests[1].Type != ctxmmu.EpisodeMutate {
		t.Errorf("digest 1 type = %s; want mutate", digests[1].Type)
	}
	if digests[1].ToolCallCount != 1 {
		t.Errorf("digest 1 tool calls = %d; want 1", digests[1].ToolCallCount)
	}

	// Current active episode should be verify
	if tracker.CurrentEpisode() != ctxmmu.EpisodeVerify {
		t.Errorf("current episode = %s; want verify", tracker.CurrentEpisode())
	}
}
