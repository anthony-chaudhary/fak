package dogfoodscore

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/stopfailure"
)

func TestTurnBoundaryRefusesFreshStopHookFailure(t *testing.T) {
	transcript := []byte(asstLine("Implementation is ready for final verification.") + "\n" +
		`{"type":"user","isMeta":true,"message":{"role":"user","content":"Stop hook error: verification failed"}}`)

	got := CheckTurnBoundary(transcript)
	if got.AllowFinal || !got.FreshFailure {
		t.Fatalf("fresh failure must refuse final narration: %+v", got)
	}
	if got.HarnessLine == "" {
		t.Fatalf("refusal must carry the harness evidence: %+v", got)
	}
}

func TestTurnBoundaryClearsAfterFailureIsHandled(t *testing.T) {
	transcript := []byte(asstLine("Ready.") + "\n" +
		`{"type":"user","isMeta":true,"message":{"role":"user","content":"Stop hook error: verification failed"}}` + "\n" +
		asstLine("I handled the hook failure and reran verification."))

	got := CheckTurnBoundary(transcript)
	if !got.AllowFinal || got.FreshFailure {
		t.Fatalf("a subsequent assistant turn must clear the pre-final refusal: %+v", got)
	}
}

func TestTurnBoundaryIgnoresAssistantQuote(t *testing.T) {
	got := CheckTurnBoundary([]byte(asstLine("The log example says Stop hook error: verification failed.")))
	if !got.AllowFinal || got.FreshFailure {
		t.Fatalf("assistant prose is not harness evidence: %+v", got)
	}
}

// writeTranscript drops one transcript into a temp Claude home laid out the way
// transcriptRoots resolves it, and returns that home.
func writeTranscript(t *testing.T, name, body string) string {
	t.Helper()
	home := t.TempDir()
	project := filepath.Join(home, ".claude-fixture", "projects", stopfailure.DefaultTranscriptNamespace)
	if err := os.MkdirAll(project, 0o755); err != nil {
		t.Fatalf("mkdir transcript project: %v", err)
	}
	if err := os.WriteFile(filepath.Join(project, name), []byte(body), 0o644); err != nil {
		t.Fatalf("write transcript: %v", err)
	}
	return home
}

func TestTurnBoundaryLatestRefusesFreshFailure(t *testing.T) {
	home := writeTranscript(t, "live.jsonl", asstLine("All checks pass.")+"\n"+
		`{"type":"user","isMeta":true,"message":{"role":"user","content":"Stop hook error: verification failed"}}`)

	got := CheckTurnBoundaryLatest(Options{ClaudeHome: home})
	if got.AllowFinal || !got.FreshFailure {
		t.Fatalf("a live fresh failure must refuse final narration: %+v", got)
	}
	if !got.Reachable || got.Transcript == "" {
		t.Fatalf("refusal must name the transcript it read: %+v", got)
	}
}

func TestTurnBoundaryLatestUnreachableIsNotGreen(t *testing.T) {
	got := CheckTurnBoundaryLatest(Options{ClaudeHome: t.TempDir()})
	if got.Reachable {
		t.Fatalf("an empty home has no transcript to read: %+v", got)
	}
	if !got.AllowFinal || got.FreshFailure {
		t.Fatalf("unreachable must not invent a refusal: %+v", got)
	}
	if got.Reason != turnBoundaryUnreachable {
		t.Fatalf("unreachable must say it is unwitnessed, not clean: %+v", got)
	}
}
