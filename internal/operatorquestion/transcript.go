package operatorquestion

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/resume/transcript"
)

// LastFromTranscript reads a harness transcript and normalizes the last native
// operator-question tool request. found=false means no recognized operator gate occurred.
func LastFromTranscript(path, harnessCommand string) (q OperatorQuestion, found bool, err error) {
	f, err := os.Open(path)
	if err != nil {
		return OperatorQuestion{}, false, err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	for {
		var record transcript.Record
		if err := dec.Decode(&record); err != nil {
			if err == io.EOF {
				break
			}
			return OperatorQuestion{}, false, fmt.Errorf("operatorquestion transcript: %w", err)
		}
		for _, use := range record.ToolUses() {
			candidate, normalizeErr := Normalize(NativeGate{HarnessCommand: harnessCommand, Tool: use.Name, Payload: use.Input})
			if normalizeErr == nil {
				q, found = candidate, true
			}
		}
	}
	return q, found, nil
}

// LastFromTranscriptAny tries the registered first-class harness projections and returns
// the last recognized gate. A transcript belongs to one harness in practice; this helper
// lets Stop-hook consumers remain harness-agnostic when the hook payload names only a path.
func LastFromTranscriptAny(path string) (OperatorQuestion, bool, error) {
	for _, harness := range []string{"claude", "codex"} {
		q, found, err := LastFromTranscript(path, harness)
		if err != nil {
			return OperatorQuestion{}, false, err
		}
		if found {
			return q, true, nil
		}
	}
	return OperatorQuestion{}, false, nil
}
