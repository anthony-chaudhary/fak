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
