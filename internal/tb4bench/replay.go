package tb4bench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// ReplayViewer renders executed agent trajectories to terminal output.
type ReplayViewer struct {
	Compact bool
}

// NewReplayViewer creates a trace replay viewer.
func NewReplayViewer(compact bool) *ReplayViewer {
	return &ReplayViewer{Compact: compact}
}

// LoadTranscriptJSONL loads an ArmExecutionResult or series of TurnRecords from JSONL.
func LoadTranscriptJSONL(path string) (*ArmExecutionResult, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var result ArmExecutionResult
	result.Status = "COMPLETED"

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(bytesTrim(line)) == 0 {
			continue
		}

		// Try decoding as ArmExecutionResult
		var fullRes ArmExecutionResult
		if err := json.Unmarshal(line, &fullRes); err == nil && fullRes.ArmID != "" {
			return &fullRes, nil
		}

		// Try decoding as single TurnRecord
		var turn TurnRecord
		if err := json.Unmarshal(line, &turn); err == nil && turn.Turn > 0 {
			result.Turns = append(result.Turns, turn)
		}
	}

	result.TotalTurns = len(result.Turns)
	return &result, scanner.Err()
}

func bytesTrim(b []byte) []byte {
	return []byte(strings.TrimSpace(string(b)))
}

// RenderTurn formats a single turn into an ANSI-compatible card.
func (v *ReplayViewer) RenderTurn(t TurnRecord, maxTurns int) string {
	var sb strings.Builder

	// Turn banner
	sb.WriteString(fmt.Sprintf("┌── [Turn %d", t.Turn))
	if maxTurns > 0 {
		sb.WriteString(fmt.Sprintf(" / %d", maxTurns))
	}
	sb.WriteString(fmt.Sprintf("] [Duration: %dms] [Tokens: +%d prompt / +%d comp] ──\n",
		t.DurationMs, t.PromptTokens, t.CompletionTokens))

	// Model text / reasoning
	if t.ModelText != "" {
		sb.WriteString("│ Model Thoughts:\n")
		lines := strings.Split(t.ModelText, "\n")
		for _, l := range lines {
			sb.WriteString(fmt.Sprintf("│   %s\n", l))
		}
	}

	// Tool proposals and adjudication
	if len(t.ToolCalls) > 0 {
		sb.WriteString("│\n│ Tool Invocations:\n")
		for i, tc := range t.ToolCalls {
			badge := "[ALLOWED]"
			if t.AdjudicationVerdict != "" {
				badge = fmt.Sprintf("[%s]", t.AdjudicationVerdict)
			}
			sb.WriteString(fmt.Sprintf("│   (%d) %s %s args: %s\n", i+1, badge, tc.Name, tc.Arguments))
		}
	}

	// Tool outputs
	if !v.Compact && len(t.ToolResults) > 0 {
		sb.WriteString("│\n│ Tool Outputs:\n")
		for _, res := range t.ToolResults {
			out := res.Stdout
			if res.Stderr != "" {
				out += " [stderr: " + res.Stderr + "]"
			}
			lines := strings.Split(out, "\n")
			for _, l := range lines {
				if len(l) > 120 {
					l = l[:117] + "..."
				}
				sb.WriteString(fmt.Sprintf("│   > %s\n", l))
			}
		}
	}

	sb.WriteString("└───\n")
	return sb.String()
}

// RenderTrajectory renders the entire trajectory in headless mode.
func (v *ReplayViewer) RenderTrajectory(w io.Writer, res *ArmExecutionResult) {
	_, _ = fmt.Fprintf(w, "========================================================\n")
	_, _ = fmt.Fprintf(w, "TB4 Run Replay: Task %s | Arm: %s | Status: %s\n", res.TaskID, res.ArmID, res.Status)
	_, _ = fmt.Fprintf(w, "Total Turns: %d | Duration: %dms | Tokens: %d prompt, %d comp\n",
		res.TotalTurns, res.DurationMs, res.TotalPromptTokens, res.TotalCompletionTokens)
	if res.PolicyBlocks > 0 {
		_, _ = fmt.Fprintf(w, "Policy Blocks: %d\n", res.PolicyBlocks)
	}
	_, _ = fmt.Fprintf(w, "========================================================\n\n")

	for _, turn := range res.Turns {
		_, _ = fmt.Fprint(w, v.RenderTurn(turn, res.TotalTurns))
	}

	_, _ = fmt.Fprintf(w, "\n[Final Verdict: %s]\n", res.Status)
}

// RenderComparativeSideBySide renders dual-column progression comparing Arm A and Arm B.
func (v *ReplayViewer) RenderComparativeSideBySide(w io.Writer, resA, resB *ArmExecutionResult) {
	_, _ = fmt.Fprintf(w, "========================================================================================\n")
	_, _ = fmt.Fprintf(w, "TB4 Comparative Replay: Task %s\n", resA.TaskID)
	_, _ = fmt.Fprintf(w, "%-42s | %-42s\n", "Arm A (fak native)", "Arm B (OpenCode reference)")
	_, _ = fmt.Fprintf(w, "Status: %-34s | Status: %-34s\n", resA.Status, resB.Status)
	_, _ = fmt.Fprintf(w, "Turns: %-35d | Turns: %-35d\n", resA.TotalTurns, resB.TotalTurns)
	_, _ = fmt.Fprintf(w, "========================================================================================\n\n")

	maxTurns := len(resA.Turns)
	if len(resB.Turns) > maxTurns {
		maxTurns = len(resB.Turns)
	}

	for turnNum := 1; turnNum <= maxTurns; turnNum++ {
		var actionA, actionB string

		if turnNum <= len(resA.Turns) {
			tA := resA.Turns[turnNum-1]
			if len(tA.ToolCalls) > 0 {
				actionA = fmt.Sprintf("Tool: %s", tA.ToolCalls[0].Name)
			} else {
				actionA = "Thinking / Finalizing"
			}
		} else {
			actionA = "(Finished)"
		}

		if turnNum <= len(resB.Turns) {
			tB := resB.Turns[turnNum-1]
			if len(tB.ToolCalls) > 0 {
				actionB = fmt.Sprintf("Tool: %s", tB.ToolCalls[0].Name)
			} else {
				actionB = "Thinking / Finalizing"
			}
		} else {
			actionB = "(Finished)"
		}

		_, _ = fmt.Fprintf(w, "Turn %02d: %-33s | Turn %02d: %-33s\n", turnNum, actionA, turnNum, actionB)
	}
}
