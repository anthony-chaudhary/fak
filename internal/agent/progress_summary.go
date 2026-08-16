package agent

import (
	"encoding/json"
	"fmt"
	"strings"
)

const maxProgressSummaryBytes = 2048

func progressResultSummary(tool, content string) string {
	if tool == "Read" || content == "" {
		return ""
	}
	if tool == "Bash" {
		var result struct {
			Stdout   string `json:"stdout"`
			Stderr   string `json:"stderr"`
			ExitCode int    `json:"exit_code"`
			TimedOut bool   `json:"timed_out"`
		}
		if json.Unmarshal([]byte(content), &result) == nil {
			text := strings.TrimSpace(result.Stdout)
			if result.Stderr != "" {
				if text != "" {
					text += "\n"
				}
				text += strings.TrimSpace(result.Stderr)
			}
			if text == "" {
				text = "no output"
			}
			return boundProgressSummary(fmt.Sprintf("exit %d: %s", result.ExitCode, text))
		}
	}
	return boundProgressSummary(strings.TrimSpace(content))
}

func boundProgressSummary(summary string) string {
	if len(summary) <= maxProgressSummaryBytes {
		return summary
	}
	return summary[:maxProgressSummaryBytes] + "…[truncated]"
}
