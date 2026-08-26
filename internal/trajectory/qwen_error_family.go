package trajectory

import (
	"sort"
	"strings"
)

// QwenToolErrorEvent identifies a tool error and its position in a trajectory.
type QwenToolErrorEvent struct {
	Content string `json:"content"`
	Index   int    `json:"index"`
	Tokens  uint64 `json:"tokens"`

	repeatedFailures int
	mutationChurn    int
}

// QwenToolErrorFamily summarizes occurrences of one classified error family.
type QwenToolErrorFamily struct {
	Family           string `json:"family"`
	Count            int    `json:"count"`
	FirstIndex       int    `json:"first_index"`
	LastIndex        int    `json:"last_index"`
	Tokens           uint64 `json:"tokens"`
	RepeatedFailures int    `json:"repeated_failures"`
	MutationChurn    int    `json:"mutation_churn"`
}

func classifyQwenToolError(content string) string {
	text := strings.ToLower(content)
	switch {
	case strings.Contains(text, "timeout"), strings.Contains(text, "deadline"):
		return "timeout"
	case strings.Contains(text, "permission"), strings.Contains(text, "denied"), strings.Contains(text, "policy_block"):
		return "permission"
	case strings.Contains(text, "not found"), strings.Contains(text, "no such"):
		return "not_found"
	default:
		return "unknown"
	}
}

func rankQwenToolErrorFamilies(events []QwenToolErrorEvent) []QwenToolErrorFamily {
	byFamily := make(map[string]*QwenToolErrorFamily)
	for _, event := range events {
		family := classifyQwenToolError(event.Content)
		summary := byFamily[family]
		if summary == nil {
			summary = &QwenToolErrorFamily{
				Family:     family,
				FirstIndex: event.Index,
				LastIndex:  event.Index,
			}
			byFamily[family] = summary
		}
		summary.Count++
		summary.LastIndex = event.Index
		summary.Tokens += event.Tokens
		summary.RepeatedFailures += event.repeatedFailures
		summary.MutationChurn += event.mutationChurn
	}

	families := make([]QwenToolErrorFamily, 0, len(byFamily))
	for _, family := range byFamily {
		families = append(families, *family)
	}
	sort.Slice(families, func(i, j int) bool {
		if families[i].Count != families[j].Count {
			return families[i].Count > families[j].Count
		}
		return families[i].Family < families[j].Family
	})
	return families
}
