package trajectory

import (
	"sort"
	"strings"
)

// QwenToolErrorEvent identifies a tool error and its position in a trajectory.
type QwenToolErrorEvent struct {
	Content string
	Index   int
	Tokens  uint64
}

// QwenToolErrorFamily summarizes occurrences of one classified error family.
type QwenToolErrorFamily struct {
	Family     string
	Count      int
	FirstIndex int
	LastIndex  int
	Tokens     uint64
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
