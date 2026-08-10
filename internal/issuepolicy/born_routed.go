package issuepolicy

import "strings"

const (
	BornLaneMissing          = "born_lane_missing"
	BornClassLabelMissing    = "born_class_label_missing"
	BornPriorityLabelMissing = "born_priority_label_missing"
)

// BornRouted is the issue-creation routing readout attached to every review.
type BornRouted struct {
	Lane          string   `json:"lane,omitempty"`
	ClassLabel    string   `json:"class_label,omitempty"`
	PriorityLabel string   `json:"priority_label,omitempty"`
	Flags         []string `json:"flags,omitempty"`
}

func bornRouted(c Candidate) BornRouted {
	r := BornRouted{Lane: strings.TrimSpace(c.Lane)}
	for _, raw := range c.Labels {
		label := strings.TrimSpace(raw)
		lower := strings.ToLower(label)
		if r.ClassLabel == "" && strings.HasPrefix(lower, "class:") && len(label) > len("class:") {
			r.ClassLabel = label
		}
		if r.PriorityLabel == "" && strings.HasPrefix(lower, "priority/p") && len(label) > len("priority/p") {
			r.PriorityLabel = label
		}
	}
	if r.Lane == "" {
		r.Flags = append(r.Flags, BornLaneMissing)
	}
	if r.ClassLabel == "" {
		r.Flags = append(r.Flags, BornClassLabelMissing)
	}
	if r.PriorityLabel == "" {
		r.Flags = append(r.Flags, BornPriorityLabelMissing)
	}
	return r
}
