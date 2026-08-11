package issuepolicy

import (
	"fmt"
	"regexp"
	"strings"
)

const TaskBriefSchema = "fak-task-brief-readiness/1"

var typedUnknownRE = regexp.MustCompile(`(?i)^unknown(?:\s*\((.+)\)|\s*[-�:]\s*(?:reason\s*:\s*)?(.+))$`)

var briefRepairs = map[string]string{
	"outcome":      "add an Outcome section naming the observable result",
	"scope":        "add a Scope / tree section naming in-scope paths and exclusions",
	"dependencies": "add a Dependencies section naming issue/graph edges or explicitly saying none",
	"acceptance":   "add an Acceptance section with checkable done conditions",
	"witness":      "add a Witness / proof section naming the proof artifact or command",
	"placement":    "add a Placement section naming generation, priority, milestone, and lane",
}

// AssessIssueBrief classifies the six shift-left task-brief decisions.
func AssessIssueBrief(d IssueDraft) BriefReadiness {
	c := CandidateFromIssueDraft(d)
	return assessIssueBrief(d, c)
}

func assessIssueBrief(d IssueDraft, c Candidate) BriefReadiness {
	sections := markdownSections(d.Body)
	values := map[string]string{
		"outcome":      firstNonEmpty(sections["outcome"], c.RootPoint, c.WhyNow),
		"scope":        firstNonEmpty(sections["scope / tree"], sections["scope"], c.InScope),
		"dependencies": firstNonEmpty(sections["dependencies"], dependencyDeclaration(c)),
		"acceptance":   firstNonEmpty(sections["acceptance"], sections["definition of done"], c.DoneCondition, c.AcceptanceGate),
		"witness":      firstNonEmpty(sections["witness / proof"], sections["witness"], c.Witness),
		"placement":    firstNonEmpty(sections["placement"], placementDeclaration(c)),
	}
	enforced := hasBriefVocabulary(sections) || strings.EqualFold(strings.TrimSpace(c.Schema), "fak-task-brief/1")
	out := BriefReadiness{Ready: true, Enforced: enforced, Fields: make(map[string]BriefField, len(values))}
	for _, name := range []string{"outcome", "scope", "dependencies", "acceptance", "witness", "placement"} {
		field := classifyBriefField(values[name], briefRepairs[name])
		out.Fields[name] = field
		if field.Status == "missing" {
			out.Ready = false
			out.RepairActions = append(out.RepairActions, fmt.Sprintf("%s: %s", name, field.RepairAction))
		}
	}
	return out
}

func classifyBriefField(value, repair string) BriefField {
	value = strings.TrimSpace(value)
	if value == "" {
		return BriefField{Status: "missing", RepairAction: repair}
	}
	if m := typedUnknownRE.FindStringSubmatch(value); m != nil {
		reason := strings.TrimSpace(firstNonEmpty(m[1], m[2]))
		reason = strings.TrimSpace(strings.TrimPrefix(reason, "reason:"))
		if reason == "" {
			return BriefField{Status: "missing", RepairAction: repair + "; unknown requires a reason"}
		}
		return BriefField{Status: "unknown", Reason: reason}
	}
	return BriefField{Status: "present"}
}

func hasBriefVocabulary(sections map[string]string) bool {
	// These headings are unique to the #6418 shift-left template. Generic
	// Outcome/Dependencies sections predate it and remain migration-compatible.
	for _, name := range []string{"scope / tree", "witness / proof", "placement"} {
		if _, ok := sections[name]; ok {
			return true
		}
	}
	return false
}

func dependencyDeclaration(c Candidate) string {
	if len(c.Dependencies) > 0 || strings.TrimSpace(c.ParentRef) != "" {
		return "declared"
	}
	return ""
}

func placementDeclaration(c Candidate) string {
	if strings.TrimSpace(c.Lane) != "" && (strings.TrimSpace(c.Generation) != "" || len(c.Labels) > 0) {
		return "declared"
	}
	return ""
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
