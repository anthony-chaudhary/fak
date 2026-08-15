package issuepolicy

import (
	"fmt"
	"regexp"
	"strings"
)

const ProblemFrameSchema = "fak-problem-frame/1"

const (
	CentralityCore         = "core"
	CentralityEnabling     = "enabling"
	CentralityStewardship  = "stewardship"
	CentralityPeripheral   = "peripheral"
	CentralityUnclassified = "unclassified"
)

const (
	ProblemCheckAdvanced  = "advanced"
	ProblemCheckPreserved = "preserved"
	ProblemCheckNA        = "n/a"
)

var problemCheckRE = regexp.MustCompile(`(?i)^(advanced|preserved|n\s*/\s*a)\b\s*(?:[-:]+\s*)?(.*)$`)

type ProblemCheck struct {
	ID           string `json:"id"`
	Status       string `json:"status,omitempty"`
	Evidence     string `json:"evidence,omitempty"`
	Valid        bool   `json:"valid"`
	Reason       string `json:"reason,omitempty"`
	RepairAction string `json:"repair_action,omitempty"`
}

type ProblemFrame struct {
	Schema           string                  `json:"schema"`
	Ready            bool                    `json:"ready"`
	Enforced         bool                    `json:"enforced"`
	Centrality       string                  `json:"centrality"`
	CentralityTarget string                  `json:"centrality_target,omitempty"`
	Checks           map[string]ProblemCheck `json:"checks"`
	Reasons          []string                `json:"reasons,omitempty"`
	RepairActions    []string                `json:"repair_actions,omitempty"`
}

// AssessProblemFrame validates the problem-centrality class and all four
// problem checks at the shift-left task-creation boundary.
func AssessProblemFrame(d IssueDraft) ProblemFrame {
	sections := markdownSections(d.Body)
	value := labeledBriefValues(sections["value"])
	centralityRaw := firstNonEmpty(value["centrality"], labeledLineValue(d.Body, "centrality"))
	out := ProblemFrame{
		Schema:     ProblemFrameSchema,
		Ready:      true,
		Enforced:   hasProblemFrameVocabulary(d.Body) || hasBriefVocabulary(sections),
		Centrality: CentralityUnclassified,
		Checks:     make(map[string]ProblemCheck, 4),
	}

	out.Centrality, out.CentralityTarget = parseCentrality(centralityRaw)
	if out.Enforced {
		switch out.Centrality {
		case CentralityCore, CentralityPeripheral:
		case CentralityEnabling:
			if out.CentralityTarget == "" {
				out.addProblemReason("problem_centrality_target_missing", "centrality: name the Core outcome this Enabling work unlocks")
			}
		case CentralityStewardship:
			if out.CentralityTarget == "" {
				out.addProblemReason("problem_centrality_obligation_missing", "centrality: name the obligation this Stewardship work satisfies")
			}
		default:
			out.addProblemReason("problem_centrality_invalid", "centrality: use Core, Enabling (<named Core outcome>), Stewardship (<obligation>), or Peripheral")
		}
	}

	for _, id := range []string{"p1", "p2", "p3", "p4"} {
		raw := labeledLineValue(d.Body, id)
		check := parseProblemCheck(id, raw)
		out.Checks[id] = check
		if out.Enforced && !check.Valid {
			out.addProblemReason("problem_check_"+id+"_"+check.Reason, id+": "+check.RepairAction)
		}
	}
	return out
}

func parseCentrality(raw string) (string, string) {
	raw = strings.TrimSpace(raw)
	lower := strings.ToLower(raw)
	for _, item := range []struct{ prefix, class string }{
		{"stewardship", CentralityStewardship},
		{"peripheral", CentralityPeripheral},
		{"enabling", CentralityEnabling},
		{"core", CentralityCore},
	} {
		if lower == item.prefix {
			return item.class, ""
		}
		if strings.HasPrefix(lower, item.prefix) {
			rest := strings.TrimSpace(raw[len(item.prefix):])
			rest = strings.TrimSpace(strings.Trim(rest, "()[]{}:- "))
			return item.class, rest
		}
	}
	return CentralityUnclassified, ""
}

func parseProblemCheck(id, raw string) ProblemCheck {
	check := ProblemCheck{ID: id, Valid: false}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		check.Reason = "missing"
		check.RepairAction = "declare advanced, preserved, or N/A - <concrete reason>"
		return check
	}
	m := problemCheckRE.FindStringSubmatch(raw)
	if m == nil {
		check.Reason = "invalid"
		check.RepairAction = "start with advanced, preserved, or N/A - <concrete reason>"
		return check
	}
	status := strings.ToLower(strings.ReplaceAll(m[1], " ", ""))
	if status == "n/a" {
		check.Status = ProblemCheckNA
	} else {
		check.Status = status
	}
	check.Evidence = strings.TrimSpace(m[2])
	if check.Evidence == "" {
		check.Reason = "ceremonial"
		check.RepairAction = fmt.Sprintf("explain how %s is %s; a bare label is not evidence", strings.ToUpper(id), check.Status)
		return check
	}
	check.Valid = true
	return check
}

func hasProblemFrameVocabulary(body string) bool {
	return labeledLinePresent(body, "centrality") ||
		labeledLinePresent(body, "p1") || labeledLinePresent(body, "p2") ||
		labeledLinePresent(body, "p3") || labeledLinePresent(body, "p4")
}

func labeledLinePresent(body, label string) bool {
	label = strings.ToLower(strings.TrimSpace(label))
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(line), "-*"))
		key, _, ok := strings.Cut(line, ":")
		if ok && strings.EqualFold(strings.TrimSpace(key), label) {
			return true
		}
	}
	return false
}

func labeledLineValue(body, label string) string {
	label = strings.ToLower(strings.TrimSpace(label))
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(line), "-*"))
		key, value, ok := strings.Cut(line, ":")
		if ok && strings.EqualFold(strings.TrimSpace(key), label) {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func (p *ProblemFrame) addProblemReason(reason, repair string) {
	p.Ready = false
	p.Reasons = append(p.Reasons, reason)
	p.RepairActions = append(p.RepairActions, repair)
}
