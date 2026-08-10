package flowmetrics

import (
	"regexp"
	"strings"
)

var (
	// checkboxRe matches a GitHub task-list item in any of the three bullet
	// forms, at any indentation.
	checkboxRe = regexp.MustCompile(`^\s*[-*+]\s+\[([ xX])\]\s`)
	// dodRe matches the headings that declare a done-condition. `dod` needs
	// word boundaries so it does not fire inside unrelated words.
	dodRe = regexp.MustCompile(`(?i)(definition of done|acceptance criteria|\bdod\b)`)
	// fenceRe matches a fenced code block delimiter, whose contents must not
	// be mined for checkboxes.
	fenceRe = regexp.MustCompile("^\\s*(```|~~~)")
)

// Checklist is the declared done-condition of an issue, counted from its body.
type Checklist struct {
	Checked   int `json:"checked"`
	Unchecked int `json:"unchecked"`
}

// Total is the number of declared done-conditions.
func (c Checklist) Total() int { return c.Checked + c.Unchecked }

// PercentComplete reports progress against the declared checklist, rounded
// down, and whether that number means anything.
//
// known is false when the issue declares no checklist, and callers must
// propagate that rather than substituting zero. There is no fallback estimate
// on purpose: the available proxy would be commit count, and commit count
// rises under thrash exactly as it does under progress, so a fallback would
// make the metric most confident when it is most wrong.
func (c Checklist) PercentComplete() (pct int, known bool) {
	total := c.Total()
	if total == 0 {
		return 0, false
	}
	return c.Checked * 100 / total, true
}

// ParseChecklist counts the task-list items in an issue body, skipping fenced
// code blocks so a checklist pasted as an example does not inflate the scope.
func ParseChecklist(body string) Checklist {
	var list Checklist
	inFence := false
	for _, line := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		if fenceRe.MatchString(line) {
			inFence = !inFence
			continue
		}
		if inFence {
			continue
		}
		m := checkboxRe.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		if m[1] == " " {
			list.Unchecked++
			continue
		}
		list.Checked++
	}
	return list
}

// HasDoD reports whether the body names a done-condition at all. An issue
// without one cannot be verified complete by anything but opinion.
func HasDoD(body string) bool { return dodRe.MatchString(body) }

// ScopeClass rates how atomic a ticket's DECLARED scope is. It reads intent,
// not outcome: the companion after-the-fact measure is [IssueFlow.Atomic],
// and the gap between the two is where estimation is going wrong.
type ScopeClass string

const (
	// ScopeUndeclared has no checklist, so its size is unknown and it cannot
	// be scheduled honestly. This is the class to eliminate first: it is the
	// only one where nobody can tell whether the ticket is atomic.
	ScopeUndeclared ScopeClass = "undeclared"
	// ScopeAtomic is one landable unit: 1-4 done-conditions.
	ScopeAtomic ScopeClass = "atomic"
	// ScopeCompound is 5-9 done-conditions — splittable, and usually should be.
	ScopeCompound ScopeClass = "compound"
	// ScopeEpic is 10+ done-conditions: a container, not a unit of work.
	// An epic may stay open indefinitely without counting as WIP, provided
	// only its leaves are ever started.
	ScopeEpic ScopeClass = "epic"
)

// Scope thresholds. AtomicMax is 4 because a four-item done-condition still
// fits one review; EpicMin is 10 because past that the list is a plan.
const (
	AtomicMax = 4
	EpicMin   = 10
)

// ClassifyScope buckets an issue body by its declared done-condition count.
func ClassifyScope(body string) ScopeClass {
	total := ParseChecklist(body).Total()
	switch {
	case total == 0:
		return ScopeUndeclared
	case total <= AtomicMax:
		return ScopeAtomic
	case total < EpicMin:
		return ScopeCompound
	default:
		return ScopeEpic
	}
}

// CountableWIP reports whether an issue in this scope class should count
// against a WIP cap when it is in flight.
//
// Epics are excluded, and that exclusion is what makes a WIP cap survive
// contact with a real roadmap: the big picture stays visible as an open
// container while only its leaves consume capacity. The rule it depends on is
// that an epic is never itself started — if commits land directly on an epic,
// [IssueFlow] will show it in flight and the exemption is being abused.
func CountableWIP(class ScopeClass) bool { return class != ScopeEpic }
