package agenticbench

import (
	"fmt"
	"sort"
	"strings"
)

// toolCallClass declares the normalized effect class used as a coverage denominator.
type toolCallClass struct {
	Name      string
	ToolNames []string
}

// importedToolCall is an observed call from an external trajectory. Name and
// CallID are retained verbatim in the joined receipt.
type importedToolCall struct {
	CallID     string `json:"call_id"`
	Name       string `json:"name"`
	Observable bool   `json:"observable"`
}

// fakToolEvent proves that an observed call crossed FAK's policy/context seam.
type fakToolEvent struct {
	EventID string `json:"event_id"`
	CallID  string `json:"call_id"`
}

// externalToolDisposition explicitly accounts for a call that did not cross FAK.
type externalToolDisposition struct {
	CallID string `json:"call_id"`
	Reason string `json:"reason"`
}

type toolCallDisposition string

const (
	toolCallMediated     toolCallDisposition = "mediated"
	toolCallUnmediated   toolCallDisposition = "unmediated"
	toolCallUnknown      toolCallDisposition = "unknown"
	toolCallUnobservable toolCallDisposition = "unobservable"
)

// joinedToolCall preserves source identity while adding normalized attribution.
type joinedToolCall struct {
	CallID       string              `json:"call_id"`
	OriginalName string              `json:"original_name"`
	Class        string              `json:"class"`
	Disposition  toolCallDisposition `json:"disposition"`
	FAKEventID   string              `json:"fak_event_id,omitempty"`
	Reason       string              `json:"reason,omitempty"`
}

type toolCallCoverageCounts struct {
	Observed     int `json:"observed"`
	Mediated     int `json:"mediated"`
	Unmediated   int `json:"unmediated"`
	Unknown      int `json:"unknown"`
	Unobservable int `json:"unobservable"`
}

func (c toolCallCoverageCounts) MediatedPercent() float64 {
	if c.Observed == 0 {
		return 0
	}
	return 100 * float64(c.Mediated) / float64(c.Observed)
}

// toolCallCoverageReceipt reports effect/tool-call coverage only. Endpoint or
// model-call coverage belongs in a separate receipt and cannot satisfy this one.
type toolCallCoverageReceipt struct {
	Calls             []joinedToolCall                  `json:"calls"`
	ByClass           map[string]toolCallCoverageCounts `json:"by_class"`
	Overall           toolCallCoverageCounts            `json:"overall"`
	FAKGovernedRun    bool                              `json:"fak_governed_run"`
	GovernanceRefusal string                            `json:"governance_refusal,omitempty"`
}

// joinToolCallCoverage joins a trajectory denominator to FAK policy/context
// events. An unmatched observable call is unknown, never implicitly mediated.
func joinToolCallCoverage(calls []importedToolCall, classes []toolCallClass, events []fakToolEvent, external []externalToolDisposition, claimGoverned bool) (toolCallCoverageReceipt, error) {
	toolClass := make(map[string]string)
	for _, class := range classes {
		name := strings.TrimSpace(class.Name)
		if name == "" {
			return toolCallCoverageReceipt{}, fmt.Errorf("tool class name is required")
		}
		for _, tool := range class.ToolNames {
			key := strings.ToLower(strings.TrimSpace(tool))
			if key == "" {
				return toolCallCoverageReceipt{}, fmt.Errorf("tool class %q contains an empty tool name", name)
			}
			if prior, ok := toolClass[key]; ok {
				return toolCallCoverageReceipt{}, fmt.Errorf("tool %q is declared in both %q and %q", tool, prior, name)
			}
			toolClass[key] = name
		}
	}
	eventByCall := make(map[string]fakToolEvent)
	eventIDs := make(map[string]bool)
	for _, event := range events {
		if strings.TrimSpace(event.CallID) == "" || strings.TrimSpace(event.EventID) == "" {
			return toolCallCoverageReceipt{}, fmt.Errorf("FAK event requires call_id and event_id")
		}
		if _, ok := eventByCall[event.CallID]; ok {
			return toolCallCoverageReceipt{}, fmt.Errorf("duplicate FAK event for call_id %q", event.CallID)
		}
		if eventIDs[event.EventID] {
			return toolCallCoverageReceipt{}, fmt.Errorf("duplicate FAK event_id %q", event.EventID)
		}
		eventByCall[event.CallID], eventIDs[event.EventID] = event, true
	}
	externalByCall := make(map[string]externalToolDisposition)
	for _, disposition := range external {
		if strings.TrimSpace(disposition.CallID) == "" || strings.TrimSpace(disposition.Reason) == "" {
			return toolCallCoverageReceipt{}, fmt.Errorf("external disposition requires call_id and reason")
		}
		if _, ok := externalByCall[disposition.CallID]; ok {
			return toolCallCoverageReceipt{}, fmt.Errorf("duplicate external disposition for call_id %q", disposition.CallID)
		}
		externalByCall[disposition.CallID] = disposition
	}

	receipt := toolCallCoverageReceipt{ByClass: make(map[string]toolCallCoverageCounts)}
	seenCalls := make(map[string]bool)
	for _, call := range calls {
		if strings.TrimSpace(call.CallID) == "" {
			return toolCallCoverageReceipt{}, fmt.Errorf("imported tool call is missing call_id")
		}
		if seenCalls[call.CallID] {
			return toolCallCoverageReceipt{}, fmt.Errorf("duplicate imported call_id %q", call.CallID)
		}
		seenCalls[call.CallID] = true
		class, ok := toolClass[strings.ToLower(strings.TrimSpace(call.Name))]
		if !ok {
			return toolCallCoverageReceipt{}, fmt.Errorf("tool %q has no declared normalized class", call.Name)
		}
		joined := joinedToolCall{CallID: call.CallID, OriginalName: call.Name, Class: class}
		event, mediated := eventByCall[call.CallID]
		disposition, explicitlyExternal := externalByCall[call.CallID]
		if mediated && explicitlyExternal {
			return toolCallCoverageReceipt{}, fmt.Errorf("call_id %q is both mediated and externally disposed", call.CallID)
		}
		switch {
		case !call.Observable:
			joined.Disposition = toolCallUnobservable
			joined.Reason = "source adapter declared the call unobservable"
		case mediated:
			joined.Disposition, joined.FAKEventID = toolCallMediated, event.EventID
		case explicitlyExternal:
			joined.Disposition, joined.Reason = toolCallUnmediated, disposition.Reason
		default:
			joined.Disposition = toolCallUnknown
			joined.Reason = "no FAK event or external disposition matched call_id"
		}
		receipt.Calls = append(receipt.Calls, joined)
		counts := receipt.ByClass[class]
		addToolCallCount(&counts, joined.Disposition)
		receipt.ByClass[class] = counts
		addToolCallCount(&receipt.Overall, joined.Disposition)
	}
	for callID := range eventByCall {
		if !seenCalls[callID] {
			return toolCallCoverageReceipt{}, fmt.Errorf("FAK event references unobserved call_id %q", callID)
		}
	}
	for callID := range externalByCall {
		if !seenCalls[callID] {
			return toolCallCoverageReceipt{}, fmt.Errorf("external disposition references unobserved call_id %q", callID)
		}
	}
	if claimGoverned && (receipt.Overall.Unmediated > 0 || receipt.Overall.Unknown > 0) {
		receipt.GovernanceRefusal = fmt.Sprintf("refused fak_governed_run=true: observed tool-call effects include %d unmediated and %d unknown calls; endpoint/model-call coverage is not effect/tool-call coverage", receipt.Overall.Unmediated, receipt.Overall.Unknown)
	} else if claimGoverned {
		receipt.FAKGovernedRun = true
	}
	return receipt, nil
}

func addToolCallCount(counts *toolCallCoverageCounts, disposition toolCallDisposition) {
	counts.Observed++
	switch disposition {
	case toolCallMediated:
		counts.Mediated++
	case toolCallUnmediated:
		counts.Unmediated++
	case toolCallUnknown:
		counts.Unknown++
	case toolCallUnobservable:
		counts.Unobservable++
	}
}

// toolCallCoverageClasses returns stable class ordering for renderers.
func toolCallCoverageClasses(receipt toolCallCoverageReceipt) []string {
	classes := make([]string, 0, len(receipt.ByClass))
	for class := range receipt.ByClass {
		classes = append(classes, class)
	}
	sort.Strings(classes)
	return classes
}
