package sessionreset

import (
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/taskdecision"
)

func init() {
	Register(taskDecisionLog{})
}

type taskDecisionLog struct{}

func (taskDecisionLog) Name() string { return "task_decision_log" }

func (taskDecisionLog) Contribute(in Input) (Part, bool) {
	text := taskdecision.Render(in.DecisionLog)
	if text == "" {
		return Part{Name: "task_decision_log", Order: 30}, false
	}
	meta := map[string]string{"entries": strconv.Itoa(len(in.DecisionLog))}
	if len(in.DecisionLog) > 0 {
		meta["task_id"] = in.DecisionLog[0].TaskID
	}
	return Part{Name: "task_decision_log", Order: 30, Text: text, Meta: meta}, true
}
