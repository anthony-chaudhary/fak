package loopdrive

import (
	"fmt"
	"strconv"
	"strings"
)

// Spec is the parseable on-disk GOAL.md contract the loop driver re-reads each
// turn. Completion is judged against Witness, while Plan is the durable
// cross-turn checklist memory.
type Spec struct {
	Loop      string
	Witness   string
	Budget    Budget
	Objective string
	Plan      []PlanItem
	Scratch   string
}

// Budget carries the bounded loop budget from frontmatter.
type Budget struct {
	MaxIters  int
	MaxTokens int64
}

// PlanItem is one markdown checklist row under # Plan.
type PlanItem struct {
	Checked bool
	Text    string
}

// Parse reads the minimal GOAL.md format:
//
//	---
//	loop: id
//	witness: commit-audit
//	budget: { max_iters: 20 }
//	---
//	# Objective
//	...
//	# Plan
//	- [ ] ...
func Parse(data []byte) (Spec, error) {
	text := strings.TrimPrefix(strings.ReplaceAll(string(data), "\r\n", "\n"), "\ufeff")
	fm, body, ok := splitFrontmatter(text)
	if !ok {
		return Spec{}, fmt.Errorf("goal spec: missing frontmatter delimited by ---")
	}
	spec, err := parseFrontmatter(fm)
	if err != nil {
		return Spec{}, err
	}
	parseBody(&spec, body)
	if strings.TrimSpace(spec.Loop) == "" {
		return Spec{}, fmt.Errorf("goal spec: frontmatter loop is required")
	}
	if strings.TrimSpace(spec.Witness) == "" {
		return Spec{}, fmt.Errorf("goal spec: frontmatter witness is required")
	}
	if strings.TrimSpace(spec.Objective) == "" {
		return Spec{}, fmt.Errorf("goal spec: # Objective is required")
	}
	return spec, nil
}

// Render emits a canonical GOAL.md form. Parsing a rendered Spec yields the same
// known frontmatter, objective, checklist, and scratch fields.
func (s Spec) Render() []byte {
	var b strings.Builder
	fmt.Fprintf(&b, "---\nloop: %s\nwitness: %s\n", s.Loop, s.Witness)
	if s.Budget.MaxIters > 0 || s.Budget.MaxTokens > 0 {
		var parts []string
		if s.Budget.MaxIters > 0 {
			parts = append(parts, fmt.Sprintf("max_iters: %d", s.Budget.MaxIters))
		}
		if s.Budget.MaxTokens > 0 {
			parts = append(parts, fmt.Sprintf("max_tokens: %d", s.Budget.MaxTokens))
		}
		fmt.Fprintf(&b, "budget: { %s }\n", strings.Join(parts, ", "))
	}
	b.WriteString("---\n# Objective\n")
	writeBlock(&b, s.Objective)
	b.WriteString("\n# Plan\n")
	for _, item := range s.Plan {
		mark := " "
		if item.Checked {
			mark = "x"
		}
		fmt.Fprintf(&b, "- [%s] %s\n", mark, strings.TrimSpace(item.Text))
	}
	b.WriteString("\n# Scratch / last-refusal\n")
	writeBlock(&b, s.Scratch)
	return []byte(b.String())
}

// Template returns a parseable starter GOAL.md.
func Template(loopID string) []byte {
	loopID = strings.TrimSpace(loopID)
	if loopID == "" {
		loopID = "goal"
	}
	return []byte(fmt.Sprintf(`---
loop: %s
witness: commit-audit
	budget: { max_iters: 20 }
---
# Objective
State one unit of work the loop should complete.

# Plan
- [ ] Write the smallest concrete first step.

# Scratch / last-refusal
`, loopID))
}

// NextUnchecked returns the next unchecked plan item and its zero-based index.
func (s Spec) NextUnchecked() (int, PlanItem, bool) {
	for i, item := range s.Plan {
		if !item.Checked {
			return i, item, true
		}
	}
	return -1, PlanItem{}, false
}

// Action is the pure driver's next move.
type Action string

const (
	ActionRunTurn       Action = "run_turn"
	ActionStopWitnessed Action = "stop_witnessed"
	ActionStopBudget    Action = "stop_budget"
)

const (
	ReasonWitnessedDone = "WITNESSED_DONE"
	ReasonBudgetSpent   = "BUDGET_SPENT"
)

// PolicyInput is the spawn-free state the loop driver needs to decide whether
// another fresh-context turn may run.
type PolicyInput struct {
	Iterations       int
	MaxIters         int
	TokensUsed       int64
	MaxTokens        int64
	NowUnixNano      int64
	DeadlineUnixNano int64
	Witnessed        bool
}

// Decision is the spawn-free loop policy verdict.
type Decision struct {
	Action  Action
	Reason  string
	Summary string
}

// Decide applies the loop-drive termination policy. It is intentionally pure:
// no spawning, no filesystem, no clock read. The command shell supplies the
// current counters and clock.
func Decide(in PolicyInput) Decision {
	if in.Witnessed {
		return Decision{Action: ActionStopWitnessed, Reason: ReasonWitnessedDone, Summary: "exit gate witnessed done"}
	}
	if in.MaxIters > 0 && in.Iterations >= in.MaxIters {
		return Decision{Action: ActionStopBudget, Reason: ReasonBudgetSpent,
			Summary: fmt.Sprintf("iteration budget spent (%d/%d)", in.Iterations, in.MaxIters)}
	}
	if in.MaxTokens > 0 && in.TokensUsed >= in.MaxTokens {
		return Decision{Action: ActionStopBudget, Reason: ReasonBudgetSpent,
			Summary: fmt.Sprintf("token budget spent (%d/%d)", in.TokensUsed, in.MaxTokens)}
	}
	if in.DeadlineUnixNano > 0 && in.NowUnixNano >= in.DeadlineUnixNano {
		return Decision{Action: ActionStopBudget, Reason: ReasonBudgetSpent, Summary: "deadline spent"}
	}
	return Decision{Action: ActionRunTurn, Reason: "TURN_READY", Summary: "budget remains"}
}

// NextWork returns the next durable plan item to expose to a fresh-context turn.
// If the checklist is fully checked, the loop still runs a turn so the agent can
// settle the witness gate; completion is never inferred from checked markdown.
func (s Spec) NextWork() (int, PlanItem, bool) {
	if idx, item, ok := s.NextUnchecked(); ok {
		return idx, item, true
	}
	return len(s.Plan), PlanItem{Text: "settle the witness gate"}, false
}

func splitFrontmatter(text string) (string, string, bool) {
	lines := strings.Split(text, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return "", "", false
	}
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			return strings.Join(lines[1:i], "\n"), strings.Join(lines[i+1:], "\n"), true
		}
	}
	return "", "", false
}

func parseFrontmatter(text string) (Spec, error) {
	var spec Spec
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(stripFrontmatterComment(line))
		if line == "" {
			continue
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			return Spec{}, fmt.Errorf("goal spec: malformed frontmatter line %q", line)
		}
		key = strings.ToLower(strings.TrimSpace(key))
		value = unquote(strings.TrimSpace(value))
		switch key {
		case "loop":
			spec.Loop = value
		case "witness":
			spec.Witness = value
		case "budget":
			b, err := parseBudget(value)
			if err != nil {
				return Spec{}, err
			}
			spec.Budget = b
		}
	}
	return spec, nil
}

func parseBudget(value string) (Budget, error) {
	value = strings.TrimSpace(value)
	value = strings.TrimPrefix(value, "{")
	value = strings.TrimSuffix(value, "}")
	var b Budget
	for _, part := range strings.Split(value, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		key, raw, ok := strings.Cut(part, ":")
		if !ok {
			return Budget{}, fmt.Errorf("goal spec: malformed budget item %q", part)
		}
		key = strings.TrimSpace(key)
		if key != "max_iters" && key != "max_tokens" {
			continue
		}
		n, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 64)
		if err != nil || n < 0 {
			return Budget{}, fmt.Errorf("goal spec: budget %s must be a non-negative integer", key)
		}
		switch key {
		case "max_iters":
			b.MaxIters = int(n)
		case "max_tokens":
			b.MaxTokens = n
		}
	}
	return b, nil
}

func parseBody(spec *Spec, body string) {
	var objective, plan, scratch []string
	section := ""
	for _, line := range strings.Split(body, "\n") {
		if next, ok := bodySection(line); ok {
			section = next
			continue
		}
		switch section {
		case "objective":
			objective = append(objective, line)
		case "plan":
			plan = append(plan, line)
		case "scratch":
			scratch = append(scratch, line)
		}
	}
	spec.Objective = trimBlock(objective)
	spec.Scratch = trimBlock(scratch)
	spec.Plan = parsePlan(plan)
}

func bodySection(line string) (string, bool) {
	line = strings.TrimSpace(line)
	if !strings.HasPrefix(line, "#") {
		return "", false
	}
	title := strings.ToLower(strings.TrimSpace(strings.TrimLeft(line, "#")))
	switch {
	case strings.HasPrefix(title, "objective"):
		return "objective", true
	case strings.HasPrefix(title, "plan"):
		return "plan", true
	case strings.HasPrefix(title, "scratch"):
		return "scratch", true
	default:
		return "", false
	}
}

func parsePlan(lines []string) []PlanItem {
	var items []PlanItem
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if len(line) < len("- [ ] ") || !strings.HasPrefix(line, "- [") || line[4] != ']' || line[5] != ' ' {
			continue
		}
		mark := line[3]
		if mark != ' ' && mark != 'x' && mark != 'X' {
			continue
		}
		items = append(items, PlanItem{
			Checked: mark == 'x' || mark == 'X',
			Text:    strings.TrimSpace(line[6:]),
		})
	}
	return items
}

func stripFrontmatterComment(line string) string {
	for i, r := range line {
		if r == '#' && (i == 0 || line[i-1] == ' ' || line[i-1] == '\t') {
			return line[:i]
		}
	}
	return line
}

func unquote(s string) string {
	if len(s) >= 2 {
		if (s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'') {
			return s[1 : len(s)-1]
		}
	}
	return s
}

func trimBlock(lines []string) string {
	for len(lines) > 0 && strings.TrimSpace(lines[0]) == "" {
		lines = lines[1:]
	}
	for len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) == "" {
		lines = lines[:len(lines)-1]
	}
	return strings.Join(lines, "\n")
}

func writeBlock(b *strings.Builder, s string) {
	s = strings.Trim(s, "\n")
	if s != "" {
		b.WriteString(s)
	}
	b.WriteByte('\n')
}
