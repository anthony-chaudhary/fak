package taskmgr

import (
	"fmt"
	"io"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const SchemaQueue = "fak-task-queue/1"

type IssueLabel struct {
	Name string `json:"name"`
}
type IssueMilestone struct {
	Title string `json:"title"`
}
type QueueIssue struct {
	Number    int             `json:"number"`
	Title     string          `json:"title"`
	State     string          `json:"state"`
	Body      string          `json:"body"`
	Labels    []IssueLabel    `json:"labels"`
	Milestone *IssueMilestone `json:"milestone,omitempty"`
}

type Attempt struct {
	Holder      string `json:"holder"`
	PID         int    `json:"pid,omitempty"`
	Account     string `json:"account,omitempty"`
	Token       string `json:"token,omitempty"`
	HeartbeatAt string `json:"heartbeat_at,omitempty"`
	AcquiredAt  string `json:"acquired_at,omitempty"`
	Lane        string `json:"lane,omitempty"`
}

type QueueLeaf struct {
	Number       int       `json:"number"`
	State        string    `json:"state"`
	DurableState string    `json:"durable_state"`
	Priority     string    `json:"priority"`
	Generation   string    `json:"generation"`
	Lane         string    `json:"lane"`
	Title        string    `json:"title"`
	Outcome      string    `json:"outcome"`
	Requires     []int     `json:"requires"`
	Witness      string    `json:"witness"`
	Parent       int       `json:"parent,omitempty"`
	Attempts     []Attempt `json:"attempts,omitempty"`
}

type Queue struct {
	Schema string      `json:"schema"`
	Leaves []QueueLeaf `json:"leaves"`
}

var issueRefRE = regexp.MustCompile(`#([0-9]+)`)

// BuildQueue folds durable issue contracts and ephemeral attempts without allowing
// attempt telemetry to mutate the durable leaf state.
func BuildQueue(issues []QueueIssue, attempts []Attempt) Queue {
	states := make(map[int]string, len(issues))
	for _, issue := range issues {
		states[issue.Number] = strings.ToUpper(issue.State)
	}
	out := Queue{Schema: SchemaQueue}
	for _, issue := range issues {
		leaf := QueueLeaf{Number: issue.Number, Title: queueOneLine(issue.Title), Outcome: section(issue.Body, "Outcome"), Witness: witnessText(issue.Body)}
		leaf.Priority, leaf.Generation, leaf.Lane = labels(issue.Labels)
		leaf.Requires = refs(section(issue.Body, "Dependencies"))
		leaf.Parent = firstRef(section(issue.Body, "Parent"))
		leaf.DurableState = durableState(issue.State, leaf.Requires, states)
		leaf.State = leaf.DurableState
		for _, attempt := range attempts {
			if attemptIssue(attempt.Holder) == issue.Number {
				leaf.Attempts = append(leaf.Attempts, attempt)
				if leaf.Lane == "" {
					leaf.Lane = attempt.Lane
				}
			}
		}
		if leaf.DurableState == "ready" && len(leaf.Attempts) > 0 {
			leaf.State = "active"
		}
		out.Leaves = append(out.Leaves, leaf)
	}
	sort.Slice(out.Leaves, func(i, j int) bool { return out.Leaves[i].Number < out.Leaves[j].Number })
	return out
}

func RenderQueue(w io.Writer, q Queue, drilldown bool) {
	for _, leaf := range q.Leaves {
		fmt.Fprintf(w, "#%d state=%s priority=%s generation=%s lane=%s title=%q outcome=%q requires=%s witness=%q parent=%s\n",
			leaf.Number, value(leaf.State), value(leaf.Priority), value(leaf.Generation), value(leaf.Lane), value(leaf.Title), value(leaf.Outcome), refList(leaf.Requires), value(leaf.Witness), ref(leaf.Parent))
		if drilldown {
			for _, a := range leaf.Attempts {
				fmt.Fprintf(w, "  attempt holder=%s pid=%d account=%s token=%s heartbeat=%s acquired=%s lane=%s\n", value(a.Holder), a.PID, value(a.Account), value(a.Token), value(a.HeartbeatAt), value(a.AcquiredAt), value(a.Lane))
			}
		}
	}
}

func durableState(state string, requires []int, states map[int]string) string {
	if strings.EqualFold(state, "closed") {
		return "done"
	}
	for _, n := range requires {
		if !strings.EqualFold(states[n], "closed") {
			return "held"
		}
	}
	return "ready"
}
func labels(ls []IssueLabel) (priority, generation, lane string) {
	for _, l := range ls {
		n := strings.TrimSpace(l.Name)
		switch {
		case strings.HasPrefix(n, "priority/"):
			priority = strings.TrimPrefix(n, "priority/")
		case strings.HasPrefix(n, "gen/"):
			generation = strings.TrimPrefix(n, "gen/")
		case strings.HasPrefix(n, "lane/"):
			lane = strings.TrimPrefix(n, "lane/")
		case strings.HasPrefix(n, "lane:"):
			lane = strings.TrimPrefix(n, "lane:")
		}
	}
	return
}
func section(body, name string) string {
	lines := strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n")
	in := false
	var got []string
	for _, line := range lines {
		if strings.HasPrefix(line, "## ") {
			if in {
				break
			}
			in = strings.EqualFold(strings.TrimSpace(strings.TrimPrefix(line, "## ")), name)
			continue
		}
		if in && strings.TrimSpace(line) != "" {
			got = append(got, strings.TrimSpace(strings.TrimLeft(line, "-* ")))
		}
	}
	return queueOneLine(strings.Join(got, " "))
}
func witnessText(body string) string {
	if s := section(body, "Witness"); s != "" {
		return s
	}
	if s := section(body, "Definition of done"); s != "" {
		return "definition-of-done"
	}
	return "unknown"
}
func refs(s string) []int {
	ms := issueRefRE.FindAllStringSubmatch(s, -1)
	out := make([]int, 0, len(ms))
	seen := map[int]bool{}
	for _, m := range ms {
		n, _ := strconv.Atoi(m[1])
		if n > 0 && !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	return out
}
func firstRef(s string) int {
	r := refs(s)
	if len(r) > 0 {
		return r[0]
	}
	return 0
}
func attemptIssue(holder string) int {
	for _, prefix := range []string{"issue-", "codex-", "#"} {
		if i := strings.LastIndex(holder, prefix); i >= 0 {
			holder = "#" + holder[i+len(prefix):]
			break
		}
	}
	m := issueRefRE.FindStringSubmatch(holder)
	if len(m) < 2 {
		return 0
	}
	n, _ := strconv.Atoi(m[1])
	return n
}
func queueOneLine(s string) string { return strings.Join(strings.Fields(s), " ") }
func value(s string) string {
	if strings.TrimSpace(s) == "" {
		return "unknown"
	}
	return queueOneLine(s)
}
func ref(n int) string {
	if n == 0 {
		return "none"
	}
	return fmt.Sprintf("#%d", n)
}
func refList(ns []int) string {
	if len(ns) == 0 {
		return "none"
	}
	xs := make([]string, len(ns))
	for i, n := range ns {
		xs[i] = ref(n)
	}
	return strings.Join(xs, ",")
}
