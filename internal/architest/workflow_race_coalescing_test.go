package architest

// #12482 — ordinary-main race coverage must coalesce onto the newest head while
// per-commit fast/build evidence and a trusted pinned release witness survive.
//
// This fixture DERIVES the concurrency group keys from the workflow text and
// replays GitHub's documented concurrency semantics over them, rather than
// restating the expected keys. A future edit to a group expression therefore
// re-evaluates here instead of silently passing.
//
// GitHub semantics modelled (docs: "Using concurrency"):
//   - at most one run is in progress per group;
//   - a new arrival with cancel-in-progress cancels the running run;
//   - otherwise it becomes the single pending run, and any previously pending
//     run in that group is cancelled.
//
// The expression evaluator implements only the operator subset these workflows
// use. && and || return operand VALUES (JS-style), which is load-bearing:
// `github.head_ref || github.sha` must yield the SHA when head_ref is empty.

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// --- workflow text extraction --------------------------------------------

type concurrencyContract struct {
	group  string
	cancel string
}

// jobBlock returns the raw lines of a named job, from its header to the next
// job header (a line indented exactly two spaces).
func jobBlock(text, job string) (string, error) {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	header := "  " + job + ":"
	start := -1
	for i, ln := range lines {
		if ln == header {
			start = i
			break
		}
	}
	if start < 0 {
		return "", fmt.Errorf("job %q not found", job)
	}
	end := len(lines)
	for i := start + 1; i < len(lines); i++ {
		ln := lines[i]
		if len(ln) > 2 && ln[0] == ' ' && ln[1] == ' ' && ln[2] != ' ' && ln[2] != '#' {
			end = i
			break
		}
	}
	return strings.Join(lines[start:end], "\n"), nil
}

// extractConcurrency pulls the `concurrency:` block for a top-level workflow
// (job == "") or for a named job. It is deliberately a line scanner, not a YAML
// parser: the contract under test is the literal text of the group and
// cancel-in-progress expressions, and a YAML round-trip would normalize it.
func extractConcurrency(text, job string) (concurrencyContract, error) {
	scope := text
	keyIndent := "  "
	blockIndent := "concurrency:"
	if job != "" {
		b, err := jobBlock(text, job)
		if err != nil {
			return concurrencyContract{}, err
		}
		scope = b
		keyIndent = "      "
		blockIndent = "    concurrency:"
	}
	lines := strings.Split(scope, "\n")
	ci := -1
	for i, ln := range lines {
		if ln == blockIndent {
			ci = i
			break
		}
	}
	if ci < 0 {
		return concurrencyContract{}, fmt.Errorf("no concurrency block (job=%q)", job)
	}
	var c concurrencyContract
	for i := ci + 1; i < len(lines); i++ {
		ln := lines[i]
		if strings.TrimSpace(ln) == "" {
			continue
		}
		if !strings.HasPrefix(ln, keyIndent) {
			break
		}
		body := strings.TrimSpace(strings.TrimPrefix(ln, keyIndent))
		switch {
		case strings.HasPrefix(body, "group:"):
			c.group = strings.TrimSpace(strings.TrimPrefix(body, "group:"))
		case strings.HasPrefix(body, "cancel-in-progress:"):
			c.cancel = strings.TrimSpace(strings.TrimPrefix(body, "cancel-in-progress:"))
		}
		if c.group != "" && c.cancel != "" {
			break
		}
	}
	if c.group == "" {
		return concurrencyContract{}, fmt.Errorf("concurrency block has no group (job=%q)", job)
	}
	if c.cancel == "" {
		return concurrencyContract{}, fmt.Errorf("concurrency block has no cancel-in-progress (job=%q)", job)
	}
	return c, nil
}

// --- minimal GitHub Actions expression evaluator --------------------------

type ghTok struct {
	kind string // ident | str | num | op | punct
	text string
}

func isIdentStart(c byte) bool {
	return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func isIdentPart(c byte) bool {
	return isIdentStart(c) || (c >= '0' && c <= '9') || c == '-'
}

func ghTokenize(s string) ([]ghTok, error) {
	var toks []ghTok
	i := 0
	for i < len(s) {
		c := s[i]
		switch {
		case c == ' ' || c == '\t' || c == '\n' || c == '\r':
			i++
		case c == '\'':
			j := i + 1
			var b strings.Builder
			closed := false
			for j < len(s) {
				if s[j] == '\'' {
					if j+1 < len(s) && s[j+1] == '\'' { // '' escapes a quote
						b.WriteByte('\'')
						j += 2
						continue
					}
					closed = true
					break
				}
				b.WriteByte(s[j])
				j++
			}
			if !closed {
				return nil, fmt.Errorf("unterminated string literal in %q", s)
			}
			toks = append(toks, ghTok{"str", b.String()})
			i = j + 1
		case c >= '0' && c <= '9':
			j := i
			for j < len(s) && s[j] >= '0' && s[j] <= '9' {
				j++
			}
			toks = append(toks, ghTok{"num", s[i:j]})
			i = j
		case isIdentStart(c):
			j := i
			for j < len(s) && (isIdentPart(s[j]) || s[j] == '.') {
				j++
			}
			toks = append(toks, ghTok{"ident", s[i:j]})
			i = j
		default:
			if i+1 < len(s) {
				switch s[i : i+2] {
				case "==", "!=", "&&", "||":
					toks = append(toks, ghTok{"op", s[i : i+2]})
					i += 2
					continue
				}
			}
			switch c {
			case '!', '(', ')', ',':
				toks = append(toks, ghTok{"punct", string(c)})
				i++
			default:
				return nil, fmt.Errorf("unexpected character %q in expression %q", string(c), s)
			}
		}
	}
	return toks, nil
}

type ghParser struct {
	toks []ghTok
	pos  int
}

func (p *ghParser) peek() (ghTok, bool) {
	if p.pos >= len(p.toks) {
		return ghTok{}, false
	}
	return p.toks[p.pos], true
}

func ghEval(expr string, ctx map[string]any) (any, error) {
	toks, err := ghTokenize(expr)
	if err != nil {
		return nil, err
	}
	p := &ghParser{toks: toks}
	v, err := p.parseOr(ctx)
	if err != nil {
		return nil, err
	}
	if p.pos != len(p.toks) {
		return nil, fmt.Errorf("trailing tokens in expression %q", expr)
	}
	return v, nil
}

func (p *ghParser) parseOr(ctx map[string]any) (any, error) {
	left, err := p.parseAnd(ctx)
	if err != nil {
		return nil, err
	}
	for {
		t, ok := p.peek()
		if !ok || t.text != "||" {
			return left, nil
		}
		p.pos++
		right, err := p.parseAnd(ctx)
		if err != nil {
			return nil, err
		}
		if !ghTruthy(left) {
			left = right
		}
	}
}

func (p *ghParser) parseAnd(ctx map[string]any) (any, error) {
	left, err := p.parseEquality(ctx)
	if err != nil {
		return nil, err
	}
	for {
		t, ok := p.peek()
		if !ok || t.text != "&&" {
			return left, nil
		}
		p.pos++
		right, err := p.parseEquality(ctx)
		if err != nil {
			return nil, err
		}
		if ghTruthy(left) {
			left = right
		}
	}
}

func (p *ghParser) parseEquality(ctx map[string]any) (any, error) {
	left, err := p.parseUnary(ctx)
	if err != nil {
		return nil, err
	}
	for {
		t, ok := p.peek()
		if !ok || (t.text != "==" && t.text != "!=") {
			return left, nil
		}
		p.pos++
		right, err := p.parseUnary(ctx)
		if err != nil {
			return nil, err
		}
		eq := ghLooseEqual(left, right)
		if t.text == "!=" {
			eq = !eq
		}
		left = eq
	}
}

func (p *ghParser) parseUnary(ctx map[string]any) (any, error) {
	if t, ok := p.peek(); ok && t.text == "!" {
		p.pos++
		v, err := p.parseUnary(ctx)
		if err != nil {
			return nil, err
		}
		return !ghTruthy(v), nil
	}
	return p.parsePrimary(ctx)
}

func (p *ghParser) parsePrimary(ctx map[string]any) (any, error) {
	t, ok := p.peek()
	if !ok {
		return nil, fmt.Errorf("unexpected end of expression")
	}
	switch t.kind {
	case "str":
		p.pos++
		return t.text, nil
	case "num":
		p.pos++
		f, err := strconv.ParseFloat(t.text, 64)
		if err != nil {
			return nil, err
		}
		return f, nil
	case "ident":
		p.pos++
		if nxt, ok := p.peek(); ok && nxt.text == "(" {
			return p.parseCall(t.text, ctx)
		}
		return ghLookup(ctx, t.text), nil
	case "punct":
		if t.text == "(" {
			p.pos++
			v, err := p.parseOr(ctx)
			if err != nil {
				return nil, err
			}
			nxt, ok := p.peek()
			if !ok || nxt.text != ")" {
				return nil, fmt.Errorf("missing ) in expression")
			}
			p.pos++
			return v, nil
		}
	}
	return nil, fmt.Errorf("unexpected token %q", t.text)
}

func (p *ghParser) parseCall(name string, ctx map[string]any) (any, error) {
	p.pos++ // consume '('
	var args []any
	if nxt, ok := p.peek(); ok && nxt.text == ")" {
		p.pos++
	} else {
		for {
			v, err := p.parseOr(ctx)
			if err != nil {
				return nil, err
			}
			args = append(args, v)
			nxt, ok := p.peek()
			if !ok {
				return nil, fmt.Errorf("unterminated call to %s", name)
			}
			if nxt.text == "," {
				p.pos++
				continue
			}
			if nxt.text == ")" {
				p.pos++
				break
			}
			return nil, fmt.Errorf("unexpected token %q in call to %s", nxt.text, name)
		}
	}
	if name != "format" {
		return nil, fmt.Errorf("unsupported function %q", name)
	}
	if len(args) == 0 {
		return nil, fmt.Errorf("format() requires a template")
	}
	out := ghString(args[0])
	for i := 1; i < len(args); i++ {
		out = strings.ReplaceAll(out, fmt.Sprintf("{%d}", i-1), ghString(args[i]))
	}
	return out, nil
}

// ghLookup walks a dotted path into the nested context. A missing member yields
// null, matching GitHub's behaviour for absent properties.
func ghLookup(ctx map[string]any, path string) any {
	var cur any = ctx
	for _, part := range strings.Split(path, ".") {
		m, ok := cur.(map[string]any)
		if !ok {
			return nil
		}
		cur, ok = m[part]
		if !ok {
			return nil
		}
	}
	return cur
}

func ghTruthy(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case string:
		return x != ""
	case float64:
		return x != 0
	}
	return true
}

func ghString(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	case bool:
		if x {
			return "true"
		}
		return "false"
	case float64:
		if x == math.Trunc(x) {
			return strconv.FormatInt(int64(x), 10)
		}
		return strconv.FormatFloat(x, 'f', -1, 64)
	}
	return fmt.Sprintf("%v", v)
}

// ghLooseEqual mirrors GitHub's loose equality: null equals only null, numbers
// compare numerically, and everything else compares by canonical string form.
func ghLooseEqual(a, b any) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	af, aok := a.(float64)
	bf, bok := b.(float64)
	if aok && bok {
		return af == bf
	}
	return ghString(a) == ghString(b)
}

// ghResolveGroup renders a `group:` template: literal text with embedded
// ${{ ... }} segments evaluated against ctx.
func ghResolveGroup(tmpl string, ctx map[string]any) (string, error) {
	var b strings.Builder
	rest := tmpl
	for {
		i := strings.Index(rest, "${{")
		if i < 0 {
			b.WriteString(rest)
			return b.String(), nil
		}
		b.WriteString(rest[:i])
		j := strings.Index(rest[i:], "}}")
		if j < 0 {
			return "", fmt.Errorf("unterminated ${{ in group template %q", tmpl)
		}
		inner := rest[i+3 : i+j]
		v, err := ghEval(inner, ctx)
		if err != nil {
			return "", fmt.Errorf("evaluate %q: %w", inner, err)
		}
		b.WriteString(ghString(v))
		rest = rest[i+j+2:]
	}
}

// ghResolveBool resolves a cancel-in-progress value: a literal true/false or a
// ${{ ... }} expression.
func ghResolveBool(tmpl string, ctx map[string]any) (bool, error) {
	t := strings.TrimSpace(tmpl)
	if strings.HasPrefix(t, "${{") {
		inner := strings.TrimSuffix(strings.TrimPrefix(t, "${{"), "}}")
		v, err := ghEval(inner, ctx)
		if err != nil {
			return false, err
		}
		return ghTruthy(v), nil
	}
	return t == "true", nil
}

// --- concurrency simulator ------------------------------------------------

type ghRun struct {
	name   string
	status string // running | pending | cancelled | completed
}

type ghGroup struct {
	running *ghRun
	pending *ghRun
}

type ghConcurrency struct {
	groups map[string]*ghGroup
}

func newGHConcurrency() *ghConcurrency {
	return &ghConcurrency{groups: map[string]*ghGroup{}}
}

// enqueue applies GitHub's documented concurrency semantics for one group.
func (c *ghConcurrency) enqueue(group string, cancel bool, r *ghRun) {
	g := c.groups[group]
	if g == nil {
		g = &ghGroup{}
		c.groups[group] = g
	}
	switch {
	case g.running == nil:
		g.running = r
		r.status = "running"
	case cancel:
		g.running.status = "cancelled"
		if g.pending != nil {
			g.pending.status = "cancelled"
			g.pending = nil
		}
		g.running = r
		r.status = "running"
	default:
		if g.pending != nil {
			g.pending.status = "cancelled"
		}
		g.pending = r
		r.status = "pending"
	}
}

// drain completes the running run and promotes the pending one, repeating until
// the group is quiescent. It returns the runs in the order they reached a
// terminal state.
func (c *ghConcurrency) drain(group string) []*ghRun {
	var done []*ghRun
	g := c.groups[group]
	for g != nil && g.running != nil {
		g.running.status = "completed"
		done = append(done, g.running)
		g.running = g.pending
		g.pending = nil
		if g.running != nil {
			g.running.status = "running"
		}
	}
	return done
}

// --- event context --------------------------------------------------------

type ghEvent struct {
	workflow  string
	eventName string
	ref       string
	sha       string
	headRef   string
	runID     string
	prNumber  any
	intent    any
	inputSHA  any
	shard     string
}

func (e ghEvent) ctx() map[string]any {
	gh := map[string]any{
		"workflow":   e.workflow,
		"event_name": e.eventName,
		"ref":        e.ref,
		"sha":        e.sha,
		"head_ref":   e.headRef,
		"run_id":     e.runID,
	}
	if e.prNumber != nil {
		gh["event"] = map[string]any{
			"pull_request": map[string]any{"number": e.prNumber},
		}
	}
	ctx := map[string]any{"github": gh}
	if e.intent != nil || e.inputSHA != nil {
		inputs := map[string]any{}
		if e.intent != nil {
			inputs["intent"] = e.intent
		}
		if e.inputSHA != nil {
			inputs["sha"] = e.inputSHA
		}
		ctx["inputs"] = inputs
	}
	if e.shard != "" {
		ctx["matrix"] = map[string]any{"shard": e.shard}
	}
	return ctx
}

func resolveContract(t *testing.T, c concurrencyContract, ev ghEvent) (string, bool) {
	t.Helper()
	ctx := ev.ctx()
	group, err := ghResolveGroup(c.group, ctx)
	if err != nil {
		t.Fatalf("resolve group %q: %v", c.group, err)
	}
	cancel, err := ghResolveBool(c.cancel, ctx)
	if err != nil {
		t.Fatalf("resolve cancel-in-progress %q: %v", c.cancel, err)
	}
	return group, cancel
}

type enqueued struct {
	group string
	run   *ghRun
}

// --- the fixture ----------------------------------------------------------

func TestRaceConcurrencyCoalescesOrdinaryMainAndIsolatesPRs(t *testing.T) {
	root := filepath.Dir(internalDir(t))
	ciPath := filepath.Join(root, ".github", "workflows", "ci.yml")
	fastPath := filepath.Join(root, ".github", "workflows", "ci-fast.yml")

	ciRaw, err := os.ReadFile(ciPath)
	if err != nil {
		t.Fatalf("read %s: %v", ciPath, err)
	}
	fastRaw, err := os.ReadFile(fastPath)
	if err != nil {
		t.Fatalf("read %s: %v", fastPath, err)
	}
	ciText := strings.ReplaceAll(string(ciRaw), "\r\n", "\n")
	fastText := strings.ReplaceAll(string(fastRaw), "\r\n", "\n")

	topLevel, err := extractConcurrency(ciText, "")
	if err != nil {
		t.Fatalf("ci.yml top-level concurrency: %v", err)
	}
	raceJob, err := extractConcurrency(ciText, "race-detector")
	if err != nil {
		t.Fatalf("ci.yml race-detector concurrency: %v", err)
	}
	fast, err := extractConcurrency(fastText, "")
	if err != nil {
		t.Fatalf("ci-fast.yml concurrency: %v", err)
	}

	// The race job must be per-shard and must not cancel in progress: that pair
	// is what yields "one running plus one pending leg per shard per ref".
	if !strings.Contains(raceJob.group, "matrix.shard") {
		t.Fatalf("race-detector group must be keyed per shard; got %q", raceJob.group)
	}
	if strings.TrimSpace(raceJob.cancel) != "false" {
		t.Fatalf("race-detector cancel-in-progress must be the literal false; got %q", raceJob.cancel)
	}

	t.Run("per-commit fast/build evidence is preserved", func(t *testing.T) {
		sim := newGHConcurrency()
		var enq []enqueued
		for i := 0; i < 10; i++ {
			ev := ghEvent{
				workflow: "CI", eventName: "push", ref: "refs/heads/main",
				sha: fmt.Sprintf("sha%02d", i), runID: fmt.Sprintf("%d", i),
			}
			group, cancel := resolveContract(t, topLevel, ev)
			r := &ghRun{name: ev.sha}
			sim.enqueue(group, cancel, r)
			enq = append(enq, enqueued{group, r})
		}
		seen := map[string]int{}
		for _, e := range enq {
			seen[e.group]++
		}
		if len(seen) != 10 {
			t.Fatalf("per-commit fast/build evidence must occupy 10 distinct groups (one per SHA); got %d: %v", len(seen), seen)
		}
		for _, e := range enq {
			if e.run.status != "running" {
				t.Fatalf("run %s in group %s must remain running; got %s", e.run.name, e.group, e.run.status)
			}
		}
	})

	t.Run("ordinary-main race coalesces to one running plus one pending", func(t *testing.T) {
		sim := newGHConcurrency()
		var enq []enqueued
		for i := 0; i < 10; i++ {
			ev := ghEvent{
				workflow: "CI", eventName: "push", ref: "refs/heads/main",
				sha: fmt.Sprintf("sha%02d", i), runID: fmt.Sprintf("%d", i), shard: "cli-a",
			}
			group, cancel := resolveContract(t, raceJob, ev)
			r := &ghRun{name: ev.sha}
			sim.enqueue(group, cancel, r)
			enq = append(enq, enqueued{group, r})
		}
		groups := map[string]bool{}
		for _, e := range enq {
			groups[e.group] = true
		}
		if len(groups) != 1 {
			t.Fatalf("ordinary-main race runs must share one group per shard; got %d groups: %v", len(groups), groups)
		}
		var running, pending, cancelled int
		for _, e := range enq {
			switch e.run.status {
			case "running":
				running++
			case "pending":
				pending++
			case "cancelled":
				cancelled++
			}
		}
		if running != 1 || pending != 1 {
			t.Fatalf("ordinary-main race must admit exactly one running plus one pending run; got running=%d pending=%d cancelled=%d", running, pending, cancelled)
		}
		if cancelled != 8 {
			t.Fatalf("the eight superseded ordinary-main race runs must be cancelled; got %d", cancelled)
		}
		done := sim.drain(enq[0].group)
		if len(done) != 2 {
			t.Fatalf("draining the group must complete exactly the running and pending runs; got %d", len(done))
		}
		newest := enq[len(enq)-1].run
		if newest.status != "completed" {
			t.Fatalf("the newest main head's race run must reach a terminal conclusion; got %s", newest.status)
		}
	})

	t.Run("PR race jobs cannot cancel another PR", func(t *testing.T) {
		sim := newGHConcurrency()
		pr1 := ghEvent{
			workflow: "CI", eventName: "pull_request", ref: "refs/pull/1/merge",
			sha: "pr1sha", runID: "1", prNumber: float64(1), shard: "cli-a",
		}
		pr2 := ghEvent{
			workflow: "CI", eventName: "pull_request", ref: "refs/pull/2/merge",
			sha: "pr2sha", runID: "2", prNumber: float64(2), shard: "cli-a",
		}
		g1, c1 := resolveContract(t, raceJob, pr1)
		g2, c2 := resolveContract(t, raceJob, pr2)
		if g1 == g2 {
			t.Fatalf("distinct PRs must occupy distinct race groups; both resolved to %q", g1)
		}
		r1 := &ghRun{name: "pr1"}
		r2 := &ghRun{name: "pr2"}
		sim.enqueue(g1, c1, r1)
		sim.enqueue(g2, c2, r2)
		if r1.status != "running" {
			t.Fatalf("PR #1's race run must be unaffected by PR #2; got %s", r1.status)
		}
		if r2.status != "running" {
			t.Fatalf("PR #2's race run must start independently; got %s", r2.status)
		}
		r1b := &ghRun{name: "pr1b"}
		sim.enqueue(g1, c1, r1b)
		if r1b.status != "pending" {
			t.Fatalf("a second push to PR #1 must queue behind PR #1's own run; got %s", r1b.status)
		}
		if r2.status != "running" {
			t.Fatalf("PR #2 must remain running when PR #1 queues; got %s", r2.status)
		}
	})

	t.Run("a pinned release-intent run survives later main pushes", func(t *testing.T) {
		sim := newGHConcurrency()
		rel := ghEvent{
			workflow: "ci-fast", eventName: "workflow_dispatch", ref: "refs/heads/main",
			sha: "mainsha", runID: "900", intent: "release", inputSHA: "candidateSHA",
		}
		rg, rc := resolveContract(t, fast, rel)
		if rc {
			t.Fatalf("a pinned release-intent run must not be cancellable by later pushes; cancel-in-progress resolved true")
		}
		rr := &ghRun{name: "release"}
		sim.enqueue(rg, rc, rr)
		if rr.status != "running" {
			t.Fatalf("the pinned release-intent run must start; got %s", rr.status)
		}
		for i := 0; i < 10; i++ {
			ev := ghEvent{
				workflow: "ci-fast", eventName: "push", ref: "refs/heads/main",
				sha: fmt.Sprintf("sha%02d", i), runID: fmt.Sprintf("%d", i),
			}
			g, c := resolveContract(t, fast, ev)
			if g == rg {
				t.Fatalf("an ordinary main push must not share the pinned release group %q", rg)
			}
			sim.enqueue(g, c, &ghRun{name: ev.sha})
		}
		if rr.status != "running" {
			t.Fatalf("the pinned release-intent run must survive ten later main pushes; got %s", rr.status)
		}
		// The pinned group is keyed by the candidate SHA, so a different
		// candidate is a different group and cannot coalesce onto it.
		rel2 := rel
		rel2.inputSHA = "otherSHA"
		rel2.runID = "901"
		rg2, _ := resolveContract(t, fast, rel2)
		if rg2 == rg {
			t.Fatalf("release-intent runs for different candidate SHAs must not coalesce; both resolved to %q", rg)
		}
	})

	t.Run("the race job cannot carry release-intent evidence", func(t *testing.T) {
		block, err := jobBlock(ciText, "race-detector")
		if err != nil {
			t.Fatalf("race-detector job block: %v", err)
		}
		// The race job is guarded to push/PR events, so a release-intent
		// workflow_dispatch never enters its concurrency domain at all: there is
		// no release-intent race run that a later push could supersede.
		if !strings.Contains(block, "if: github.event_name != 'workflow_dispatch'") {
			t.Fatalf("race-detector must be guarded to non-dispatch events so release-intent evidence cannot be superseded")
		}
	})
}
