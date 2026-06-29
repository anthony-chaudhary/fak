package gardenbundle

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Schema is the stable control-pane schema identifier for the bundle envelope.
const Schema = "fak-garden-bundle/1"

// offValues is the off-switch vocabulary, shared with the fleet's other
// default-on guards. Any of these as FAK_GARDEN means "skip the bundle".
var offValues = map[string]bool{
	"0": true, "off": true, "false": true, "no": true,
	"disable": true, "disabled": true,
}

// loopAuditNames is the loop-audit subset used by the opt-in --deep member.
// Restricted to the FAST, LOCAL, deterministic loops -- no network, no external
// tool -- so the deep pass stays safe on any clone.
var loopAuditNames = []string{
	"readme-freshness", // tools/readme_freshness_audit.py -- local tree
	"gofmt-debt-audit", // tools/gofmt_debt_audit.py -- gofmt over .go, local
	"public-leak-scan", // tools/scrub_public_copy.py --audit-tree -- local scan
}

// Member binds a label to the argv that produces its control-pane payload and
// whether a RED verdict from that member GATES the bundle (--check exits
// non-zero). The scorecard ratchet is the only hard gate today; the others are
// advisory panes whose ACTION verdict is a surfaced condition, not a broken
// garden.
type Member struct {
	Key   string
	Label string
	Argv  []string
	Gates bool
	Kind  string // "envelope" or "loop_audit"
	// Exec selects how Argv is run. "" / "python" (the default) runs Argv[0] as a
	// repo python script under the resolved interpreter -- the shape every original
	// member uses. "command" runs Argv[0] as a direct executable with Argv[1:] as
	// its args (e.g. `go run ./cmd/fak ...`), so a Go-native member can join the
	// bundle without a python shim.
	Exec string
}

// Members is the DEFAULT bundle: the two fast, canonical folds that already
// speak the same control-pane envelope -- the scorecard control pane and
// fresh-status. It is deliberately small so a scheduled tick is fast and
// host-safe.
var Members = []Member{
	{
		Key:   "scorecard",
		Label: "scorecard control pane",
		Argv:  []string{"tools/scorecard_control_pane.py", "--check", "--json"},
		Gates: true,
		Kind:  "envelope",
	},
	{
		Key:   "fresh_status",
		Label: "fresh status",
		Argv:  []string{"tools/fresh_status.py", "--json"},
		Gates: false,
		Kind:  "envelope",
	},
	{
		// The closure rung: review the latest guarded session's decision journal
		// and route its worst bucket into a pickable findings-queue row. Re-invokes
		// the running fak binary (Argv[0]=="fak" -> os.Executable in RunMember), NOT
		// `go run`, so it stays green even when a peer's uncommitted edit leaves the
		// tree uncompilable. Non-gating: a routed finding is the pass WORKING, not a
		// broken garden. Passes --no-issues so an unattended garden tick never tries
		// to open a gh issue from a host that may lack gh auth -- issue filing is
		// reserved for the operator-invoked `fak guard-verdict-rsi route` (issues on
		// by default there).
		Key:   "guard_route",
		Label: "guard-session route",
		Argv:  []string{"fak", "guard-verdict-rsi", "route", "--no-issues", "--json"},
		Gates: false,
		Kind:  "envelope",
		Exec:  "command",
	},
	{
		// The learn rung for session observability (#1161): consume the committed
		// scrubbed corpus and surface the value-vs-waste behavior contrast. Missing
		// corpus is advisory ACTION, not a red garden, because the member itself still
		// measured and reported the missing feed.
		Key:   "sessions_learn",
		Label: "sessions learn",
		Argv:  []string{"fak", "sessions", "learn", "--corpus", "experiments/sessionobs/corpus.jsonl", "--json"},
		Gates: false,
		Kind:  "envelope",
		Exec:  "command",
	},
	{
		// Stale-work rung 1: orphaned/unwitnessed dispatched runs. Re-invokes the on-disk
		// fak binary (Exec=command, Argv[0]=fak -> os.Executable), reads the loop ledger
		// TOLERANTLY (a forked seq chain no longer takes recover down), and surfaces the
		// recovery worklist. Non-gating: a found orphan is the pass WORKING, not a broken
		// garden — the operator re-dispatches/re-verifies. --control-pane emits ok/verdict/reason.
		Key:   "orphaned_runs",
		Label: "orphaned runs",
		Argv:  []string{"fak", "loop", "recover", "--control-pane"},
		Gates: false,
		Kind:  "envelope",
		Exec:  "command",
	},
	{
		// Stale-work rung 2: publish freshness. `@latest` rots silently as the trunk moves
		// past the last release tag; this makes the lag a loud, GATING red so an unattended
		// garden tick flags "adopters are on a stale binary". The member already speaks the
		// control-pane envelope (ok/verdict/reason) with no extra flag.
		Key:   "release_staleness",
		Label: "release staleness",
		Argv:  []string{"fak", "release-staleness", "--json"},
		Gates: true,
		Kind:  "envelope",
		Exec:  "command",
	},
	{
		// Stale-work rung 3: expired cross-machine leases under refs/fak/locks/*. READ-ONLY
		// audit — it reaps NOTHING (reaping stays the explicit `fak leaseref reap` so a
		// read-only garden fold never mutates lock state). Non-gating: an expired lease is an
		// advisory ACTION (run reap), not a red. A crashed holder's lapsed lease is surfaced.
		Key:   "stale_leases",
		Label: "stale leases",
		Argv:  []string{"fak", "leaseref", "audit"},
		Gates: false,
		Kind:  "envelope",
		Exec:  "command",
	},
}

// DeepMember is the opt-in deep member (added by --deep). Non-gating advisory.
// It goes through fleet_control_pane's per-loop orchestration, which carries
// enough overhead that it doesn't belong on the default tick.
var DeepMember = Member{
	Key:   "loop_audit",
	Label: "fleet loop-audit",
	Argv: []string{"tools/fleet_control_pane.py", "loop-audit",
		"--names", strings.Join(loopAuditNames, ","), "--json"},
	Gates: false,
	Kind:  "loop_audit",
}

// GardenOff reports whether FAK_GARDEN names an off value (the env-side governor
// brake).
func GardenOff() bool {
	return offValues[strings.ToLower(strings.TrimSpace(os.Getenv("FAK_GARDEN")))]
}

// MemberResult is the bundle's normalization of one member into a uniform row.
// State is one of:
//   - "ok"      -- the member ran and reports nothing to do
//   - "action"  -- the member ran and surfaces a real condition (advisory)
//   - "red"     -- the member's gate tripped (a hard regression)
//   - "errored" -- the member could not run / produced no usable payload
type MemberResult struct {
	Key      string
	Label    string
	Gates    bool
	ExitCode int
	State    string
	OK       bool
	Verdict  string
	Detail   string
	// Counts carries the loop-audit bucket counts; nil for envelope members.
	Counts map[string]int
}

// MarshalJSON emits the same field set and order as the Python member row.
func (r MemberResult) MarshalJSON() ([]byte, error) {
	var b strings.Builder
	b.WriteByte('{')
	writeJSONField(&b, "key", r.Key, true)
	writeJSONField(&b, "label", r.Label, false)
	writeJSONField(&b, "gates", r.Gates, false)
	writeJSONField(&b, "exit_code", r.ExitCode, false)
	writeJSONField(&b, "state", r.State, false)
	writeJSONField(&b, "ok", r.OK, false)
	writeJSONField(&b, "verdict", r.Verdict, false)
	writeJSONField(&b, "detail", r.Detail, false)
	if r.Counts != nil {
		writeJSONField(&b, "counts", r.Counts, false)
	}
	b.WriteByte('}')
	return []byte(b.String()), nil
}

func writeJSONField(b *strings.Builder, name string, value any, first bool) {
	if !first {
		b.WriteByte(',')
	}
	k, _ := json.Marshal(name)
	v, _ := json.Marshal(value)
	b.Write(k)
	b.WriteByte(':')
	b.Write(v)
}

// Interpret folds one member's raw payload into a uniform member-result row.
// payload is nil when the member produced no usable payload.
func Interpret(member Member, payload map[string]any, exitCode int, err string) MemberResult {
	base := MemberResult{
		Key:      member.Key,
		Label:    member.Label,
		Gates:    member.Gates,
		ExitCode: exitCode,
	}
	if err != "" || payload == nil {
		base.State = "errored"
		base.OK = false
		base.Verdict = "ERROR"
		if err != "" {
			base.Detail = err
		} else {
			base.Detail = "no payload"
		}
		return base
	}

	if member.Kind == "loop_audit" {
		counts := asIntMap(payload["counts"])
		broken := counts["broken"]
		action := counts["action"]
		// loop-audit is a NON-GATING advisory member: it ran (it produced a
		// payload), so a broken sub-loop is a condition to surface, NOT the
		// bundle's own inability to measure. A peripheral check that can't run
		// on this host must not be able to gate the whole garden red. Only a
		// member that produces NO payload at all is `errored` (handled above).
		if broken > 0 || action > 0 {
			base.State = "action"
			base.OK = true
			base.Verdict = "ACTION"
			var bits []string
			if broken > 0 {
				bits = append(bits, fmt.Sprintf("%d loop(s) broken", broken))
			}
			if action > 0 {
				bits = append(bits, fmt.Sprintf("%d surfacing a condition", action))
			}
			base.Detail = strings.Join(bits, "; ") + " (advisory; does not gate)"
		} else {
			base.State = "ok"
			base.OK = true
			base.Verdict = "OK"
			base.Detail = fmt.Sprintf("%d loop(s) healthy", counts["healthy"])
		}
		base.Counts = counts
		return base
	}

	// Standard control-pane envelope (scorecard, fresh_status).
	ok := asBool(payload["ok"])
	verdict := asString(payload["verdict"])
	detail := asString(payload["reason"])
	switch {
	case member.Gates && !ok:
		base.State = "red"
	case !ok:
		base.State = "action"
	default:
		base.State = "ok"
	}
	base.OK = ok
	if verdict != "" {
		base.Verdict = verdict
	} else if ok {
		base.Verdict = "OK"
	} else {
		base.Verdict = "ACTION"
	}
	base.Detail = detail
	return base
}

// Payload is one folded garden-bundle control-pane envelope.
type Payload struct {
	OK          bool
	Verdict     string
	Finding     string
	Reason      string
	NextAction  string
	Workspace   string
	Commit      string
	Members     []MemberResult
	MemberCount int
	Gating      []string
	Skipped     bool
	// gate fields, set only for the --check --json envelope
	gateExit    *int
	gateMessage string
}

// MarshalJSON emits the same field set and order as the Python payload.
func (p Payload) MarshalJSON() ([]byte, error) {
	gating := p.Gating
	if gating == nil {
		gating = []string{}
	}
	members := p.Members
	if members == nil {
		members = []MemberResult{}
	}
	var b strings.Builder
	b.WriteByte('{')
	writeJSONField(&b, "schema", Schema, true)
	writeJSONField(&b, "ok", p.OK, false)
	writeJSONField(&b, "verdict", p.Verdict, false)
	writeJSONField(&b, "finding", p.Finding, false)
	writeJSONField(&b, "reason", p.Reason, false)
	writeJSONField(&b, "next_action", p.NextAction, false)
	writeJSONField(&b, "workspace", p.Workspace, false)
	writeJSONField(&b, "commit", p.Commit, false)
	writeJSONField(&b, "members", members, false)
	writeJSONField(&b, "member_count", p.MemberCount, false)
	writeJSONField(&b, "gating", gating, false)
	if p.Skipped {
		writeJSONField(&b, "skipped", true, false)
	}
	if p.gateExit != nil {
		writeJSONField(&b, "gate_exit", *p.gateExit, false)
		writeJSONField(&b, "gate_message", p.gateMessage, false)
	}
	b.WriteByte('}')
	return []byte(b.String()), nil
}

// Fold folds member results into one garden-bundle control-pane payload.
func Fold(results []MemberResult, workspace, commit string) Payload {
	var errored, red, action []MemberResult
	for _, r := range results {
		switch r.State {
		case "errored":
			errored = append(errored, r)
		case "red":
			red = append(red, r)
		case "action":
			action = append(action, r)
		}
	}

	p := Payload{
		Workspace:   workspace,
		Commit:      commit,
		Members:     results,
		MemberCount: len(results),
		Gating:      gatingKeys(results),
	}

	switch {
	case len(errored) > 0:
		p.OK, p.Verdict, p.Finding = false, "ACTION", "garden_member_unmeasured"
		p.Reason = "garden bundle could not measure " + joinLabels(errored) +
			" -- a gardening pass failed to run, so the garden is not proven tended"
		p.NextAction = "repair the failing member(s): " + joinLabelDetails(errored)
	case len(red) > 0:
		p.OK, p.Verdict, p.Finding = false, "ACTION", "garden_gate_red"
		p.Reason = "garden gate RED -- " + joinLabelParens(red)
		p.NextAction = "retire the regression worst-first with the owning scorecard's " +
			"skill, then re-run go run ./cmd/fak garden --check"
	case len(action) > 0:
		p.OK, p.Verdict, p.Finding = true, "OK", "garden_advisory"
		p.Reason = "garden tended; advisory conditions surfaced by " + joinLabels(action) +
			" (these don't gate -- a pass surfacing a condition is the pass working)"
		p.NextAction = "optional: address the advisory condition(s) -- " + joinLabelDetails(action)
	default:
		p.OK, p.Verdict, p.Finding = true, "OK", "garden_clear"
		p.Reason = fmt.Sprintf("garden tended; all %d members clear", len(results))
		p.NextAction = "hold the line; the scheduled garden tick keeps it tended"
	}
	return p
}

// SkippedPayload is the well-formed payload for an FAK_GARDEN=off skip (still
// ok=True).
func SkippedPayload(workspace, commit string) Payload {
	return Payload{
		OK:          true,
		Verdict:     "OK",
		Finding:     "garden_skipped",
		Reason:      "FAK_GARDEN is set off; garden bundle skipped (env-side governor brake)",
		NextAction:  "unset FAK_GARDEN (or set it on) to run the bundle",
		Workspace:   workspace,
		Commit:      commit,
		Members:     []MemberResult{},
		MemberCount: 0,
		Gating:      []string{},
		Skipped:     true,
	}
}

// CheckGate is the CI gate decision over a folded payload (pure: exit code +
// message).
//
//	0  garden tended (clear or advisory-only)
//	1  a gating member regressed, or a member failed to run
func CheckGate(p Payload) (int, string) {
	switch p.Finding {
	case "garden_clear", "garden_advisory", "garden_skipped":
		return 0, "GARDEN OK: " + p.Reason
	default:
		return 1, "GARDEN RED: " + p.Reason
	}
}

// WithGate returns a copy of the payload reconciled to a CheckGate decision, for
// the --check --json envelope.
func (p Payload) WithGate(code int, message string) Payload {
	q := p
	q.OK = code == 0
	if code == 0 {
		q.Verdict = "OK"
	} else {
		q.Verdict = "ACTION"
	}
	c := code
	q.gateExit = &c
	q.gateMessage = message
	return q
}

// Render produces the human snapshot, mirroring the Python render().
func Render(p Payload) string {
	icon := map[string]string{"ok": "+", "action": ".", "red": "x", "errored": "x"}
	lines := []string{
		fmt.Sprintf("garden bundle -- %s (%s)  @%s", p.Verdict, p.Finding, p.Commit),
		"",
	}
	if p.Skipped {
		lines = append(lines, "  (skipped: FAK_GARDEN is off)")
	}
	for _, m := range p.Members {
		gate := ""
		if m.Gates {
			gate = " [gates]"
		}
		ic := icon[m.State]
		if ic == "" {
			ic = "?"
		}
		lines = append(lines, fmt.Sprintf("  %s %-24s %-8s%s  %s", ic, m.Label, m.State, gate, m.Detail))
	}
	lines = append(lines, "", "  -> "+p.NextAction)
	return strings.Join(lines, "\n")
}

// --- live runner -----------------------------------------------------------

// RunMember runs one member and parses its JSON stdout. A default ("" / "python")
// member runs Argv[0] as a repo python script under the interpreter; a "command"
// member runs Argv[0] as a direct executable. A "command" member whose Argv[0] is
// the bare token "fak" is rewritten to the CURRENTLY-RUNNING executable
// (os.Executable), so a Go-native member re-invokes the same built binary -- never
// `go run`, which would recompile the whole tree and so error whenever a peer's
// uncommitted edit leaves cmd/fak uncompilable on the shared trunk. It returns the
// parsed payload (nil on any failure), the process exit code, and an error string
// (empty on success).
func RunMember(root string, member Member, python string, timeout time.Duration) (map[string]any, int, string) {
	argv := member.Argv
	var cmd *exec.Cmd
	if member.Exec == "command" {
		bin := argv[0]
		if bin == "fak" {
			if self, err := os.Executable(); err == nil && self != "" {
				bin = self
			}
		}
		cmd = exec.Command(bin, argv[1:]...)
	} else {
		script := filepath.Join(root, argv[0])
		if _, err := os.Stat(script); err != nil {
			return nil, -1, "missing member script: " + argv[0]
		}
		args := append([]string{script}, argv[1:]...)
		cmd = exec.Command(python, args...)
	}
	cmd.Dir = root
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return nil, -1, err.Error()
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case <-time.After(timeout):
		_ = cmd.Process.Kill()
		<-done
		return nil, -1, fmt.Sprintf("timed out after %ds", int(timeout.Seconds()))
	case err := <-done:
		code := cmd.ProcessState.ExitCode()
		var payload map[string]any
		if jerr := json.Unmarshal([]byte(stdout.String()), &payload); jerr == nil && payload != nil {
			return payload, code, ""
		}
		_ = err
		tail := lastLine(stderr.String())
		if tail == "" {
			tail = lastLine(stdout.String())
		}
		if len(tail) > 160 {
			tail = tail[:160]
		}
		return nil, code, fmt.Sprintf("non-JSON output (exit %d): %s", code, tail)
	}
}

// Collect runs every member (plus the deep member when deep is set) and folds
// each into a MemberResult.
func Collect(root, python string, timeout time.Duration, deep bool) []MemberResult {
	if python == "" {
		python = defaultPython()
	}
	members := append([]Member{}, Members...)
	if deep {
		members = append(members, DeepMember)
	}
	results := make([]MemberResult, 0, len(members))
	for _, m := range members {
		payload, code, err := RunMember(root, m, python, timeout)
		results = append(results, Interpret(m, payload, code, err))
	}
	return results
}

// HeadCommit returns the short HEAD commit of root, or "unknown".
func HeadCommit(root string) string {
	cmd := exec.Command("git", "rev-parse", "--short", "HEAD")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return "unknown"
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return "unknown"
	}
	return s
}

// --- helpers ---------------------------------------------------------------

func defaultPython() string {
	if p := os.Getenv("FAK_PYTHON"); p != "" {
		return p
	}
	return "python3"
}

// DefaultPython is the exported form of defaultPython, so sibling packages
// (e.g. cadencereport) can share this one resolver instead of copying it.
func DefaultPython() string { return defaultPython() }

func gatingKeys(results []MemberResult) []string {
	out := []string{}
	for _, r := range results {
		if r.Gates {
			out = append(out, r.Key)
		}
	}
	return out
}

func joinLabels(rows []MemberResult) string {
	parts := make([]string, len(rows))
	for i, r := range rows {
		parts[i] = r.Label
	}
	return strings.Join(parts, ", ")
}

func joinLabelDetails(rows []MemberResult) string {
	parts := make([]string, len(rows))
	for i, r := range rows {
		parts[i] = r.Label + ": " + r.Detail
	}
	return strings.Join(parts, "; ")
}

func joinLabelParens(rows []MemberResult) string {
	parts := make([]string, len(rows))
	for i, r := range rows {
		parts[i] = fmt.Sprintf("%s (%s)", r.Label, r.Detail)
	}
	return strings.Join(parts, ", ")
}

func asBool(v any) bool {
	b, ok := v.(bool)
	return ok && b
}

func asString(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

// asIntMap normalizes a decoded "counts" object into an int map, tolerating the
// JSON-number (float64) form and string forms.
func asIntMap(v any) map[string]int {
	out := map[string]int{}
	m, ok := v.(map[string]any)
	if !ok {
		return out
	}
	for k, raw := range m {
		out[k] = asInt(raw)
	}
	return out
}

func asInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	case string:
		i, _ := strconv.Atoi(strings.TrimSpace(n))
		return i
	default:
		return 0
	}
}

func lastLine(s string) string {
	s = strings.TrimRight(s, "\n")
	if s == "" {
		return ""
	}
	if i := strings.LastIndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[i+1:])
	}
	return strings.TrimSpace(s)
}
