package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/codetools"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// codetools_loop_test.go — the OWNED-LOOP witness for #6703.
//
// The other tests in internal/codetools drive the toolset's own methods. This one drives
// the real agent loop: a scripted planner proposes Read / Grep / Glob (and an escaping
// Read) exactly as a model would, RunArm carries each through k.Syscall, the codetools
// rung adjudicates and pins the engine, and the kernel dispatches to the registered
// engine. That distinction is the whole acceptance criterion — a test that called
// ts.read() directly would prove the engine works, not that the LOOP reaches it.
//
// What makes this a proof of MEDIATION rather than of execution:
//
//   - the kernel's own EngineCalls counter (k.Counters(), folded into ArmMetrics by
//     finalizeFak) is what reports the dispatches, so the number comes from the kernel
//     rather than from the harness observing itself;
//   - every allowed call's trace row carries By="codetools", naming the rung that decided
//     it — a call executed outside the kernel would have no rung and no row;
//   - the escaping Read is a kernel DENY, counted in Denies, and the loop hands the model
//     a typed deny receipt instead of file bytes.

var lastRecordingMessages []Message

type recordingCodePlanner struct {
	turns    []*Completion
	n        int
	messages []Message
}

func (p *recordingCodePlanner) Complete(_ context.Context, messages []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.messages = append([]Message(nil), messages...)
	lastRecordingMessages = p.messages
	c := bindLatestCodeToolVersion(p.turns[p.n], messages)
	if p.n < len(p.turns)-1 {
		p.n++
	}
	return c, nil
}
func (p *recordingCodePlanner) Model() string { return "recording-code" }

func resultsFromMessages(messages []Message) (read, grep, glob string) {
	for _, m := range messages {
		if m.Role != "tool" {
			continue
		}
		switch m.Name {
		case codetools.ToolRead:
			if read == "" {
				read = m.Content
			}
		case codetools.ToolGrep:
			grep = m.Content
		case codetools.ToolGlob:
			glob = m.Content
		}
	}
	return
}

func lastResultFromMessages(messages []Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "tool" {
			return messages[i].Content
		}
	}
	return ""
}

// codeToolScript is one scripted model turn: the tool the planner emits and its raw args.
type codeToolScript struct {
	tool string
	args string
}

// runCodeToolLoop arms the coding tools over root, drives RunArm through the scripted
// turns, and returns the arm metrics plus the trace rows.
func runCodeToolLoop(t *testing.T, root string, script []codeToolScript) (ArmMetrics, []traceEvent) {
	t.Helper()
	catalog, err := ArmCodeTools(root)
	if err != nil {
		t.Fatalf("ArmCodeTools: %v", err)
	}
	t.Cleanup(DisarmCodeTools)
	if len(catalog) != len(codetools.Catalog())+1 {
		t.Fatalf("armed catalog has %d tools, want %d", len(catalog), len(codetools.Catalog())+1)
	}

	turns := make([]*Completion, 0, len(script)+1)
	for _, s := range script {
		turns = append(turns, toolCallTurn(s.tool, s.args))
	}
	turns = append(turns, &Completion{Message: Message{Content: "done"}})

	var log []traceEvent
	planner := &recordingCodePlanner{turns: turns}
	m, err := RunArm(context.Background(), planner, "inspect the workspace",
		true, len(turns)+1, &log, WithToolCatalog(catalog))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	return m, log
}

// seedCodeToolFixture builds the scratch workspace the loop reads: a small Go-shaped tree
// plus a secret OUTSIDE it that the escape turn will try to reach.
func seedCodeToolFixture(t *testing.T) (root, outside string) {
	t.Helper()
	root = t.TempDir()
	outside = t.TempDir()
	write := func(path, body string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatalf("write %s: %v", path, err)
		}
	}
	write(filepath.Join(root, "main.go"), "package main\n\nfunc main() {\n\tgreet()\n}\n")
	write(filepath.Join(root, "greet", "greet.go"), "package greet\n\nfunc Greet() string { return \"hi\" }\n")
	write(filepath.Join(root, "README.md"), "# fixture\n")
	write(filepath.Join(outside, "secret.txt"), "CLASSIFIED-OUT-OF-TREE")
	return root, outside
}

// TestOwnedLoopDispatchesCodeToolsThroughKernelEngines is the captured #6703 witness: a
// scratch-fixture coding session driven entirely by the owned loop, proving Read/Grep/Glob
// reach registered kernel engines and that an escape is refused before one runs.
func TestOwnedLoopDispatchesCodeToolsThroughKernelEngines(t *testing.T) {
	root, outside := seedCodeToolFixture(t)
	escape := filepath.Join(outside, "secret.txt")

	m, log := runCodeToolLoop(t, root, []codeToolScript{
		{codetools.ToolRead, `{"file_path":"main.go"}`},
		{codetools.ToolGrep, `{"pattern":"func Greet","glob":"*.go"}`},
		{codetools.ToolGlob, `{"pattern":"**/*.go"}`},
		{codetools.ToolRead, `{"file_path":` + mustJSON(t, escape) + `}`},
	})

	rows := codeToolRows(log)
	if len(rows) != 4 {
		t.Fatalf("loop recorded %d coding-tool rows, want 4: %+v", len(rows), rows)
	}

	// 1. Every call was decided by the codetools rung — the loop did not execute a single
	//    one outside the kernel.
	for _, r := range rows {
		if r.By != codetools.RungName {
			t.Fatalf("%s row decided By=%q, want %q (call did not cross the codetools rung)",
				r.Tool, r.By, codetools.RungName)
		}
	}

	// 2. The three admitted calls dispatched to registered engines, counted by the KERNEL.
	if m.EngineCalls < 3 {
		t.Fatalf("kernel counted %d engine calls, want >= 3 (Read/Grep/Glob dispatched)", m.EngineCalls)
	}

	// 3. The escaping Read was DENIED — counted by the kernel, never dispatched.
	if m.Denies != 1 {
		t.Fatalf("kernel counted %d denies, want exactly 1 (the out-of-tree Read)", m.Denies)
	}
	deny := rows[3]
	if deny.Verdict != "DENY" {
		t.Fatalf("out-of-tree Read verdict = %q, want DENY", deny.Verdict)
	}

	// 4. The results the model saw are the engines' real output, and the denied one
	//    carries no file bytes.
	read, grep, glob := resultsFromMessages(lastRecordingMessages)
	if !strings.Contains(read, "func main()") {
		t.Fatalf("Read result did not carry the fixture's content: %s", read)
	}
	if !strings.Contains(grep, "greet/greet.go") {
		t.Fatalf("Grep result did not name the matching file: %s", grep)
	}
	if !strings.Contains(glob, "main.go") || !strings.Contains(glob, "greet/greet.go") {
		t.Fatalf("Glob result did not list the fixture's Go files: %s", glob)
	}
}

// TestOwnedLoopRefusesOutOfTreeReadWithoutReadingIt pins the safety half separately from
// the happy path: the denied call must not leak the file it was refused, and the refusal
// must be structural (a kernel verdict) rather than a prose note the model could ignore.
func TestOwnedLoopRefusesOutOfTreeReadWithoutReadingIt(t *testing.T) {
	root, outside := seedCodeToolFixture(t)
	escape := filepath.Join(outside, "secret.txt")

	for _, tc := range []struct{ name, args string }{
		{"absolute path outside the root", `{"file_path":` + mustJSON(t, escape) + `}`},
		{"lexical traversal", `{"file_path":"../../etc/passwd"}`},
		{"traversal through a real subdirectory", `{"file_path":"greet/../../secret.txt"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, log := runCodeToolLoop(t, root, []codeToolScript{{codetools.ToolRead, tc.args}})
			if m.Denies != 1 {
				t.Fatalf("denies = %d, want 1", m.Denies)
			}
			if m.EngineCalls != 0 {
				t.Fatalf("engine calls = %d, want 0 — a refused read reached an engine", m.EngineCalls)
			}
			body := lastTraceNote(log)
			if strings.Contains(body, "CLASSIFIED-OUT-OF-TREE") {
				t.Fatalf("refused Read leaked the out-of-tree file: %s", body)
			}
			if !strings.Contains(body, "POLICY_BLOCK") {
				t.Fatalf("refusal is not a typed policy verdict: %s", body)
			}
		})
	}
}

// TestOwnedLoopDeniesUnarmedCodeTools pins that the surface is OFF by default: with no
// ArmCodeTools call, a Read proposed by the model is refused by the loop's default-deny
// floor rather than quietly reaching a filesystem engine.
func TestOwnedLoopDeniesUnarmedCodeTools(t *testing.T) {
	DisarmCodeTools()
	var log []traceEvent
	m, err := RunArm(context.Background(), &scriptedPlanner{turns: []*Completion{
		toolCallTurn(codetools.ToolRead, `{"file_path":"main.go"}`),
		{Message: Message{Content: "done"}},
	}}, "read a file", true, 4, &log)
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.Denies != 1 {
		t.Fatalf("unarmed Read denies = %d, want 1", m.Denies)
	}
	if m.EngineCalls != 0 {
		t.Fatalf("unarmed Read reached %d engines, want 0", m.EngineCalls)
	}
}

// TestCodeToolWitnessArtifact captures the owned-loop run as a machine-readable artifact
// and checks the artifact itself carries the mediation evidence. Setting
// FAK_CODETOOLS_WITNESS_OUT writes it to that path, which is how the committed proof under
// docs/proofs/ is produced; unset, it round-trips through a scratch file so the capture
// path is exercised on every run rather than only when someone remembers to regenerate.
func TestCodeToolWitnessArtifact(t *testing.T) {
	root, outside := seedCodeToolFixture(t)
	escape := filepath.Join(outside, "secret.txt")

	m, log := runCodeToolLoop(t, root, []codeToolScript{
		{codetools.ToolRead, `{"file_path":"main.go"}`},
		{codetools.ToolGrep, `{"pattern":"func Greet","glob":"*.go"}`},
		{codetools.ToolGlob, `{"pattern":"**/*.go"}`},
		{codetools.ToolRead, `{"file_path":` + mustJSON(t, escape) + `}`},
	})

	art := codeToolWitness{
		Schema:      "fak-codetools-owned-loop-witness/1",
		Issue:       "#6703",
		Arm:         m.Arm,
		Tools:       []string{codetools.ToolRead, codetools.ToolGrep, codetools.ToolGlob},
		EngineCalls: m.EngineCalls,
		Denies:      m.Denies,
		VDSOHits:    m.VDSOHits,
	}
	for _, r := range codeToolRows(log) {
		art.Calls = append(art.Calls, codeToolWitnessCall{
			Tool: r.Tool, Verdict: r.Verdict, By: r.By, Reason: r.Reason, Args: oneLine(r.RawArgs, 120),
		})
	}

	body, err := json.MarshalIndent(art, "", "  ")
	if err != nil {
		t.Fatalf("marshal witness: %v", err)
	}
	out := os.Getenv("FAK_CODETOOLS_WITNESS_OUT")
	if out == "" {
		out = filepath.Join(t.TempDir(), "codetools-owned-loop-witness.json")
	}
	if err := os.WriteFile(out, append(body, '\n'), 0o644); err != nil {
		t.Fatalf("write witness %s: %v", out, err)
	}
	t.Logf("owned-loop witness written to %s", out)

	// Read the artifact back and assert on THAT, so what is committed is what was checked.
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read back witness: %v", err)
	}
	var got codeToolWitness
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("witness is not valid JSON: %v", err)
	}
	if got.Arm != "fak" {
		t.Fatalf("witness arm = %q, want fak (the kernel-mediated arm)", got.Arm)
	}
	if got.EngineCalls+got.VDSOHits < 3 || got.Denies != 1 || len(got.Calls) != 4 {
		t.Fatalf("witness does not show 3 dispatches + 1 deny over 4 calls: %+v", got)
	}
	for _, c := range got.Calls {
		if c.Verdict == "DENY" && c.By != codetools.RungName {
			t.Fatalf("witness denied call %+v was not decided by the codetools rung", c)
		}
		if c.Verdict == "ALLOW" && c.By != codetools.RungName && c.By != "vdso" {
			t.Fatalf("witness allowed call %+v bypassed codetools/vDSO", c)
		}
	}
}

// codeToolWitness is the captured artifact's shape.
type codeToolWitness struct {
	Schema      string                `json:"schema"`
	Issue       string                `json:"issue"`
	Arm         string                `json:"arm"`
	Tools       []string              `json:"tools"`
	EngineCalls int                   `json:"engine_calls"`
	Denies      int                   `json:"denies"`
	VDSOHits    int                   `json:"vdso_hits"`
	Calls       []codeToolWitnessCall `json:"calls"`
}

// codeToolWitnessCall is one adjudicated coding-tool call in the artifact.
type codeToolWitnessCall struct {
	Tool    string `json:"tool"`
	Verdict string `json:"verdict"`
	By      string `json:"by"`
	Reason  string `json:"reason,omitempty"`
	Args    string `json:"args,omitempty"`
}

// codeToolRows filters a trace to the coding-tool calls on the fak arm.
func codeToolRows(log []traceEvent) []traceEvent {
	out := make([]traceEvent, 0, len(log))
	for _, e := range log {
		switch e.Tool {
		case codetools.ToolRead, codetools.ToolWrite, codetools.ToolEdit, codetools.ToolBash, codetools.ToolGrep, codetools.ToolGlob:
			out = append(out, e)
		}
	}
	return out
}

// mustJSON renders s as a JSON string literal so a Windows path's backslashes survive
// being embedded in a scripted call's raw arguments.
func lastTraceNote(log []traceEvent) string {
	for i := len(log) - 1; i >= 0; i-- {
		if log[i].Tool != "" {
			return log[i].Note
		}
	}
	return ""
}
func mustJSON(t *testing.T, s string) string {
	t.Helper()
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("marshal %q: %v", s, err)
	}
	return string(b)
}

func TestOwnedLoopMutatesScratchRepoThroughKernelEngines(t *testing.T) {
	root := t.TempDir()
	metrics, log := runCodeToolLoop(t, root, []codeToolScript{
		{tool: codetools.ToolWrite, args: `{"file_path":"pkg/a.go","content":"package pkg\n\nconst Value = 1\n","mode":"create"}`},
		{tool: codetools.ToolEdit, args: `{"file_path":"pkg/a.go","old_string":"Value = 1","new_string":"Value = 2"}`},
	})
	body, err := os.ReadFile(filepath.Join(root, "pkg", "a.go"))
	if err != nil {
		t.Fatalf("read witness: %v", err)
	}
	if string(body) != "package pkg\n\nconst Value = 2\n" {
		t.Fatalf("mutated file = %q", body)
	}
	if metrics.EngineCalls < 2 {
		t.Fatalf("EngineCalls=%d, want >=2", metrics.EngineCalls)
	}
	allowed := 0
	for _, ev := range log {
		if ev.By == codetools.RungName && ev.Verdict == "ALLOW" {
			allowed++
		}
	}
	if allowed < 2 {
		t.Fatalf("codetools ALLOW rows=%d, log=%+v", allowed, log)
	}
}

func TestOwnedLoopCarriesReadVersionThroughStaleAndFreshEdit(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "value.go")
	if err := os.WriteFile(path, []byte("package fixture\n\nconst Value = 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	firstCatalog, err := ArmCodeTools(root)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(DisarmCodeTools)
	var editSchema map[string]any
	for _, def := range firstCatalog {
		if def.Function.Name == codetools.ToolEdit {
			if err := json.Unmarshal(def.Function.Parameters, &editSchema); err != nil {
				t.Fatal(err)
			}
		}
	}
	properties, _ := editSchema["properties"].(map[string]any)
	if _, ok := properties["expected_version"]; !ok {
		t.Fatal("owned-loop catalog omits Edit.expected_version")
	}
	DisarmCodeTools()

	readMetrics, readLog := runCodeToolLoop(t, root, []codeToolScript{{codetools.ToolRead, `{"file_path":"value.go"}`}})
	if readMetrics.EngineCalls != 1 || len(codeToolRows(readLog)) != 1 {
		t.Fatalf("owned-loop Read mediation: metrics=%+v log=%+v", readMetrics, readLog)
	}
	readReceipt := lastResultFromMessages(lastRecordingMessages)
	var observed struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal([]byte(readReceipt), &observed); err != nil || observed.Version == "" {
		t.Fatalf("owned-loop Read receipt = %s, err=%v", readReceipt, err)
	}
	if err := os.WriteFile(path, []byte("package fixture\n\nconst Value = 10\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	vdso.Default.BumpWorld()

	staleMetrics, staleLog := runCodeToolLoop(t, root, []codeToolScript{{codetools.ToolEdit,
		mustSelfcheckArgs(map[string]any{"file_path": "value.go", "old_string": "Value = 1", "new_string": "Value = 2", "expected_version": observed.Version})}})
	if staleMetrics.EngineCalls != 1 || len(codeToolRows(staleLog)) != 1 {
		t.Fatalf("owned-loop stale Edit mediation: metrics=%+v log=%+v", staleMetrics, staleLog)
	}
	staleReceipt := lastResultFromMessages(lastRecordingMessages)
	if !strings.Contains(staleReceipt, codetools.CodeStaleVersion) {
		t.Fatalf("owned-loop stale Edit receipt = %s", staleReceipt)
	}
	if got, _ := os.ReadFile(path); !strings.Contains(string(got), "Value = 10") {
		t.Fatalf("stale Edit changed peer bytes to %q", got)
	}

	freshMetrics, freshLog := runCodeToolLoop(t, root, []codeToolScript{
		{codetools.ToolRead, `{"file_path":"value.go"}`},
		{codetools.ToolEdit, `{"file_path":"value.go","old_string":"Value = 10","new_string":"Value = 2"}`},
	})
	if (freshMetrics.EngineCalls+freshMetrics.VDSOHits) != 2 || len(codeToolRows(freshLog)) != 2 {
		t.Fatalf("owned-loop fresh flow mediation: metrics=%+v log=%+v", freshMetrics, freshLog)
	}
	freshReceipt := lastResultFromMessages(lastRecordingMessages)
	var applied struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal([]byte(freshReceipt), &applied); err != nil || applied.Version == "" || applied.Version == observed.Version {
		t.Fatalf("owned-loop fresh Edit receipt = %s, err=%v", freshReceipt, err)
	}
	final, _ := os.ReadFile(path)
	if !strings.Contains(string(final), "Value = 2") {
		t.Fatalf("fresh Edit final bytes = %q", final)
	}
	t.Logf("read=%s stale_edit=%s fresh_edit=%s final=%q", readReceipt, staleReceipt, freshReceipt, final)
}

func TestOwnedLoopRunsBashThroughKernelEngine(t *testing.T) {
	root := t.TempDir()
	command := "printf kernel>witness.txt"
	if runtime.GOOS == "windows" {
		command = "echo kernel>witness.txt"
	}
	metrics, log := runCodeToolLoop(t, root, []codeToolScript{{tool: codetools.ToolBash, args: `{"command":` + mustJSON(t, command) + `,"cwd":"."}`}})
	body, err := os.ReadFile(filepath.Join(root, "witness.txt"))
	if err != nil || !strings.Contains(string(body), "kernel") {
		t.Fatalf("process witness=%q err=%v", body, err)
	}
	if metrics.EngineCalls < 1 {
		t.Fatalf("EngineCalls=%d, want >=1", metrics.EngineCalls)
	}
	found := false
	for _, ev := range log {
		if ev.Tool == codetools.ToolBash && ev.Verdict == "ALLOW" && ev.By == codetools.RungName {
			found = true
		}
	}
	if !found {
		t.Fatalf("no kernel-mediated Bash trace: %+v", log)
	}
}

func TestIntegratedCodeToolWitnessArtifact(t *testing.T) {
	root := t.TempDir()
	command := "printf processed>processed.txt"
	if runtime.GOOS == "windows" {
		command = "echo processed>processed.txt"
	}
	m, log := runCodeToolLoop(t, root, []codeToolScript{
		{tool: codetools.ToolWrite, args: `{"file_path":"pkg/value.go","content":"package pkg\n\nconst Value = 1\n","mode":"create"}`},
		{tool: codetools.ToolRead, args: `{"file_path":"pkg/value.go"}`},
		{tool: codetools.ToolGrep, args: `{"pattern":"Value = 1","glob":"*.go"}`},
		{tool: codetools.ToolEdit, args: `{"file_path":"pkg/value.go","old_string":"Value = 1","new_string":"Value = 2"}`},
		{tool: codetools.ToolGlob, args: `{"pattern":"**/*.go"}`},
		{tool: codetools.ToolBash, args: `{"command":` + mustJSON(t, command) + `,"cwd":"."}`},
	})
	art := codeToolWitness{Schema: "fak-codetools-owned-loop-witness/2", Issue: "#6658", Arm: m.Arm,
		Tools:       []string{codetools.ToolRead, codetools.ToolWrite, codetools.ToolEdit, codetools.ToolBash, codetools.ToolGrep, codetools.ToolGlob},
		EngineCalls: m.EngineCalls, Denies: m.Denies, VDSOHits: m.VDSOHits}
	for _, r := range codeToolRows(log) {
		art.Calls = append(art.Calls, codeToolWitnessCall{Tool: r.Tool, Verdict: r.Verdict, By: r.By, Reason: r.Reason, Args: oneLine(r.RawArgs, 120)})
	}
	body, err := json.MarshalIndent(art, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	out := os.Getenv("FAK_CODETOOLS_INTEGRATED_WITNESS_OUT")
	if out == "" {
		out = filepath.Join(t.TempDir(), "integrated.json")
	}
	if err = os.WriteFile(out, append(body, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var got codeToolWitness
	if err = json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got.Arm != "fak" || len(got.Calls) != 6 || got.EngineCalls+got.VDSOHits < 6 {
		t.Fatalf("integrated witness=%+v", got)
	}
	seen := map[string]bool{}
	for _, c := range got.Calls {
		if c.Verdict != "ALLOW" || (c.By != codetools.RungName && c.By != "vdso") {
			t.Fatalf("bypass=%+v", c)
		}
		seen[c.Tool] = true
	}
	for _, tool := range got.Tools {
		if !seen[tool] {
			t.Fatalf("tool %s absent: %+v", tool, got)
		}
	}
	mutated, _ := os.ReadFile(filepath.Join(root, "pkg", "value.go"))
	if !strings.Contains(string(mutated), "Value = 2") {
		t.Fatalf("mutation=%q", mutated)
	}
	processed, _ := os.ReadFile(filepath.Join(root, "processed.txt"))
	if !strings.Contains(string(processed), "processed") {
		t.Fatalf("process=%q", processed)
	}
}
