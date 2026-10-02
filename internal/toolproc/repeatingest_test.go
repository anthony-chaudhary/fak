package toolproc

// repeatingest_test.go — proof for the rollout INGESTION front (#4764 DoD "stream
// native Codex rollout logs and normalize tool calls"). A hermetic fixture mirrors
// the real rollout JSONL shape (confirmed against captured sessions: outer
// {timestamp,type,payload}; payload types local_shell_call / function_call /
// custom_tool_call and their *_output twins joined by call_id). The test proves the
// end-to-end path: raw rollout bytes → normalized CallRecords → the classifier's
// typed inventory, with output SIZES joined and no body retained.

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// a Codex rollout fixture: a bash-wrapped `git status` poll, the same immutable
// skill read twice (proving the fold survives ingestion), a `git push` write, a
// call whose output never arrives, a non-tool payload, and a malformed line.
const rolloutFixture = `
{"timestamp":"2026-07-14T21:25:00.000Z","type":"session_meta","payload":{"type":"session_meta","id":"s1"}}
{"timestamp":"2026-07-14T21:25:00.100Z","type":"response_item","payload":{"type":"local_shell_call","call_id":"c1","action":{"type":"exec","command":["bash","-lc","git status --short --branch"]}}}
{"timestamp":"2026-07-14T21:25:00.150Z","type":"response_item","payload":{"type":"local_shell_call_output","call_id":"c1","output":"On branch main"}}
{"timestamp":"2026-07-14T21:25:01.000Z","type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"c2","arguments":"{\"command\":[\"bash\",\"-lc\",\"cat super-loop/SKILL.md\"]}"}}
{"timestamp":"2026-07-14T21:25:01.050Z","type":"response_item","payload":{"type":"function_call_output","call_id":"c2","output":"SKILLBODY__"}}
NOT-JSON-a-torn-line{
{"timestamp":"2026-07-14T21:25:02.000Z","type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"c3","arguments":"{\"command\":[\"bash\",\"-lc\",\"cat super-loop/SKILL.md\"]}"}}
{"timestamp":"2026-07-14T21:25:02.050Z","type":"response_item","payload":{"type":"function_call_output","call_id":"c3","output":"SKILLBODY__"}}
{"timestamp":"2026-07-14T21:25:03.000Z","type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"c4","arguments":"{\"command\":[\"bash\",\"-lc\",\"git push origin main\"]}"}}
{"timestamp":"2026-07-14T21:25:03.050Z","type":"response_item","payload":{"type":"function_call_output","call_id":"c4","output":"Everything up-to-date"}}
{"timestamp":"2026-07-14T21:25:04.000Z","type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"c5","arguments":"{\"command\":[\"bash\",\"-lc\",\"git diff\"]}"}}
`

// TestIngestRolloutNormalizesAndFeedsClassifier proves the whole ingestion path.
func TestIngestRolloutNormalizesAndFeedsClassifier(t *testing.T) {
	recs := IngestRollout(strings.NewReader(rolloutFixture))

	// Five tool calls (c1..c5); the session_meta, the output rows, and the torn
	// line are not calls.
	if len(recs) != 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST — closed fixture/contract cardinality
		t.Fatalf("want 5 ingested calls, got %d: %+v", len(recs), recs)
	}

	byRaw := map[string][]CallRecord{}
	for _, r := range recs {
		byRaw[r.Raw] = append(byRaw[r.Raw], r)
	}

	// The bash wrapper is unwrapped: the classifier sees `git status ...`, not `bash`.
	gs := byRaw["git status --short --branch"]
	if len(gs) != 1 || gs[0].Tool != "shell_command" {
		t.Fatalf("git status: want 1 shell_command record, got %+v", gs)
	}
	if gs[0].OutputBytes != int64(len("On branch main")) {
		t.Errorf("git status output size: want %d, got %d", len("On branch main"), gs[0].OutputBytes)
	}

	// The immutable skill read appears twice with an identical unwrapped command.
	cat := byRaw["cat super-loop/SKILL.md"]
	if len(cat) != 2 {
		t.Fatalf("skill read: want 2 records, got %d", len(cat))
	}
	if cat[0].OutputBytes != int64(len("SKILLBODY__")) {
		t.Errorf("skill read output size: want %d, got %d", len("SKILLBODY__"), cat[0].OutputBytes)
	}

	// c5 (`git diff`) has no output row → OutputBytes 0, never a crash.
	gd := byRaw["git diff"]
	if len(gd) != 1 || gd[0].OutputBytes != 0 {
		t.Errorf("missing-output call must ingest with 0 bytes, got %+v", gd)
	}

	// End to end: the ingested records classify exactly as hand-authored ones would.
	rep := ClassifyRepeats(recs, RepeatConfig{})
	read := findGroup(t, rep, "read:super-loop/SKILL.md")
	if read.Class != ClassImmutableRead || read.Count != 2 || read.AvoidableSpawns != 1 {
		t.Errorf("skill read group: want IMMUTABLE_READ count=2 avoidable=1, got class=%s count=%d avoidable=%d",
			read.Class, read.Count, read.AvoidableSpawns)
	}
	push := findGroup(t, rep, "write:git push origin main")
	if push.Reuse != ReuseNever {
		t.Errorf("push must be NEVER-reuse, got %s", push.Reuse)
	}
	status := findGroup(t, rep, "query:git status --branch --short")
	if status.Class != ClassMutableQuery {
		t.Errorf("git status: want MUTABLE_QUERY, got %s", status.Class)
	}
}

// nativeShellCommandFixture mirrors the shape the CAPTURED rollouts actually carry,
// which differs from the fixture above in both fields that matter: the function is
// named `shell_command` (not `shell`), and its `command` argument is a plain STRING
// with `workdir`/`timeout_ms` siblings (not a ["bash","-lc",…] array). This is the
// spelling #4764's own audit counted when it reported shell_command dominating the
// workload, so it is the spelling the replay has to ingest.
const nativeShellCommandFixture = `
{"timestamp":"2026-07-14T21:25:00.000Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","call_id":"n1","arguments":"{\"command\":\"git status --short --branch\",\"workdir\":\"C:\\\\work\\\\fak\",\"timeout_ms\":10000}"}}
{"timestamp":"2026-07-14T21:25:00.050Z","type":"response_item","payload":{"type":"function_call_output","call_id":"n1","output":"On branch main"}}
{"timestamp":"2026-07-14T21:25:01.000Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","call_id":"n2","arguments":"{\"command\":\"cat super-loop/SKILL.md\",\"workdir\":\"C:\\\\work\\\\fak\",\"timeout_ms\":10000}"}}
{"timestamp":"2026-07-14T21:25:01.050Z","type":"response_item","payload":{"type":"function_call_output","call_id":"n2","output":"SKILLBODY__"}}
{"timestamp":"2026-07-14T21:25:02.000Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","call_id":"n3","arguments":"{\"command\":\"cat super-loop/SKILL.md\",\"workdir\":\"C:\\\\work\\\\fak\",\"timeout_ms\":10000}"}}
{"timestamp":"2026-07-14T21:25:02.050Z","type":"response_item","payload":{"type":"function_call_output","call_id":"n3","output":"SKILLBODY__"}}
{"timestamp":"2026-07-14T21:25:03.000Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","call_id":"n4","arguments":"{\"command\":\"git push origin main\",\"workdir\":\"C:\\\\work\\\\fak\",\"timeout_ms\":10000}"}}
{"timestamp":"2026-07-14T21:25:03.050Z","type":"response_item","payload":{"type":"function_call_output","call_id":"n4","output":"Everything up-to-date"}}
`

// TestIngestRolloutClassifiesNativeShellCommand is the replay gate for #5120: the
// top-100 workload is ~95% `shell_command`, so if that name does not reach
// commandFromArgs the command never gets extracted, every call keys on the raw
// argument blob, and the whole inventory collapses to a single UNKNOWN class — no
// command-frequency table, no reproducible headline. It also pins the blob-leak
// consequence: an unextracted call carries its argument JSON (patch bodies and all)
// into the group key that the report renders.
func TestIngestRolloutClassifiesNativeShellCommand(t *testing.T) {
	recs := IngestRollout(strings.NewReader(nativeShellCommandFixture))
	if len(recs) != 4 { //boundarylint:ignore CHANGE_DETECTOR_TEST — closed fixture/contract cardinality
		t.Fatalf("want 4 ingested calls, got %d: %+v", len(recs), recs)
	}

	// Every native call must land as a shell command with the wrapper stripped —
	// not as a tool named `shell_command` carrying an argument blob.
	for _, r := range recs {
		if r.Tool != "shell_command" {
			t.Fatalf("native call: want Tool=shell_command, got %q (raw %q)", r.Tool, r.Raw)
		}
		if strings.Contains(r.Raw, "workdir") || strings.Contains(r.Raw, "timeout_ms") {
			t.Fatalf("argument blob leaked into the group key instead of the command: %q", r.Raw)
		}
	}

	rep := ClassifyRepeats(recs, RepeatConfig{})

	// The whole point: the inventory is TYPED, not one UNKNOWN bucket.
	if n := rep.Totals.PerClass[ClassUnknown]; n != 0 {
		t.Errorf("native shell_command workload must not fall through to UNKNOWN, got %d groups", n)
	}
	read := findGroup(t, rep, "read:super-loop/SKILL.md")
	if read.Class != ClassImmutableRead || read.Count != 2 || read.AvoidableSpawns != 1 {
		t.Errorf("skill read group: want IMMUTABLE_READ count=2 avoidable=1, got class=%s count=%d avoidable=%d",
			read.Class, read.Count, read.AvoidableSpawns)
	}
	if push := findGroup(t, rep, "write:git push origin main"); push.Reuse != ReuseNever {
		t.Errorf("push must be NEVER-reuse, got %s", push.Reuse)
	}
	if status := findGroup(t, rep, "query:git status --branch --short"); status.Class != ClassMutableQuery {
		t.Errorf("git status: want MUTABLE_QUERY, got %s", status.Class)
	}
}

// TestIngestRolloutRetainsNoOutputBody proves the analytics contract at the ingest
// boundary: even a large output body is reduced to its SIZE — the body text never
// appears in any returned CallRecord field.
func TestIngestRolloutRetainsNoOutputBody(t *testing.T) {
	body := strings.Repeat("SECRETLIKE-", 500) // ~5.5 KB of body text
	line := `{"timestamp":"2026-07-14T21:25:00.000Z","type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"x","arguments":"{\"command\":[\"bash\",\"-lc\",\"cat f\"]}"}}` + "\n" +
		`{"timestamp":"2026-07-14T21:25:00.050Z","type":"response_item","payload":{"type":"function_call_output","call_id":"x","output":"` + body + `"}}`
	recs := IngestRollout(strings.NewReader(line))
	if len(recs) != 1 {
		t.Fatalf("want 1 record, got %d", len(recs))
	}
	if recs[0].OutputBytes != int64(len(body)) {
		t.Errorf("output size: want %d, got %d", len(body), recs[0].OutputBytes)
	}
	if strings.Contains(recs[0].Raw, "SECRETLIKE") || strings.Contains(recs[0].Tool, "SECRETLIKE") {
		t.Fatalf("output body leaked into a retained field: %+v", recs[0])
	}
}

// TestIngestRolloutEvidence pins #13416: a lossless evidence view that
// distinguishes a completely understood zero-call transcript from incomplete or
// unsupported input, preserving original tool-call inputs without retaining any
// output body. All fixtures are inline and synthetic; no live rollout is read.
func TestIngestRolloutEvidence(t *testing.T) {
	tests := []struct {
		name        string
		in          string
		wantCalls   int
		wantKind    []RolloutCallKind
		wantCompl   bool
		wantUnsup   int
		wantMal     int
		wantRecords int
	}{
		{
			name: "complete metadata-only zero is Complete with no calls",
			in: `{"timestamp":"2026-09-30T00:00:00Z","payload":{"type":"session_meta","id":"s1"}}
{"timestamp":"2026-09-30T00:00:01Z","payload":{"type":"turn_context","model":"gpt-5"}}
{"timestamp":"2026-09-30T00:00:02Z","payload":{"type":"event_msg","message":"ready"}}
{"timestamp":"2026-09-30T00:00:03Z","payload":{"type":"compacted"}}
`,
			wantCalls: 0, wantCompl: true, wantRecords: 4,
		},
		{
			name: "function_call preserves exact arguments",
			in: `{"timestamp":"2026-09-30T00:00:00Z","payload":{"type":"function_call","name":"shell_command","arguments":"{\"command\":\"cmd\",\"x\":1}","call_id":"c1"}}
{"timestamp":"2026-09-30T00:00:01Z","payload":{"type":"function_call_output","call_id":"c1","output":"ok"}}
`,
			wantCalls: 1, wantKind: []RolloutCallKind{RolloutKindFunctionCall}, wantCompl: true, wantRecords: 2,
		},
		{
			name: "custom_tool_call preserves raw input",
			in: `{"timestamp":"2026-09-30T00:00:00Z","payload":{"type":"custom_tool_call","name":"apply_patch","input":"*** Begin Patch\nx\n*** End Patch","call_id":"c2"}}
`,
			wantCalls: 1, wantKind: []RolloutCallKind{RolloutKindCustomTool}, wantCompl: true, wantRecords: 1,
		},
		{
			name: "local_shell_call copies the command array",
			in: `{"timestamp":"2026-09-30T00:00:00Z","payload":{"type":"local_shell_call","call_id":"c3","action":{"command":["bash","-lc","git status"]}}}
`,
			wantCalls: 1, wantKind: []RolloutCallKind{RolloutKindLocalShell}, wantCompl: true, wantRecords: 1,
		},
		{
			name: "response_item wrapping a call is unwrapped",
			in: `{"timestamp":"2026-09-30T00:00:00Z","payload":{"type":"response_item","payload":{"type":"function_call","name":"f","arguments":"{}","call_id":"c4"}}}
`,
			wantCalls: 1, wantKind: []RolloutCallKind{RolloutKindFunctionCall}, wantCompl: true, wantRecords: 1,
		},
		{
			name: "unknown top-level kind makes coverage incomplete",
			in: `{"timestamp":"2026-09-30T00:00:00Z","payload":{"type":"brand_new_kind"}}
`,
			wantCalls: 0, wantCompl: false, wantUnsup: 1, wantRecords: 1,
		},
		{
			name: "malformed row makes coverage incomplete and is not a zero",
			in: `{"timestamp":"2026-09-30T00:00:00Z","payload":{"type":"session_meta"}}
not json at all
`,
			wantCalls: 0, wantCompl: false, wantMal: 1, wantRecords: 2,
		},
		{
			name:      "empty input is incomplete, never a witnessed zero",
			in:        ``,
			wantCalls: 0, wantCompl: false, wantRecords: 0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ev := IngestRolloutEvidence(strings.NewReader(tc.in))
			if len(ev.Calls) != tc.wantCalls {
				t.Fatalf("calls = %d, want %d (%+v)", len(ev.Calls), tc.wantCalls, ev.Calls)
			}
			if ev.Complete != tc.wantCompl {
				t.Fatalf("Complete = %v, want %v", ev.Complete, tc.wantCompl)
			}
			if ev.Unsupported != tc.wantUnsup {
				t.Fatalf("Unsupported = %d, want %d", ev.Unsupported, tc.wantUnsup)
			}
			if ev.Malformed != tc.wantMal {
				t.Fatalf("Malformed = %d, want %d", ev.Malformed, tc.wantMal)
			}
			if ev.Records != tc.wantRecords {
				t.Fatalf("Records = %d, want %d", ev.Records, tc.wantRecords)
			}
			for i, k := range tc.wantKind {
				if ev.Calls[i].Kind != k {
					t.Fatalf("call[%d].Kind = %q, want %q", i, ev.Calls[i].Kind, k)
				}
			}
		})
	}

	// Exact bytes and ordering, plus no output-body retention.
	ev := IngestRolloutEvidence(strings.NewReader(
		`{"timestamp":"2026-09-30T00:00:00Z","payload":{"type":"function_call","name":"shell_command","arguments":"{\"command\":\"cmd\",\"x\":1}","call_id":"c1"}}
{"timestamp":"2026-09-30T00:00:01Z","payload":{"type":"custom_tool_call","name":"apply_patch","input":"RAW","call_id":"c2"}}
{"timestamp":"2026-09-30T00:00:02Z","payload":{"type":"function_call_output","call_id":"c1","output":"a secret body that must not be retained"}}
`))
	if ev.Calls[0].Arguments != `{"command":"cmd","x":1}` {
		t.Fatalf("Arguments = %q, want exact bytes", ev.Calls[0].Arguments)
	}
	if ev.Calls[1].Input != "RAW" {
		t.Fatalf("Input = %q, want RAW", ev.Calls[1].Input)
	}
	// The evidence view exposes no field carrying the output body.
	blob, _ := json.Marshal(ev)
	if strings.Contains(string(blob), "secret body") {
		t.Fatalf("evidence view retained an output body: %s", blob)
	}

	// A local_shell command must be a COPY, not an alias of the decoder buffer.
	ev2 := IngestRolloutEvidence(strings.NewReader(
		`{"payload":{"type":"local_shell_call","call_id":"c9","action":{"command":["bash","-lc","x"]}}}`))
	ev2.Calls[0].Command[0] = "mutated"
	ev3 := IngestRolloutEvidence(strings.NewReader(
		`{"payload":{"type":"local_shell_call","call_id":"c9","action":{"command":["bash","-lc","x"]}}}`))
	if ev3.Calls[0].Command[0] != "bash" {
		t.Fatalf("command array was aliased across calls: %v", ev3.Calls[0].Command)
	}

	// A reader failure must make coverage incomplete via ReadError.
	ev4 := IngestRolloutEvidence(&failingReader{after: []byte(`{"payload":{"type":"session_meta"}}` + "\n")})
	if ev4.Complete || ev4.ReadError == nil {
		t.Fatalf("reader failure must be incomplete with a ReadError, got %+v", ev4)
	}
}

// failingReader yields its prefix then returns a non-EOF error.
type failingReader struct {
	after []byte
	done  bool
}

func (f *failingReader) Read(p []byte) (int, error) {
	if !f.done {
		f.done = true
		n := copy(p, f.after)
		return n, nil
	}
	return 0, errors.New("simulated read failure")
}
