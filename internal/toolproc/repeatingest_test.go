package toolproc

// repeatingest_test.go — proof for the rollout INGESTION front (#4764 DoD "stream
// native Codex rollout logs and normalize tool calls"). A hermetic fixture mirrors
// the real rollout JSONL shape (confirmed against captured sessions: outer
// {timestamp,type,payload}; payload types local_shell_call / function_call /
// custom_tool_call and their *_output twins joined by call_id). The test proves the
// end-to-end path: raw rollout bytes → normalized CallRecords → the classifier's
// typed inventory, with output SIZES joined and no body retained.

import (
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
