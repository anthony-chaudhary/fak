package main

// toolproc_repeats_test.go — end-to-end proof for `fak toolproc repeats` (#5121):
// a native-shaped rollout fixture on disk folds through the shell into the
// classifier's report, in both human and --json form; a missing file exits 1.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

const repeatsCLIFixture = `{"timestamp":"2026-07-14T21:25:00.100Z","type":"response_item","payload":{"type":"local_shell_call","call_id":"c1","action":{"type":"exec","command":["bash","-lc","git status --short --branch"]}}}
{"timestamp":"2026-07-14T21:25:00.150Z","type":"response_item","payload":{"type":"local_shell_call_output","call_id":"c1","output":"On branch main"}}
{"timestamp":"2026-07-14T21:25:01.000Z","type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"c2","arguments":"{\"command\":[\"bash\",\"-lc\",\"cat sub/SKILL.md\"]}"}}
{"timestamp":"2026-07-14T21:25:01.050Z","type":"response_item","payload":{"type":"function_call_output","call_id":"c2","output":"SKILLBODY__"}}
{"timestamp":"2026-07-14T21:25:02.000Z","type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"c3","arguments":"{\"command\":[\"bash\",\"-lc\",\"cat sub/SKILL.md\"]}"}}
{"timestamp":"2026-07-14T21:25:02.050Z","type":"response_item","payload":{"type":"function_call_output","call_id":"c3","output":"SKILLBODY__"}}
`

func TestRunToolprocRepeatsFoldsRolloutFixture(t *testing.T) {
	dir := t.TempDir()
	roll := filepath.Join(dir, "rollout.jsonl")
	if err := os.WriteFile(roll, []byte(repeatsCLIFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	// The read target exists under --root, so the digest rung fires.
	if err := os.MkdirAll(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "sub", "SKILL.md"), []byte("skill content"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb strings.Builder
	if rc := runToolprocRepeats(&out, &errb, []string{"--json", "--root", dir, roll}); rc != 0 {
		t.Fatalf("exit %d, stderr: %s", rc, errb.String())
	}
	var rep toolproc.RepeatReport
	if err := json.Unmarshal([]byte(out.String()), &rep); err != nil {
		t.Fatalf("--json output must be a RepeatReport: %v\n%s", err, out.String())
	}
	if rep.Totals.Records != 3 || rep.Totals.Groups != 2 {
		t.Fatalf("want 3 records / 2 groups, got %d / %d", rep.Totals.Records, rep.Totals.Groups)
	}
	var read *toolproc.RepeatGroup
	for i := range rep.Groups {
		if rep.Groups[i].Class == toolproc.ClassImmutableRead {
			read = &rep.Groups[i]
		}
	}
	if read == nil {
		t.Fatalf("no IMMUTABLE_READ group in %+v", rep.Groups)
	}
	if !strings.HasPrefix(read.Digest, "sha256:") {
		t.Errorf("read group must carry the resolved content digest, got %q", read.Digest)
	}
	if read.Count != 2 || read.AvoidableSpawns != 1 {
		t.Errorf("read group: want count=2 avoidable=1, got count=%d avoidable=%d", read.Count, read.AvoidableSpawns)
	}

	// Human form: totals line, no output body.
	out.Reset()
	errb.Reset()
	if rc := runToolprocRepeats(&out, &errb, []string{"--root", dir, roll}); rc != 0 {
		t.Fatalf("human form exit %d, stderr: %s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "repeats: records=3") || strings.Contains(out.String(), "SKILLBODY__") {
		t.Errorf("human render wrong or leaked a body:\n%s", out.String())
	}

	// A named-but-missing rollout refuses with exit 1.
	out.Reset()
	errb.Reset()
	if rc := runToolprocRepeats(&out, &errb, []string{filepath.Join(dir, "absent.jsonl")}); rc != 1 {
		t.Errorf("missing rollout: want exit 1, got %d (stderr: %s)", rc, errb.String())
	}
	// No positional args is usage (2).
	if rc := runToolprocRepeats(&out, &errb, nil); rc != 2 {
		t.Errorf("no args: want exit 2, got %d", rc)
	}
}
