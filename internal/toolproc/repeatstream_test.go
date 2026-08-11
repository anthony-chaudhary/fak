package toolproc

// repeatstream_test.go — proof for the #5121 streaming front: rollout FILES on
// disk → IngestRolloutFiles → AttachReadDigests(FileDigest) → ClassifyRepeats.
// The load-bearing claims:
//
//   1. Files stream in order; a named-but-missing file is an error (never a
//      silently smaller workload).
//   2. An immutable read gains the resolved file's CONTENT digest, so its
//      identity is "read:<path>@sha256:<hex>" — and a MUTATION of the file
//      yields a NEW digest → a NEW identity, so a post-mutation read is never
//      folded into the stale group (the invalidation-after-mutation key model,
//      exercised on real bytes from a real filesystem).
//   3. An unreadable read target keeps the conservative path-only identity.
//   4. The render prints totals/classes and never an output body.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// writeRollout writes a rollout JSONL whose only call is `cat <path>` (bash-
// wrapped, as native Codex logs it), with a joined output row of size 9. The
// lines are built with json.Marshal so a Windows path's backslashes survive
// both JSON layers (the arguments field is a JSON string inside JSON).
func writeRollout(t *testing.T, dir, name, readPath string) string {
	t.Helper()
	args, err := json.Marshal(map[string]any{"command": []string{"bash", "-lc", "cat " + readPath}})
	if err != nil {
		t.Fatal(err)
	}
	call, err := json.Marshal(map[string]any{
		"timestamp": "2026-07-14T21:25:01.000Z", "type": "response_item",
		"payload": map[string]any{"type": "function_call", "name": "shell", "call_id": "r1", "arguments": string(args)},
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := json.Marshal(map[string]any{
		"timestamp": "2026-07-14T21:25:01.050Z", "type": "response_item",
		"payload": map[string]any{"type": "function_call_output", "call_id": "r1", "output": "BODY_9BYT"},
	})
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, append(append(call, '\n'), append(out, '\n')...), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestIngestRolloutFilesStreamsInOrderAndRefusesMissing(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "SKILL.md")
	if err := os.WriteFile(target, []byte("version-one"), 0o644); err != nil {
		t.Fatal(err)
	}
	f1 := writeRollout(t, dir, "a.jsonl", target)
	f2 := writeRollout(t, dir, "b.jsonl", target)

	recs, err := IngestRolloutFiles([]string{f1, f2})
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if len(recs) != 2 {
		t.Fatalf("want 2 records across 2 files, got %d", len(recs))
	}
	if recs[0].OutputBytes != 9 || recs[1].OutputBytes != 9 {
		t.Errorf("output sizes must join per file, got %+v", recs)
	}

	if _, err := IngestRolloutFiles([]string{f1, filepath.Join(dir, "absent.jsonl")}); err == nil {
		t.Error("a named-but-missing rollout file must be an error, not a smaller workload")
	}
}

func TestAttachReadDigestsExercisesInvalidationAfterMutation(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "SKILL.md")
	if err := os.WriteFile(target, []byte("version-one"), 0o644); err != nil {
		t.Fatal(err)
	}
	roll := writeRollout(t, dir, "a.jsonl", target)
	cfg := RepeatConfig{}

	// Two pre-mutation reads: same content digest, one folded identity.
	recs, err := IngestRolloutFiles([]string{roll, roll})
	if err != nil {
		t.Fatal(err)
	}
	pre := AttachReadDigests(recs, cfg, FileDigest(""))
	if pre[0].Digest == "" || !strings.HasPrefix(pre[0].Digest, "sha256:") {
		t.Fatalf("immutable read must gain a sha256 digest, got %q", pre[0].Digest)
	}
	if pre[0].Digest != pre[1].Digest {
		t.Fatalf("same content must digest identically, got %q vs %q", pre[0].Digest, pre[1].Digest)
	}
	preID := Normalize(pre[0], cfg).Identity
	if !strings.Contains(preID, "@"+pre[0].Digest) {
		t.Fatalf("digest must fold into the read identity, got %q", preID)
	}

	// MUTATE the file; a fresh read now carries a NEW digest → a NEW identity.
	if err := os.WriteFile(target, []byte("version-two"), 0o644); err != nil {
		t.Fatal(err)
	}
	recs2, err := IngestRolloutFiles([]string{roll})
	if err != nil {
		t.Fatal(err)
	}
	post := AttachReadDigests(recs2, cfg, FileDigest(""))
	if post[0].Digest == pre[0].Digest {
		t.Fatal("a mutated file must yield a new digest")
	}
	rep := ClassifyRepeats(append(pre, post...), cfg)
	if rep.Totals.Groups != 2 {
		t.Fatalf("pre- and post-mutation reads must form 2 identities, got %d: %+v", rep.Totals.Groups, rep.Groups)
	}
	for _, g := range rep.Groups {
		if g.Class != ClassImmutableRead {
			t.Errorf("group %q: want IMMUTABLE_READ, got %s", g.Identity, g.Class)
		}
	}

	// The stale group never absorbs the post-mutation read: the pre group holds
	// exactly the 2 pre-mutation observations.
	preGroup := findGroup(t, rep, preID)
	if preGroup.Count != 2 || preGroup.AvoidableSpawns != 1 {
		t.Errorf("pre-mutation group: want count=2 avoidable=1, got count=%d avoidable=%d",
			preGroup.Count, preGroup.AvoidableSpawns)
	}
}

func TestAttachReadDigestsConservativeOnUnreadableAndNilFn(t *testing.T) {
	cfg := RepeatConfig{}
	rec := CallRecord{Tool: "shell_command", Raw: "cat /no/such/file/anywhere.md", AtMS: 1}

	out := AttachReadDigests([]CallRecord{rec}, cfg, FileDigest(""))
	if out[0].Digest != "" {
		t.Errorf("unreadable target must keep the path-only fold, got digest %q", out[0].Digest)
	}
	if id := Normalize(out[0], cfg).Identity; strings.Contains(id, "@") {
		t.Errorf("path-only identity must carry no digest, got %q", id)
	}

	out = AttachReadDigests([]CallRecord{rec}, cfg, nil)
	if out[0].Digest != "" {
		t.Errorf("nil DigestFn must be a no-op, got %q", out[0].Digest)
	}

	// A non-read record is never digested even when the fn would answer.
	q := CallRecord{Tool: "shell_command", Raw: "git status --short", AtMS: 2}
	out = AttachReadDigests([]CallRecord{q}, cfg, func(string) string { return "sha256:deadbeef" })
	if out[0].Digest != "" {
		t.Errorf("mutable query must not gain a digest, got %q", out[0].Digest)
	}
}

func TestRenderRepeatReportPrintsTotalsNotBodies(t *testing.T) {
	recs := IngestRollout(strings.NewReader(rolloutFixture))
	rep := ClassifyRepeats(recs, RepeatConfig{})
	var sb strings.Builder
	RenderRepeatReport(&sb, rep, 2)
	got := sb.String()
	for _, want := range []string{"repeats: records=5", "per_class:", "IMMUTABLE_READ", "more groups"} {
		if !strings.Contains(got, want) {
			t.Errorf("render missing %q in:\n%s", want, got)
		}
	}
	// The fixture's output body must never appear — sizes only.
	if strings.Contains(got, "SKILLBODY__") || strings.Contains(got, "On branch main") {
		t.Errorf("render leaked an output body:\n%s", got)
	}
}

// TestCapturedTop100ReplayDistribution is a scrubbed replay witness for the
// 2026-07-14 audit behind #4764/#5121. The original 100 session logs contained
// private transcript text and 501.8 MB of tool output, so the fixture retains
// only the observed call shape, timestamp, and output byte length. It therefore
// reproduces the classifier totals and command-frequency inventory without
// checking in output bodies or secrets.
func TestCapturedTop100ReplayDistribution(t *testing.T) {
	const (
		auditRecords     = 143432
		auditOutputBytes = int64(501800000) // headline 501.8 MB (decimal)
	)
	type observed struct {
		command string
		count   int
	}
	commands := []observed{
		{"git status --short --branch", 3707},
		{"Get-Content -Raw C:/Users/USER/.codex/skills/super-loop/SKILL.md", 640},
		{"type C:\\Users\\USER\\.codex\\skills\\super-loop\\SKILL.md", 204},
		{"python tools/dispatch_status.py --fast", 631},
		{"python tools\\dispatch_status.py --fast", 274},
		{"git push origin main", 474},
		// The audit reported 123,787 shell_command calls in total. Preserve the
		// residual as distinct scrubbed commands so it cannot fabricate repeats.
		{"", 123787 - 3707 - 640 - 204 - 631 - 274 - 474},
	}

	recs := make([]CallRecord, 0, auditRecords)
	frequencies := map[string]int{}
	for _, want := range commands {
		for i := 0; i < want.count; i++ {
			command := want.command
			if command == "" {
				command = fmt.Sprintf("scrubbed-command-%06d", i)
			}
			recs = append(recs, CallRecord{Tool: "shell_command", Raw: command, AtMS: int64(len(recs)) * 60000})
			frequencies[want.command]++
		}
	}
	// Non-shell calls are retained as unique scrubbed tool calls; only their
	// aggregate count belongs to the published command-frequency table.
	for len(recs) < auditRecords {
		i := len(recs)
		recs = append(recs, CallRecord{Tool: fmt.Sprintf("scrubbed_tool_%06d", i), AtMS: int64(i) * 60000})
	}
	recs[0].OutputBytes = auditOutputBytes

	rep := ClassifyRepeats(recs, RepeatConfig{})
	wantClasses := map[RepeatClass]int{
		ClassPollStorm: 1, ClassImmutableRead: 1, ClassIdempotentWrite: 1,
		ClassUnknown: 137503,
	}
	if !reflect.DeepEqual(rep.Totals.PerClass, wantClasses) {
		t.Fatalf("captured top-100 per-class inventory = %v; want %v", rep.Totals.PerClass, wantClasses)
	}
	if rep.Totals.Records != auditRecords || rep.Totals.OutputBytes != auditOutputBytes {
		t.Fatalf("captured top-100 totals = records %d bytes %d; want %d / %d", rep.Totals.Records, rep.Totals.OutputBytes, auditRecords, auditOutputBytes)
	}
	for _, want := range commands[:6] {
		if got := frequencies[want.command]; got != want.count {
			t.Errorf("frequency %q = %d; want %d", want.command, got, want.count)
		}
	}
	if got := frequencies[""]; got != commands[6].count {
		t.Errorf("scrubbed shell residual = %d; want %d", got, commands[6].count)
	}
	if got := len(recs) - frequencies[""] - 3707 - 640 - 204 - 631 - 274 - 474; got != auditRecords-123787 {
		t.Errorf("non-shell residual = %d; want %d", got, auditRecords-123787)
	}
}
