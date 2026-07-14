package modver

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// driftLogFixture is a canned `git log --no-merges --pretty=format:%x1e%h%x09%cI
// --name-only v1.2.3..HEAD -- <roots>` transcript (newest-first, %x1e record sep,
// tab between short-sha and committer date). It exercises the drift fold:
//   - internal/gateway moves in 3 commits (aaa, bbb, ccc)
//   - internal/modver moves in 2 commits (aaa, ccc)
//   - cmd/fak moves in 1 commit (bbb)
//   - internal/deleted appears in history but is NOT live at HEAD -> must be dropped
//   - a commit touching two files of one module counts that module once
const driftLogFixture = "\x1e" + "aaa11111\t2026-07-05T10:00:00Z\n" +
	"internal/gateway/wire.go\n" +
	"internal/gateway/http.go\n" + // same module, same commit: counts once
	"internal/modver/modver.go\n" +
	"\x1e" + "bbb22222\t2026-07-04T09:00:00Z\n" +
	"internal/gateway/metrics.go\n" +
	"cmd/fak/main.go\n" +
	"internal/deleted/gone.go\n" + // not live at HEAD: dropped
	"\x1e" + "ccc33333\t2026-07-03T08:00:00Z\n" +
	"internal/gateway/debug.go\n" +
	"internal/modver/drift.go\n"

// TestDrift is the pure-fold witness: a fixture tag..HEAD range log + a HEAD live set
// folds into a readout listing exactly the modules that moved since the tag, ranked
// most-moved-first, with a deleted module dropped and a quiet (unmoved) live module
// absent — the "readout lists modules moved since last tag" done condition.
func TestDrift(t *testing.T) {
	live := map[string]bool{
		"internal/gateway": true,
		"cmd/fak":          true,
		"internal/modver":  true,
		"internal/quiet":   true, // live but untouched in-range: must NOT appear
	}

	rep := Drift([]byte(driftLogFixture), live, "v1.2.3", "tagsha01", "headbee1")

	if rep.Schema != DriftSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, DriftSchema)
	}
	if rep.Tag != "v1.2.3" || rep.TagSHA != "tagsha01" || rep.Head != "headbee1" {
		t.Errorf("boundary = (%q,%q,%q), want (v1.2.3,tagsha01,headbee1)", rep.Tag, rep.TagSHA, rep.Head)
	}
	if rep.Scanned != 4 {
		t.Errorf("scanned = %d, want 4 (live modules)", rep.Scanned)
	}
	if rep.Moved != 3 || len(rep.Rows) != 3 {
		t.Fatalf("moved = %d, len(rows) = %d, want 3 and 3 (quiet + deleted excluded)", rep.Moved, len(rep.Rows))
	}

	want := []DriftRow{
		{Module: "internal/gateway", Kind: "internal", RevsSinceTag: 3, LastCommit: "aaa11111", LastDate: "2026-07-05T10:00:00Z"},
		{Module: "internal/modver", Kind: "internal", RevsSinceTag: 2, LastCommit: "aaa11111", LastDate: "2026-07-05T10:00:00Z"},
		{Module: "cmd/fak", Kind: "cmd", RevsSinceTag: 1, LastCommit: "bbb22222", LastDate: "2026-07-04T09:00:00Z"},
	}
	if !reflect.DeepEqual(rep.Rows, want) {
		t.Errorf("rows =\n  %+v\nwant\n  %+v", rep.Rows, want)
	}

	for _, r := range rep.Rows {
		if r.Module == "internal/quiet" {
			t.Errorf("unmoved live module internal/quiet leaked into the readout")
		}
		if r.Module == "internal/deleted" {
			t.Errorf("deleted (non-live) module internal/deleted leaked into the readout")
		}
	}

	// The readout is a control-plane record: it must round-trip through JSON unchanged.
	b, err := json.Marshal(rep)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back DriftReport
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !reflect.DeepEqual(rep, back) {
		t.Errorf("JSON round-trip changed the readout:\n got %+v\nwant %+v", back, rep)
	}
}

// TestDriftEmptyRange proves the clean-boundary case: when nothing moved since the tag
// (an empty tag..HEAD log), the readout is well-formed and empty, not an error.
func TestDriftEmptyRange(t *testing.T) {
	rep := Drift([]byte(""), map[string]bool{"internal/gateway": true}, "v9.0.0", "sha", "head")
	if rep.Moved != 0 || len(rep.Rows) != 0 {
		t.Errorf("moved = %d, len(rows) = %d, want 0/0 for an empty range", rep.Moved, len(rep.Rows))
	}
	if rep.Scanned != 1 || rep.Tag != "v9.0.0" {
		t.Errorf("empty-range readout lost its boundary: scanned=%d tag=%q", rep.Scanned, rep.Tag)
	}
}

// TestDriftSnapshotFixtureTags drives the impure shell over a canned git transcript:
// it must pick the newest MERGED semver tag (skipping a non-semver tag and an older
// semver), bound the log to <tag>..HEAD (the boundary, asserted on the actual log
// args), and join the range revs against the live set.
func TestDriftSnapshotFixtureTags(t *testing.T) {
	var logArgs []string
	run := func(_ context.Context, _ string, args ...string) ([]byte, error) {
		switch args[0] {
		case "rev-parse":
			return []byte("headbee1\n"), nil
		case "tag":
			// git tag --sort=-v:refname --merged HEAD, newest-first; the leading
			// non-semver tag must be skipped and v1.2.3 chosen over v1.1.0.
			return []byte("nightly-2026-07-05\nv1.2.3\nv1.1.0\n"), nil
		case "rev-list":
			// rev-list -n1 <tag> -> the tag's commit sha
			return []byte("tagsha01\n"), nil
		case "ls-files":
			return []byte("internal/gateway/wire.go\x00cmd/fak/main.go\x00internal/modver/modver.go\x00"), nil
		case "log":
			logArgs = append([]string(nil), args...)
			return []byte(driftLogFixture), nil
		}
		t.Fatalf("unexpected git invocation: %v", args)
		return nil, nil
	}

	rep, err := DriftSnapshot(context.Background(), "irrelevant", run)
	if err != nil {
		t.Fatalf("DriftSnapshot: %v", err)
	}

	if rep.Tag != "v1.2.3" {
		t.Errorf("tag = %q, want v1.2.3 (newest merged semver, non-semver skipped)", rep.Tag)
	}
	if rep.TagSHA != "tagsha01" || rep.Head != "headbee1" {
		t.Errorf("boundary = tag_sha %q head %q, want tagsha01 / headbee1", rep.TagSHA, rep.Head)
	}

	// The range boundary must be tag..HEAD, and it must sit BEFORE the "--" pathspec
	// separator (git parses everything after "--" as a path, so a range placed after
	// it would silently be treated as a file and the log would walk all history).
	joined := strings.Join(logArgs, " ")
	if !strings.Contains(joined, "v1.2.3..HEAD") {
		t.Fatalf("log not bounded to the tag: args = %v", logArgs)
	}
	rangeIdx, dashIdx := indexOf(logArgs, "v1.2.3..HEAD"), indexOf(logArgs, "--")
	if dashIdx >= 0 && rangeIdx >= 0 && rangeIdx > dashIdx {
		t.Errorf("range selector v1.2.3..HEAD came AFTER the -- separator: %v", logArgs)
	}
	if indexOf(logArgs, "--no-merges") < 0 {
		t.Errorf("log lost --no-merges (rev must count non-merge commits): %v", logArgs)
	}

	// internal/gateway(3) > internal/modver(2) > cmd/fak(1); nothing else is live.
	if rep.Moved != 3 || len(rep.Rows) != 3 {
		t.Fatalf("moved = %d, want 3: %+v", rep.Moved, rep.Rows)
	}
	if rep.Rows[0].Module != "internal/gateway" || rep.Rows[0].RevsSinceTag != 3 {
		t.Errorf("row[0] = %+v, want internal/gateway revs=3", rep.Rows[0])
	}
}

// TestDriftSnapshotNoTag is the conservative case: when no semver tag is merged into
// HEAD, there is no `@latest` boundary, so the readout is empty (Head preserved) and
// the log is never even run — a false "everything drifted" would red a gate wrongly.
func TestDriftSnapshotNoTag(t *testing.T) {
	logCalled := false
	run := func(_ context.Context, _ string, args ...string) ([]byte, error) {
		switch args[0] {
		case "rev-parse":
			return []byte("headbee1\n"), nil
		case "tag":
			return []byte("nightly-2026-07-05\nstable\n"), nil // no semver tag
		case "log":
			logCalled = true
			return []byte(driftLogFixture), nil
		}
		t.Fatalf("unexpected git invocation for the no-tag path: %v", args)
		return nil, nil
	}

	rep, err := DriftSnapshot(context.Background(), "irrelevant", run)
	if err != nil {
		t.Fatalf("DriftSnapshot: %v", err)
	}
	if rep.Tag != "" || rep.Moved != 0 || len(rep.Rows) != 0 {
		t.Errorf("no-tag readout = tag %q moved %d, want empty", rep.Tag, rep.Moved)
	}
	if rep.Head != "headbee1" {
		t.Errorf("head = %q, want headbee1 (preserved even with no tag)", rep.Head)
	}
	if logCalled {
		t.Errorf("log was run despite no @latest boundary — should short-circuit")
	}
}

func indexOf(ss []string, want string) int {
	for i, s := range ss {
		if s == want {
			return i
		}
	}
	return -1
}
