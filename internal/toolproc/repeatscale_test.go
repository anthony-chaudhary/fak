package toolproc

// repeatscale_test.go — the SCALE proof for #5120's replay bullet: ClassifyRepeats
// must fold the real top-100 rollout workload (~10^5 records over ~10^4 identities)
// in one pass, not once per group.
//
// The fold used to reproject each group's per-observation output sizes by
// re-scanning the WHOLE record stream and re-running Normalize on every record —
// O(groups x records). On the captured corpus that is ~10^9 Normalize calls, so the
// replay the issue asks for never terminates; #5410 deleted the replay witness
// rather than run it. The sizes are now accumulated inline with the group's
// observation times, so the two tests below pin both halves of that change:
//
//   - SCALE: a workload the old shape could not finish now folds well inside a
//     deadline, so the captured replay is actually runnable.
//   - EQUIVALENCE: the accumulated sizes stay aligned to the observation times, so
//     the freshness-bounded saving (the one scoring that reads bytes positionally,
//     out of stream order) is unchanged.

import (
	"fmt"
	"testing"
	"time"
)

// TestClassifyRepeatsFoldsRolloutScaleWorkload replays a workload with the shape of
// the captured top-100 corpus — tens of thousands of records spread over tens of
// thousands of distinct identities, which is the worst case for a per-group
// re-scan — and requires the fold to finish inside a deadline the O(groups x
// records) shape could not meet. At this size the old fold paid ~8x10^8 Normalize
// calls; the single-pass fold pays 4x10^4.
func TestClassifyRepeatsFoldsRolloutScaleWorkload(t *testing.T) {
	if testing.Short() {
		t.Skip("scale fold: skipped under -short")
	}
	const (
		groups   = 20000
		perGroup = 2
		deadline = 30 * time.Second
	)
	recs := make([]CallRecord, 0, groups*perGroup)
	for i := 0; i < groups; i++ {
		path := fmt.Sprintf("C:/repo/skills/skill-%05d/SKILL.md", i)
		for r := 0; r < perGroup; r++ {
			recs = append(recs, CallRecord{
				Tool:        "shell_command",
				Raw:         "cat " + path,
				AtMS:        int64(i*perGroup + r),
				OutputBytes: 1024,
				Digest:      fmt.Sprintf("sha256:%05d", i),
			})
		}
	}

	start := time.Now()
	rep := ClassifyRepeats(recs, RepeatConfig{})
	elapsed := time.Since(start)

	if elapsed > deadline {
		t.Fatalf("ClassifyRepeats took %v over %d records / %d identities, past the %v deadline: the fold is re-scanning the record stream per group again", elapsed, len(recs), groups, deadline)
	}
	if rep.Totals.Records != groups*perGroup {
		t.Errorf("Totals.Records = %d, want %d", rep.Totals.Records, groups*perGroup)
	}
	if rep.Totals.Groups != groups {
		t.Errorf("Totals.Groups = %d, want %d", rep.Totals.Groups, groups)
	}
	// Every identity is an immutable read observed twice, so exactly one repeat per
	// group is avoidable and it carries that observation's bytes.
	if want := groups * (perGroup - 1); rep.Totals.AvoidableSpawns != want {
		t.Errorf("Totals.AvoidableSpawns = %d, want %d", rep.Totals.AvoidableSpawns, want)
	}
	if want := int64(groups*(perGroup-1)) * 1024; rep.Totals.AvoidableInputBytes != want {
		t.Errorf("Totals.AvoidableInputBytes = %d, want %d", rep.Totals.AvoidableInputBytes, want)
	}
}

// TestFreshnessSavingKeepsBytesAlignedToObservations pins the alignment the
// single-pass accumulation has to preserve. The freshness-bounded scoring walks a
// group's observations in TIME order while indexing the sizes by their STREAM
// position, so a misalignment silently attributes the wrong tool-result size to a
// coalesced poll. Here two identities interleave and the records arrive out of time
// order with distinct sizes, so only a correctly aligned sizes slice reproduces the
// expected saving.
func TestFreshnessSavingKeepsBytesAlignedToObservations(t *testing.T) {
	cfg := RepeatConfig{DefaultFreshnessMS: 1000}
	// Interleaved, out of time order. For `git status` the time-ordered
	// observations are 0(9000) 500(11) 900(7) 2500(13): 500 and 900 fall inside the
	// 1000ms window of the fetch at 0 and coalesce; 2500 is past it and is a fresh
	// fetch. So the avoidable bytes are exactly 11+7 = 18 — the sizes of the two
	// coalesced polls, not of their stream neighbours.
	recs := []CallRecord{
		{Tool: "shell_command", Raw: "git status --short --branch", AtMS: 0, OutputBytes: 9000},
		{Tool: "shell_command", Raw: "git rev-parse HEAD", AtMS: 100, OutputBytes: 41},
		{Tool: "shell_command", Raw: "git status --short --branch", AtMS: 2500, OutputBytes: 13},
		{Tool: "shell_command", Raw: "git rev-parse HEAD", AtMS: 300, OutputBytes: 41},
		{Tool: "shell_command", Raw: "git status --short --branch", AtMS: 500, OutputBytes: 11},
		{Tool: "shell_command", Raw: "git status --short --branch", AtMS: 900, OutputBytes: 7},
	}
	rep := ClassifyRepeats(recs, cfg)

	var status RepeatGroup
	for _, g := range rep.Groups {
		if g.Path == "" && g.Count == 4 {
			status = g
		}
	}
	if status.Count != 4 {
		t.Fatalf("expected the 4-observation status group in %d groups", len(rep.Groups))
	}
	if status.Reuse != ReuseFreshnessBounded {
		t.Fatalf("status group Reuse = %v, want %v", status.Reuse, ReuseFreshnessBounded)
	}
	if status.AvoidableSpawns != 2 {
		t.Errorf("AvoidableSpawns = %d, want 2 (the two polls inside the freshness window)", status.AvoidableSpawns)
	}
	if status.AvoidableInputBytes != 18 {
		t.Errorf("AvoidableInputBytes = %d, want 18 (11+7, the coalesced observations' own sizes)", status.AvoidableInputBytes)
	}
}
