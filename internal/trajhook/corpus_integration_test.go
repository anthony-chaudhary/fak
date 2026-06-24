package trajhook

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// sampleCorpus is the shipped demo corpus the `fak traj` toolkit and the
// trajectory-garden skill run against. This path is relative to this test file
// (internal/trajhook -> repo root -> examples/trajectory).
const sampleCorpus = "../../examples/trajectory/sample-corpus.jsonl"

// TestSampleCorpusGardening is the END-TO-END proof of the gardening flow: load the
// shipped corpus exactly as `fak traj score`/`gc` do (trajectory.ImportFrom), run the
// default scorers, and assert the three signals the corpus was built to surface:
//
//   - a near-DUPLICATE query (sess-a:3 re-asks sess-a:1's KB lookup),
//   - a COST outlier (sess-b:2's 9800-token summarize),
//   - a high-DENY-RATE trace (sess-c refused all 3 destructive SQL turns).
//
// If this passes, the data file is well-formed AND the whole pipeline the CLI drives
// works, without needing the (gateway-coupled) binary to link.
func TestSampleCorpusGardening(t *testing.T) {
	f, err := os.Open(sampleCorpus)
	if err != nil {
		t.Skipf("sample corpus not found (run from repo root): %v", err)
	}
	defer f.Close()
	rec, n, err := trajectory.ImportFrom(f)
	if err != nil || n == 0 {
		t.Fatalf("import sample corpus: n=%d err=%v", n, err)
	}

	findings := Default().Run(rec.Turns())
	got := map[string]bool{}
	for _, fnd := range findings {
		got[fnd.Label] = true
	}
	for _, want := range []string{"duplicate_query", "cost_outlier", "high_deny_rate"} {
		if !got[want] {
			t.Errorf("expected a %q finding from the sample corpus; got labels %v", want, keys(got))
		}
	}

	// The destructive-SQL trace must be the worst high-deny-rate finding (3/3 refused).
	var denyFinding *Finding
	for i := range findings {
		if findings[i].Label == "high_deny_rate" {
			denyFinding = &findings[i]
			break
		}
	}
	if denyFinding == nil || denyFinding.TraceID != "sess-c" {
		t.Fatalf("high-deny-rate trace = %v, want sess-c", denyFinding)
	}
}

// TestSampleCorpusGCProposes — the gc flow proposes pruning sess-a:3 (the later
// KB-lookup duplicate), keeping sess-a:1, and never deletes anything.
func TestSampleCorpusGCProposes(t *testing.T) {
	f, err := os.Open(sampleCorpus)
	if err != nil {
		t.Skipf("sample corpus not found: %v", err)
	}
	defer f.Close()
	rec, _, err := trajectory.ImportFrom(f)
	if err != nil {
		t.Fatalf("import: %v", err)
	}

	reg := NewRegistry()
	reg.Register("duplicate_query", DuplicateQuery(DefaultDuplicateThreshold))
	findings := reg.Run(rec.Turns())

	var prunedA3 bool
	for _, fnd := range findings {
		if fnd.TraceID == "sess-a" && fnd.Seq == 3 && fnd.Related == "sess-a:1" {
			prunedA3 = true
		}
	}
	if !prunedA3 {
		t.Fatalf("gc did not propose pruning sess-a:3 as a duplicate of sess-a:1; findings=%+v", findings)
	}
}

func keys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
