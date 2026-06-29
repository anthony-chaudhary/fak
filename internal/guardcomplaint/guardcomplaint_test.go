package guardcomplaint

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
)

func sampleComplaint() Complaint {
	return Complaint{
		Kind:      "false-positive",
		Reason:    "FILE_ADMISSION",
		Tool:      "Bash",
		Summary:   "blocked committing a curated docs/notes file",
		Rationale: "The path is a genuine curated note in docs/notes/, not operator-private telemetry; the marker heuristic misfired.",
	}
}

func TestNormalizeKind(t *testing.T) {
	if k, err := NormalizeKind(""); err != nil || k != DefaultKind {
		t.Fatalf("empty kind => (%q,%v), want default %q", k, err, DefaultKind)
	}
	if k, err := NormalizeKind("  Over-Broad "); err != nil || k != "over-broad" {
		t.Fatalf("over-broad normalize => (%q,%v)", k, err)
	}
	if _, err := NormalizeKind("nonsense"); err == nil {
		t.Fatal("unknown kind must error")
	}
}

func TestKeyStableAndDiscriminating(t *testing.T) {
	c := sampleComplaint()
	want := "guard-complaint/false-positive/file-admission/bash/blocked-committing-a-curated-docs-notes-file"
	if got := c.Key(); got != want {
		t.Fatalf("key = %q, want %q", got, want)
	}
	// Trivial wording / case drift in fields that are slugged identically must NOT split.
	c2 := c
	c2.Reason = "file_admission"
	if c2.Key() != c.Key() {
		t.Fatalf("reason case drift split the key: %q vs %q", c2.Key(), c.Key())
	}
	// A genuinely different summary MUST split into its own issue.
	c3 := c
	c3.Summary = "refused a totally different write"
	if c3.Key() == c.Key() {
		t.Fatal("different summary must yield a different key")
	}
	// Empty reason/tool fold to sentinels, not a panic or collision with a real token.
	c4 := Complaint{Kind: "latency", Summary: "slow gate"}
	if got := c4.Key(); got != "guard-complaint/latency/none/any/slow-gate" {
		t.Fatalf("sentinel key = %q", got)
	}
}

func TestBodyMarkerRoundTripAndOccurrences(t *testing.T) {
	c := sampleComplaint()
	body := c.Body(1)
	if MarkerKey(body) != c.Key() {
		t.Fatalf("marker key %q != complaint key %q", MarkerKey(body), c.Key())
	}
	if occurrencesOf(body) != 1 {
		t.Fatalf("fresh body occurrences = %d, want 1", occurrencesOf(body))
	}
	body5 := c.Body(5)
	if occurrencesOf(body5) != 5 {
		t.Fatalf("occurrences read-back = %d, want 5", occurrencesOf(body5))
	}
	if !strings.Contains(body, c.Rationale) {
		t.Fatal("body must carry the agent's rationale verbatim")
	}
	if !strings.Contains(body, "FILE_ADMISSION") {
		t.Fatal("body must name the appealed reason")
	}
}

func TestBodyEvidenceBlock(t *testing.T) {
	c := sampleComplaint()
	// No evidence => honest disclaimer, not a fabricated witness.
	if !strings.Contains(c.Body(1), "No journal verdict attached") {
		t.Fatal("evidence-less body must disclose the missing witness")
	}
	c.Evidence = &Evidence{
		Source: "journal", JournalPath: "/x/guard-audit.jsonl",
		Verdict: "DENY", Reason: "FILE_ADMISSION", Tool: "Bash", By: "ifc-sink",
		TraceID: "trace-7", ArgsDigest: "sha256:abc", Seq: 42,
	}
	body := c.Body(1)
	for _, want := range []string{"DENY", "ifc-sink", "trace-7", "sha256:abc", "guard-audit.jsonl"} {
		if !strings.Contains(body, want) {
			t.Fatalf("evidence body missing %q", want)
		}
	}
}

func TestBuildPlanCreateThenUpdateEscalates(t *testing.T) {
	c := sampleComplaint()

	// No existing issue => create at occurrences 1.
	row := BuildPlan(c, nil)
	if row.Action != "create" || row.Number != nil || row.Occurrences != 1 {
		t.Fatalf("first file = %+v, want create/nil/1", row)
	}

	// Re-file with the prior body present => update in place, occurrences bumped.
	existing := []dogfoodissues.Issue{{
		Number: 314, State: "OPEN", Title: c.Title(), Body: c.Body(1),
	}}
	row2 := BuildPlan(c, existing)
	if row2.Action != "update" {
		t.Fatalf("re-file action = %q, want update", row2.Action)
	}
	if row2.Number == nil || *row2.Number != 314 {
		t.Fatalf("re-file number = %v, want 314", row2.Number)
	}
	if row2.Occurrences != 2 {
		t.Fatalf("re-file occurrences = %d, want 2", row2.Occurrences)
	}
	if occurrencesOf(row2.Body) != 2 {
		t.Fatalf("re-file body occurrences = %d, want 2", occurrencesOf(row2.Body))
	}

	// An unrelated existing issue must not match.
	other := []dogfoodissues.Issue{{Number: 9, Body: "<!-- fak-guard-complaint-key: guard-complaint/other/none/any/x -->"}}
	if BuildPlan(c, other).Action != "create" {
		t.Fatal("non-matching key must still be a create")
	}
}

func TestSyncCreateUsesInjectedRunnerAndLabels(t *testing.T) {
	c := sampleComplaint()
	row := BuildPlan(c, nil)
	var gotArgs []string
	runner := func(args []string) (string, string, bool) {
		gotArgs = args
		return "https://example/issues/1", "", true
	}
	out := Sync(row, "owner/repo", []string{Label}, runner)
	if !out.OK || out.Action != "create" {
		t.Fatalf("sync result = %+v", out)
	}
	joined := strings.Join(gotArgs, " ")
	for _, want := range []string{"issue create", "--title", "--label " + Label, "--repo owner/repo"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("gh args %q missing %q", joined, want)
		}
	}
}

func TestSyncUpdateEditsByNumber(t *testing.T) {
	c := sampleComplaint()
	existing := []dogfoodissues.Issue{{Number: 88, State: "OPEN", Body: c.Body(3)}}
	row := BuildPlan(c, existing)
	var gotArgs []string
	runner := func(args []string) (string, string, bool) {
		gotArgs = args
		return "", "", true
	}
	if out := Sync(row, "", nil, runner); !out.OK || out.Action != "update" {
		t.Fatalf("update sync = %+v", out)
	}
	joined := strings.Join(gotArgs, " ")
	if !strings.Contains(joined, "issue edit 88") {
		t.Fatalf("update must edit by number; got %q", joined)
	}
}

func TestLatestDenialFiltersAndPicksMostRecent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "guard-audit.jsonl")
	lines := []string{
		`{"seq":1,"ts_unix_nano":100,"kind":"DECIDE","tool":"Read","verdict":"ALLOW"}`,
		`{"seq":2,"ts_unix_nano":200,"kind":"DENY","tool":"Bash","verdict":"DENY","reason":"FILE_ADMISSION","by":"ifc-sink","trace_id":"t-old","args_digest":"sha256:old"}`,
		`{"seq":3,"ts_unix_nano":300,"kind":"DENY","tool":"Write","verdict":"DENY","reason":"OUT_OF_TREE_WRITE","by":"monitor","trace_id":"t-write"}`,
		`{"seq":4,"ts_unix_nano":400,"kind":"DENY","tool":"Bash","verdict":"DENY","reason":"FILE_ADMISSION","by":"ifc-sink","trace_id":"t-new","args_digest":"sha256:new"}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	paths := []string{path}

	// Unfiltered => most recent denial overall (seq 4).
	if e := LatestDenial(paths, "", ""); e == nil || e.TraceID != "t-new" {
		t.Fatalf("unfiltered latest = %+v, want trace t-new", e)
	}
	// Filter by reason picks the matching reason's most recent (still seq 4).
	if e := LatestDenial(paths, "file_admission", ""); e == nil || e.Seq != 4 || e.ArgsDigest != "sha256:new" {
		t.Fatalf("reason-filtered latest = %+v", e)
	}
	// Filter by a different reason picks that row.
	if e := LatestDenial(paths, "OUT_OF_TREE_WRITE", ""); e == nil || e.TraceID != "t-write" {
		t.Fatalf("out-of-tree filter = %+v", e)
	}
	// Filter by tool.
	if e := LatestDenial(paths, "", "Write"); e == nil || e.Tool != "Write" {
		t.Fatalf("tool filter = %+v", e)
	}
	// No match => nil, an honest no-witness.
	if e := LatestDenial(paths, "NO_SUCH_REASON", ""); e != nil {
		t.Fatalf("no-match must be nil, got %+v", e)
	}
}

func TestRenderShowsActionAndDryRunHint(t *testing.T) {
	c := sampleComplaint()
	r := Result{Schema: Schema, Mode: "dry-run", Planned: []PlanRow{BuildPlan(c, nil)}}
	out := Render(r)
	if !strings.Contains(out, "[create]") || !strings.Contains(out, "occurrences=1") {
		t.Fatalf("render missing plan line: %s", out)
	}
	if !strings.Contains(out, "pass --live") {
		t.Fatalf("dry-run render must hint --live: %s", out)
	}
}
