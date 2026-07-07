package eveimport_test

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/eveimport"
)

var update = flag.Bool("update", false, "regenerate the golden import witness artifact")

// fixtureBodyMarker prefixes every message/reasoning body in BOTH fixtures. Its
// absence from an artifact is the byte-level proof that redaction-by-default held:
// no prose from the evidence leaked into the reconstruction, the ledger rows, or the
// operator summary.
const fixtureBodyMarker = "FIXTURE-"

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return data
}

func importNDJSONFixture(t *testing.T, opt eveimport.Options) eveimport.Run {
	t.Helper()
	return eveimport.ImportNDJSON("testdata/eve-session-stream.ndjson",
		readFixture(t, "eve-session-stream.ndjson"), opt)
}

func importOTelFixture(t *testing.T, opt eveimport.Options) eveimport.Run {
	t.Helper()
	return eveimport.ImportOTelSpans("testdata/eve-otel-spans.json",
		readFixture(t, "eve-otel-spans.json"), opt)
}

// TestNDJSONReconstructsSessionTree is the lineage acceptance: the fixture NDJSON
// stream reconstructs the root session, its turns, the subagent child with correct
// parentage, model ids, per-turn token/cache-read counters, tool counts, and the
// failure event — all from framework-owned tags only.
func TestNDJSONReconstructsSessionTree(t *testing.T) {
	r := importNDJSONFixture(t, eveimport.Options{})

	if r.Observation != eveimport.ObservationComplete {
		t.Fatalf("observation = %q, want complete (diagnostics: %+v)", r.Observation, r.Diagnostics)
	}
	if r.Root == nil {
		t.Fatal("no root session reconstructed")
	}
	if r.Root.SessionID != "ses_root" || r.Root.ModelID != "claude-opus-4-fixture" {
		t.Errorf("root = %s model=%s, want ses_root / claude-opus-4-fixture", r.Root.SessionID, r.Root.ModelID)
	}
	if len(r.Root.Turns) != 2 {
		t.Fatalf("root turns = %d, want 2", len(r.Root.Turns))
	}
	if got := r.Root.Turns[0].Usage; got != (eveimport.Usage{PromptTokens: 812, CompletionTokens: 94, CacheReadTokens: 512}) {
		t.Errorf("root turn 1 usage = %+v", got)
	}
	if r.Root.ToolCalls != 2 {
		t.Errorf("root tool calls = %d, want 2", r.Root.ToolCalls)
	}
	if len(r.Root.Children) != 1 {
		t.Fatalf("root children = %d, want 1 subagent", len(r.Root.Children))
	}
	sub := r.Root.Children[0]
	if sub.SessionID != "ses_sub1" || sub.ParentID != "ses_root" {
		t.Errorf("subagent lineage = %s (parent %s), want ses_sub1 under ses_root", sub.SessionID, sub.ParentID)
	}
	if sub.ModelID != "claude-haiku-fixture" {
		t.Errorf("subagent model = %q", sub.ModelID)
	}
	if len(sub.Failures) != 1 || sub.Failures[0].Reason != "tool grep: exit status 2" {
		t.Errorf("subagent failures = %+v, want the fixture grep failure", sub.Failures)
	}
	if got := sub.Usage; got != (eveimport.Usage{PromptTokens: 301, CompletionTokens: 57, CacheReadTokens: 128}) {
		t.Errorf("subagent usage = %+v", got)
	}
}

// TestRedactionDefaultDropsBodies proves the default posture on BOTH wires: every
// message/reasoning body is redacted (text dropped, sha256+bytes witness kept), and
// no fixture prose survives anywhere in the marshaled artifact, the ledger rows, or
// the summary.
func TestRedactionDefaultDropsBodies(t *testing.T) {
	for name, r := range map[string]eveimport.Run{
		"ndjson": importNDJSONFixture(t, eveimport.Options{}),
		"otel":   importOTelFixture(t, eveimport.Options{}),
	} {
		t.Run(name, func(t *testing.T) {
			if r.BodiesIncluded {
				t.Error("bodies_included = true under default options")
			}
			bodies := 0
			var walk func(s *eveimport.Session)
			walk = func(s *eveimport.Session) {
				for _, turn := range s.Turns {
					for _, b := range turn.Bodies {
						bodies++
						if !b.Redacted {
							t.Errorf("session %s turn %d: body not redacted by default", s.SessionID, turn.Index)
						}
						if b.Text != "" {
							t.Errorf("session %s turn %d: body text carried despite redaction", s.SessionID, turn.Index)
						}
						if b.SHA256 == "" || b.Bytes == 0 {
							t.Errorf("session %s turn %d: redacted body lost its sha256/bytes witness", s.SessionID, turn.Index)
						}
					}
				}
				for _, c := range s.Children {
					walk(c)
				}
			}
			walk(r.Root)
			if bodies == 0 {
				t.Fatal("fixture reconstructed zero bodies; the redaction assertions did not run")
			}
			artifact := mustJSON(t, struct {
				Run     eveimport.Run         `json:"run"`
				Ledger  []eveimport.LedgerRow `json:"ledger_rows"`
				Summary string                `json:"summary"`
			}{r, eveimport.JoinLedger(r), eveimport.Summary(r)})
			if bytes.Contains(artifact, []byte(fixtureBodyMarker)) {
				t.Errorf("fixture prose leaked into the default artifact:\n%s", artifact)
			}
		})
	}
}

// TestIncludeBodiesIsExplicitOptIn is the other half of the redaction contract: only
// an explicit Options.IncludeBodies (a fixture/test decision, never a default)
// carries body text, and the artifact says so via bodies_included.
func TestIncludeBodiesIsExplicitOptIn(t *testing.T) {
	r := importNDJSONFixture(t, eveimport.Options{IncludeBodies: true})
	if !r.BodiesIncluded {
		t.Error("bodies_included = false despite explicit opt-in")
	}
	found := false
	for _, turn := range r.Root.Turns {
		for _, b := range turn.Bodies {
			if b.Redacted {
				t.Errorf("turn %d: body still redacted under explicit opt-in", turn.Index)
			}
			if strings.HasPrefix(b.Text, fixtureBodyMarker) {
				found = true
			}
		}
	}
	if !found {
		t.Error("opt-in carried no fixture body text")
	}
}

// TestOTelMissingTagIsPartialObservation is the partial-tag acceptance: the fixture
// span set's turn 2 carries no usage counters, which must degrade to a MISSING_TAG
// diagnostic and a "partial" observation — never a fake complete — while turn 1's
// counters still import.
func TestOTelMissingTagIsPartialObservation(t *testing.T) {
	r := importOTelFixture(t, eveimport.Options{})

	if r.Observation != eveimport.ObservationPartial {
		t.Fatalf("observation = %q, want partial (diagnostics: %+v)", r.Observation, r.Diagnostics)
	}
	foundMissing := false
	for _, d := range r.Diagnostics {
		if d.Code == eveimport.DiagMissingTag && d.SessionID == "ses_otel_root" && d.Turn == 2 {
			foundMissing = true
		}
	}
	if !foundMissing {
		t.Errorf("no MISSING_TAG diagnostic for ses_otel_root turn 2: %+v", r.Diagnostics)
	}
	if r.Root == nil || r.Root.SessionID != "ses_otel_root" {
		t.Fatalf("root = %+v, want ses_otel_root", r.Root)
	}
	if got := r.Root.Turns[0].Usage; got != (eveimport.Usage{PromptTokens: 640, CompletionTokens: 88, CacheReadTokens: 256}) {
		t.Errorf("otel turn 1 usage = %+v (the $eve.-prefixed counters must import)", got)
	}
	if len(r.Root.Turns) != 2 || r.Root.Turns[1].UsageObserved {
		t.Errorf("turn 2 must exist with usage_observed=false, got %+v", r.Root.Turns)
	}
	if len(r.Root.Children) != 1 || len(r.Root.Children[0].Failures) != 1 {
		t.Fatalf("subagent failure event not reconstructed: %+v", r.Root.Children)
	}
	if got := r.Root.Children[0].Failures[0].Reason; got != "subagent turn aborted: tool timeout" {
		t.Errorf("failure reason = %q", got)
	}
}

// TestNoRootIsIndeterminate: evidence from which no session is reconstructible must
// return the INDETERMINATE verdict, mint NO ledger rows, and say so in the summary —
// the "not a fake success" clause of #2606.
func TestNoRootIsIndeterminate(t *testing.T) {
	r := eveimport.ImportNDJSON("testdata/none.ndjson", []byte("not json at all\n"), eveimport.Options{})
	if r.Observation != eveimport.ObservationIndeterminate {
		t.Fatalf("observation = %q, want INDETERMINATE", r.Observation)
	}
	if rows := eveimport.JoinLedger(r); rows != nil {
		t.Errorf("INDETERMINATE run minted %d ledger rows", len(rows))
	}
	if s := eveimport.Summary(r); !strings.Contains(s, eveimport.ObservationIndeterminate) {
		t.Errorf("summary hides the INDETERMINATE verdict: %q", s)
	}

	// An unreadable span export degrades the same way.
	r = eveimport.ImportOTelSpans("testdata/none.json", []byte("{broken"), eveimport.Options{})
	if r.Observation != eveimport.ObservationIndeterminate {
		t.Fatalf("otel observation = %q, want INDETERMINATE", r.Observation)
	}
}

// TestJoinLedgerRows pins the join shape: root-first depth-first rows carrying ids,
// lineage, model, counters, and failed-step counts — and structurally no prose.
func TestJoinLedgerRows(t *testing.T) {
	r := importNDJSONFixture(t, eveimport.Options{})
	rows := eveimport.JoinLedger(r)
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want 2 (root + subagent)", len(rows))
	}
	root, sub := rows[0], rows[1]
	if root.SessionID != "ses_root" || root.Depth != 0 || root.RootSessionID != "ses_root" {
		t.Errorf("root row = %+v", root)
	}
	if root.Turns != 2 || root.Subagents != 1 || root.ToolCalls != 2 || root.FailedSteps != 0 {
		t.Errorf("root row counters = %+v", root)
	}
	if root.PromptTokens != 1992 || root.CompletionTokens != 327 || root.CacheReadTokens != 1536 {
		t.Errorf("root row tokens = %+v", root)
	}
	if sub.SessionID != "ses_sub1" || sub.ParentSessionID != "ses_root" || sub.RootSessionID != "ses_root" || sub.Depth != 1 {
		t.Errorf("subagent row lineage = %+v", sub)
	}
	if sub.FailedSteps != 1 || sub.ToolCalls != 1 || sub.CacheReadTokens != 128 {
		t.Errorf("subagent row counters = %+v", sub)
	}
	for _, row := range rows {
		if row.Schema != eveimport.LedgerRowSchema {
			t.Errorf("row schema = %q", row.Schema)
		}
		if row.EvidencePath != "testdata/eve-session-stream.ndjson" || row.EvidenceKind != "eve-ndjson" {
			t.Errorf("row evidence = %s:%s", row.EvidenceKind, row.EvidencePath)
		}
	}
}

// goldenArtifact is the pinned witness: both wires' full reconstruction, their ledger
// rows, and their operator summaries, under DEFAULT (redacting) options.
type goldenArtifact struct {
	NDJSON goldenArm `json:"ndjson"`
	OTel   goldenArm `json:"otel_spans"`
}

type goldenArm struct {
	Run     eveimport.Run         `json:"run"`
	Ledger  []eveimport.LedgerRow `json:"ledger_rows"`
	Summary string                `json:"summary"`
}

// TestGoldenImportWitness pins the whole deterministic reconstruction byte-for-byte —
// tree, counters, diagnostics, ledger rows, summaries — and re-proves, at the byte
// level, that no fixture prose is anywhere in the default artifact.
func TestGoldenImportWitness(t *testing.T) {
	nd := importNDJSONFixture(t, eveimport.Options{})
	ot := importOTelFixture(t, eveimport.Options{})
	got := mustJSON(t, goldenArtifact{
		NDJSON: goldenArm{Run: nd, Ledger: eveimport.JoinLedger(nd), Summary: eveimport.Summary(nd)},
		OTel:   goldenArm{Run: ot, Ledger: eveimport.JoinLedger(ot), Summary: eveimport.Summary(ot)},
	})
	if bytes.Contains(got, []byte(fixtureBodyMarker)) {
		t.Fatalf("fixture prose leaked into the golden artifact:\n%s", got)
	}

	goldenPath := filepath.Join("testdata", "import-witness.golden.json")
	if *update {
		if err := os.WriteFile(goldenPath, got, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote golden %s", goldenPath)
		return
	}
	want, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden (run `go test -run Golden -update` to create it): %v", err)
	}
	if !bytes.Equal(bytes.TrimSpace(got), bytes.TrimSpace(want)) {
		t.Errorf("import artifact drifted from golden %s.\n--- got ---\n%s", goldenPath, got)
	}
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return b
}
