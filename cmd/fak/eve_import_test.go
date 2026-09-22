package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/eveimport"
)

// eveImportJSONEnvelope is the typed shape of `fak eve import --json` stdout. The
// test unmarshals it and asserts fields rather than string-matching the blob, so a
// formatting change cannot fake a pass.
type eveImportJSONEnvelope struct {
	Run        eveimport.Run         `json:"run"`
	LedgerRows []eveimport.LedgerRow `json:"ledger_rows"`
}

// writeEveImportFixture writes bytes to a named file inside a fresh temp dir and
// returns the full path. Fixtures are inlined here (never read from
// internal/eveimport/testdata) so the test is self-contained at cmd/fak.
func writeEveImportFixture(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// runEveImportCase calls the handler directly, the way sibling CLI tests do, and
// returns the exit code plus the captured streams.
func runEveImportCase(t *testing.T, argv ...string) (int, string, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := runEveImport(&stdout, &stderr, argv)
	return code, stdout.String(), stderr.String()
}

// eveImportNDJSONFixture is a minimal but valid two-session NDJSON stream: a root
// session with one subagent child, usage counters, a tool call, and a failure.
const eveImportNDJSONFixture = `{"type":"session.start","session_id":"ses_root","model_id":"model-root-fixture"}
{"type":"turn.start","session_id":"ses_root","turn":1}
{"type":"tool.call","session_id":"ses_root","turn":1,"tool":"search"}
{"type":"usage","session_id":"ses_root","turn":1,"prompt_tokens":100,"completion_tokens":20,"cache_read_tokens":40}
{"type":"session.start","session_id":"ses_sub","parent_session_id":"ses_root","model_id":"model-sub-fixture"}
{"type":"turn.start","session_id":"ses_sub","turn":1}
{"type":"tool.call","session_id":"ses_sub","turn":1,"tool":"grep"}
{"type":"usage","session_id":"ses_sub","turn":1,"prompt_tokens":30,"completion_tokens":5,"cache_read_tokens":10}
{"type":"failure","session_id":"ses_sub","turn":1,"reason":"tool grep: exit status 2"}
{"type":"session.end","session_id":"ses_sub"}
{"type":"session.end","session_id":"ses_root"}
`

// eveImportOTelFixture is a minimal span export with a root session span and a
// subagent span, using the $eve.-prefixed attribute namespace.
const eveImportOTelFixture = `{
  "spans": [
    {
      "name": "eve.session",
      "span_id": "sp-root",
      "status": {"code": "OK", "message": ""},
      "attributes": {"eve.session.id": "ses_otel_root", "eve.model.id": "model-otel-fixture"}
    },
    {
      "name": "eve.turn",
      "span_id": "sp-turn1",
      "parent_span_id": "sp-root",
      "status": {"code": "OK", "message": ""},
      "attributes": {
        "$eve.session.id": "ses_otel_root",
        "$eve.turn.index": 1,
        "$eve.usage.prompt_tokens": 50,
        "$eve.usage.completion_tokens": 10,
        "$eve.usage.cache_read_tokens": 20
      }
    },
    {
      "name": "eve.session",
      "span_id": "sp-sub",
      "parent_span_id": "sp-root",
      "status": {"code": "OK", "message": ""},
      "attributes": {
        "eve.session.id": "ses_otel_sub",
        "eve.session.parent_id": "ses_otel_root",
        "eve.model.id": "model-otel-mini"
      }
    },
    {
      "name": "eve.turn",
      "span_id": "sp-sub-turn1",
      "parent_span_id": "sp-sub",
      "status": {"code": "OK", "message": ""},
      "attributes": {
        "eve.session.id": "ses_otel_sub",
        "eve.turn.index": 1,
        "eve.usage.prompt_tokens": 5,
        "eve.usage.completion_tokens": 1,
        "eve.usage.cache_read_tokens": 2
      }
    }
  ]
}
`

// TestEveImportHappyPathNDJSON: a small valid NDJSON stream imports with exit 0,
// the operator summary names the root session and carries non-zero tokens, and
// stderr carries the one-line observation verdict.
func TestEveImportHappyPathNDJSON(t *testing.T) {
	path := writeEveImportFixture(t, "session.ndjson", eveImportNDJSONFixture)
	code, stdout, stderr := runEveImportCase(t, path)
	if code != 0 {
		t.Fatalf("expected exit 0, got %d (stderr: %s)", code, stderr)
	}
	if !strings.Contains(stdout, "ses_root") {
		t.Fatalf("summary must name the root session, got:\n%s", stdout)
	}
	if !strings.Contains(stdout, "prompt=") {
		t.Fatalf("summary must carry token totals, got:\n%s", stdout)
	}
	if !strings.Contains(stderr, "observation=") {
		t.Fatalf("stderr must carry the observation= verdict line, got %q", stderr)
	}
}

// TestEveImportOTelByExtensionInference: with no --kind, a .json file is inferred
// as the OTel wire and the output identifies that wire.
func TestEveImportOTelByExtensionInference(t *testing.T) {
	path := writeEveImportFixture(t, "spans.json", eveImportOTelFixture)
	code, stdout, stderr := runEveImportCase(t, path)
	if code != 0 {
		t.Fatalf("expected exit 0, got %d (stderr: %s)", code, stderr)
	}
	combined := stdout + stderr
	if !strings.Contains(combined, "eve-otel-spans") {
		t.Fatalf("output must identify the OTel wire, got:\nstdout=%s\nstderr=%s", stdout, stderr)
	}
	if !strings.Contains(stdout, "ses_otel_root") {
		t.Fatalf("OTel summary must name the root session, got:\n%s", stdout)
	}
}

// TestEveImportJSONShape: --json emits a document whose top-level keys are
// `run` and `ledger_rows`; the root ledger row carries depth 0 and the subagent
// child depth 1 with the root session id.
func TestEveImportJSONShape(t *testing.T) {
	path := writeEveImportFixture(t, "session.ndjson", eveImportNDJSONFixture)
	code, stdout, stderr := runEveImportCase(t, "--json", path)
	if code != 0 {
		t.Fatalf("expected exit 0, got %d (stderr: %s)", code, stderr)
	}
	if !strings.Contains(stdout, `"run"`) || !strings.Contains(stdout, `"ledger_rows"`) {
		t.Fatalf("--json must expose top-level run and ledger_rows, got:\n%s", stdout)
	}
	var env eveImportJSONEnvelope
	if err := json.Unmarshal([]byte(stdout), &env); err != nil {
		t.Fatalf("stdout is not the typed JSON envelope: %v\n%s", err, stdout)
	}
	if env.Run.Root == nil || env.Run.Root.SessionID != "ses_root" {
		t.Fatalf("run.root must be ses_root, got %+v", env.Run.Root)
	}
	var root, sub *eveimport.LedgerRow
	for i := range env.LedgerRows {
		switch env.LedgerRows[i].SessionID {
		case "ses_root":
			root = &env.LedgerRows[i]
		case "ses_sub":
			sub = &env.LedgerRows[i]
		}
	}
	if root == nil || sub == nil {
		t.Fatalf("expected ledger rows for ses_root and ses_sub, got %+v", env.LedgerRows)
	}
	if root.Depth != 0 || root.RootSessionID != "ses_root" {
		t.Fatalf("root row must be depth 0 rooted at ses_root, got %+v", *root)
	}
	if sub.Depth != 1 || sub.RootSessionID != "ses_root" || sub.ParentSessionID != "ses_root" {
		t.Fatalf("subagent row must be depth 1 under ses_root, got %+v", *sub)
	}
}

// eveImportSentinel NDJSON embeds a unique body string whose presence/absence is
// the byte-level witness for the redaction contract.
const eveImportSentinel = "TOP-SECRET-SENTINEL-9f3a"

const eveImportSentinelNDJSON = `{"type":"session.start","session_id":"ses_secret","model_id":"model-secret"}
{"type":"turn.start","session_id":"ses_secret","turn":1}
{"type":"message","session_id":"ses_secret","turn":1,"role":"assistant","body":"` + eveImportSentinel + ` must never reach the default artifact"}
{"type":"reasoning","session_id":"ses_secret","turn":1,"body":"` + eveImportSentinel + ` private reasoning"}
{"type":"usage","session_id":"ses_secret","turn":1,"prompt_tokens":7,"completion_tokens":3,"cache_read_tokens":1}
{"type":"session.end","session_id":"ses_secret"}
`

// TestEveImportRedactsBodiesByDefault: the default artifact on both the text and
// JSON surfaces drops the body text but keeps a redacted witness (sha256 + bytes).
func TestEveImportRedactsBodiesByDefault(t *testing.T) {
	path := writeEveImportFixture(t, "secret.ndjson", eveImportSentinelNDJSON)

	_, textOut, _ := runEveImportCase(t, path)
	if strings.Contains(textOut, eveImportSentinel) {
		t.Fatalf("default text summary leaked the body sentinel:\n%s", textOut)
	}

	code, jsonOut, _ := runEveImportCase(t, "--json", path)
	if code != 0 {
		t.Fatalf("expected exit 0, got %d", code)
	}
	if strings.Contains(jsonOut, eveImportSentinel) {
		t.Fatalf("default --json artifact leaked the body sentinel:\n%s", jsonOut)
	}
	var env eveImportJSONEnvelope
	if err := json.Unmarshal([]byte(jsonOut), &env); err != nil {
		t.Fatalf("stdout is not the typed JSON envelope: %v", err)
	}
	if env.Run.BodiesIncluded {
		t.Fatal("bodies_included must be false by default")
	}
	if env.Run.Root == nil || len(env.Run.Root.Turns) == 0 {
		t.Fatalf("expected a root session with turns, got %+v", env.Run.Root)
	}
	bodies := 0
	for _, turn := range env.Run.Root.Turns {
		for _, b := range turn.Bodies {
			bodies++
			if !b.Redacted || b.Text != "" {
				t.Fatalf("default body must be redacted with no text, got %+v", b)
			}
			if b.SHA256 == "" || b.Bytes <= 0 {
				t.Fatalf("redacted body must keep a sha256 and positive byte count, got %+v", b)
			}
		}
	}
	if bodies == 0 {
		t.Fatal("the redaction assertions never ran: no bodies reconstructed")
	}
}

// TestEveImportIncludeBodiesOptIn: --include-bodies is the explicit opt-in that
// carries the body text and flags bodies_included.
func TestEveImportIncludeBodiesOptIn(t *testing.T) {
	path := writeEveImportFixture(t, "secret.ndjson", eveImportSentinelNDJSON)
	code, stdout, stderr := runEveImportCase(t, "--include-bodies", "--json", path)
	if code != 0 {
		t.Fatalf("expected exit 0, got %d (stderr: %s)", code, stderr)
	}
	if !strings.Contains(stdout, eveImportSentinel) {
		t.Fatalf("--include-bodies must carry the body text, got:\n%s", stdout)
	}
	var env eveImportJSONEnvelope
	if err := json.Unmarshal([]byte(stdout), &env); err != nil {
		t.Fatalf("stdout is not the typed JSON envelope: %v", err)
	}
	if !env.Run.BodiesIncluded {
		t.Fatal("bodies_included must be true under --include-bodies")
	}
	if env.Run.Root == nil || len(env.Run.Root.Turns) == 0 {
		t.Fatalf("expected a root session with turns, got %+v", env.Run.Root)
	}
	// The opt-in must actually CLEAR the redaction witness: at least one body must
	// come back un-redacted carrying its text, not merely be counted as included.
	clear := 0
	for _, turn := range env.Run.Root.Turns {
		for _, b := range turn.Bodies {
			if !b.Redacted && b.Text != "" {
				clear++
			}
		}
	}
	if clear == 0 {
		t.Fatalf("--include-bodies must clear the redaction witness on some body, got %+v", env.Run.Root.Turns)
	}
}

// TestEveImportIndeterminateIsNotAnError: non-JSON garbage reconstructs no root,
// which must be a legible INDETERMINATE verdict at exit 0 - never a fake success
// and never a nonzero error.
func TestEveImportIndeterminateIsNotAnError(t *testing.T) {
	path := writeEveImportFixture(t, "garbage.ndjson", "this is not json\nnor is this\n")
	code, stdout, stderr := runEveImportCase(t, "--json", path)
	if code != 0 {
		t.Fatalf("INDETERMINATE must be exit 0, got %d (stderr: %s)", code, stderr)
	}
	var env eveImportJSONEnvelope
	if err := json.Unmarshal([]byte(stdout), &env); err != nil {
		t.Fatalf("stdout is not the typed JSON envelope: %v\n%s", err, stdout)
	}
	if env.Run.Observation != eveimport.ObservationIndeterminate {
		t.Fatalf("observation = %q, want %q", env.Run.Observation, eveimport.ObservationIndeterminate)
	}
	if !strings.Contains(stdout, eveimport.ObservationIndeterminate) {
		t.Fatalf("output must surface the INDETERMINATE verdict, got:\n%s", stdout)
	}

	// The DEFAULT text path (no --json) must surface the verdict too: the stderr
	// observation= line carries it, and stdout names it in the no-root summary.
	textCode, textOut, textErr := runEveImportCase(t, path)
	if textCode != 0 {
		t.Fatalf("INDETERMINATE text path must be exit 0, got %d (stderr: %s)", textCode, textErr)
	}
	if !strings.Contains(textOut, eveimport.ObservationIndeterminate) {
		t.Fatalf("default text stdout must surface %s, got:\n%s", eveimport.ObservationIndeterminate, textOut)
	}
	if !strings.Contains(textErr, "observation=INDETERMINATE") {
		t.Fatalf("default text stderr must carry observation=INDETERMINATE, got %q", textErr)
	}
}

// TestEveImportKindOverridesExtensionInference: --kind wins over extension
// inference. A valid NDJSON stream mislabelled `.json` is inferred as the OTel
// wire (and honestly degrades to INDETERMINATE), but --kind ndjson reconstructs
// it as the NDJSON wire.
func TestEveImportKindOverridesExtensionInference(t *testing.T) {
	path := writeEveImportFixture(t, "mislabelled.json", eveImportNDJSONFixture)

	// Without --kind: extension inference picks otel, which cannot parse the
	// NDJSON stream and degrades to INDETERMINATE - still exit 0.
	code, stdout, stderr := runEveImportCase(t, "--json", path)
	if code != 0 {
		t.Fatalf("inferred degradation must be exit 0, got %d (stderr: %s)", code, stderr)
	}
	var inferred eveImportJSONEnvelope
	if err := json.Unmarshal([]byte(stdout), &inferred); err != nil {
		t.Fatalf("stdout is not the typed JSON envelope: %v\n%s", err, stdout)
	}
	if inferred.Run.Source.Kind != "eve-otel-spans" {
		t.Fatalf("expected .json inference to pick the OTel wire, got kind=%q", inferred.Run.Source.Kind)
	}
	if inferred.Run.Observation != eveimport.ObservationIndeterminate {
		t.Fatalf("mislabelled NDJSON parsed as OTel must be INDETERMINATE, got %q", inferred.Run.Observation)
	}

	// WITH --kind ndjson: the override reads the same bytes as the NDJSON wire.
	code, stdout, stderr = runEveImportCase(t, "--kind", "ndjson", "--json", path)
	if code != 0 {
		t.Fatalf("expected exit 0, got %d (stderr: %s)", code, stderr)
	}
	var overridden eveImportJSONEnvelope
	if err := json.Unmarshal([]byte(stdout), &overridden); err != nil {
		t.Fatalf("stdout is not the typed JSON envelope: %v\n%s", err, stdout)
	}
	if overridden.Run.Source.Kind != "eve-ndjson" {
		t.Fatalf("--kind ndjson must override inference, got kind=%q", overridden.Run.Source.Kind)
	}
	if overridden.Run.Observation != eveimport.ObservationComplete {
		t.Fatalf("observation = %q, want %q", overridden.Run.Observation, eveimport.ObservationComplete)
	}
	if overridden.Run.Sessions != 2 {
		t.Fatalf("sessions = %d, want 2", overridden.Run.Sessions)
	}
}

// TestEveImportUsageAndExitEdges: zero args, extra positionals, a bogus --kind,
// and a nonexistent FILE each map to their distinct exit code.
func TestEveImportUsageAndExitEdges(t *testing.T) {
	valid := writeEveImportFixture(t, "session.ndjson", eveImportNDJSONFixture)
	cases := []struct {
		name string
		argv []string
		want int
	}{
		{"zero args", nil, 2},
		{"two positionals", []string{valid, valid}, 2},
		{"bogus kind", []string{"--kind", "bogus", valid}, 2},
		{"nonexistent file", []string{filepath.Join(t.TempDir(), "absent.ndjson")}, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			code, _, _ := runEveImportCase(t, tc.argv...)
			if code != tc.want {
				t.Fatalf("argv=%v: exit = %d, want %d", tc.argv, code, tc.want)
			}
		})
	}
}

// TestEveImportPartialTag: a span set whose turn is missing usage counters must
// yield the partial observation and a MISSING_TAG diagnostic - still at exit 0.
func TestEveImportPartialTag(t *testing.T) {
	const partialSpans = `{
  "spans": [
    {
      "name": "eve.session",
      "span_id": "sp-root",
      "status": {"code": "OK", "message": ""},
      "attributes": {"eve.session.id": "ses_partial", "eve.model.id": "model-partial"}
    },
    {
      "name": "eve.turn",
      "span_id": "sp-turn1",
      "parent_span_id": "sp-root",
      "status": {"code": "OK", "message": ""},
      "attributes": {"eve.session.id": "ses_partial", "eve.turn.index": 1}
    }
  ]
}
`
	path := writeEveImportFixture(t, "partial.json", partialSpans)
	code, stdout, stderr := runEveImportCase(t, "--json", path)
	if code != 0 {
		t.Fatalf("partial must be exit 0, got %d (stderr: %s)", code, stderr)
	}
	var env eveImportJSONEnvelope
	if err := json.Unmarshal([]byte(stdout), &env); err != nil {
		t.Fatalf("stdout is not the typed JSON envelope: %v\n%s", err, stdout)
	}
	if env.Run.Observation != eveimport.ObservationPartial {
		t.Fatalf("observation = %q, want %q (diagnostics: %+v)", env.Run.Observation, eveimport.ObservationPartial, env.Run.Diagnostics)
	}
	found := false
	for _, d := range env.Run.Diagnostics {
		if d.Code == "MISSING_TAG" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected a MISSING_TAG diagnostic, got %+v", env.Run.Diagnostics)
	}
}
