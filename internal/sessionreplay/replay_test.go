package sessionreplay

import (
	"os"
	"path/filepath"
	"testing"
)

var (
	benchVerdictSink Verdict
	benchFixtureSink Fixture
	benchBytesSink   []byte
)

const goldenFixture = "regime_conditioned_turn.json"

// TestReplayRegressionFreezesRegimeVerdict loads the golden (turn,
// active_regime=plan) fixture, replays the decision path deterministically, and
// asserts it produces the frozen verdict. This is the permanent witness a fixed
// mode bug asked for: a later refactor of the plan/propose-only floor that
// changes this decision fails HERE instead of regressing silently.
func TestReplayRegressionFreezesRegimeVerdict(t *testing.T) {
	f, err := LoadFixture(filepath.Join("testdata", goldenFixture))
	if err != nil {
		t.Fatalf("load golden fixture: %v", err)
	}
	if f.ActiveRegime != "plan" {
		t.Fatalf("golden active_regime = %q, want plan", f.ActiveRegime)
	}
	// Belt-and-suspenders: the checked-in golden must itself carry the known
	// frozen verdict, so a silent weakening of the fixture is caught too.
	frozen := Verdict{Kind: "DENY", Reason: "DEFAULT_DENY"}
	if !f.Expect.Equal(frozen) {
		t.Fatalf("golden Expect = %s, want frozen %s (the checked-in golden was weakened)", f.Expect, frozen)
	}

	got, err := Replay(f)
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	if !got.Equal(f.Expect) {
		t.Fatalf("replay verdict = %s, want frozen %s", got, f.Expect)
	}
}

// TestReplayIsRegimeConditioned proves the fixture is genuinely regime-
// conditioned, not regime-blind: flipping active_regime ALONE flips the verdict.
// The same captured write the plan regime refuses is admitted under the
// autonomous regime. A fixture whose verdict did not move with the regime could
// not catch a mode-conditioned bug — the entire reason the regime is captured.
func TestReplayIsRegimeConditioned(t *testing.T) {
	f, err := LoadFixture(filepath.Join("testdata", goldenFixture))
	if err != nil {
		t.Fatalf("load golden fixture: %v", err)
	}

	planVerdict, err := Replay(f)
	if err != nil {
		t.Fatalf("replay under plan: %v", err)
	}

	// Flip ONLY the active regime; the captured turn is untouched.
	f.ActiveRegime = "autonomous"
	autoVerdict, err := Replay(f)
	if err != nil {
		t.Fatalf("replay under autonomous: %v", err)
	}

	if autoVerdict.Equal(planVerdict) {
		t.Fatalf("verdict did not change with the regime: plan=%s autonomous=%s — the fixture is regime-blind", planVerdict, autoVerdict)
	}
	if autoVerdict.Kind != "ALLOW" {
		t.Fatalf("autonomous regime verdict = %s, want ALLOW (the write is admitted under the broad loop)", autoVerdict)
	}
}

// TestFixtureRoundTripsAndCaptures pins the fak.sessionreplay.v1 (de)serialization
// and the Capture helper: a loaded golden re-marshals and re-parses to the same
// record, and Capture builds the same shape from the same inputs.
func TestFixtureRoundTripsAndCaptures(t *testing.T) {
	f, err := LoadFixture(filepath.Join("testdata", goldenFixture))
	if err != nil {
		t.Fatalf("load golden fixture: %v", err)
	}
	b, err := f.Marshal()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseFixture(b)
	if err != nil {
		t.Fatalf("re-parse: %v", err)
	}
	if got.Schema != SchemaV1 || got.Turn.Tool != f.Turn.Tool ||
		got.ActiveRegime != f.ActiveRegime || !got.Expect.Equal(f.Expect) {
		t.Fatalf("round-trip drifted:\n got %+v\nwant %+v", got, f)
	}

	captured := Capture(f.Turn, f.ActiveRegime, f.Expect)
	if captured.Schema != SchemaV1 || captured.ActiveRegime != f.ActiveRegime || !captured.Expect.Equal(f.Expect) {
		t.Fatalf("Capture built an unexpected fixture: %+v", captured)
	}
}

// TestReplayRefusesUnknownRegime proves an unresolvable regime is a loud refusal,
// not a silently-guessed floor.
func TestReplayRefusesUnknownRegime(t *testing.T) {
	f := Capture(
		DecisionInputs{Tool: "Write", Args: []byte(`{"path":"workspace/report.txt"}`)},
		"no-such-regime",
		Verdict{Kind: "DENY"},
	)
	if _, err := Replay(f); err == nil {
		t.Fatal("replay under an unknown regime returned nil error, want a refusal")
	}
}

// BenchmarkReplay_Plan measures end-to-end replay throughput under the "plan"
// regime, exercising regime resolution, policy parsing, and adjudication to DENY.
func BenchmarkReplay_Plan(b *testing.B) {
	f, err := LoadFixture(filepath.Join("testdata", goldenFixture))
	if err != nil {
		b.Fatalf("load golden fixture: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := Replay(f)
		if err != nil {
			b.Fatalf("replay: %v", err)
		}
		benchVerdictSink = v
	}
}

// BenchmarkReplay_Autonomous measures end-to-end replay throughput under the
// "autonomous" regime, exercising policy parsing and adjudication to ALLOW.
func BenchmarkReplay_Autonomous(b *testing.B) {
	f, err := LoadFixture(filepath.Join("testdata", goldenFixture))
	if err != nil {
		b.Fatalf("load golden fixture: %v", err)
	}
	f.ActiveRegime = "autonomous"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := Replay(f)
		if err != nil {
			b.Fatalf("replay: %v", err)
		}
		benchVerdictSink = v
	}
}

// BenchmarkReplay_DiverseTurns measures replay across alternating tool calls
// and regimes (Read, Write, and bash under plan and autonomous floors).
func BenchmarkReplay_DiverseTurns(b *testing.B) {
	fixtures := []Fixture{
		Capture(
			DecisionInputs{Tool: "Read", Args: []byte(`{"path":"workspace/notes.txt"}`)},
			"plan",
			Verdict{Kind: "ALLOW"},
		),
		Capture(
			DecisionInputs{Tool: "Write", Args: []byte(`{"path":"workspace/notes.txt","content":"hello"}`)},
			"plan",
			Verdict{Kind: "DENY", Reason: "DEFAULT_DENY"},
		),
		Capture(
			DecisionInputs{Tool: "Write", Args: []byte(`{"path":"workspace/notes.txt","content":"hello"}`)},
			"autonomous",
			Verdict{Kind: "ALLOW"},
		),
		Capture(
			DecisionInputs{Tool: "bash", Args: []byte(`{"command":"ls -la"}`)},
			"plan",
			Verdict{Kind: "DENY", Reason: "DEFAULT_DENY"},
		),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := fixtures[i%len(fixtures)]
		v, err := Replay(f)
		if err != nil {
			b.Fatalf("replay: %v", err)
		}
		benchVerdictSink = v
	}
}

// BenchmarkParseFixture measures fixture unmarshaling throughput with strict
// schema tagging and unknown-field rejection.
func BenchmarkParseFixture(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", goldenFixture))
	if err != nil {
		b.Fatalf("read golden fixture: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f, err := ParseFixture(raw)
		if err != nil {
			b.Fatalf("parse fixture: %v", err)
		}
		benchFixtureSink = f
	}
}

// BenchmarkMarshalFixture measures indented JSON serialization throughput
// for golden regression fixtures.
func BenchmarkMarshalFixture(b *testing.B) {
	f, err := LoadFixture(filepath.Join("testdata", goldenFixture))
	if err != nil {
		b.Fatalf("load golden fixture: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := f.Marshal()
		if err != nil {
			b.Fatalf("marshal fixture: %v", err)
		}
		benchBytesSink = data
	}
}

// BenchmarkCapture measures in-memory fixture construction throughput.
func BenchmarkCapture(b *testing.B) {
	turn := DecisionInputs{
		Tool: "Write",
		Args: []byte(`{"path":"workspace/report.txt","content":"shipped"}`),
	}
	expect := Verdict{Kind: "DENY", Reason: "DEFAULT_DENY"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := Capture(turn, "plan", expect)
		benchFixtureSink = f
	}
}
