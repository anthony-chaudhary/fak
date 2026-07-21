package sessionctl

// fatigue_test.go — the #4427 witness: the rubber-stamp flag must FIRE for a gate
// that is waved through without inspection, and CLEAR for one that is actually
// inspected. Both halves matter: a detector that only ever fires is a stuck needle,
// not a signal.

import (
	"encoding/json"
	"strings"
	"testing"
)

// stopEvents builds n firings of one gate. approved waves them through; inspected
// makes the first half of the approvals follow an inspect-class tool call.
func stopEvents(session, stage, kind, disposition string, n int, approved bool, inspectedHalf bool) []FatigueEvent {
	out := make([]FatigueEvent, 0, n)
	for i := 0; i < n; i++ {
		tool := "TaskCreate" // a real tool from the stream, but not inspect-class
		if inspectedHalf && i%2 == 0 {
			tool = "Read"
		}
		out = append(out, FatigueEvent{
			Schema:      FatigueEventSchema,
			Session:     session,
			Stage:       stage,
			Kind:        kind,
			Disposition: disposition,
			Mode:        "enforce",
			Blocked:     !approved,
			Transcript:  &FatigueTranscript{Read: true, LastToolUse: tool},
		})
	}
	return out
}

func rowFor(t *testing.T, rep FatigueReport, key string) FatigueRow {
	t.Helper()
	for _, r := range rep.Rows {
		if r.Key == key {
			return r
		}
	}
	t.Fatalf("no row for gate %q in report %+v", key, rep.Rows)
	return FatigueRow{}
}

// TestFatigueFlagFiresAndClears is the issue's stated witness: a gate fired 20 and
// approved 20 with nothing inspected reads fatigue 1.0 and RUBBER_STAMPED; a
// sibling gate with the same fire count but half its approvals preceded by a Read
// reads 0.5 and stays unflagged.
func TestFatigueFlagFiresAndClears(t *testing.T) {
	var events []FatigueEvent
	events = append(events, stopEvents("sess-1", "allow", "clean", "clean_completion", 20, true, false)...)
	events = append(events, stopEvents("sess-1", "allow", "continue", "handoff_block", 20, true, true)...)

	rep := FoldFatigue(events, FatigueOptions{})

	stamped := rowFor(t, rep, "allow/clean/clean_completion")
	if stamped.Fires != 20 || stamped.Approved != 20 || stamped.ApprovedWithoutInspection != 20 {
		t.Fatalf("rubber-stamped gate counts wrong: %+v", stamped)
	}
	if stamped.Rate != 1.0 {
		t.Fatalf("want fatigue 1.0 for an always-approved, never-inspected gate, got %v (%+v)", stamped.Rate, stamped)
	}
	if !stamped.RubberStamped || stamped.Flag != RubberStampedFlag {
		t.Fatalf("want the gate flagged %s, got rubber_stamped=%v flag=%q", RubberStampedFlag, stamped.RubberStamped, stamped.Flag)
	}
	if stamped.Coarsen == "" {
		t.Fatalf("a flagged gate must name its coarsening target: %+v", stamped)
	}

	inspected := rowFor(t, rep, "allow/continue/handoff_block")
	if inspected.Fires != 20 || inspected.Approved != 20 || inspected.ApprovedWithoutInspection != 10 {
		t.Fatalf("inspected gate counts wrong: %+v", inspected)
	}
	if inspected.Rate != 0.5 {
		t.Fatalf("want fatigue 0.5 for a half-inspected gate, got %v (%+v)", inspected.Rate, inspected)
	}
	if inspected.RubberStamped || inspected.Flag != "" {
		t.Fatalf("a half-inspected gate is below the soft bar and must NOT be flagged: %+v", inspected)
	}

	if len(rep.Flagged) != 1 || rep.Flagged[0] != "allow/clean/clean_completion" {
		t.Fatalf("flagged worklist should name exactly the rubber-stamped gate, got %v", rep.Flagged)
	}
	// One row per gate identity, never one per firing (the batch-policy fence).
	if len(rep.Rows) != 2 {
		t.Fatalf("want one row per gate identity, got %d rows for 2 gates", len(rep.Rows))
	}
	if rep.Events != 40 {
		t.Fatalf("want 40 folded events, got %d", rep.Events)
	}
	// Most fatigued first, so the strongest coarsening target reads at the top.
	if rep.Rows[0].Key != "allow/clean/clean_completion" {
		t.Fatalf("want the most-fatigued gate ranked first, got %q", rep.Rows[0].Key)
	}
}

// TestFatigueJSONRenderCarriesBothRows shows the --json payload the CLI emits:
// both gates present, with the per-gate counts and the literal RUBBER_STAMPED token.
func TestFatigueJSONRenderCarriesBothRows(t *testing.T) {
	var events []FatigueEvent
	events = append(events, stopEvents("sess-1", "allow", "clean", "clean_completion", 20, true, false)...)
	events = append(events, stopEvents("sess-1", "allow", "continue", "handoff_block", 20, true, true)...)

	b, err := json.MarshalIndent(FoldFatigue(events, FatigueOptions{}), "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)
	for _, want := range []string{
		`"schema": "fak.gate-fatigue.v1"`,
		`"key": "allow/clean/clean_completion"`,
		`"key": "allow/continue/handoff_block"`,
		`"fires": 20`,
		`"approved": 20`,
		`"approved_without_inspection": 20`,
		`"approved_without_inspection": 10`,
		`"fatigue": 1`,
		`"fatigue": 0.5`,
		`"flag": "RUBBER_STAMPED"`,
		`"rubber_stamped": false`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("--json payload missing %s\npayload:\n%s", want, got)
		}
	}
}

// TestFatigueMinFiresFloorHoldsTheFlag pins the sample floor: a gate waved through
// every time it fired is NOT a habit until it has fired enough to be one.
func TestFatigueMinFiresFloorHoldsTheFlag(t *testing.T) {
	events := stopEvents("sess-1", "allow", "clean", "clean_completion", 3, true, false)
	rep := FoldFatigue(events, FatigueOptions{})
	row := rowFor(t, rep, "allow/clean/clean_completion")
	if row.Rate != 1.0 {
		t.Fatalf("want rate 1.0, got %v", row.Rate)
	}
	if row.RubberStamped {
		t.Fatalf("3 fires is below the min-fires floor (%d) and must not flag: %+v", DefaultFatigueMinFires, row)
	}
	if len(rep.Flagged) != 0 {
		t.Fatalf("want an empty worklist under the floor, got %v", rep.Flagged)
	}
}

// TestFatigueBlockedFiringsAreNotApprovals pins the denominator: a gate that keeps
// BLOCKING is the gate working, and must never read as rubber-stamped.
func TestFatigueBlockedFiringsAreNotApprovals(t *testing.T) {
	events := stopEvents("sess-1", "allow", "continue", "handoff_block", 20, false, false)
	rep := FoldFatigue(events, FatigueOptions{})
	row := rowFor(t, rep, "allow/continue/handoff_block")
	if row.Fires != 20 || row.Approved != 0 || row.ApprovedWithoutInspection != 0 {
		t.Fatalf("blocked firings must count as fires but never approvals: %+v", row)
	}
	if row.Rate != 0 || row.RubberStamped {
		t.Fatalf("an always-blocking gate must not be flagged: %+v", row)
	}
}

// TestFatigueUnknownInspectionDilutesRatherThanFlags is the honest-direction fence:
// a stop whose transcript could not be read is NOT evidence of habituation, so it
// must lower the rate, never raise it.
func TestFatigueUnknownInspectionDilutesRatherThanFlags(t *testing.T) {
	events := make([]FatigueEvent, 0, 20)
	for i := 0; i < 20; i++ {
		e := FatigueEvent{
			Schema: FatigueEventSchema, Session: "sess-1",
			Stage: "allow", Kind: "clean", Disposition: "clean_completion",
		}
		if i < 10 {
			e.Transcript = &FatigueTranscript{Read: true, LastToolUse: "TaskCreate"}
		} else if i%2 == 0 {
			e.Transcript = &FatigueTranscript{Read: false} // a transcript we could not parse
		} // else: no transcript block at all
		events = append(events, e)
	}
	rep := FoldFatigue(events, FatigueOptions{})
	row := rowFor(t, rep, "allow/clean/clean_completion")
	if row.Approved != 20 || row.ApprovedWithoutInspection != 10 || row.InspectionUnknown != 10 {
		t.Fatalf("unknown-inspection accounting wrong: %+v", row)
	}
	if row.Rate != 0.5 {
		t.Fatalf("unknown must dilute the rate to 0.5, not inflate it to 1.0: %+v", row)
	}
	if row.RubberStamped {
		t.Fatalf("unknown inspection must never carry a gate over the bar: %+v", row)
	}
}

// TestParseFatigueEventsReadsTheRealStream decodes rows in the exact on-disk shape
// the Stop hook writes, including a foreign line and a superset of fields, proving
// the detector reads the LIVE stream rather than a bespoke fixture format.
func TestParseFatigueEventsReadsTheRealStream(t *testing.T) {
	content := strings.Join([]string{
		`{"schema":"fak.guard-stop.v1","ts":"2026-07-09T18:26:15Z","session":"e286b0fa","disposition":"handoff_block","kind":"continue","stage":"allow","signal":"same-issue","mode":"enforce","exit":2,"blocked":true,"bound":6,"transcript":{"read":true,"assistant_turns":34,"last_had_tool_use":true,"last_tool_use":"Read"}}`,
		``,
		`not json at all`,
		`{"schema":"fak.some-other.v1","disposition":"ignored"}`,
		`{"schema":"fak.guard-stop.v1","ts":"2026-07-09T18:26:35Z","session":"e286b0fa","disposition":"clean_completion","kind":"clean","stage":"allow","mode":"enforce","exit":0,"transcript":{"read":true,"truncated":true,"assistant_turns":34,"last_had_tool_use":true,"last_tool_use":"TaskCreate"}}`,
	}, "\n")

	events := ParseFatigueEvents(content)
	if len(events) != 2 {
		t.Fatalf("want 2 guard-stop rows (foreign/blank/malformed skipped), got %d: %+v", len(events), events)
	}
	if events[0].Disposition != "handoff_block" || !events[0].Blocked {
		t.Fatalf("first row decoded wrong: %+v", events[0])
	}
	if events[1].Transcript == nil || events[1].Transcript.LastToolUse != "TaskCreate" {
		t.Fatalf("second row transcript decoded wrong: %+v", events[1])
	}
	if events[1].Blocked {
		t.Fatalf("an absent blocked field must decode as an approval: %+v", events[1])
	}

	rep := FoldFatigue(events, FatigueOptions{Session: "e286b0fa"})
	if rep.Events != 2 || len(rep.Rows) != 2 {
		t.Fatalf("want both gates folded, got events=%d rows=%d", rep.Events, len(rep.Rows))
	}
	// Session filtering must actually filter.
	if other := FoldFatigue(events, FatigueOptions{Session: "nobody"}); other.Events != 0 || len(other.Rows) != 0 {
		t.Fatalf("session filter should have excluded every row, got %+v", other)
	}
}
