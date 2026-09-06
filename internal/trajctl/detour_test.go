package trajctl

import (
	"path/filepath"
	"strings"
	"testing"
)

// dev is a terse stream-event builder for the table cases: a target with a
// space reads as a shell command (Bash), anything else as a file touch (Read).
func dev(target string, isErr bool) ToolEvent {
	tool := "Read"
	if strings.Contains(target, " ") {
		tool = "Bash"
	}
	return ToolEvent{Tool: tool, Target: target, IsError: isErr}
}

// detourBaseline is the parent shape every table case starts from: five
// healthy calls whose topics are {internal/trajctl, go}.
func detourBaseline() []ToolEvent {
	return []ToolEvent{
		dev("internal/trajctl/a.go", false),
		dev("internal/trajctl/b.go", false),
		dev("go test ./...", false),
		dev("internal/trajctl/a.go", false),
		dev("go build ./...", false),
	}
}

func TestDetourDetector_ErrorDetourTranscriptOpensAndClosesOnce(t *testing.T) {
	path := filepath.Join("testdata", "detour-session.jsonl")
	events, err := ReadToolStream(path)
	if err != nil {
		t.Fatalf("ReadToolStream: %v", err)
	}
	if len(events) != 14 {
		t.Fatalf("extracted %d events, want 14", len(events))
	}

	spans := DetectDetourSpans(events)
	if len(spans) != 1 {
		t.Fatalf("DetectDetourSpans = %+v, want exactly 1 span", spans)
	}
	sp := spans[0]
	if sp.BurstIndex != 5 || sp.OpenIndex != 8 || sp.CloseIndex != 12 || sp.Errors != 3 {
		t.Fatalf("span = %+v, want burst 5, open 8, close 12, errors 3", sp)
	}
	if want := []string{"c:/programdata/proxy", "netsh"}; strings.Join(sp.Topics, ",") != strings.Join(want, ",") {
		t.Fatalf("span.Topics = %v, want %v", sp.Topics, want)
	}

	obj := fourPhaseObjective()
	rows := DetourRows(obj, spans, path, 4242, Stamp{SessionID: "sess-detour", RunID: "run-detour"})
	if len(rows) != 6 {
		t.Fatalf("DetourRows returned %d rows, want 6 (open trio + close trio)", len(rows))
	}

	// While the detour is open — fold the open trio only — the parent is
	// paused and the budgeted child is active under it.
	mid := Fold(append([]Row{ObjectiveRecord(obj)}, rows[:3]...))
	childID := obj.ID + "-detour-1"
	child, ok := mid.Objectives[childID]
	if !ok {
		t.Fatalf("open fold: child %q not declared; objectives = %v", childID, mid.ObjectiveIDs())
	}
	if child.ParentID != obj.ID || child.Status != StatusActive || child.Budget != DefaultDetourBudget() {
		t.Fatalf("open child = %+v, want active under %q with the default budget", child, obj.ID)
	}
	if child.Statement == "" {
		t.Fatalf("open child carries no statement")
	}
	if got := mid.Objectives[obj.ID].Status; got != StatusPaused {
		t.Fatalf("parent status while detour open = %q, want %q", got, StatusPaused)
	}

	// The full trail round-trips the ledger: child met, parent resumed, and
	// the child's W2 repair curve has both endpoints with transcript evidence.
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	if err := Append(ledger, ObjectiveRecord(obj)); err != nil {
		t.Fatalf("append parent: %v", err)
	}
	if n, err := AppendDetourRows(ledger, rows); err != nil || n != 6 {
		t.Fatalf("AppendDetourRows = (%d, %v), want (6, nil)", n, err)
	}
	st := Fold(ReadLedgerFile(ledger))
	if got := st.Objectives[childID].Status; got != StatusMet {
		t.Fatalf("closed child status = %q, want %q", got, StatusMet)
	}
	if got := st.Objectives[obj.ID].Status; got != StatusActive {
		t.Fatalf("parent status after close = %q, want %q (resumed)", got, StatusActive)
	}
	scores := st.ScoresFor(childID)
	if len(scores) != 2 || scores[0].Value != 0 || scores[1].Value != 1 {
		t.Fatalf("child scores = %+v, want W2 endpoints 0 then 1", scores)
	}
	for _, s := range scores {
		if s.Witness != W2 || s.Method != DetourDetectorMethod || s.Version != DetourDetectorVersion {
			t.Fatalf("marker identity = witness %q method %q version %q", s.Witness, s.Method, s.Version)
		}
		if len(s.Evidence) != 1 || s.Evidence[0].Kind != "transcript-span" || s.Evidence[0].Ref != path {
			t.Fatalf("marker evidence = %+v, want transcript-span ref %q", s.Evidence, path)
		}
		if s.SessionID != "sess-detour" || s.RunID != "run-detour" || s.UnixMillis != 4242 {
			t.Fatalf("marker stamp = %+v, want the caller's stamp", s)
		}
	}
}

func TestDetourDetector_HealthyTranscriptsOpenNone(t *testing.T) {
	for _, fixture := range []string{"healthy-session.jsonl", "stalled-session.jsonl"} {
		t.Run(fixture, func(t *testing.T) {
			events, err := ReadToolStream(filepath.Join("testdata", fixture))
			if err != nil {
				t.Fatalf("ReadToolStream: %v", err)
			}
			if spans := DetectDetourSpans(events); len(spans) != 0 {
				t.Fatalf("healthy transcript opened %+v, want none", spans)
			}
		})
	}
}

func TestDetectDetourSpans_StreamShapes(t *testing.T) {
	base := detourBaseline()
	errs := func(n int) []ToolEvent {
		out := make([]ToolEvent, 0, n)
		for i := 0; i < n; i++ {
			out = append(out, dev("go test ./...", true))
		}
		return out
	}
	cat := func(parts ...[]ToolEvent) []ToolEvent {
		out := make([]ToolEvent, 0)
		for _, p := range parts {
			out = append(out, p...)
		}
		return out
	}
	off := func(dir string, n int) []ToolEvent {
		out := make([]ToolEvent, 0, n)
		for i := 0; i < n; i++ {
			out = append(out, dev(dir+"/x.conf", false))
		}
		return out
	}

	for _, tc := range []struct {
		name   string
		events []ToolEvent
		want   []DetourSpan
	}{
		{
			name:   "in-place retry burst without topic shift opens nothing",
			events: cat(base, errs(3), []ToolEvent{dev("go test ./...", false), dev("internal/trajctl/a.go", false)}),
			want:   nil,
		},
		{
			name:   "topic shift without an error burst opens nothing",
			events: cat(base, off("pkg/other", 5)),
			want:   nil,
		},
		{
			name:   "shift starting past the horizon is new work, not this burst's detour",
			events: cat(base, errs(3), append(off("internal/trajctl", 13), off("pkg/other", 3)...)),
			want:   nil,
		},
		{
			name:   "burst with no established parent shape opens nothing",
			events: cat(errs(3), off("pkg/other", 3)),
			want:   nil,
		},
		{
			name:   "a two-call stray off-topic probe is not sustained",
			events: cat(base, errs(3), off("pkg/other", 2), []ToolEvent{dev("internal/trajctl/a.go", false), dev("go test ./...", false)}),
			want:   nil,
		},
		{
			name:   "no return before stream end leaves the span open",
			events: cat(base, errs(3), off("ops/net", 3)),
			want:   []DetourSpan{{BurstIndex: 5, OpenIndex: 8, CloseIndex: -1, Errors: 3, Topics: []string{"ops/net"}}},
		},
		{
			name: "a second burst after a proven return opens a second span",
			events: cat(
				base, errs(3), off("ops/a", 3),
				[]ToolEvent{dev("internal/trajctl/a.go", false), dev("go test ./...", false)},
				errs(3), off("ops/b", 3),
				[]ToolEvent{dev("internal/trajctl/b.go", false), dev("go test ./...", false)},
			),
			want: []DetourSpan{
				{BurstIndex: 5, OpenIndex: 8, CloseIndex: 11, Errors: 3, Topics: []string{"ops/a"}},
				{BurstIndex: 13, OpenIndex: 16, CloseIndex: 19, Errors: 3, Topics: []string{"ops/b"}},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := DetectDetourSpans(tc.events)
			if len(got) != len(tc.want) {
				t.Fatalf("spans = %+v, want %d span(s)", got, len(tc.want))
			}
			for i := range tc.want {
				g, w := got[i], tc.want[i]
				if g.BurstIndex != w.BurstIndex || g.OpenIndex != w.OpenIndex || g.CloseIndex != w.CloseIndex || g.Errors != w.Errors {
					t.Fatalf("span[%d] = %+v, want %+v", i, g, w)
				}
				if strings.Join(g.Topics, ",") != strings.Join(w.Topics, ",") {
					t.Fatalf("span[%d].Topics = %v, want %v", i, g.Topics, w.Topics)
				}
			}
		})
	}
}

func TestDetourRows_OpenEndedDetourLeavesParentPaused(t *testing.T) {
	obj := fourPhaseObjective()
	span := DetourSpan{BurstIndex: 5, OpenIndex: 8, CloseIndex: -1, Errors: 3, Topics: []string{"ops/net"}}
	rows := DetourRows(obj, []DetourSpan{span}, "", 99, Stamp{})
	if len(rows) != 3 {
		t.Fatalf("open-ended DetourRows returned %d rows, want 3 (no close trio)", len(rows))
	}

	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	if err := Append(ledger, ObjectiveRecord(obj)); err != nil {
		t.Fatalf("append parent: %v", err)
	}
	if n, err := AppendDetourRows(ledger, rows); err != nil || n != 3 {
		t.Fatalf("AppendDetourRows = (%d, %v), want (3, nil)", n, err)
	}
	st := Fold(ReadLedgerFile(ledger))
	if got := st.Objectives[obj.ID].Status; got != StatusPaused {
		t.Fatalf("parent status = %q, want %q while the detour stays open", got, StatusPaused)
	}
	if got := st.Objectives[obj.ID+"-detour-1"].Status; got != StatusActive {
		t.Fatalf("child status = %q, want %q", got, StatusActive)
	}
	if scores := st.ScoresFor(obj.ID + "-detour-1"); len(scores) != 1 || scores[0].Value != 0 || scores[0].Evidence[0].Ref != "tool-stream" {
		t.Fatalf("open marker = %+v, want one value-0 row with the tool-stream fallback ref", scores)
	}
}

func TestDetourRows_UndeclarableParentYieldsNil(t *testing.T) {
	span := DetourSpan{OpenIndex: 1, CloseIndex: -1, Errors: 3}
	if rows := DetourRows(Objective{ID: "", Statement: "x"}, []DetourSpan{span}, "", 0, Stamp{}); rows != nil {
		t.Fatalf("id-less parent produced rows: %+v", rows)
	}
	if rows := DetourRows(Objective{ID: "p", Statement: ""}, []DetourSpan{span}, "", 0, Stamp{}); rows != nil {
		t.Fatalf("statement-less parent produced rows: %+v", rows)
	}
}

// TestTurnEndDetourRows_LivePassOpensClosesAndDedupesOnReplay is the #3669 done
// condition at the producer level: the shipped detour fixture run through the live
// turn-end fold TWICE against the SAME ledger folds to exactly one MET detour child
// under its root, root ACTIVE — the replay double-opens nothing — and the
// no-transcript (empty stream) case is a total no-op.
func TestTurnEndDetourRows_LivePassOpensClosesAndDedupesOnReplay(t *testing.T) {
	path := filepath.Join("testdata", "detour-session.jsonl")
	events, err := ReadToolStream(path)
	if err != nil {
		t.Fatalf("ReadToolStream: %v", err)
	}
	root := fourPhaseObjective() // a top-level (ParentID "") active objective
	childID := root.ID + "-detour-1"
	stamp := Stamp{SessionID: "sess-detour", RunID: "run-detour"}

	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	if err := Append(ledger, ObjectiveRecord(root)); err != nil {
		t.Fatalf("seed root: %v", err)
	}

	// Pass 1: the closed detour opens and closes in one fold — the full 6-row trail.
	rows1 := TurnEndDetourRows(Fold(ReadLedgerFile(ledger)), events, path, 4242, stamp)
	if len(rows1) != 6 {
		t.Fatalf("pass 1 produced %d rows, want 6 (open trio + close trio)", len(rows1))
	}
	if n, err := AppendDetourRows(ledger, rows1); err != nil || n != 6 {
		t.Fatalf("append pass 1 = (%d, %v), want (6, nil)", n, err)
	}

	// Pass 2 (replay): the same transcript against the now-populated ledger adds nothing.
	rows2 := TurnEndDetourRows(Fold(ReadLedgerFile(ledger)), events, path, 9999, stamp)
	if len(rows2) != 0 {
		t.Fatalf("replay produced %d rows, want 0 (no double-open)", len(rows2))
	}

	st := Fold(ReadLedgerFile(ledger))
	if got := st.Objectives[childID].Status; got != StatusMet {
		t.Fatalf("detour child status = %q, want %q after the live pass", got, StatusMet)
	}
	if got := st.Objectives[root.ID].Status; got != StatusActive {
		t.Fatalf("root status = %q, want %q (resumed after the detour closed)", got, StatusActive)
	}
	if child := st.Objectives[childID]; child.ParentID != root.ID || child.Budget != DefaultDetourBudget() {
		t.Fatalf("detour child = %+v, want a budgeted child under %q", child, root.ID)
	}
	scores := st.ScoresFor(childID)
	if len(scores) != 2 || scores[0].Value != 0 || scores[1].Value != 1 {
		t.Fatalf("child repair curve = %+v, want W2 endpoints 0 then 1", scores)
	}
	for _, s := range scores {
		if s.Evidence[0].Kind != "transcript-span" || s.Evidence[0].Ref != path || s.SessionID != stamp.SessionID {
			t.Fatalf("marker = %+v, want transcript-span evidence for %q stamped %q", s, path, stamp.SessionID)
		}
	}

	// No-transcript / empty stream: a total no-op.
	if rows := TurnEndDetourRows(st, nil, path, 1, stamp); len(rows) != 0 {
		t.Fatalf("empty stream produced %+v, want no rows", rows)
	}
	// A state with no OPEN root also yields nothing even with real spans.
	if rows := TurnEndDetourRows(State{Objectives: map[string]Objective{}}, events, path, 1, stamp); len(rows) != 0 {
		t.Fatalf("rootless state produced %+v, want no rows", rows)
	}
}

// TestLiveDetourRows_ClosesADetourOpenedOnAnEarlierTurn proves the live transition
// the batch DetourRows cannot express: a span still OPEN when first folded (child
// active, root paused) is CLOSED on a later turn once the growing transcript
// includes the return — the close trio fires exactly once, the root resumes, and a
// further replay adds nothing.
func TestLiveDetourRows_ClosesADetourOpenedOnAnEarlierTurn(t *testing.T) {
	root := fourPhaseObjective()
	childID := root.ID + "-detour-1"
	stamp := Stamp{SessionID: "s", RunID: "r"}
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	if err := Append(ledger, ObjectiveRecord(root)); err != nil {
		t.Fatalf("seed root: %v", err)
	}

	// Turn A: the detour is still OPEN (no return yet) — open trio only, root paused.
	openSpan := DetourSpan{BurstIndex: 5, OpenIndex: 8, CloseIndex: -1, Errors: 3, Topics: []string{"ops/net"}}
	stA := Fold(ReadLedgerFile(ledger))
	rowsA := LiveDetourRows(stA.Objectives[root.ID], []DetourSpan{openSpan}, stA, "", 1, stamp)
	if len(rowsA) != 3 {
		t.Fatalf("turn A produced %d rows, want 3 (open trio only)", len(rowsA))
	}
	if n, err := AppendDetourRows(ledger, rowsA); err != nil || n != 3 {
		t.Fatalf("append turn A = (%d, %v), want (3, nil)", n, err)
	}
	if got := Fold(ReadLedgerFile(ledger)).Objectives[root.ID].Status; got != StatusPaused {
		t.Fatalf("root status after open = %q, want %q", got, StatusPaused)
	}

	// Turn B: the transcript grew and the span now RETURNS — close trio only, root resumes.
	closedSpan := DetourSpan{BurstIndex: 5, OpenIndex: 8, CloseIndex: 12, Errors: 3, Topics: []string{"ops/net"}}
	stB := Fold(ReadLedgerFile(ledger))
	rowsB := LiveDetourRows(stB.Objectives[root.ID], []DetourSpan{closedSpan}, stB, "", 2, stamp)
	if len(rowsB) != 3 {
		t.Fatalf("turn B produced %d rows, want 3 (close trio only)", len(rowsB))
	}
	if n, err := AppendDetourRows(ledger, rowsB); err != nil || n != 3 {
		t.Fatalf("append turn B = (%d, %v), want (3, nil)", n, err)
	}

	// Turn C (replay of the closed span): nothing more to add.
	stC := Fold(ReadLedgerFile(ledger))
	if rows := LiveDetourRows(stC.Objectives[root.ID], []DetourSpan{closedSpan}, stC, "", 3, stamp); len(rows) != 0 {
		t.Fatalf("turn C replay produced %d rows, want 0", len(rows))
	}

	final := Fold(ReadLedgerFile(ledger))
	if got := final.Objectives[childID].Status; got != StatusMet {
		t.Fatalf("child status = %q, want %q", got, StatusMet)
	}
	if got := final.Objectives[root.ID].Status; got != StatusActive {
		t.Fatalf("root status = %q, want %q (resumed)", got, StatusActive)
	}
	if scores := final.ScoresFor(childID); len(scores) != 2 || scores[0].Value != 0 || scores[1].Value != 1 {
		t.Fatalf("child repair curve = %+v, want W2 endpoints 0 then 1", scores)
	}
}

// TestDetourOverrun_FixtureReturnToMainAndWarn verifies issue #2552 acceptance criteria:
// A fixture detour crossing its budget triggers exactly one return nudge referencing
// the paused parent, repeated overrun escalates one rung to warn, and all decisions
// are ledgered.
func TestDetourOverrun_FixtureReturnToMainAndWarn(t *testing.T) {
	fixturePath := filepath.Join("testdata", "curve", "detour-overrun.jsonl")
	fixtureRows := ReadLedgerFile(fixturePath)
	if len(fixtureRows) == 0 {
		t.Fatalf("failed to read fixture %s", fixturePath)
	}

	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	for _, row := range fixtureRows {
		if err := Append(ledger, row); err != nil {
			t.Fatalf("seed fixture row: %v", err)
		}
	}

	rec := &countingSteer{}
	stamp := Stamp{SessionID: "sess-detour", RunID: "run-detour"}

	// Boundary 1: detour crosses its budget (3 turns scored > 2-turn budget).
	// Triggers exactly one return-to-main nudge referencing the paused parent.
	st1 := Fold(ReadLedgerFile(ledger))
	ds1 := st1.SteerSweep(stamp, 4000, rec.fn())
	var d1 *SteerDecision
	for i := range ds1 {
		if ds1[i].ObjectiveID == "traj-detour" {
			d1 = &ds1[i]
			break
		}
	}
	if d1 == nil {
		t.Fatalf("sweep 1 produced no decision for traj-detour: %+v", ds1)
	}
	if d1.Action != ActionNudge || d1.Signal != SignalDetourOverrun {
		t.Fatalf("sweep 1 decision = %+v, want ActionNudge on SignalDetourOverrun", d1)
	}
	if !d1.Delivered || d1.DeliverErr != "" {
		t.Fatalf("sweep 1 decision not delivered: %+v", d1)
	}
	if rec.calls != 1 {
		t.Fatalf("steer calls after sweep 1 = %d, want 1", rec.calls)
	}
	// Verify packet references the paused parent (ID and statement) and return-to-main.
	for _, want := range []string{
		"return-to-main",
		"traj-epic",
		"the parent objective",
		"traj-detour",
		"repair the broken linker flag",
	} {
		if !strings.Contains(d1.Packet, want) {
			t.Errorf("sweep 1 packet missing %q:\n%s", want, d1.Packet)
		}
	}
	if n, err := AppendSteerDecisions(ledger, ds1); err != nil || n != len(ds1) {
		t.Fatalf("AppendSteerDecisions sweep 1 = (%d, %v)", n, err)
	}

	// Boundary 2: repeated overrun (still over budget).
	// Escalates one rung to warn.
	st2 := Fold(ReadLedgerFile(ledger))
	ds2 := st2.SteerSweep(stamp, 5000, rec.fn())
	var d2 *SteerDecision
	for i := range ds2 {
		if ds2[i].ObjectiveID == "traj-detour" {
			d2 = &ds2[i]
			break
		}
	}
	if d2 == nil {
		t.Fatalf("sweep 2 produced no decision for traj-detour: %+v", ds2)
	}
	if d2.Action != ActionWarn || d2.Signal != SignalDetourOverrun {
		t.Fatalf("sweep 2 decision = %+v, want ActionWarn on SignalDetourOverrun", d2)
	}
	if !d2.Delivered || d2.DeliverErr != "" {
		t.Fatalf("sweep 2 decision not delivered: %+v", d2)
	}
	if rec.calls != 2 {
		t.Fatalf("steer calls after sweep 2 = %d, want 2", rec.calls)
	}
	for _, want := range []string{
		"WARN",
		"traj-epic",
		"the parent objective",
		"traj-detour",
	} {
		if !strings.Contains(d2.Packet, want) {
			t.Errorf("sweep 2 packet missing %q:\n%s", want, d2.Packet)
		}
	}
	if n, err := AppendSteerDecisions(ledger, ds2); err != nil || n != len(ds2) {
		t.Fatalf("AppendSteerDecisions sweep 2 = (%d, %v)", n, err)
	}

	// Boundary 3: repeated overrun again after warn delivered.
	// Holds at ActionNone (warn is outstanding; does not hammer the channel).
	st3 := Fold(ReadLedgerFile(ledger))
	ds3 := st3.SteerSweep(stamp, 6000, rec.fn())
	var d3 *SteerDecision
	for i := range ds3 {
		if ds3[i].ObjectiveID == "traj-detour" {
			d3 = &ds3[i]
			break
		}
	}
	if d3 == nil {
		t.Fatalf("sweep 3 produced no decision for traj-detour: %+v", ds3)
	}
	if d3.Action != ActionNone || d3.Signal != SignalDetourOverrun {
		t.Fatalf("sweep 3 decision = %+v, want ActionNone on SignalDetourOverrun", d3)
	}
	if !strings.Contains(d3.Reason, "outstanding") {
		t.Errorf("sweep 3 reason = %q, want hold reason naming outstanding warn", d3.Reason)
	}
	if rec.calls != 2 {
		t.Fatalf("steer calls after sweep 3 = %d, want 2 (held, no new delivery)", rec.calls)
	}
	if n, err := AppendSteerDecisions(ledger, ds3); err != nil || n != len(ds3) {
		t.Fatalf("AppendSteerDecisions sweep 3 = (%d, %v)", n, err)
	}

	// Verify ledger contents: exactly the 3 expected steer decisions for traj-detour.
	finalSt := Fold(ReadLedgerFile(ledger))
	steers := finalSt.SteersFor("traj-detour")
	if len(steers) != 3 {
		t.Fatalf("ledgered steers for traj-detour = %d, want 3: %+v", len(steers), steers)
	}
	if steers[0].Action != ActionNudge || !steers[0].Delivered {
		t.Errorf("steers[0] = %+v, want delivered ActionNudge", steers[0])
	}
	if steers[1].Action != ActionWarn || !steers[1].Delivered {
		t.Errorf("steers[1] = %+v, want delivered ActionWarn", steers[1])
	}
	if steers[2].Action != ActionNone {
		t.Errorf("steers[2] = %+v, want held ActionNone", steers[2])
	}
}

// TestDetourOverrun_FailedDeliveryRetriesBeforeEscalating proves that an undelivered
// return nudge does not consume the nudge rung; the gate retries delivery before
// escalating to warn.
func TestDetourOverrun_FailedDeliveryRetriesBeforeEscalating(t *testing.T) {
	fixturePath := filepath.Join("testdata", "curve", "detour-overrun.jsonl")
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	for _, row := range ReadLedgerFile(fixturePath) {
		if err := Append(ledger, row); err != nil {
			t.Fatal(err)
		}
	}

	rec := &countingSteer{failures: 1}
	stamp := Stamp{SessionID: "sess-fail"}

	// Boundary 1: delivery fails.
	st1 := Fold(ReadLedgerFile(ledger))
	ds1 := st1.SteerSweep(stamp, 4000, rec.fn())
	if ds1[0].Action != ActionNudge || ds1[0].Delivered || ds1[0].DeliverErr == "" {
		t.Fatalf("sweep 1 = %+v, want undelivered nudge with error", ds1[0])
	}
	if _, err := AppendSteerDecisions(ledger, ds1); err != nil {
		t.Fatal(err)
	}

	// Boundary 2: retries ActionNudge (does NOT escalate to warn yet).
	st2 := Fold(ReadLedgerFile(ledger))
	ds2 := st2.SteerSweep(stamp, 5000, rec.fn())
	if ds2[0].Action != ActionNudge || !ds2[0].Delivered {
		t.Fatalf("sweep 2 = %+v, want delivered retry of ActionNudge", ds2[0])
	}
	if _, err := AppendSteerDecisions(ledger, ds2); err != nil {
		t.Fatal(err)
	}

	// Boundary 3: now escalates to ActionWarn.
	st3 := Fold(ReadLedgerFile(ledger))
	ds3 := st3.SteerSweep(stamp, 6000, rec.fn())
	if ds3[0].Action != ActionWarn || !ds3[0].Delivered {
		t.Fatalf("sweep 3 = %+v, want delivered ActionWarn", ds3[0])
	}
}

// TestDetourOverrun_RecoveryRearms proves that after an overrun episode has escalated
// to warn, a subsequent healthy decision re-arms the episode.
func TestDetourOverrun_RecoveryRearms(t *testing.T) {
	st := State{
		Objectives: map[string]Objective{
			"p": {ID: "p", Statement: "parent", Status: StatusPaused},
			"d": {ID: "d", ParentID: "p", Statement: "detour", Budget: Budget{Turns: 1}, Status: StatusActive},
		},
		Scores: []ScoreRow{
			w3Progress("d", 0.1, 1000),
			w3Progress("d", 0.1, 2000),
		},
		Steers: []SteerDecision{
			{ObjectiveID: "d", Action: ActionNudge, Signal: SignalDetourOverrun, Delivered: true},
			{ObjectiveID: "d", Action: ActionWarn, Signal: SignalDetourOverrun, Delivered: true},
			{ObjectiveID: "d", Action: ActionNone, Signal: SignalHealthy, Reason: "recovered"},
		},
	}
	oc, ok := st.CurveFor("d")
	if !ok || oc.Signal != SignalDetourOverrun {
		t.Fatalf("curve = %+v, ok = %v", oc, ok)
	}
	d := st.DecideNudge(oc)
	if d.Action != ActionNudge {
		t.Fatalf("recovered episode did not re-arm to ActionNudge: %+v", d)
	}
}
