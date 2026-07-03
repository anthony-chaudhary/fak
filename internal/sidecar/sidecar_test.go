package sidecar

import (
	"encoding/json"
	"strings"
	"testing"
)

// fixtureInputs is the golden fixture: one populated pane across all four planes,
// plus one deliberately-unmeasured plane, so the parity check covers both a
// rendered fact and an UNMEASURED gap.
func fixtureInputs() Inputs {
	return Inputs{
		Workspace:   "/work/fak",
		Host:        "test-host",
		GeneratedAt: "2026-07-03T00:00:00Z",
		Sessions:    PlaneInput{Measured: true, Note: "census over 2 config homes"},
		SessionRows: []SessionRow{
			{Session: "sess-b", Account: "acct-2", Harness: "codex", Disposition: "stopped"},
			{Session: "sess-a", Account: "acct-1", Harness: "claude", Disposition: "live"},
		},
		Accounts: PlaneInput{Measured: true},
		AccountRows: []AccountRow{
			{Account: "acct-1", State: "usable"},
			{Account: "acct-2", State: "throttled", Detail: "resets 01:00Z"},
		},
		Lanes: PlaneInput{Measured: true, Note: "dos-top reading"},
		LaneRows: []LaneRow{
			{Lane: "cmd", Kind: "cluster", Held: true, Owner: "worker-7"},
			{Lane: "docs", Kind: "cluster", Held: false},
		},
		// Posture deliberately unmeasured — no gateway address known in the fixture.
		PostureStatus: PlaneInput{Measured: false, Note: "no gateway address"},
	}
}

// TestParityCoreFieldSequence is the issue-#2215 witness: one fixture renders to
// terminal text and Slack blocks, and BOTH carry the identical core-field
// sequence, in order — machine-checked, not eyeballed. This is the parity
// contract stated as a test: because both surfaces walk Sections(), the sequence
// of rendered fields is the same object, and this proves it end-to-end through
// the actual rendered bytes of each surface.
func TestParityCoreFieldSequence(t *testing.T) {
	p := Fold(fixtureInputs())
	core := p.CoreFields()
	if len(core) == 0 {
		t.Fatal("CoreFields() is empty")
	}

	// The rendered VALUE strings, in the order Sections() produces them. Both
	// surfaces must present these values in this same order.
	var wantValues []string
	for _, sec := range p.Sections() {
		wantValues = append(wantValues, planeTitle(sec.Plane))
		for _, ln := range sec.Lines {
			wantValues = append(wantValues, ln.Value)
		}
	}

	text := RenderText(p)
	assertOrderedContains(t, "terminal", text, wantValues)

	slackText := slackFlatText(t, RenderSlack(p))
	assertOrderedContains(t, "slack", slackText, wantValues)
}

// assertOrderedContains checks that each want string appears in body, and that
// they appear in the given order (each after the previous). This is the
// machine-check that the surface renders the core fields in the pane's order.
func assertOrderedContains(t *testing.T, surface, body string, want []string) {
	t.Helper()
	cursor := 0
	for _, w := range want {
		idx := strings.Index(body[cursor:], w)
		if idx < 0 {
			t.Fatalf("%s surface missing core field %q (or out of order)\n---body---\n%s", surface, w, body)
		}
		cursor += idx + len(w)
	}
}

// slackFlatText concatenates every mrkdwn/plain_text string in the block payload
// in document order, so the parity walk sees the Slack surface as one ordered
// stream of the same field values the terminal renders.
func slackFlatText(t *testing.T, blocks []any) string {
	t.Helper()
	var b strings.Builder
	var walk func(v any)
	walk = func(v any) {
		switch node := v.(type) {
		case map[string]any:
			// A text object: {"type":"mrkdwn"|"plain_text","text":"..."}.
			if txt, ok := node["text"].(string); ok {
				b.WriteString(txt)
				b.WriteString("\n")
			}
			for _, key := range []string{"text", "elements"} {
				if child, ok := node[key]; ok {
					if _, isStr := child.(string); !isStr {
						walk(child)
					}
				}
			}
		case []any:
			for _, e := range node {
				walk(e)
			}
		}
	}
	for _, blk := range blocks {
		walk(blk)
	}
	return b.String()
}

// TestUnmeasuredPlaneIsHonest proves an absent plane reads UNMEASURED on BOTH
// surfaces (never a silent GREEN) and that the pane's OK bit reflects the gap.
func TestUnmeasuredPlaneIsHonest(t *testing.T) {
	p := Fold(fixtureInputs())
	if p.OK {
		t.Fatal("pane OK=true despite an unmeasured plane")
	}
	if p.Unmeasured != 1 {
		t.Fatalf("Unmeasured=%d, want 1", p.Unmeasured)
	}
	if got := p.Posture.Prov; got != Unmeasured {
		t.Fatalf("posture provenance=%q, want UNMEASURED", got)
	}
	text := RenderText(p)
	if !strings.Contains(text, "UNMEASURED") || !strings.Contains(text, "no gateway address") {
		t.Fatalf("terminal surface did not surface the unmeasured posture gap:\n%s", text)
	}
	slackText := slackFlatText(t, RenderSlack(p))
	if !strings.Contains(slackText, "UNMEASURED") {
		t.Fatalf("slack surface did not surface the unmeasured posture gap:\n%s", slackText)
	}
}

// TestFoldProvenanceLabels pins the per-plane provenance discipline: the census
// planes read from authored artifacts are WITNESSED; the live-read planes (lanes
// from dos-top, posture from the gateway) are OBSERVED — the pane renders lane
// occupancy, it does not re-adjudicate it.
func TestFoldProvenanceLabels(t *testing.T) {
	in := fixtureInputs()
	in.PostureStatus = PlaneInput{Measured: true}
	in.Posture = Posture{CachePosture: "managed", Compactions: 3, Elisions: 1, SessionsJoined: 2}
	p := Fold(in)

	cases := []struct {
		name string
		got  string
		want string
	}{
		{"sessions", p.Sessions.Prov, Witnessed},
		{"accounts", p.Accounts.Prov, Witnessed},
		{"lanes", p.Lanes.Prov, Observed},
		{"posture", p.Posture.Prov, Observed},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s provenance=%q, want %q", c.name, c.got, c.want)
		}
	}
}

// TestFoldTallies checks the summary counts that back the headline.
func TestFoldTallies(t *testing.T) {
	p := Fold(fixtureInputs())
	if p.Sessions.Total != 2 || p.Sessions.Live != 1 {
		t.Errorf("sessions total/live = %d/%d, want 2/1", p.Sessions.Total, p.Sessions.Live)
	}
	if p.Accounts.Usable != 1 || p.Accounts.Throttled != 1 || p.Accounts.Blocked != 0 {
		t.Errorf("accounts usable/throttled/blocked = %d/%d/%d, want 1/1/0",
			p.Accounts.Usable, p.Accounts.Throttled, p.Accounts.Blocked)
	}
	if p.Lanes.Held != 1 || p.Lanes.Free != 1 {
		t.Errorf("lanes held/free = %d/%d, want 1/1", p.Lanes.Held, p.Lanes.Free)
	}
	// Sessions must sort deterministically by (account, session) regardless of
	// input order — a parity prerequisite.
	if p.Sessions.Rows[0].Account != "acct-1" {
		t.Errorf("sessions not sorted by account: first=%q", p.Sessions.Rows[0].Account)
	}
}

// TestPaneJSONRoundTrips guards the machine-readable envelope stays a stable
// object (the --json surface).
func TestPaneJSONRoundTrips(t *testing.T) {
	p := Fold(fixtureInputs())
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back Pane
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Schema != Schema {
		t.Fatalf("schema round-trip: got %q", back.Schema)
	}
}
