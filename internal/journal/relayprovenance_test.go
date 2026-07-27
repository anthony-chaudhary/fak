package journal

import (
	"path/filepath"
	"strings"
	"testing"
)

// relaySession is the fixture conversation for these tests: one session key whose
// messages hop Telegram -> Slack -> CLI, in both directions. It is the exact shape
// #2851 names as the thing an authenticated socket cannot prove.
const relaySession = "sess-cross-platform-1"

// hop is one relayed message in a fixture conversation.
type hop struct {
	direction   string
	platform    string
	user        string
	turn        string
	destination string
	verdict     string
	reason      string
	body        string
	redactions  int
}

// crossPlatformHops is a conversation that genuinely crosses platforms: a user opens
// on Telegram, the agent answers there, the thread moves to Slack (where one outbound
// message is DENIED by the delivery floor), and finishes on the CLI.
var crossPlatformHops = []hop{
	{RelayInbound, "telegram", "tg:4471", "turn-1", "", RelayAllow, "", "where did the deploy go?", 0},
	{RelayOutbound, "telegram", "tg:4471", "turn-1", "chat:9001", RelayAllow, "", "it rolled back at 14:02", 0},
	{RelayInbound, "slack", "U07HUMAN", "turn-2", "", RelayAllow, "", "moving this to #ops", 0},
	{RelayOutbound, "slack", "U07HUMAN", "turn-2", "C0NOTALLOWED", RelayDeny, "DELIVERY_BLOCK", "token is xoxb-…", 1},
	{RelayInbound, "cli", "local", "turn-3", "", RelayAllow, "", "show me the trail", 0},
}

// recordHops appends hops to j under chain c for sessionKey and returns the rows.
func recordHops(t *testing.T, c *RelayChain, j *Journal, sessionKey string, hops []hop) []Row {
	t.Helper()
	out := make([]Row, 0, len(hops))
	for i, h := range hops {
		row := c.Append(j, RelayProvenance{
			Direction:   h.direction,
			Platform:    h.platform,
			UserID:      h.user,
			SessionKey:  sessionKey,
			TurnID:      h.turn,
			Destination: h.destination,
			Verdict:     h.verdict,
			Reason:      h.reason,
			BodyDigest:  RelayBodyDigest(h.body),
			Redactions:  h.redactions,
		})
		if row.Seq == 0 {
			t.Fatalf("hop %d (%s/%s): Append returned an uncommitted row", i, h.platform, h.direction)
		}
		out = append(out, row)
	}
	return out
}

// TestRelayProvenanceCrossPlatformTrailIsChainLinked is the issue's witness (#2851):
// a conversation spanning Telegram -> Slack -> CLI produces one chain-linked
// provenance record per relayed message, readable back as an ordered trail for the
// session and verifiable against the journal's own hash chain.
func TestRelayProvenanceCrossPlatformTrailIsChainLinked(t *testing.T) {
	j := OpenMemory()
	c := NewRelayChain()
	recordHops(t, c, j, relaySession, crossPlatformHops)

	rows := j.Recent(0)
	trail := RelayProvenanceFor(rows, relaySession)
	if len(trail) != len(crossPlatformHops) {
		t.Fatalf("trail length = %d, want %d", len(trail), len(crossPlatformHops))
	}

	// Every axis the issue names is on the record: platform, user id, session key,
	// turn id, and the adjudication verdict.
	for i, e := range trail {
		want := crossPlatformHops[i]
		r := e.Record
		if r.Schema != RelayProvenanceSchema {
			t.Errorf("entry %d: schema = %q, want %q", i, r.Schema, RelayProvenanceSchema)
		}
		if r.Platform != want.platform || r.Direction != want.direction {
			t.Errorf("entry %d: platform/direction = %s/%s, want %s/%s",
				i, r.Platform, r.Direction, want.platform, want.direction)
		}
		if r.UserID != want.user || r.TurnID != want.turn {
			t.Errorf("entry %d: user/turn = %s/%s, want %s/%s", i, r.UserID, r.TurnID, want.user, want.turn)
		}
		if r.SessionKey != relaySession {
			t.Errorf("entry %d: session key = %q, want %q", i, r.SessionKey, relaySession)
		}
		if r.Verdict != want.verdict || r.Reason != want.reason {
			t.Errorf("entry %d: verdict/reason = %s/%s, want %s/%s",
				i, r.Verdict, r.Reason, want.verdict, want.reason)
		}
		// The body is content-addressed, never transcribed.
		if r.BodyDigest != RelayBodyDigest(want.body) {
			t.Errorf("entry %d: body digest = %q, want %q", i, r.BodyDigest, RelayBodyDigest(want.body))
		}
		if strings.Contains(r.BodyDigest, want.body) {
			t.Errorf("entry %d: body digest leaks the body", i)
		}
	}

	// The per-session links form a chain: genesis first, then each entry naming its
	// predecessor's committed Seq/Hash.
	if trail[0].Record.PrevSeq != 0 || trail[0].Record.PrevHash != "" {
		t.Errorf("first entry is not genesis: prev_seq=%d prev_hash=%q",
			trail[0].Record.PrevSeq, trail[0].Record.PrevHash)
	}
	for i := 1; i < len(trail); i++ {
		if trail[i].Record.PrevSeq != trail[i-1].Seq {
			t.Errorf("entry %d: prev_seq = %d, want %d", i, trail[i].Record.PrevSeq, trail[i-1].Seq)
		}
		if trail[i].Record.PrevHash != trail[i-1].Hash {
			t.Errorf("entry %d: prev_hash = %s, want %s", i, trail[i].Record.PrevHash, trail[i-1].Hash)
		}
	}

	if err := VerifyRelayTrail(rows, trail); err != nil {
		t.Fatalf("VerifyRelayTrail: %v", err)
	}

	// It really did cross platforms, in order.
	got := RelayPlatforms(trail)
	wantPlatforms := []string{"telegram", "slack", "cli"}
	if len(got) != len(wantPlatforms) {
		t.Fatalf("platforms = %v, want %v", got, wantPlatforms)
	}
	for i := range got {
		if got[i] != wantPlatforms[i] {
			t.Fatalf("platforms = %v, want %v", got, wantPlatforms)
		}
	}

	// The chained head advanced with the conversation.
	seq, hash, ok := c.Head(relaySession)
	if !ok || seq != trail[len(trail)-1].Seq || hash != trail[len(trail)-1].Hash {
		t.Errorf("Head = (%d, %s, %v), want (%d, %s, true)",
			seq, hash, ok, trail[len(trail)-1].Seq, trail[len(trail)-1].Hash)
	}
}

// TestRelayProvenanceDeniedDeliveryIsWitnessed proves the audit trail records what the
// floor REFUSED, not just what it sent — the denied Slack hop is a first-class row
// carrying its verdict and refusal class on the CHAINED fields.
func TestRelayProvenanceDeniedDeliveryIsWitnessed(t *testing.T) {
	j := OpenMemory()
	c := NewRelayChain()
	rows := recordHops(t, c, j, relaySession, crossPlatformHops)

	denied := rows[3] // the Slack outbound the floor blocked
	if denied.Kind != KindRelayMsg {
		t.Fatalf("kind = %q, want %q", denied.Kind, KindRelayMsg)
	}
	if denied.Verdict != RelayDeny {
		t.Errorf("chained verdict = %q, want %q", denied.Verdict, RelayDeny)
	}
	if denied.Reason != "DELIVERY_BLOCK" {
		t.Errorf("chained reason = %q, want DELIVERY_BLOCK", denied.Reason)
	}
	if denied.Tool != "slack" {
		t.Errorf("chained tool (platform) = %q, want slack", denied.Tool)
	}
	if denied.TraceID != relaySession {
		t.Errorf("chained trace id (session key) = %q, want %q", denied.TraceID, relaySession)
	}
	if denied.By != "relay-"+RelayOutbound {
		t.Errorf("chained by (direction) = %q, want relay-%s", denied.By, RelayOutbound)
	}
	if denied.ArgsDigest != denied.Relay.BodyDigest {
		t.Errorf("chained args digest = %q, want the body digest %q", denied.ArgsDigest, denied.Relay.BodyDigest)
	}
	if denied.Relay.Redactions != 1 {
		t.Errorf("redactions = %d, want 1", denied.Relay.Redactions)
	}
}

// TestRelayProvenanceReadPathIsolatesSessions proves the read-path answers for ONE
// conversation even when the journal interleaves another session's relayed messages
// and unrelated row kinds — the realistic case for a relay fronting many platforms.
func TestRelayProvenanceReadPathIsolatesSessions(t *testing.T) {
	j := OpenMemory()
	c := NewRelayChain()
	const other = "sess-unrelated-2"

	// Interleave: ours, theirs, an unrelated row kind, ours, theirs.
	c.Append(j, RelayProvenance{Direction: RelayInbound, Platform: "telegram", SessionKey: relaySession, Verdict: RelayAllow})
	c.Append(j, RelayProvenance{Direction: RelayInbound, Platform: "discord", SessionKey: other, Verdict: RelayAllow})
	j.append(Row{Kind: "DECIDE", Tool: "Read", TraceID: relaySession, Verdict: "allow"})
	c.Append(j, RelayProvenance{Direction: RelayOutbound, Platform: "slack", SessionKey: relaySession, Verdict: RelayAllow})
	c.Append(j, RelayProvenance{Direction: RelayOutbound, Platform: "discord", SessionKey: other, Verdict: RelayAllow})

	rows := j.Recent(0)

	ours := RelayProvenanceFor(rows, relaySession)
	if len(ours) != 2 {
		t.Fatalf("our trail length = %d, want 2 (the DECIDE row must not fold in)", len(ours))
	}
	for i, e := range ours {
		if e.Record.SessionKey != relaySession {
			t.Errorf("our entry %d leaked session %q", i, e.Record.SessionKey)
		}
	}
	if err := VerifyRelayTrail(rows, ours); err != nil {
		t.Fatalf("VerifyRelayTrail(ours): %v", err)
	}

	theirs := RelayProvenanceFor(rows, other)
	if len(theirs) != 2 {
		t.Fatalf("their trail length = %d, want 2", len(theirs))
	}
	if err := VerifyRelayTrail(rows, theirs); err != nil {
		t.Fatalf("VerifyRelayTrail(theirs): %v", err)
	}
	// Each session's chain is independent: theirs starts at its OWN genesis even
	// though our row was committed first.
	if theirs[0].Record.PrevSeq != 0 || theirs[0].Record.PrevHash != "" {
		t.Errorf("the other session's first entry is not genesis: prev_seq=%d", theirs[0].Record.PrevSeq)
	}
	if theirs[1].Record.PrevSeq != theirs[0].Seq {
		t.Errorf("the other session's link crossed sessions: prev_seq=%d, want %d",
			theirs[1].Record.PrevSeq, theirs[0].Seq)
	}

	// An unknown session is an honest empty answer, not an error.
	if got := RelayProvenanceFor(rows, "sess-never-relayed"); got != nil {
		t.Errorf("unknown session returned %d entries, want none", len(got))
	}
}

// TestRelayProvenanceTamperedLinkIsCaught is the point of anchoring to the journal
// rather than keeping a side log: rewriting the correlation link (or the body digest,
// or the verdict) is DETECTED, because the payload only NAMES a predecessor and
// verification re-derives the name from the journal's own chained hashes.
func TestRelayProvenanceTamperedLinkIsCaught(t *testing.T) {
	build := func() ([]Row, []RelayEntry) {
		j := OpenMemory()
		c := NewRelayChain()
		recordHops(t, c, j, relaySession, crossPlatformHops)
		rows := j.Recent(0)
		return rows, RelayProvenanceFor(rows, relaySession)
	}

	// Baseline: untampered verifies.
	rows, trail := build()
	if err := VerifyRelayTrail(rows, trail); err != nil {
		t.Fatalf("baseline VerifyRelayTrail: %v", err)
	}

	cases := []struct {
		name string
		bend func(rows []Row, trail []RelayEntry)
		want string
	}{
		{
			// Re-point the correlation link at an earlier message to hide a hop.
			name: "prev_seq rewritten to hide a hop",
			bend: func(_ []Row, trail []RelayEntry) { trail[2].Record.PrevSeq = trail[0].Seq },
			want: "prev_seq",
		},
		{
			name: "prev_hash rewritten",
			bend: func(_ []Row, trail []RelayEntry) { trail[2].Record.PrevHash = trail[0].Hash },
			want: "prev_hash",
		},
		{
			// Rewrite a CHAINED field on the committed row: the hash no longer re-derives.
			name: "chained verdict flipped on the committed row",
			bend: func(rows []Row, _ []RelayEntry) {
				for i := range rows {
					if rows[i].Kind == KindRelayMsg && rows[i].Verdict == RelayDeny {
						rows[i].Verdict = RelayAllow
						return
					}
				}
				t.Fatal("fixture has no denied row to flip")
			},
			want: "does not re-derive",
		},
		{
			// Rewrite the chained body digest: same detection, so a swapped message body
			// cannot ride an otherwise-valid provenance record.
			name: "chained body digest swapped on the committed row",
			bend: func(rows []Row, _ []RelayEntry) {
				for i := range rows {
					if rows[i].Kind == KindRelayMsg {
						rows[i].ArgsDigest = RelayBodyDigest("a different message entirely")
						return
					}
				}
			},
			want: "does not re-derive",
		},
		{
			// Genesis forged: claim a predecessor that never existed.
			name: "genesis claims a predecessor",
			bend: func(_ []Row, trail []RelayEntry) {
				trail[0].Record.PrevSeq = 99
				trail[0].Record.PrevHash = "deadbeef"
			},
			want: "not genesis",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rows, trail := build()
			tc.bend(rows, trail)
			err := VerifyRelayTrail(rows, trail)
			if err == nil {
				t.Fatalf("tampering went undetected")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want it to name %q", err, tc.want)
			}
		})
	}
}

// TestRelayProvenanceJournalChainStaysVerifiable proves this rung is additive: a
// journal carrying RELAY_MSG rows interleaved with ordinary decision rows still
// verifies end-to-end with the existing chain verifier, and survives a file
// round-trip. That is what "anchored to the journal" has to mean to be worth anything.
func TestRelayProvenanceJournalChainStaysVerifiable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "relay-journal.jsonl")
	j, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	c := NewRelayChain()

	j.append(Row{Kind: "DECIDE", Tool: "Read", TraceID: relaySession, Verdict: "allow"})
	recordHops(t, c, j, relaySession, crossPlatformHops)
	j.append(Row{Kind: "DENY", Tool: "Bash", TraceID: relaySession, Verdict: "deny", Reason: "POLICY_BLOCK"})

	if err := j.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	n, err := Verify(path)
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if want := len(crossPlatformHops) + 2; n != want {
		t.Fatalf("verified %d rows, want %d", n, want)
	}

	// The trail survives the round-trip through disk, payload and all.
	rows, err := ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	trail := RelayProvenanceFor(rows, relaySession)
	if len(trail) != len(crossPlatformHops) {
		t.Fatalf("trail length after round-trip = %d, want %d", len(trail), len(crossPlatformHops))
	}
	if trail[0].Record.Platform != "telegram" || trail[len(trail)-1].Record.Platform != "cli" {
		t.Errorf("trail did not survive the round-trip: %s -> %s",
			trail[0].Record.Platform, trail[len(trail)-1].Record.Platform)
	}
	if err := VerifyRelayTrail(rows, trail); err != nil {
		t.Fatalf("VerifyRelayTrail after round-trip: %v", err)
	}
}

// TestRelayProvenanceNilSafe pins the unconditional-call contract: a caller that has
// not wired a journal (or a chain) may still call Append, and gets a zero Row.
func TestRelayProvenanceNilSafe(t *testing.T) {
	var nilChain *RelayChain
	if row := nilChain.Append(OpenMemory(), RelayProvenance{SessionKey: relaySession}); row.Seq != 0 {
		t.Errorf("nil chain: Append returned seq %d, want the zero Row", row.Seq)
	}
	if _, _, ok := nilChain.Head(relaySession); ok {
		t.Error("nil chain: Head reported a head")
	}
	if row := NewRelayChain().Append(nil, RelayProvenance{SessionKey: relaySession}); row.Seq != 0 {
		t.Errorf("nil journal: Append returned seq %d, want the zero Row", row.Seq)
	}
	if err := VerifyRelayTrail(nil, nil); err != nil {
		t.Errorf("empty trail: VerifyRelayTrail = %v, want nil", err)
	}
	if got := RelayPlatforms(nil); got != nil {
		t.Errorf("RelayPlatforms(nil) = %v, want nil", got)
	}
}

// TestRelayBodyDigest pins the content-address contract: an absent body is
// distinguishable from a present one, and equal bodies digest equally.
func TestRelayBodyDigest(t *testing.T) {
	if got := RelayBodyDigest(""); got != "" {
		t.Errorf("empty body digested to %q, want \"\"", got)
	}
	a := RelayBodyDigest("the deploy rolled back at 14:02")
	if !strings.HasPrefix(a, "sha256:") || len(a) != len("sha256:")+64 {
		t.Fatalf("digest = %q, want sha256:<64 hex>", a)
	}
	if b := RelayBodyDigest("the deploy rolled back at 14:02"); a != b {
		t.Errorf("digest is not deterministic: %q vs %q", a, b)
	}
	if b := RelayBodyDigest("the deploy rolled back at 14:03"); a == b {
		t.Error("different bodies digested equally")
	}
}
