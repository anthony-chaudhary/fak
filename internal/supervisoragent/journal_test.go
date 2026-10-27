package supervisoragent

import (
	"encoding/json"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"
)

// walkRecordFields flattens a type into a stable, sorted set of "path:kind"
// leaf descriptors, mirroring input_test.go's walkFields with one extension:
// time.Time reduces to "time" instead of expanding its unexported
// implementation fields, which are not part of the contract.
func walkRecordFields(t reflect.Type, path string, out *[]string) {
	if t == reflect.TypeOf(time.Time{}) {
		*out = append(*out, path+":time")
		return
	}
	switch t.Kind() {
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			child := f.Name
			if path != "" {
				child = path + "." + f.Name
			}
			walkRecordFields(f.Type, child, out)
		}
	case reflect.Slice, reflect.Array:
		walkRecordFields(t.Elem(), path+"[]", out)
	default:
		*out = append(*out, path+":"+t.Kind().String())
	}
}

// TestDecisionRecordClosedJSONKeySet is the golden closed-field pin (S1's
// Witness): marshaling a DecisionRecord yields EXACTLY these top-level keys —
// no transcript, log body, free-text rationale, or payload key may appear.
// Adding a field fails this test until the closed set is DELIBERATELY updated,
// forcing a reviewer to see the payload creeping in.
func TestDecisionRecordClosedJSONKeySet(t *testing.T) {
	rec := DecisionRecord{
		Seq:             7,
		Topic:           "wake://fleet/cmd",
		InputDigest:     DigestInput(presentInput()),
		ProposedVerb:    ActionReplace,
		ReasonToken:     ReasonReplaceStaleLease,
		EnvelopeVerdict: EnvelopeVerdict{Action: ActionReplace, Mode: ModeEnforce, Auth: AuthUnattended, Demand: 3, Earned: 3, Reason: ReasonSuperEnvelope},
		Outcome:         OutcomeLowered,
		ArtifactRef:     "admit:RID-2#4480",
		DecidedAt:       time.Date(2026, 9, 21, 12, 0, 0, 0, time.UTC),
	}
	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	got := make([]string, 0, len(m))
	for k := range m {
		got = append(got, k)
	}
	sort.Strings(got)
	want := []string{
		"ArtifactRef",
		"DecidedAt",
		"EnvelopeVerdict",
		"InputDigest",
		"Outcome",
		"ProposedVerb",
		"ReasonToken",
		"Seq",
		"Topic",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("DecisionRecord JSON key set drifted from the closed contract.\n got: %v\nwant: %v", got, want)
	}
}

// TestDecisionRecordClosedFieldSet is the reflection-side twin of the JSON pin:
// the full leaf field set (including the embedded EnvelopeVerdict) is exactly
// the closed list. The two pins together make adding prose a two-place,
// obviously-wrong change.
func TestDecisionRecordClosedFieldSet(t *testing.T) {
	want := []string{
		"ArtifactRef:string",
		"DecidedAt:time",
		"EnvelopeVerdict.Action:string",
		"EnvelopeVerdict.Auth:string",
		"EnvelopeVerdict.Demand:int",
		"EnvelopeVerdict.Earned:int",
		"EnvelopeVerdict.Finding:string",
		"EnvelopeVerdict.Mode:string",
		"EnvelopeVerdict.Reason:string",
		"InputDigest:string",
		"Outcome:string",
		"ProposedVerb:string",
		"ReasonToken:string",
		"Seq:uint64",
		"Topic:string",
	}
	sort.Strings(want)

	var got []string
	walkRecordFields(reflect.TypeOf(DecisionRecord{}), "", &got)
	sort.Strings(got)

	if !reflect.DeepEqual(got, want) {
		t.Errorf("DecisionRecord field set drifted from the closed contract.\n got: %v\nwant: %v", got, want)
	}
}

// TestDecisionRecordNoPayloadFieldName is the intent guard paired with the
// pins: no field name anywhere in the record may name a transcript / message /
// body / free-text / log payload.
func TestDecisionRecordNoPayloadFieldName(t *testing.T) {
	forbidden := []string{
		"transcript", "message", "msg", "body", "payload", "prose",
		"freetext", "rawlog", "log", "content", "chat", "narrative",
		"summary", "comment", "text",
	}
	var names []string
	collectFieldNames(reflect.TypeOf(DecisionRecord{}), &names)
	for _, n := range names {
		low := strings.ToLower(n)
		for _, bad := range forbidden {
			if strings.Contains(low, bad) {
				t.Errorf("record field %q contains forbidden payload token %q — a transcript/payload field turns the journal into a store", n, bad)
			}
		}
	}
}

// TestMemoryJournalGoldenBehavior pins the golden behavior every durable
// Journal must match: seq monotonic from 1 (the journal owns the sequence and
// overwrites any caller-set Seq), replay stable across calls and resumable
// from a cursor, and last after append.
func TestMemoryJournalGoldenBehavior(t *testing.T) {
	j := NewMemoryJournal()
	if _, ok := j.Last(); ok {
		t.Fatal("Last() on an empty journal must report false")
	}

	base := time.Date(2026, 9, 21, 12, 0, 0, 0, time.UTC)
	digest := DigestInput(presentInput())
	for i := 0; i < 5; i++ {
		err := j.Append(DecisionRecord{
			Seq:          999, // caller-set Seq must be ignored: the journal owns the sequence
			Topic:        "wake://fleet/cmd",
			InputDigest:  digest,
			ProposedVerb: ActionHold,
			ReasonToken:  ReasonHoldGreen,
			Outcome:      OutcomeHold,
			DecidedAt:    base.Add(time.Duration(i) * time.Minute),
		})
		if err != nil {
			t.Fatalf("append %d: %v", i, err)
		}
	}

	// Seq monotonic from 1.
	all, err := j.Replay(0)
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	if len(all) != 5 {
		t.Fatalf("Replay(0) returned %d records; want 5", len(all))
	}
	for i, r := range all {
		if r.Seq != uint64(i+1) {
			t.Errorf("record %d has Seq %d; want %d (monotonic from 1)", i, r.Seq, i+1)
		}
	}

	// Replay stable: same cursor, same records, in order, across calls.
	again, err := j.Replay(0)
	if err != nil {
		t.Fatalf("replay again: %v", err)
	}
	if !reflect.DeepEqual(all, again) {
		t.Error("Replay(0) is not stable across calls")
	}

	// Cursor resume: Replay(k) returns the tail from Seq k.
	tail, err := j.Replay(4)
	if err != nil {
		t.Fatalf("replay tail: %v", err)
	}
	if len(tail) != 2 || tail[0].Seq != 4 || tail[1].Seq != 5 {
		t.Errorf("Replay(4) = %+v; want the tail starting at Seq 4", tail)
	}

	// A cursor past the tail yields an empty (non-nil) slice.
	past, err := j.Replay(6)
	if err != nil {
		t.Fatalf("replay past: %v", err)
	}
	if past == nil || len(past) != 0 {
		t.Errorf("Replay past the tail = %v; want an empty non-nil slice", past)
	}

	// Last after append.
	last, ok := j.Last()
	if !ok || last.Seq != 5 || !last.DecidedAt.Equal(base.Add(4*time.Minute)) {
		t.Errorf("Last() = %+v, %v; want the fifth record", last, ok)
	}
}

// TestDigestInputDeterministicAndSensitive pins the digest seam: deterministic
// for equal inputs, sensitive to a surface flipping present/absent (a green
// absence must not collide with a green presence) and to content changes, and
// hex-shaped.
func TestDigestInputDeterministicAndSensitive(t *testing.T) {
	a := presentInput()
	if DigestInput(a) != DigestInput(a) {
		t.Error("DigestInput is not deterministic for an equal input")
	}

	b := withholdWorkers(a)
	if DigestInput(a) == DigestInput(b) {
		t.Error("digest collides between a present and an absent surface")
	}

	c := a
	c.Liveness.Value.Commits = 4
	if DigestInput(a) == DigestInput(c) {
		t.Error("digest collides across different input contents")
	}

	d := DigestInput(a)
	if len(d) != 64 {
		t.Fatalf("digest %q is not 64 hex chars", d)
	}
	for i := 0; i < len(d); i++ {
		ch := d[i]
		if (ch < '0' || ch > '9') && (ch < 'a' || ch > 'f') {
			t.Fatalf("digest %q is not lowercase hex", d)
		}
	}
}
