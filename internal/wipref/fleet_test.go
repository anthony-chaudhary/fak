package wipref

import (
	"strings"
	"testing"
)

// ---- the object-size gate (#3880) ----

func TestClassifyPublishGatesOnlyAProvenOversizeDelta(t *testing.T) {
	for _, tc := range []struct {
		name  string
		bytes int64
		bound int64
		want  PublishClass
	}{
		{"under the bound rides the push", 100, 1000, PublishFull},
		{"exactly at the bound rides the push", 1000, 1000, PublishFull},
		{"over the bound is withheld", 1001, 1000, PublishMetadataOnly},
		{"unmeasured is never withheld", 0, 1000, PublishFull},
		{"a zero bound is unbounded", 1 << 30, 0, PublishFull},
		{"a negative bound is unbounded", 1 << 30, -1, PublishFull},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyPublish(Stamp{DeltaBytes: tc.bytes}, tc.bound)
			if got != tc.want {
				t.Fatalf("ClassifyPublish(%d, bound %d) = %s, want %s", tc.bytes, tc.bound, got, tc.want)
			}
		})
	}
}

// A pre-#3880 stamp carries no DeltaBytes at all. The gate must leave such a namespace
// behaving exactly as it did before the gate existed — otherwise turning the default
// bound on would silently stop replicating every ref already in the namespace.
func TestPlanPublishLeavesALegacyNamespaceUngated(t *testing.T) {
	recs := []RefRecord{
		{Ref: SessionRef("a"), Object: "aaa", Stamp: Stamp{SessionID: "a"}},
		{Ref: SessionRef("b"), Object: "bbb", Stamp: Stamp{SessionID: "b"}},
	}
	plan := PlanPublish(recs, DefaultMaxDeltaBytes)
	if plan.Gated() || plan.MetadataOnly != 0 || plan.Full != 2 {
		t.Fatalf("legacy namespace graded %+v, want ungated with 2 FULL", plan)
	}
	batches, err := PushRefspecs(plan, nil)
	if err != nil {
		t.Fatalf("PushRefspecs: %v", err)
	}
	if len(batches) != 1 || len(batches[0]) != 1 || batches[0][0] != PushRefspec {
		t.Fatalf("ungated plan pushed %v, want the single glob %q", batches, PushRefspec)
	}
}

func TestPlanPublishCountsSortsAndDropsUnsafeRecords(t *testing.T) {
	recs := []RefRecord{
		{Ref: SessionRef("zeta"), Object: "z1", Stamp: Stamp{SessionID: "zeta", DeltaBytes: 10}},
		{Ref: SessionRef("alpha"), Object: "a1", Stamp: Stamp{SessionID: "alpha", DeltaBytes: 5000}},
		{Ref: "refs/fak/wip/bad name", Object: "b1", Stamp: Stamp{SessionID: "bad name"}},
		{Ref: SessionRef("noobj"), Object: "", Stamp: Stamp{SessionID: "noobj"}},
	}
	plan := PlanPublish(recs, 1000)
	if len(plan.Entries) != 2 {
		t.Fatalf("kept %d entries, want 2 (an unsafe session id and an objectless ref are dropped): %+v", len(plan.Entries), plan.Entries)
	}
	if plan.Entries[0].Session != "alpha" || plan.Entries[1].Session != "zeta" {
		t.Fatalf("entries not session-sorted: %+v", plan.Entries)
	}
	if plan.Entries[0].Class != PublishMetadataOnly || plan.Entries[1].Class != PublishFull {
		t.Fatalf("classes wrong: %+v", plan.Entries)
	}
	if plan.Full != 1 || plan.MetadataOnly != 1 || !plan.Gated() {
		t.Fatalf("counts wrong: %+v", plan)
	}
}

// The load-bearing property of the gate: once ANY delta is withheld the glob refspec is
// off the table, because git has no negative refspec and the glob would carry exactly the
// objects being suppressed.
func TestPushRefspecsGatedSubstitutesTheMetadataObjectAndNeverGlobs(t *testing.T) {
	plan := PlanPublish([]RefRecord{
		{Ref: SessionRef("fat"), Object: "f1", Stamp: Stamp{SessionID: "fat", DeltaBytes: 9999}},
		{Ref: SessionRef("thin"), Object: "t1", Stamp: Stamp{SessionID: "thin", DeltaBytes: 1}},
	}, 100)
	batches, err := PushRefspecs(plan, map[string]string{"fat": "meta1"})
	if err != nil {
		t.Fatalf("PushRefspecs: %v", err)
	}
	if len(batches) != 1 {
		t.Fatalf("got %d batches, want 1", len(batches))
	}
	got := strings.Join(batches[0], " ")
	if strings.Contains(got, "*") {
		t.Fatalf("a gated push must not carry a glob refspec: %q", got)
	}
	wantFat := "+meta1:" + SessionRef("fat")
	wantThin := "+" + SessionRef("thin") + ":" + SessionRef("thin")
	if !strings.Contains(got, wantFat) {
		t.Fatalf("withheld session did not push its metadata object: %q wants %q", got, wantFat)
	}
	if !strings.Contains(got, wantThin) {
		t.Fatalf("full session did not push its own ref: %q wants %q", got, wantThin)
	}
	if strings.Contains(got, "f1") {
		t.Fatalf("the withheld checkpoint object leaked into the refspecs: %q", got)
	}
}

// A withheld session with no substitute must FAIL. Skipping it would drop the session out
// of the fleet view silently; falling back to a full push would publish the very bytes the
// gate exists to withhold.
func TestPushRefspecsRefusesAWithheldSessionWithNoMetadataObject(t *testing.T) {
	plan := PlanPublish([]RefRecord{
		{Ref: SessionRef("fat"), Object: "f1", Stamp: Stamp{SessionID: "fat", DeltaBytes: 9999}},
	}, 100)
	if _, err := PushRefspecs(plan, nil); err == nil {
		t.Fatal("PushRefspecs accepted a gated plan with no metadata object; want an error")
	}
	if _, err := PushRefspecs(plan, map[string]string{"fat": "   "}); err == nil {
		t.Fatal("PushRefspecs accepted a blank metadata object; want an error")
	}
}

func TestPushRefspecsBatchesAGatedPushSoArgvStaysBounded(t *testing.T) {
	var recs []RefRecord
	for i := 0; i < pushRefspecBatch*2+3; i++ {
		id := "s" + strings.Repeat("0", 3-len(itoa(i))) + itoa(i)
		recs = append(recs, RefRecord{Ref: SessionRef(id), Object: "o" + itoa(i), Stamp: Stamp{SessionID: id, DeltaBytes: 1}})
	}
	// One oversize ref is enough to force the explicit-refspec path for the whole plan.
	recs[0].Stamp.DeltaBytes = 1 << 20
	plan := PlanPublish(recs, 1024)
	batches, err := PushRefspecs(plan, map[string]string{recs[0].Stamp.SessionID: "meta"})
	if err != nil {
		t.Fatalf("PushRefspecs: %v", err)
	}
	if len(batches) != 3 {
		t.Fatalf("got %d batches for %d entries, want 3 at batch size %d", len(batches), len(plan.Entries), pushRefspecBatch)
	}
	total := 0
	for _, b := range batches {
		if len(b) > pushRefspecBatch {
			t.Fatalf("batch of %d exceeds the %d cap", len(b), pushRefspecBatch)
		}
		total += len(b)
	}
	if total != len(plan.Entries) {
		t.Fatalf("batches carry %d refspecs, want %d — a session was dropped", total, len(plan.Entries))
	}
}

func TestPushRefspecsEmptyPlanPushesNothing(t *testing.T) {
	batches, err := PushRefspecs(PlanPublish(nil, DefaultMaxDeltaBytes), nil)
	if err != nil {
		t.Fatalf("PushRefspecs: %v", err)
	}
	if len(batches) != 0 {
		t.Fatalf("an empty namespace produced %v, want no push at all", batches)
	}
}

// The metadata stub travels as a commit message like any checkpoint, so the fleet fields
// must survive the same encode/decode the spine already uses. A stub whose MetadataOnly
// bit was lost in transit would read on the peer as a REAL, reclaimable delta over an
// empty tree — the single most dangerous misread this ticket can produce.
func TestMetadataStampRoundTripsThroughTheCommitMessage(t *testing.T) {
	real := Stamp{
		SessionID: "sess", StartSHA: "deadbeef", Leaves: []string{"internal/wipref"},
		Scope: []string{"internal/wipref/fleet.go"}, Buildable: true,
		CheckpointedAt: 1700, Host: "box-7", DeltaBytes: 5 << 20,
	}
	msg, err := EncodeStamp(MetadataStamp(real, "cafebabe"))
	if err != nil {
		t.Fatalf("EncodeStamp: %v", err)
	}
	got, ok := DecodeStamp("some prose a hook appended\n" + msg + "\nmore prose")
	if !ok {
		t.Fatal("DecodeStamp did not find the stub stamp")
	}
	if !got.MetadataOnly {
		t.Fatal("MetadataOnly was lost — a peer would read the stub as recoverable work")
	}
	if got.DeltaObject != "cafebabe" {
		t.Fatalf("DeltaObject = %q, want cafebabe", got.DeltaObject)
	}
	if got.Host != "box-7" || got.DeltaBytes != 5<<20 || got.SessionID != "sess" || got.StartSHA != "deadbeef" {
		t.Fatalf("identity fields did not survive: %+v", got)
	}
	if real.MetadataOnly || real.DeltaObject != "" {
		t.Fatalf("MetadataStamp mutated its input: %+v", real)
	}
}

// ---- the fleet fold (#3880) ----

// mirrored builds the record a fetch leaves in THIS clone's mirror of remote.
func mirrored(remote, session string, s Stamp) RefRecord {
	s.SessionID = session
	return RefRecord{Ref: MirrorSessionRef(remote, session), Object: "obj-" + session, Stamp: s}
}

// The two-clone fold this ticket's done condition names: clone B fetched clone A's
// namespace, and every distinct kind of row it can now hold must grade correctly in ONE
// pass — a peer's recoverable delta, a peer's size-gated stub, B's own session coming back
// over the mirror, and a ref whose bytes never arrived.
func TestFoldFleetEnumeratesAPeerHostsStrandedCheckpoint(t *testing.T) {
	const remote = "origin"
	recs := []RefRecord{
		mirrored(remote, "crashed", Stamp{Host: "box-a", StartSHA: "base1", CheckpointedAt: 900, Leaves: []string{"internal/wipref"}, Buildable: true, DeltaBytes: 4096}),
		mirrored(remote, "huge", MetadataStamp(Stamp{Host: "box-a", CheckpointedAt: 950, DeltaBytes: 9 << 20}, "withheld-oid")),
		mirrored(remote, "mine", Stamp{Host: "box-b", CheckpointedAt: 990}),
		mirrored(remote, "gone", Stamp{Host: "box-c", CheckpointedAt: 800}),
	}
	present := map[string]bool{"obj-crashed": true, "obj-huge": true, "obj-mine": true}
	rep := FoldFleet(remote, recs, map[string]bool{"mine": true}, present, 1000)

	if rep.Count != 4 || rep.Remote != remote {
		t.Fatalf("report shape wrong: %+v", rep)
	}
	bySession := map[string]FleetRow{}
	for _, r := range rep.Rows {
		bySession[r.Session] = r
	}
	// The headline: a peer host's crashed-session WIP is enumerable and named as
	// recoverable, with the OWNER HOST attached — the thing no verb could report before.
	crashed := bySession["crashed"]
	if crashed.Disposition != FleetReclaimable {
		t.Fatalf("peer crash checkpoint graded %s, want %s", crashed.Disposition, FleetReclaimable)
	}
	if crashed.Host != "box-a" {
		t.Fatalf("owner host = %q, want box-a", crashed.Host)
	}
	if crashed.AgeSeconds != 100 || crashed.StartSHA != "base1" || crashed.DeltaBytes != 4096 {
		t.Fatalf("peer row lost its facts: %+v", crashed)
	}
	// A size-gated stub is VISIBLE but must never be called recoverable from here.
	huge := bySession["huge"]
	if huge.Disposition != FleetMetadataOnly {
		t.Fatalf("size-gated stub graded %s, want %s", huge.Disposition, FleetMetadataOnly)
	}
	if huge.DeltaObject != "withheld-oid" {
		t.Fatalf("stub row cannot name the withheld delta: %+v", huge)
	}
	// This clone's own session coming back over the mirror is not a strandee.
	if bySession["mine"].Disposition != FleetOwnLocal {
		t.Fatalf("own session graded %s, want %s", bySession["mine"].Disposition, FleetOwnLocal)
	}
	// A ref whose object never arrived is fail-safe, never reclaimable.
	if bySession["gone"].Disposition != FleetObjectMissing {
		t.Fatalf("absent object graded %s, want %s", bySession["gone"].Disposition, FleetObjectMissing)
	}
	if rep.Reclaimable != 1 || rep.MetadataOnly != 1 || rep.OwnLocal != 1 || rep.ObjectMissing != 1 {
		t.Fatalf("census does not match the rows: %+v", rep)
	}
	if got := strings.Join(rep.Hosts, ","); got != "box-a,box-b,box-c" {
		t.Fatalf("hosts = %q, want box-a,box-b,box-c", got)
	}
	if !strings.Contains(FleetSummary(rep), "1 RECLAIMABLE") {
		t.Fatalf("summary does not state the reclaimable count: %q", FleetSummary(rep))
	}
}

// Ownership outranks everything: a clone that checkpointed again since its last sync sees
// its own session at a DIFFERENT object in the mirror, and must still recognize it as its
// own rather than report itself as a peer strandee.
func TestFoldFleetOwnSessionAtAStaleObjectIsStillOwn(t *testing.T) {
	rec := mirrored("origin", "s1", Stamp{Host: "box-b", CheckpointedAt: 10})
	rep := FoldFleet("origin", []RefRecord{rec}, map[string]bool{"s1": true}, map[string]bool{rec.Object: true}, 100)
	if rep.Rows[0].Disposition != FleetOwnLocal || rep.OwnLocal != 1 {
		t.Fatalf("own session at a stale object graded %s, want %s", rep.Rows[0].Disposition, FleetOwnLocal)
	}
}

// A pre-#3880 peer ref carries no host and, if its stamp did not decode at all, no session
// either. Both must degrade to a labelled row rather than an empty key that collides every
// unstamped ref onto one another.
func TestFoldFleetLabelsAnUnstampedPeerRef(t *testing.T) {
	recs := []RefRecord{
		{Ref: MirrorSessionRef("origin", "legacy1"), Object: "o1"},
		{Ref: MirrorSessionRef("origin", "legacy2"), Object: "o2"},
	}
	rep := FoldFleet("origin", recs, nil, map[string]bool{"o1": true, "o2": true}, 100)
	if rep.Count != 2 {
		t.Fatalf("unstamped refs collapsed: %+v", rep.Rows)
	}
	if rep.Rows[0].Session != "legacy1" || rep.Rows[1].Session != "legacy2" {
		t.Fatalf("unstamped refs lost their ref-name labels: %+v", rep.Rows)
	}
	for _, r := range rep.Rows {
		if r.Host != HostUnknown {
			t.Fatalf("hostless stamp reported host %q, want %s", r.Host, HostUnknown)
		}
		if r.AgeSeconds != 0 {
			t.Fatalf("an unstamped ref has no capture time; age = %d, want 0 rather than an epoch-derived lie", r.AgeSeconds)
		}
	}
}

func TestFoldFleetSortsByHostThenSession(t *testing.T) {
	recs := []RefRecord{
		mirrored("origin", "b", Stamp{Host: "box-z"}),
		mirrored("origin", "a", Stamp{Host: "box-z"}),
		mirrored("origin", "c", Stamp{Host: "box-a"}),
	}
	rep := FoldFleet("origin", recs, nil, nil, 0)
	var got []string
	for _, r := range rep.Rows {
		got = append(got, r.Host+"/"+r.Session)
	}
	if strings.Join(got, " ") != "box-a/c box-z/a box-z/b" {
		t.Fatalf("fleet rows not (host, session)-sorted: %v", got)
	}
}

// An EMPTY fleet listing is the reading this ticket most has to protect: "no peer has
// stranded work" is a claim about OTHER HOSTS, and it is only licensed by a real fetch.
func TestFoldFleetEmptyListingCarriesItsProvenance(t *testing.T) {
	rep := FoldFleet("origin", nil, nil, nil, 1000)
	if rep.Count != 0 || len(rep.Rows) != 0 {
		t.Fatalf("empty mirror folded %+v", rep)
	}
	view := ClassifyMirror("origin", MirrorStamp{}, false, 0, 1000, 0)
	rep.Mirror = &view
	if rep.Mirror.EmptyIsAbsence {
		t.Fatal("a never-synced clone was licensed to read an empty fleet listing as absence")
	}
	if MirrorCaveat(*rep.Mirror) == "" {
		t.Fatal("an unlicensed empty fleet listing carries no caveat")
	}
}

// itoa keeps the batching test free of a strconv import for one digit-string helper.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
