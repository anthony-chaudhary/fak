package wipref

import (
	"strings"
	"testing"
)

// TestMirrorStampRoundTripsAndRefusesAForeignBlob: the stamp must survive encode/decode
// exactly, and a blob that is not one of ours must NOT decode — a ref pointed at some
// other object has to read as "no stamp", never as a fabricated sync time.
func TestMirrorStampRoundTripsAndRefusesAForeignBlob(t *testing.T) {
	in := MirrorStamp{Remote: "origin", Source: MirrorFromFetch, SyncedAt: 1700000000, Refs: 7}
	body, err := EncodeMirrorStamp(in)
	if err != nil {
		t.Fatalf("EncodeMirrorStamp: %v", err)
	}
	got, ok := DecodeMirrorStamp(body)
	if !ok || got != in {
		t.Fatalf("round trip = (%+v, %v), want (%+v, true)", got, ok, in)
	}
	for _, foreign := range []string{
		"",
		"just some text\n",
		`{"remote":"origin","synced_at":1700000000}`, // valid JSON, no marker
		"fak-wip: " + `{"session_id":"a"}`,           // the CHECKPOINT marker, not ours
		mirrorStampMarker + "not json",
	} {
		if st, ok := DecodeMirrorStamp(foreign); ok {
			t.Errorf("DecodeMirrorStamp(%q) accepted a foreign body as %+v", foreign, st)
		}
	}
}

// TestMirrorStampRefIsUnreachableFromBothCheckpointSweeps is the namespace safety
// argument as an assertion. The stamp ref must not sit under either checkpoint namespace
// — string-prefix OR git's path-component matching — because one sweep enumerates
// sessions (a stamp there becomes a phantom peer session) and one of the live namespace's
// readers is a deleter.
func TestMirrorStampRefIsUnreachableFromBothCheckpointSweeps(t *testing.T) {
	for _, remote := range []string{"origin", "peer", "ssh://host/repo.git"} {
		ref := MirrorStampRef(remote)
		for _, ns := range []string{RefNamespace, RemoteMirrorNamespace, MirrorNamespace(remote)} {
			if strings.HasPrefix(ref, ns) {
				t.Errorf("stamp ref %q nests under %q", ref, ns)
			}
		}
		// The careless form too: a HasPrefix written WITHOUT the trailing slash is the
		// exact hazard sync.go's namespace choice was made to defeat, so the stamp
		// namespace must survive it as well.
		for _, loose := range []string{"refs/fak/wip", "refs/fak/remotewip"} {
			if strings.HasPrefix(ref, loose) {
				t.Errorf("stamp ref %q is reachable from a slashless HasPrefix on %q", ref, loose)
			}
		}
		// One ref per remote, keyed by the SAME segment as the mirror it describes.
		if want := MirrorStampNamespace + MirrorSegment(remote); ref != want {
			t.Errorf("MirrorStampRef(%q) = %q, want %q", remote, ref, want)
		}
	}
	// The stamp can never ride the push refspec, so a peer's picture cannot land here.
	if strings.Contains(PushRefspec, MirrorStampNamespace) {
		t.Errorf("PushRefspec %q carries the stamp namespace", PushRefspec)
	}
}

// TestClassifyMirrorSeparatesIgnoranceFromAbsence is the point of the unit. Every state
// that is NOT "a fetch surveyed the remote recently" must refuse EmptyIsAbsence, because
// under each of them an empty mirror is this clone not knowing rather than the remote not
// holding.
func TestClassifyMirrorSeparatesIgnoranceFromAbsence(t *testing.T) {
	const now = 1_000_000
	const tol = 3600

	cases := []struct {
		name       string
		stamp      MirrorStamp
		ok         bool
		wantFresh  MirrorFreshness
		wantAbsent bool
		wantAge    int64
	}{{
		name: "never synced: no stamp at all", ok: false,
		wantFresh: MirrorNeverSynced, wantAbsent: false, wantAge: -1,
	}, {
		name:  "unreadable stamp decoded to a zero time",
		stamp: MirrorStamp{Remote: "origin", Source: MirrorFromFetch}, ok: true,
		wantFresh: MirrorNeverSynced, wantAbsent: false, wantAge: -1,
	}, {
		name:  "fresh fetch: the ONLY state that licenses reading a zero",
		stamp: MirrorStamp{Remote: "origin", Source: MirrorFromFetch, SyncedAt: now - 60}, ok: true,
		wantFresh: MirrorFresh, wantAbsent: true, wantAge: 60,
	}, {
		name:  "stale fetch: looked, but long ago",
		stamp: MirrorStamp{Remote: "origin", Source: MirrorFromFetch, SyncedAt: now - tol - 1}, ok: true,
		wantFresh: MirrorStale, wantAbsent: false, wantAge: tol + 1,
	}, {
		name:  "fresh push: published, never surveyed",
		stamp: MirrorStamp{Remote: "origin", Source: MirrorFromPush, SyncedAt: now - 60}, ok: true,
		wantFresh: MirrorFresh, wantAbsent: false, wantAge: 60,
	}, {
		name:  "unknown source is not a fetch",
		stamp: MirrorStamp{Remote: "origin", Source: MirrorSource("SOMETHING"), SyncedAt: now - 60}, ok: true,
		wantFresh: MirrorFresh, wantAbsent: false, wantAge: 60,
	}, {
		name:  "stamp belongs to a remote that folds onto the same segment",
		stamp: MirrorStamp{Remote: "ssh://a/x", Source: MirrorFromFetch, SyncedAt: now - 60}, ok: true,
		wantFresh: MirrorNeverSynced, wantAbsent: false, wantAge: -1,
	}, {
		name:  "exactly at the tolerance is still fresh",
		stamp: MirrorStamp{Remote: "origin", Source: MirrorFromFetch, SyncedAt: now - tol}, ok: true,
		wantFresh: MirrorFresh, wantAbsent: true, wantAge: tol,
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			v := ClassifyMirror("origin", c.stamp, c.ok, 0, now, tol)
			if v.Freshness != c.wantFresh {
				t.Errorf("freshness = %q, want %q", v.Freshness, c.wantFresh)
			}
			if v.EmptyIsAbsence != c.wantAbsent {
				t.Errorf("EmptyIsAbsence = %v, want %v", v.EmptyIsAbsence, c.wantAbsent)
			}
			if v.AgeSeconds != c.wantAge {
				t.Errorf("AgeSeconds = %d, want %d", v.AgeSeconds, c.wantAge)
			}
			// A reader that CANNOT read a zero as absence must always be handed the
			// sentence that says so; one that can must not be nagged with a caveat.
			caveat := MirrorCaveat(v)
			if c.wantAbsent && caveat != "" {
				t.Errorf("licensed view still carries a caveat: %q", caveat)
			}
			if !c.wantAbsent && caveat == "" {
				t.Error("unlicensed view carries no caveat — a reader would print the zero unqualified")
			}
		})
	}
}

// TestClassifyMirrorNeverSyncedAgeIsNegativeNotZero pins the sentinel choice on its own,
// because it is the field a naive renderer reaches for first: a 0 there reads as "synced
// just now", which is the precise overstatement this type exists to prevent.
func TestClassifyMirrorNeverSyncedAgeIsNegativeNotZero(t *testing.T) {
	v := ClassifyMirror("origin", MirrorStamp{}, false, 0, 1_000_000, 0)
	if v.AgeSeconds >= 0 {
		t.Errorf("AgeSeconds = %d, want a negative sentinel", v.AgeSeconds)
	}
	if v.SyncedAt != 0 || v.Source != "" {
		t.Errorf("never-synced view invented provenance: synced_at=%d source=%q", v.SyncedAt, v.Source)
	}
	if v.MaxAgeSeconds != DefaultMirrorMaxAgeSeconds {
		t.Errorf("MaxAgeSeconds = %d, want the default %d", v.MaxAgeSeconds, DefaultMirrorMaxAgeSeconds)
	}
	if v.Namespace != MirrorNamespace("origin") {
		t.Errorf("Namespace = %q, want %q", v.Namespace, MirrorNamespace("origin"))
	}
}

// TestClassifyMirrorClampsAFutureStamp: a forward-dated stamp is this machine's clock
// moving, not a claim from elsewhere, so it clamps to age 0 exactly as Fold clamps a
// future checkpoint rather than producing a negative age some consumer would format as a
// countdown.
func TestClassifyMirrorClampsAFutureStamp(t *testing.T) {
	v := ClassifyMirror("origin", MirrorStamp{Remote: "origin", Source: MirrorFromFetch, SyncedAt: 2_000_000}, true, 3, 1_000_000, 3600)
	if v.AgeSeconds != 0 {
		t.Errorf("AgeSeconds = %d, want 0 (clamped)", v.AgeSeconds)
	}
	if v.Freshness != MirrorFresh {
		t.Errorf("freshness = %q, want FRESH", v.Freshness)
	}
}

// TestMirrorCaveatNamesTheCureForEachState: the four unlicensed states differ in what the
// operator should DO, so the sentence must differ too — a single generic "may be stale"
// would send a colliding-segment operator to re-sync forever.
func TestMirrorCaveatNamesTheCureForEachState(t *testing.T) {
	const now = 1_000_000
	cases := []struct {
		name  string
		view  MirrorView
		wants []string
	}{{
		name:  "never synced",
		view:  ClassifyMirror("origin", MirrorStamp{}, false, 0, now, 3600),
		wants: []string{"never synced", "IGNORANCE", "fak wip sync"},
	}, {
		name:  "push only",
		view:  ClassifyMirror("origin", MirrorStamp{Remote: "origin", Source: MirrorFromPush, SyncedAt: now - 60}, true, 1, now, 3600),
		wants: []string{"PUBLISHED", "says nothing about a peer", "fak wip sync"},
	}, {
		name:  "stale fetch",
		view:  ClassifyMirror("origin", MirrorStamp{Remote: "origin", Source: MirrorFromFetch, SyncedAt: now - 90000}, true, 2, now, 3600),
		wants: []string{"90000s ago", "staleness, not absence"},
	}, {
		name:  "segment collision",
		view:  ClassifyMirror("origin", MirrorStamp{Remote: "other", Source: MirrorFromFetch, SyncedAt: now - 60}, true, 2, now, 3600),
		wants: []string{"\"other\"", "one ref segment", "distinct names"},
	}}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := MirrorCaveat(c.view)
			for _, w := range c.wants {
				if !strings.Contains(got, w) {
					t.Errorf("caveat %q missing %q", got, w)
				}
			}
		})
	}
}

// TestClassifyMirrorDefaultToleranceIsTheGardenTickCadence: the zero-tolerance default is
// the registered garden-tick interval, and a mirror that has outlived one full tick is
// STALE. The number is grounded rather than chosen — see the constant's comment.
func TestClassifyMirrorDefaultToleranceIsTheGardenTickCadence(t *testing.T) {
	if DefaultMirrorMaxAgeSeconds != 3600 {
		t.Errorf("DefaultMirrorMaxAgeSeconds = %d, want the garden-tick cadence 3600", DefaultMirrorMaxAgeSeconds)
	}
	const now = 1_000_000
	fresh := ClassifyMirror("origin", MirrorStamp{Remote: "origin", Source: MirrorFromFetch, SyncedAt: now - DefaultMirrorMaxAgeSeconds}, true, 0, now, 0)
	if fresh.Freshness != MirrorFresh {
		t.Errorf("at exactly one tick: %q, want FRESH", fresh.Freshness)
	}
	stale := ClassifyMirror("origin", MirrorStamp{Remote: "origin", Source: MirrorFromFetch, SyncedAt: now - DefaultMirrorMaxAgeSeconds - 1}, true, 0, now, 0)
	if stale.Freshness != MirrorStale {
		t.Errorf("one second past a tick: %q, want STALE", stale.Freshness)
	}
}
