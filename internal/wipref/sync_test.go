package wipref

import "testing"

// TestRefspecsAreForcedAndNamespaceConfined pins the two refspecs a sync issues. The
// force is not cosmetic: a checkpoint commit is parented on HEAD, never on the previous
// checkpoint, so two checkpoints of one session are siblings and a non-forced update is
// rejected. The confinement is the safety contract — no branch, HEAD, or tag is ever in
// range on either end.
func TestRefspecsAreForcedAndNamespaceConfined(t *testing.T) {
	if got, want := PushRefspec, "+refs/fak/wip/*:refs/fak/wip/*"; got != want {
		t.Errorf("PushRefspec = %q, want %q", got, want)
	}
	if got, want := FetchRefspec("origin"), "+refs/fak/wip/*:refs/fak/remotewip/origin/*"; got != want {
		t.Errorf("FetchRefspec = %q, want %q", got, want)
	}
}

// TestMirrorNamespaceCannotCollideWithLiveNamespace is the divergence from leaseref
// stated as an assertion: the fetch destination must NOT sit under refs/fak/wip/, or a
// peer host's checkpoints would join the population that `wip reap` deletes from and
// `wip reconcile`/`land` apply to this working tree.
func TestMirrorNamespaceCannotCollideWithLiveNamespace(t *testing.T) {
	for _, remote := range []string{"origin", "peer", "ssh://host/repo.git"} {
		ns := MirrorNamespace(remote)
		if len(ns) >= len(RefNamespace) && ns[:len(RefNamespace)] == RefNamespace {
			t.Fatalf("mirror namespace %q for remote %q nests under the live namespace %q", ns, remote, RefNamespace)
		}
		ref := MirrorSessionRef(remote, "sessA")
		if got := SessionFromMirrorRef(remote, ref); got != "sessA" {
			t.Errorf("SessionFromMirrorRef(%q) = %q, want sessA", ref, got)
		}
	}
}

// TestMirrorSegmentIsOneSafeRefComponent covers the refname hazards the sanitizer folds
// away in one stroke: a leading dot, an embedded "..", a ".lock" suffix, a slash, and an
// empty remote. `origin` — the only remote the fleet uses — must pass through unchanged.
func TestMirrorSegmentIsOneSafeRefComponent(t *testing.T) {
	cases := map[string]string{
		"origin":                   "origin",
		"up_stream-2":              "up_stream-2",
		".hidden":                  "-hidden",
		"a..b":                     "a--b",
		"thing.lock":               "thing-lock",
		"ssh://host:22/x/y.git":    "ssh---host-22-x-y-git",
		"":                         "remote",
		"///":                      "---",
		"https://ex.com/a/b?c=d#e": "https---ex-com-a-b-c-d-e",
	}
	for in, want := range cases {
		got := MirrorSegment(in)
		if got != want {
			t.Errorf("MirrorSegment(%q) = %q, want %q", in, got, want)
		}
		for _, c := range []byte(got) {
			ok := (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-'
			if !ok {
				t.Errorf("MirrorSegment(%q) = %q contains unsafe byte %q", in, got, string(c))
			}
		}
	}
	long := make([]byte, 500)
	for i := range long {
		long[i] = 'x'
	}
	if got := len(MirrorSegment(string(long))); got > mirrorSegmentMax {
		t.Errorf("MirrorSegment length %d exceeds cap %d", got, mirrorSegmentMax)
	}
}

// TestValidRemoteIsArgvHygiene: a name and a URL pass; anything that could misparse as a
// git flag or split into two argv tokens is refused.
func TestValidRemoteIsArgvHygiene(t *testing.T) {
	for _, ok := range []string{"origin", "ssh://host/repo.git", "https://ex.com/a.git", "peer2"} {
		if !ValidRemote(ok) {
			t.Errorf("ValidRemote(%q) = false, want true", ok)
		}
	}
	for _, bad := range []string{"", "-x", "--upload-pack=evil", "a b", "a\tb", "a\nb", "a\x7fb"} {
		if ValidRemote(bad) {
			t.Errorf("ValidRemote(%q) = true, want false", bad)
		}
	}
}

// TestClassifyReplicationThreeStates is the heart of the second done-condition: the
// middle state must be distinguishable, because an operator who synced once and
// checkpointed again has an OLDER delta on the remote and reading that as "replicated"
// is the exact false comfort the column removes.
func TestClassifyReplicationThreeStates(t *testing.T) {
	rec := RefRecord{Ref: "refs/fak/wip/sessA", Object: "aaa", Stamp: Stamp{SessionID: "sessA"}}

	if got, obj := ClassifyReplication(rec, nil); got != ReplicationLocalOnly || obj != "" {
		t.Errorf("no mirror: got (%q,%q), want (LOCAL_ONLY,\"\")", got, obj)
	}
	if got, obj := ClassifyReplication(rec, map[string]string{"other": "zzz"}); got != ReplicationLocalOnly || obj != "" {
		t.Errorf("mirror without this session: got (%q,%q), want (LOCAL_ONLY,\"\")", got, obj)
	}
	if got, obj := ClassifyReplication(rec, map[string]string{"sessA": "aaa"}); got != ReplicationReplicated || obj != "aaa" {
		t.Errorf("same object: got (%q,%q), want (REPLICATED,aaa)", got, obj)
	}
	if got, obj := ClassifyReplication(rec, map[string]string{"sessA": "bbb"}); got != ReplicationStaleRemote || obj != "bbb" {
		t.Errorf("older object: got (%q,%q), want (STALE_REMOTE,bbb)", got, obj)
	}

	// A record whose stamp lost its session id is still classified, keyed off the ref
	// name — the same fallback Fold uses for its label.
	unstamped := RefRecord{Ref: "refs/fak/wip/sessB", Object: "ccc"}
	if got, _ := ClassifyReplication(unstamped, map[string]string{"sessB": "ccc"}); got != ReplicationReplicated {
		t.Errorf("unstamped record: got %q, want REPLICATED", got)
	}
}

// TestFoldWithMirrorCountsAndLabels asserts the census sums to Count and each row
// carries its own verdict.
func TestFoldWithMirrorCountsAndLabels(t *testing.T) {
	recs := []RefRecord{
		{Ref: "refs/fak/wip/c", Object: "o3", Stamp: Stamp{SessionID: "c", CheckpointedAt: 10}},
		{Ref: "refs/fak/wip/a", Object: "o1", Stamp: Stamp{SessionID: "a", CheckpointedAt: 10}},
		{Ref: "refs/fak/wip/b", Object: "o2", Stamp: Stamp{SessionID: "b", CheckpointedAt: 10}},
	}
	mirror := map[string]string{"a": "o1", "b": "OLD"}
	rep := FoldWithMirror(recs, mirror, 20)

	if rep.Count != 3 || rep.Replicated != 1 || rep.StaleRemote != 1 || rep.LocalOnly != 1 {
		t.Fatalf("census = count %d / repl %d / stale %d / local %d, want 3/1/1/1",
			rep.Count, rep.Replicated, rep.StaleRemote, rep.LocalOnly)
	}
	if rep.Replicated+rep.StaleRemote+rep.LocalOnly != rep.Count {
		t.Fatal("census does not sum to Count")
	}
	want := []string{"REPLICATED", "STALE_REMOTE", "LOCAL_ONLY"} // sorted a,b,c
	for i, s := range rep.Sessions {
		if s.Replication != want[i] {
			t.Errorf("session %q replication = %q, want %q", s.Session, s.Replication, want[i])
		}
	}
	if rep.Sessions[1].RemoteObject != "OLD" {
		t.Errorf("stale row remote_object = %q, want OLD", rep.Sessions[1].RemoteObject)
	}
	if rep.Sessions[2].RemoteObject != "" {
		t.Errorf("local-only row must carry no remote object, got %q", rep.Sessions[2].RemoteObject)
	}
}

// TestFoldWithoutMirrorNeverOverstatesDurability: the legacy Fold entry point reads no
// replication evidence, so it must report LOCAL_ONLY rather than an empty string that a
// consumer could render as "unknown, probably fine".
func TestFoldWithoutMirrorNeverOverstatesDurability(t *testing.T) {
	recs := []RefRecord{{Ref: "refs/fak/wip/a", Object: "o1", Stamp: Stamp{SessionID: "a"}}}
	rep := Fold(recs, 1)
	if rep.Sessions[0].Replication != string(ReplicationLocalOnly) {
		t.Errorf("Fold replication = %q, want LOCAL_ONLY", rep.Sessions[0].Replication)
	}
	if rep.LocalOnly != 1 || rep.Replicated != 0 {
		t.Errorf("Fold census = local %d / repl %d, want 1/0", rep.LocalOnly, rep.Replicated)
	}
}

// TestMirrorIndexKeysBySession: the index strips the remote's mirror prefix and keys by
// session id, so a mirrored record and a local one compare directly. A record carrying
// no object is dropped rather than indexed as an empty string, which would otherwise
// classify a live checkpoint LOCAL_ONLY by accident.
func TestMirrorIndexKeysBySession(t *testing.T) {
	fetched := []RefRecord{
		{Ref: MirrorSessionRef("origin", "sessA"), Object: "oA"},
		{Ref: MirrorSessionRef("origin", "sessB"), Object: ""}, // no object: skipped
	}
	idx := MirrorIndex("origin", fetched)
	if idx["sessA"] != "oA" {
		t.Errorf("MirrorIndex sessA = %q, want oA", idx["sessA"])
	}
	if _, ok := idx["sessB"]; ok {
		t.Error("MirrorIndex kept a record with no object")
	}
	if MirrorIndex("origin", nil) != nil {
		t.Error("MirrorIndex of nothing should be nil")
	}
}
