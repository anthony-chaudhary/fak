package wipref

import "testing"

func TestPlanRemoteDrainRequiresContainmentAndPeerOptIn(t *testing.T) {
	refs := []RemoteRef{{Ref: "refs/fak/wip/own", SHA: "a"}, {Ref: "refs/fak/wip/peer", SHA: "b"}}
	owned := map[string]bool{"own": true}
	rows := PlanRemoteDrain(refs, owned, func(r RemoteRef) (bool, error) { return false, nil }, false)
	by := map[string]RemoteDrainCandidate{}
	for _, r := range rows {
		by[r.Session] = r
	}
	if by["own"].State != RemoteDrainKeep || by["own"].DeleteRefspec != "" {
		t.Fatalf("unlanded own = %+v", by["own"])
	}
	if by["peer"].State != RemoteDrainPeer || by["peer"].DeleteRefspec != "" {
		t.Fatalf("peer = %+v", by["peer"])
	}
	rows = PlanRemoteDrain(refs, owned, func(r RemoteRef) (bool, error) { return true, nil }, true)
	for _, r := range rows {
		if r.State != RemoteDrainSafe || r.DeleteRefspec != ":"+r.Ref {
			t.Fatalf("safe = %+v", r)
		}
	}
}

func TestParseRemoteRefsRejectsOutsideNamespace(t *testing.T) {
	if _, err := ParseRemoteRefs("abc refs/heads/main\n"); err == nil {
		t.Fatal("accepted branch")
	}
}
