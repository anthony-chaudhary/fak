package portability

import (
	"crypto/ed25519"
	"encoding/json"
	"strings"
	"testing"
)

func fixturePackage(kind string) Package {
	return PrepareOrganizationPackage([]Object{{ID: kind + ":fixture", Kind: kind, Name: "fixture", Active: true, Payload: json.RawMessage(`{"behavior":"deterministic"}`)}})
}
func setupOrg(t *testing.T) (*OrgState, map[string]Actor, map[string]ed25519.PrivateKey) {
	t.Helper()
	actors := map[string]Actor{}
	keys := map[string]ed25519.PrivateKey{}
	specs := []struct {
		id    string
		roles []Role
	}{{"olivia", []Role{RoleOwner, RoleOperator}}, {"pavel", []Role{RolePublisher}}, {"amy", []Role{RoleApprover}}, {"cora", []Role{RoleMember}}, {"dev", []Role{RoleMember}}}
	for _, x := range specs {
		a, k := ActorFromSeed(x.id, x.roles, x.id)
		actors[x.id] = a
		keys[x.id] = k
	}
	s := NewOrganization("acme", actors["olivia"])
	for _, id := range []string{"pavel", "amy", "cora", "dev"} {
		if d := s.AddActor("olivia", actors[id]); !d.Allowed {
			t.Fatal(d)
		}
	}
	return s, actors, keys
}
func TestOrganizationGovernanceJourneyAndAttacks(t *testing.T) {
	s, _, raw := setupOrg(t)
	publisher := raw["pavel"]
	approver := raw["amy"]
	// Corporate wins over lower scopes. Same-scope disagreements fail closed and explain the conflict.
	for _, p := range []ScopeRule{{ID: "personal", Scope: ScopePersonal, Subject: "cora", MinApprovals: 0, Revision: 1}, {ID: "team", Scope: ScopeTeam, MinApprovals: 1, Revision: 1}, {ID: "corp", Scope: ScopeCorporate, MinApprovals: 1, DenyKinds: []string{"policy"}, RetentionDays: 365, Revision: 1}} {
		if d := s.SetPolicy("olivia", p); !d.Allowed {
			t.Fatal(d)
		}
	}
	p, path, err := s.AppliedRule("cora")
	if err != nil || p.ID != "corp" || len(path) != 3 {
		t.Fatalf("policy=%+v path=%v err=%v", p, path, err)
	}
	v := CollectionVersion{Organization: "acme", Collection: "review-kit", Version: 1, Package: fixturePackage("skill"), Publisher: "pavel"}
	SignCollection(publisher, &v)
	if d := s.Publish("cora", publisher, v); d.Allowed || d.Code != "UNAUTHORIZED" {
		t.Fatalf("publish=%+v", d)
	}
	if d := s.Publish("pavel", publisher, v); !d.Allowed {
		t.Fatal(d)
	}
	if d := s.Publish("pavel", publisher, v); d.Allowed || d.Code != "DOWNGRADE_REPLAY" {
		t.Fatalf("replay=%+v", d)
	}
	if d := s.StartRollout("olivia", "review-kit", 1, []string{"canary", "team"}); d.Allowed || d.Code != "APPROVAL_REQUIRED" {
		t.Fatalf("unapproved rollout=%+v", d)
	}
	a := SignApproval(approver, v, "amy")
	if d := s.Approve("amy", "review-kit", 1, a); !d.Allowed {
		t.Fatal(d)
	}
	if d := s.StartRollout("olivia", "review-kit", 1, []string{"canary", "team"}); !d.Allowed {
		t.Fatal(d)
	}
	if d := s.Install("dev", "devbox", "team", "review-kit", 1); d.Allowed || d.Code != "RING_CLOSED" {
		t.Fatalf("early install=%+v", d)
	}
	if d := s.Install("cora", "laptop", "canary", "review-kit", 1); !d.Allowed {
		t.Fatal(d)
	}
	if d := s.Activate("cora", "laptop", "review-kit", 1); !d.Allowed {
		t.Fatal(d)
	}
	if d := s.Promote("cora", "review-kit"); d.Allowed || d.Code != "UNAUTHORIZED" {
		t.Fatalf("promotion=%+v", d)
	}
	if d := s.Promote("olivia", "review-kit"); !d.Allowed {
		t.Fatal(d)
	}
	if d := s.Install("dev", "devbox", "team", "review-kit", 1); !d.Allowed {
		t.Fatal(d)
	}
	if d := s.Revoke("pavel", "review-kit", 1, "compromised signing input"); d.Allowed || d.Code != "UNAUTHORIZED" {
		t.Fatalf("revoke=%+v", d)
	}
	// Offline device diverges before reconnect. Authority revokes and deterministically quarantines on reconciliation.
	offlineBytes, _ := json.Marshal(s)
	var offline OrgState
	json.Unmarshal(offlineBytes, &offline)
	if d := s.Revoke("olivia", "review-kit", 1, "compromised package"); !d.Allowed {
		t.Fatal(d)
	}
	if d := offline.Activate("dev", "devbox", "review-kit", 1); !d.Allowed {
		t.Fatal(d)
	}
	s.ReconcileDevice(&offline, "devbox")
	in := offline.Installs["devbox/review-kit"]
	if in.Active || !in.Quarantined || !strings.Contains(in.Remediation, "install an approved") {
		t.Fatalf("quarantine=%+v", in)
	}
	if d := offline.Activate("dev", "devbox", "review-kit", 1); d.Allowed || d.Code != "REVOKED" {
		t.Fatalf("post revoke=%+v", d)
	}
	if d := offline.Install("dev", "devbox", "team", "review-kit", 1); d.Allowed || d.Code != "REVOKED" {
		t.Fatalf("new activation path=%+v", d)
	}
	if d := offline.Rollback("olivia", "devbox", "review-kit", 1); d.Allowed || d.Code != "UNSAFE_ROLLBACK" {
		t.Fatalf("rollback attack=%+v", d)
	}
	if err := offline.VerifyAudit(); err != nil {
		t.Fatal(err)
	}
	for _, r := range offline.Audit {
		b, _ := json.Marshal(r)
		if strings.Contains(string(b), "compromised package") {
			t.Fatalf("secret/reason leaked into receipt: %s", b)
		}
	}
}
func TestPolicyConflictAndViolationFailClosed(t *testing.T) {
	s, _, raw := setupOrg(t)
	pub := raw["pavel"]
	app := raw["amy"]
	s.SetPolicy("olivia", ScopeRule{ID: "corp-a", Scope: ScopeCorporate, MinApprovals: 1, Revision: 1})
	s.SetPolicy("olivia", ScopeRule{ID: "corp-b", Scope: ScopeCorporate, MinApprovals: 2, Revision: 1})
	_, path, err := s.AppliedRule("cora")
	if err == nil || !strings.Contains(err.Error(), "POLICY_"+"CONFLICT") || len(path) < 2 {
		t.Fatalf("path=%v err=%v", path, err)
	}
	// Remove conflict, then prove receiving policy rejects an imported prohibited kind while provenance survives publish.
	s.Policies = []ScopeRule{{ID: "corp", Scope: ScopeCorporate, MinApprovals: 1, DenyKinds: []string{"policy"}, Revision: 1}}
	v := CollectionVersion{Organization: "source", Collection: "unsafe", Version: 2, Package: fixturePackage("policy"), Publisher: "pavel"}
	if d := s.Import("pavel", v, "source", pub); !d.Allowed {
		t.Fatal(d)
	}
	stored := s.Collections["unsafe"][0]
	a := SignApproval(app, stored, "amy")
	if d := s.Approve("amy", "unsafe", 2, a); !d.Allowed {
		t.Fatal(d)
	}
	if d := s.StartRollout("olivia", "unsafe", 2, []string{"canary"}); d.Allowed || d.Code != "POLICY_DENY" || len(d.PolicyPath) == 0 {
		t.Fatalf("policy violation=%+v", d)
	}
	if len(stored.Provenance) != 1 || !strings.Contains(stored.Provenance[0], "source/") {
		t.Fatalf("provenance=%v", stored.Provenance)
	}
	if d := s.SetPolicy("olivia", ScopeRule{ID: "corp", Scope: ScopeCorporate, MinApprovals: 0, Revision: 1}); d.Allowed || d.Code != "POLICY_REPLAY" {
		t.Fatalf("policy replay=%+v", d)
	}
}
