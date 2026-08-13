package main

import (
	"crypto/ed25519"
	"encoding/json"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/portability"
)

func runContinuityOrgSelfcheck(stdout, stderr io.Writer, jsonOut bool) int {
	actor := func(id string, roles ...portability.Role) (portability.Actor, ed25519.PrivateKey) {
		return portability.ActorFromSeed(id, roles, id)
	}
	owner, _ := actor("olivia", portability.RoleOwner, portability.RoleOperator)
	publisher, pubKey := actor("pavel", portability.RolePublisher)
	approver, approveKey := actor("amy", portability.RoleApprover)
	canary, _ := actor("cora", portability.RoleMember)
	team, _ := actor("dev", portability.RoleMember)
	s := portability.NewOrganization("acme", owner)
	for _, a := range []portability.Actor{publisher, approver, canary, team} {
		if d := s.AddActor("olivia", a); !d.Allowed {
			return orgSelfcheckFail(stderr, d)
		}
	}
	policy := portability.ScopeRule{ID: "corp-safe", Scope: portability.ScopeCorporate, MinApprovals: 1, DenyKinds: []string{"policy"}, RetentionDays: 365, Revision: 1}
	if d := s.SetPolicy("olivia", policy); !d.Allowed {
		return orgSelfcheckFail(stderr, d)
	}
	pkg := portability.PrepareOrganizationPackage([]portability.Object{{ID: "skill:review", Kind: "skill", Name: "review", Active: true, Payload: json.RawMessage(`{"behavior":"review-concisely"}`)}})
	v := portability.CollectionVersion{Organization: "acme", Collection: "review-kit", Version: 1, Package: pkg, Publisher: "pavel"}
	portability.SignCollection(pubKey, &v)
	unauthorized := s.Publish("cora", pubKey, v)
	published := s.Publish("pavel", pubKey, v)
	unapproved := s.StartRollout("olivia", "review-kit", 1, []string{"canary", "team"})
	approved := s.Approve("amy", "review-kit", 1, portability.SignApproval(approveKey, v, "amy"))
	rollout := s.StartRollout("olivia", "review-kit", 1, []string{"canary", "team"})
	early := s.Install("dev", "devbox", "team", "review-kit", 1)
	install := s.Install("cora", "laptop", "canary", "review-kit", 1)
	activation := s.Activate("cora", "laptop", "review-kit", 1)
	promotion := s.Promote("olivia", "review-kit")
	teamInstall := s.Install("dev", "devbox", "team", "review-kit", 1)
	teamActivation := s.Activate("dev", "devbox", "review-kit", 1)
	offlineBytes, _ := json.Marshal(s)
	var offline portability.OrgState
	json.Unmarshal(offlineBytes, &offline)
	revoked := s.Revoke("olivia", "review-kit", 1, "compromised package")
	reconciled := s.ReconcileDevice(&offline, "devbox")
	post := offline.Activate("dev", "devbox", "review-kit", 1)
	quarantine := offline.Installs["devbox/review-kit"]
	checks := []portability.Decision{published, approved, rollout, install, activation, promotion, teamInstall, teamActivation, revoked, reconciled}
	for _, d := range checks {
		if !d.Allowed {
			return orgSelfcheckFail(stderr, d)
		}
	}
	for _, d := range []portability.Decision{unauthorized, unapproved, early, post} {
		if d.Allowed {
			return orgSelfcheckFail(stderr, d)
		}
	}
	if !quarantine.Quarantined || quarantine.Active {
		return orgSelfcheckFail(stderr, portability.Decision{Code: "QUARANTINE_FAILED", Explanation: "revoked active install not quarantined"})
	}
	if err := offline.VerifyAudit(); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	p, path, _ := offline.AppliedRule("dev")
	out := map[string]any{"result": "PASS", "service": "optional/none", "organization": "acme", "actors": 5, "collection": "review-kit@1", "policy": p.ID, "policy_path": path, "rollout": []string{"canary", "team"}, "fail_closed": map[string]string{"unauthorized_publish": unauthorized.Code, "unapproved_rollout": unapproved.Code, "early_install": early.Code, "revoked_activation": post.Code}, "quarantine": quarantine, "audit_receipts": len(offline.Audit), "audit_chain": "valid"}
	if jsonOut {
		b, _ := json.MarshalIndent(out, "", "  ")
		fmt.Fprintln(stdout, string(b))
		return 0
	}
	fmt.Fprintf(stdout, "PASS organization continuity: 5 actors, signed review-kit@1, no service\nPASS fail closed: publish=%s approval=%s ring=%s revoked-activation=%s\nPASS rollout: canary installed and activated, then team promoted\nPASS reconnect: revoked active devbox install quarantined; remediation=%s\npolicy precedence corporate > team > project > personal; effective=%s path=%v\naudit append-only chain valid: %d receipts link identity/package/policy/decision/actor/receipt without payload secrets\n", unauthorized.Code, unapproved.Code, early.Code, post.Code, quarantine.Remediation, p.ID, path, len(offline.Audit))
	return 0
}
func orgSelfcheckFail(w io.Writer, d portability.Decision) int {
	fmt.Fprintf(w, "FAIL organization continuity: %s: %s\n", d.Code, d.Explanation)
	return 1
}
