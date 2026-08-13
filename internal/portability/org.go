package portability

// Organization governance is an offline-first, repository-backed control plane for
// managed collections. The serialized State is the authority; a server may carry
// it, but is never required.

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

const OrgSchema = "fak-portability-organization/v1"

type Role string

const (
	RoleOwner     Role = "owner"
	RoleApprover  Role = "approver"
	RolePublisher Role = "publisher"
	RoleOperator  Role = "operator"
	RoleMember    Role = "member"
)

type Actor struct {
	ID        string `json:"id"`
	Roles     []Role `json:"roles"`
	PublicKey []byte `json:"public_key"`
}
type Scope string

const (
	ScopePersonal  Scope = "personal"
	ScopeProject   Scope = "project"
	ScopeTeam      Scope = "team"
	ScopeCorporate Scope = "corporate"
)

type ScopeRule struct {
	ID            string   `json:"id"`
	Scope         Scope    `json:"scope"`
	Subject       string   `json:"subject,omitempty"`
	MinApprovals  int      `json:"min_approvals"`
	DenyKinds     []string `json:"deny_kinds,omitempty"`
	RetentionDays int      `json:"retention_days,omitempty"`
	Revision      uint64   `json:"revision"`
}
type CollectionVersion struct {
	Organization string           `json:"organization"`
	Collection   string           `json:"collection"`
	Version      uint64           `json:"version"`
	Package      Package          `json:"package"`
	Publisher    string           `json:"publisher"`
	Signature    string           `json:"signature"`
	Approvals    []SignedApproval `json:"approvals,omitempty"`
	Provenance   []string         `json:"provenance,omitempty"`
	Deprecated   bool             `json:"deprecated,omitempty"`
	Revoked      bool             `json:"revoked,omitempty"`
	Note         string           `json:"revocation_reason,omitempty"`
}
type SignedApproval struct {
	Actor     string `json:"actor"`
	Signature string `json:"signature"`
}
type Rollout struct {
	Collection string   `json:"collection"`
	Version    uint64   `json:"version"`
	Ring       int      `json:"ring"`
	Rings      []string `json:"rings"`
}
type Installation struct {
	Device      string `json:"device"`
	Actor       string `json:"actor"`
	Collection  string `json:"collection"`
	Version     uint64 `json:"version"`
	Active      bool   `json:"active"`
	Quarantined bool   `json:"quarantined"`
	Remediation string `json:"remediation,omitempty"`
}
type AuditReceipt struct {
	ID       string `json:"id"`
	Previous string `json:"previous,omitempty"`
	Identity string `json:"identity"`
	Package  string `json:"package,omitempty"`
	Policy   string `json:"policy,omitempty"`
	Decision string `json:"decision"`
	Actor    string `json:"actor"`
	Action   string `json:"action"`
}
type OrgState struct {
	Schema       string                         `json:"schema"`
	Organization string                         `json:"organization"`
	Actors       map[string]Actor               `json:"actors"`
	Policies     []ScopeRule                    `json:"policies"`
	Collections  map[string][]CollectionVersion `json:"collections"`
	Rollouts     map[string]Rollout             `json:"rollouts"`
	Installs     map[string]Installation        `json:"installs"`
	Audit        []AuditReceipt                 `json:"audit"`
}
type Decision struct {
	Allowed     bool     `json:"allowed"`
	Code        string   `json:"code"`
	Explanation string   `json:"explanation"`
	PolicyPath  []string `json:"policy_path,omitempty"`
	Receipt     string   `json:"receipt,omitempty"`
}

func NewOrganization(id string, owner Actor) *OrgState {
	return &OrgState{Schema: OrgSchema, Organization: id, Actors: map[string]Actor{owner.ID: owner}, Collections: map[string][]CollectionVersion{}, Rollouts: map[string]Rollout{}, Installs: map[string]Installation{}}
}
func (s *OrgState) hasRole(id string, roles ...Role) bool {
	a, ok := s.Actors[id]
	if !ok {
		return false
	}
	for _, got := range a.Roles {
		for _, want := range roles {
			if got == want || got == RoleOwner {
				return true
			}
		}
	}
	return false
}
func actorPrivate(seed string) ed25519.PrivateKey {
	h := sha256.Sum256([]byte("fak-org-fixture:" + seed))
	return ed25519.NewKeyFromSeed(h[:])
}
func ActorFromSeed(id string, roles []Role, seed string) (Actor, ed25519.PrivateKey) {
	k := actorPrivate(seed)
	return Actor{ID: id, Roles: roles, PublicKey: append([]byte(nil), k.Public().(ed25519.PublicKey)...)}, k
}
func packageRef(c string, v uint64, d string) string { return fmt.Sprintf("%s@%d#%s", c, v, d) }
func signingBytes(v CollectionVersion) []byte {
	v.Signature = ""
	v.Approvals = nil
	b, _ := json.Marshal(v)
	return b
}
func approvalBytes(v CollectionVersion) []byte {
	return []byte("approve:" + packageRef(v.Collection, v.Version, v.Package.Digest))
}
func sign(k ed25519.PrivateKey, b []byte) string { return hex.EncodeToString(ed25519.Sign(k, b)) }
func verify(pub []byte, b []byte, sig string) bool {
	raw, e := hex.DecodeString(sig)
	return e == nil && len(pub) == ed25519.PublicKeySize && ed25519.Verify(ed25519.PublicKey(pub), b, raw)
}

func canonicalJSON(raw json.RawMessage) ([]byte, error) {
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, err
	}
	return json.Marshal(v)
}
func PrepareOrganizationPackage(objects []Object) Package {
	p := Package{Schema: Schema, Objects: append([]Object(nil), objects...)}
	for i := range p.Objects {
		canon, err := canonicalJSON(p.Objects[i].Payload)
		if err == nil {
			p.Objects[i].Payload = canon
			h := sha256.Sum256(canon)
			p.Objects[i].Digest = "sha256:" + hex.EncodeToString(h[:])
		}
	}
	p.Digest = packageDigest(p)
	p.ID = "pkg-" + strings.TrimPrefix(p.Digest, "sha256:")[:16]
	return p
}

func (s *OrgState) AddActor(by string, actor Actor) Decision {
	if !s.hasRole(by, RoleOwner) {
		return s.deny(by, "add-actor", "UNAUTHORIZED", "owner role required", "")
	}
	s.Actors[actor.ID] = actor
	return s.allow(by, "add-actor", "actor "+actor.ID+" added", "")
}
func (s *OrgState) RemoveActor(by, actor string) Decision {
	if !s.hasRole(by, RoleOwner) {
		return s.deny(by, "remove-actor", "UNAUTHORIZED", "owner role required", "")
	}
	if actor == by {
		return s.deny(by, "remove-actor", "OWNER_SELF_REMOVE", "owner cannot remove itself through this operation", "")
	}
	if _, ok := s.Actors[actor]; !ok {
		return s.deny(by, "remove-actor", "NOT_FOUND", "actor not found", "")
	}
	delete(s.Actors, actor)
	return s.allow(by, "remove-actor", "organization authority removed; personal state unchanged", "")
}

func (s *OrgState) SetPolicy(by string, p ScopeRule) Decision {
	if !s.hasRole(by, RoleOwner) {
		return s.deny(by, "set-policy", "UNAUTHORIZED", "owner role required", p.ID)
	}
	if p.Revision == 0 {
		return s.deny(by, "set-policy", "POLICY_REPLAY", "policy revision must be positive", p.ID)
	}
	for i, old := range s.Policies {
		if old.ID == p.ID {
			if p.Revision <= old.Revision {
				return s.deny(by, "set-policy", "POLICY_REPLAY", "policy revision must increase", p.ID)
			}
			s.Policies[i] = p
			return s.allow(by, "set-policy", "policy updated", p.ID)
		}
	}
	s.Policies = append(s.Policies, p)
	return s.allow(by, "set-policy", "policy added", p.ID)
}
func (s *OrgState) Publish(by string, k ed25519.PrivateKey, v CollectionVersion) Decision {
	if !s.hasRole(by, RolePublisher) {
		return s.deny(by, "publish", "UNAUTHORIZED", "publisher role required", "")
	}
	if v.Organization != s.Organization || v.Collection == "" || v.Version == 0 || v.Package.Digest == "" {
		return s.deny(by, "publish", "INVALID_PACKAGE", "organization, collection, version, and digest required", "")
	}
	versions := s.Collections[v.Collection]
	if len(versions) > 0 && v.Version <= versions[len(versions)-1].Version {
		return s.deny(by, "publish", "DOWNGRADE_REPLAY", "version must increase monotonically", "")
	}
	if v.Publisher != by {
		return s.deny(by, "publish", "IDENTITY_MISMATCH", "publisher must equal signing actor", "")
	}
	if v.Package.Schema != Schema || v.Package.Digest != packageDigest(v.Package) {
		return s.deny(by, "publish", "PACKAGE_INTEGRITY", "package schema or digest invalid", "")
	}
	for _, o := range v.Package.Objects {
		canon, err := canonicalJSON(o.Payload)
		if err != nil {
			return s.deny(by, "publish", "PACKAGE_INTEGRITY", "object payload is not canonical JSON", "")
		}
		h := sha256.Sum256(canon)
		if o.Digest != "sha256:"+hex.EncodeToString(h[:]) {
			return s.deny(by, "publish", "PACKAGE_INTEGRITY", "object digest invalid", "")
		}
	}
	if !verify(s.Actors[by].PublicKey, signingBytes(v), v.Signature) || len(k) > 0 && v.Signature != sign(k, signingBytes(v)) {
		return s.deny(by, "publish", "BAD_SIGNATURE", "publisher signature invalid", "")
	}
	s.Collections[v.Collection] = append(versions, v)
	return s.allowPackage(by, "publish", "signed package accepted", v, "")
}
func SignCollection(k ed25519.PrivateKey, v *CollectionVersion) {
	v.Signature = sign(k, signingBytes(*v))
}
func SignApproval(k ed25519.PrivateKey, v CollectionVersion, actor string) SignedApproval {
	return SignedApproval{Actor: actor, Signature: sign(k, approvalBytes(v))}
}
func (s *OrgState) Approve(by, collection string, version uint64, a SignedApproval) Decision {
	if !s.hasRole(by, RoleApprover) || a.Actor != by {
		return s.deny(by, "approve", "UNAUTHORIZED", "approver role required", "")
	}
	v, idx, ok := s.find(collection, version)
	if !ok {
		return s.deny(by, "approve", "NOT_FOUND", "package version not found", "")
	}
	if !verify(s.Actors[by].PublicKey, approvalBytes(v), a.Signature) {
		return s.deny(by, "approve", "BAD_SIGNATURE", "approval signature invalid", "")
	}
	for _, old := range v.Approvals {
		if old.Actor == by {
			return s.deny(by, "approve", "APPROVAL_REPLAY", "actor already approved this version", "")
		}
	}
	s.Collections[collection][idx].Approvals = append(s.Collections[collection][idx].Approvals, a)
	return s.allowPackage(by, "approve", "approval recorded", s.Collections[collection][idx], "")
}
func scopeRank(x Scope) int {
	return map[Scope]int{ScopePersonal: 1, ScopeProject: 2, ScopeTeam: 3, ScopeCorporate: 4}[x]
}
func (s *OrgState) AppliedRule(actor string) (ScopeRule, []string, error) {
	candidates := append([]ScopeRule(nil), s.Policies...)
	sort.Slice(candidates, func(i, j int) bool {
		if scopeRank(candidates[i].Scope) == scopeRank(candidates[j].Scope) {
			return candidates[i].ID < candidates[j].ID
		}
		return scopeRank(candidates[i].Scope) > scopeRank(candidates[j].Scope)
	})
	path := []string{}
	var chosen ScopeRule
	rank := 0
	for _, p := range candidates {
		if p.Subject != "" && p.Subject != actor {
			continue
		}
		path = append(path, fmt.Sprintf("%s:%s@r%d", p.Scope, p.ID, p.Revision))
		r := scopeRank(p.Scope)
		if rank == 0 {
			chosen = p
			rank = r
			continue
		}
		if r == rank && (p.MinApprovals != chosen.MinApprovals || strings.Join(p.DenyKinds, ",") != strings.Join(chosen.DenyKinds, ",")) {
			return ScopeRule{}, path, fmt.Errorf("POLICY_CONFLICT: %s and %s disagree at %s scope", chosen.ID, p.ID, p.Scope)
		}
	}
	if rank == 0 {
		chosen = ScopeRule{ID: "default-deny", Scope: ScopeCorporate, MinApprovals: 1, Revision: 1}
		path = []string{"corporate:default-deny@r1"}
	}
	return chosen, path, nil
}
func (s *OrgState) StartRollout(by, collection string, version uint64, rings []string) Decision {
	if !s.hasRole(by, RoleOperator) {
		return s.deny(by, "rollout", "UNAUTHORIZED", "operator role required", "")
	}
	v, _, ok := s.find(collection, version)
	if !ok {
		return s.deny(by, "rollout", "NOT_FOUND", "package version not found", "")
	}
	p, path, err := s.AppliedRule(by)
	if err != nil {
		return s.denyPath(by, "rollout", "POLICY_"+"CONFLICT", err.Error(), p.ID, path)
	}
	if d := validateVersion(v, p, path); !d.Allowed {
		return s.record(by, "rollout", d, v, p.ID)
	}
	if len(rings) == 0 {
		return s.deny(by, "rollout", "INVALID_RING", "at least one rollout ring required", p.ID)
	}
	s.Rollouts[collection] = Rollout{Collection: collection, Version: version, Ring: 0, Rings: append([]string(nil), rings...)}
	return s.allowPackage(by, "rollout", "ring 0 opened: "+rings[0], v, p.ID)
}
func (s *OrgState) Promote(by, collection string) Decision {
	if !s.hasRole(by, RoleOperator) {
		return s.deny(by, "promote", "UNAUTHORIZED", "operator role required", "")
	}
	r, ok := s.Rollouts[collection]
	if !ok || r.Ring+1 >= len(r.Rings) {
		return s.deny(by, "promote", "INVALID_PROMOTION", "no next rollout ring", "")
	}
	v, _, _ := s.find(collection, r.Version)
	if v.Revoked {
		return s.deny(by, "promote", "REVOKED", "revoked package cannot be promoted", "")
	}
	r.Ring++
	s.Rollouts[collection] = r
	return s.allowPackage(by, "promote", "promoted to ring "+r.Rings[r.Ring], v, "")
}
func (s *OrgState) Install(by, device, ring, collection string, version uint64) Decision {
	if !s.hasRole(by, RoleMember) {
		return s.deny(by, "install", "UNAUTHORIZED", "member role required", "")
	}
	v, _, ok := s.find(collection, version)
	if !ok {
		return s.deny(by, "install", "NOT_FOUND", "package version not found", "")
	}
	r, ok := s.Rollouts[collection]
	if !ok || r.Version != version || index(r.Rings, ring) > r.Ring || index(r.Rings, ring) < 0 {
		return s.deny(by, "install", "RING_CLOSED", "device ring is not eligible", "")
	}
	if v.Revoked {
		return s.deny(by, "install", "REVOKED", "revoked package cannot be installed", "")
	}
	key := device + "/" + collection
	if old, ok := s.Installs[key]; ok && version < old.Version {
		return s.deny(by, "install", "DOWNGRADE_REPLAY", "ordinary install cannot downgrade", "")
	}
	p, path, err := s.AppliedRule(by)
	if err != nil {
		return s.denyPath(by, "install", "POLICY_"+"CONFLICT", err.Error(), p.ID, path)
	}
	if d := validateVersion(v, p, path); !d.Allowed {
		return s.record(by, "install", d, v, p.ID)
	}
	s.Installs[key] = Installation{Device: device, Actor: by, Collection: collection, Version: version}
	return s.allowPackage(by, "install", "discovered and installed; activation remains separate", v, p.ID)
}
func (s *OrgState) Activate(by, device, collection string, version uint64) Decision {
	if !s.hasRole(by, RoleMember) {
		return s.deny(by, "activate", "UNAUTHORIZED", "member role required", "")
	}
	key := device + "/" + collection
	in, ok := s.Installs[key]
	if !ok || in.Actor != by || in.Version != version {
		return s.deny(by, "activate", "INSTALL_REQUIRED", "exact installed version and actor required", "")
	}
	v, _, ok := s.find(collection, version)
	if !ok || v.Revoked || in.Quarantined {
		return s.deny(by, "activate", "REVOKED", "revoked or quarantined package cannot activate", "")
	}
	in.Active = true
	s.Installs[key] = in
	return s.allowPackage(by, "activate", "installed package activated", v, "")
}
func (s *OrgState) Rollback(by, device, collection string, target uint64) Decision {
	if !s.hasRole(by, RoleOperator) {
		return s.deny(by, "rollback", "UNAUTHORIZED", "operator role required", "")
	}
	v, _, ok := s.find(collection, target)
	if !ok || v.Revoked {
		return s.deny(by, "rollback", "UNSAFE_ROLLBACK", "target absent or revoked", "")
	}
	p, path, err := s.AppliedRule(by)
	if err != nil {
		return s.denyPath(by, "rollback", "POLICY_"+"CONFLICT", err.Error(), p.ID, path)
	}
	if d := validateVersion(v, p, path); !d.Allowed {
		return s.record(by, "rollback", d, v, p.ID)
	}
	key := device + "/" + collection
	old, ok := s.Installs[key]
	if !ok {
		return s.deny(by, "rollback", "NOT_INSTALLED", "nothing to roll back", p.ID)
	}
	old.Version = target
	old.Active = false
	old.Quarantined = false
	old.Remediation = "operator-authorized rollback installed; explicit activation required"
	s.Installs[key] = old
	return s.allowPackage(by, "rollback", old.Remediation, v, p.ID)
}
func (s *OrgState) Revoke(by, collection string, version uint64, reason string) Decision {
	if !s.hasRole(by, RoleOwner) {
		return s.deny(by, "revoke", "UNAUTHORIZED", "owner role required", "")
	}
	if strings.TrimSpace(reason) == "" {
		return s.deny(by, "revoke", "REASON_REQUIRED", "revocation reason required", "")
	}
	v, idx, ok := s.find(collection, version)
	if !ok {
		return s.deny(by, "revoke", "NOT_FOUND", "package version not found", "")
	}
	s.Collections[collection][idx].Revoked = true
	s.Collections[collection][idx].Note = reason
	s.quarantine(collection, version)
	return s.allowPackage(by, "revoke", "revocation propagated; active installs quarantined", v, "")
}
func (s *OrgState) ReconcileDevice(local *OrgState, device string) Decision {
	// Authority wins for identity, policy, rollout, versions, and revocations. Device install state is retained then checked.
	retained := map[string]Installation{}
	for k, v := range local.Installs {
		if v.Device == device {
			retained[k] = v
		}
	}
	clone, _ := json.Marshal(s)
	_ = json.Unmarshal(clone, local)
	for k, v := range retained {
		local.Installs[k] = v
	}
	local.quarantineAll()
	return local.allow("reconcile:"+device, "reconcile", "authoritative ownership, rollout, policy, and revocation applied; local installs retained safely", "")
}
func (s *OrgState) quarantine(c string, v uint64) {
	for k, in := range s.Installs {
		if in.Collection == c && in.Version == v {
			in.Active = false
			in.Quarantined = true
			in.Remediation = "remove revoked package; install an approved non-revoked version; activate explicitly"
			s.Installs[k] = in
		}
	}
}
func (s *OrgState) quarantineAll() {
	for c, vs := range s.Collections {
		for _, v := range vs {
			if v.Revoked {
				s.quarantine(c, v.Version)
			}
		}
	}
}
func (s *OrgState) find(c string, v uint64) (CollectionVersion, int, bool) {
	for i, x := range s.Collections[c] {
		if x.Version == v {
			return x, i, true
		}
	}
	return CollectionVersion{}, 0, false
}
func index(xs []string, x string) int {
	for i, v := range xs {
		if v == x {
			return i
		}
	}
	return -1
}
func validateVersion(v CollectionVersion, p ScopeRule, path []string) Decision {
	if v.Revoked {
		return Decision{Code: "REVOKED", Explanation: "revoked package rejected", PolicyPath: path}
	}
	if len(v.Approvals) < p.MinApprovals {
		return Decision{Code: "APPROVAL_REQUIRED", Explanation: fmt.Sprintf("policy %s requires %d approval(s)", p.ID, p.MinApprovals), PolicyPath: path}
	}
	for _, o := range v.Package.Objects {
		for _, k := range p.DenyKinds {
			if o.Kind == k {
				return Decision{Code: "POLICY_DENY", Explanation: fmt.Sprintf("policy %s denies object kind %s", p.ID, k), PolicyPath: path}
			}
		}
	}
	return Decision{Allowed: true, Code: "ALLOW", PolicyPath: path}
}
func (s *OrgState) deny(a, act, code, why, p string) Decision {
	return s.denyPath(a, act, code, why, p, nil)
}
func (s *OrgState) denyPath(a, act, code, why, p string, path []string) Decision {
	return s.record(a, act, Decision{Code: code, Explanation: why, PolicyPath: path}, CollectionVersion{}, p)
}
func (s *OrgState) allow(a, act, why, p string) Decision {
	return s.record(a, act, Decision{Allowed: true, Code: "ALLOW", Explanation: why}, CollectionVersion{}, p)
}
func (s *OrgState) allowPackage(a, act, why string, v CollectionVersion, p string) Decision {
	return s.record(a, act, Decision{Allowed: true, Code: "ALLOW", Explanation: why}, v, p)
}
func (s *OrgState) record(a, act string, d Decision, v CollectionVersion, p string) Decision {
	prev := ""
	if n := len(s.Audit); n > 0 {
		prev = s.Audit[n-1].ID
	}
	r := AuditReceipt{Previous: prev, Identity: a, Actor: a, Action: act, Decision: d.Code, Policy: p}
	if v.Collection != "" {
		r.Package = packageRef(v.Collection, v.Version, v.Package.Digest)
	}
	b, _ := json.Marshal(r)
	h := sha256.Sum256(b)
	r.ID = "sha256:" + hex.EncodeToString(h[:])
	s.Audit = append(s.Audit, r)
	d.Receipt = r.ID
	return d
}
func (s *OrgState) VerifyAudit() error {
	prev := ""
	for i, r := range s.Audit {
		if r.Previous != prev {
			return fmt.Errorf("audit link %d", i)
		}
		id := r.ID
		r.ID = ""
		b, _ := json.Marshal(r)
		h := sha256.Sum256(b)
		if id != "sha256:"+hex.EncodeToString(h[:]) {
			return fmt.Errorf("audit digest %d", i)
		}
		prev = id
	}
	return nil
}
func (s *OrgState) Import(by string, v CollectionVersion, sourceOrg string, k ed25519.PrivateKey) Decision {
	if sourceOrg == "" || sourceOrg == s.Organization {
		return s.deny(by, "import", "PROVENANCE_REQUIRED", "cross-organization source required", "")
	}
	v.Organization = s.Organization
	v.Publisher = by
	v.Provenance = append(v.Provenance, sourceOrg+"/"+v.Collection+fmt.Sprintf("@%d", v.Version))
	SignCollection(k, &v)
	return s.Publish(by, k, v)
}
func (s *OrgState) Validate() error {
	if s.Schema != OrgSchema {
		return errors.New("unsupported organization schema")
	}
	return s.VerifyAudit()
}
