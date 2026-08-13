package portability

// Public package lifecycle. The transport is deliberately smaller than the trust
// policy: a filesystem, OCI adapter, or hosted registry can implement Registry,
// while every caller still performs the same local verification.

import (
	"bytes"
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

type PackageRef struct{ Namespace, Name, Version string }

func (r PackageRef) String() string { return r.Namespace + "/" + r.Name + "@" + r.Version }

type Dependency struct {
	Namespace string `json:"namespace"`
	Name      string `json:"name"`
	Version   string `json:"version"`
	Digest    string `json:"digest"`
}
type Provenance struct {
	Source      string `json:"source"`
	Revision    string `json:"revision"`
	Builder     string `json:"builder"`
	Attestation string `json:"attestation"`
}
type RegistryManifest struct {
	Schema          string       `json:"schema"`
	Namespace       string       `json:"namespace"`
	Name            string       `json:"name"`
	Version         string       `json:"version"`
	Sequence        uint64       `json:"sequence"`
	IssuedAt        int64        `json:"issued_at"`
	ExpiresAt       int64        `json:"expires_at"`
	PackageDigest   string       `json:"package_digest"`
	Provenance      Provenance   `json:"provenance"`
	License         string       `json:"license"`
	Compatibility   []string     `json:"compatibility"`
	Sensitivity     string       `json:"sensitivity"`
	Dependencies    []Dependency `json:"dependencies,omitempty"`
	Permissions     []string     `json:"permissions,omitempty"`
	Hooks           []string     `json:"hooks,omitempty"`
	BreakingChanges []string     `json:"breaking_changes,omitempty"`
	Migration       string       `json:"migration,omitempty"`
	Rollback        string       `json:"rollback"`
	Deprecated      string       `json:"deprecated,omitempty"`
	Yanked          bool         `json:"yanked,omitempty"`
	Revoked         string       `json:"revoked,omitempty"`
	Signer          string       `json:"signer"`
}
type SignedPackage struct {
	Manifest  RegistryManifest `json:"manifest"`
	Package   Package          `json:"package"`
	Signature string           `json:"signature"`
}
type Registry interface {
	Put(SignedPackage) error
	Get(PackageRef) (SignedPackage, error)
}

type LocalRegistry struct{ Root string }

func (r LocalRegistry) path(ref PackageRef) (string, error) {
	if err := validRef(ref); err != nil {
		return "", err
	}
	return filepath.Join(r.Root, ref.Namespace, ref.Name, ref.Version+".fakpkg.json"), nil
}
func (r LocalRegistry) Put(p SignedPackage) error {
	ref := PackageRef{p.Manifest.Namespace, p.Manifest.Name, p.Manifest.Version}
	path, err := r.path(ref)
	if err != nil {
		return err
	}
	b, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	if old, e := os.ReadFile(path); e == nil {
		if bytes.Equal(old, b) {
			return nil
		}
		return fmt.Errorf("IMMUTABLE_VERSION: %s already exists", ref)
	}
	if err = os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	if _, err = f.Write(b); err == nil {
		err = f.Sync()
	}
	cerr := f.Close()
	if err == nil {
		err = cerr
	}
	return err
}
func (r LocalRegistry) Get(ref PackageRef) (SignedPackage, error) {
	path, err := r.path(ref)
	if err != nil {
		return SignedPackage{}, err
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return SignedPackage{}, err
	}
	var p SignedPackage
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err = d.Decode(&p); err != nil {
		return p, fmt.Errorf("MALICIOUS_METADATA: %w", err)
	}
	return p, nil
}

var tokenRE = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,63}$`)

func validRef(r PackageRef) error {
	if !tokenRE.MatchString(r.Namespace) || !tokenRE.MatchString(r.Name) || !tokenRE.MatchString(r.Version) || strings.Contains(r.Version, "..") {
		return fmt.Errorf("INVALID_REFERENCE: namespace/name/version must be bounded lowercase tokens")
	}
	return nil
}
func manifestBytes(m RegistryManifest) ([]byte, error) { return json.Marshal(m) }

type PublishRequest struct {
	Manifest   RegistryManifest
	Package    Package
	PrivateKey ed25519.PrivateKey
	Commit     bool
}
type PublishPreview struct {
	Action       string       `json:"action"`
	Reference    string       `json:"reference"`
	Digest       string       `json:"digest"`
	Permissions  []string     `json:"permissions"`
	Dependencies []Dependency `json:"dependencies"`
	Explanations []string     `json:"explanations"`
}

func Publish(reg Registry, req PublishRequest) (PublishPreview, error) {
	m := req.Manifest
	ref := PackageRef{m.Namespace, m.Name, m.Version}
	if err := validRef(ref); err != nil {
		return PublishPreview{}, err
	}
	if m.Schema != "fak.registry/v1" || m.Provenance.Source == "" || m.Provenance.Revision == "" || m.Provenance.Builder == "" || m.Provenance.Attestation == "" || m.License == "" || len(m.Compatibility) == 0 || m.Sensitivity == "" || m.Rollback == "" || m.Signer == "" {
		return PublishPreview{}, errors.New("INCOMPLETE_METADATA: provenance, license, compatibility, sensitivity, rollback, and signer are required")
	}
	if m.Revoked != "" || m.Yanked {
		return PublishPreview{}, errors.New("INVALID_PUBLISH_STATE: cannot initially publish revoked or yanked content")
	}
	for _, o := range req.Package.Objects {
		plan, err := PreviewEgress(ChannelPublic, o.Payload)
		if err != nil {
			return PublishPreview{}, err
		}
		if !plan.Allowed {
			return PublishPreview{}, fmt.Errorf("EGRESS_DENIED: object %s contains excluded content", o.ID)
		}
	}
	digest := packageDigest(req.Package)
	m.PackageDigest = digest
	sort.Slice(m.Dependencies, func(i, j int) bool { return depKey(m.Dependencies[i]) < depKey(m.Dependencies[j]) })
	sort.Strings(m.Permissions)
	sort.Strings(m.Hooks)
	sort.Strings(m.Compatibility)
	out := PublishPreview{Action: "dry-run", Reference: ref.String(), Digest: digest, Permissions: m.Permissions, Dependencies: m.Dependencies, Explanations: []string{"non-executing inspection required before install", "install remains inactive until explicit activation"}}
	if !req.Commit {
		return out, nil
	}
	if len(req.PrivateKey) != ed25519.PrivateKeySize {
		return out, errors.New("SIGNATURE_POLICY: Ed25519 private key required")
	}
	b, _ := manifestBytes(m)
	sig := ed25519.Sign(req.PrivateKey, b)
	if err := reg.Put(SignedPackage{Manifest: m, Package: req.Package, Signature: hex.EncodeToString(sig)}); err != nil {
		return out, err
	}
	out.Action = "published"
	return out, nil
}

type RegistryTrust struct {
	Keys                 map[string]ed25519.PublicKey
	Retired              map[string]bool
	Now                  time.Time
	MaxMetadataAge       time.Duration
	MinSequence          map[string]uint64
	AllowedCompatibility map[string]bool
}
type Inspection struct {
	Reference     string       `json:"reference"`
	Digest        string       `json:"digest"`
	Provenance    Provenance   `json:"provenance"`
	License       string       `json:"license"`
	Compatibility []string     `json:"compatibility"`
	Sensitivity   string       `json:"sensitivity"`
	Dependencies  []Dependency `json:"dependencies"`
	Permissions   []string     `json:"permissions"`
	Hooks         []string     `json:"hooks"`
	Deprecated    string       `json:"deprecated,omitempty"`
	Revoked       string       `json:"revoked,omitempty"`
	Installable   bool         `json:"installable"`
	Explanations  []string     `json:"explanations"`
}

func Inspect(reg Registry, ref PackageRef, p RegistryTrust) (SignedPackage, Inspection, error) {
	sp, err := reg.Get(ref)
	if err != nil {
		return sp, Inspection{}, err
	}
	m := sp.Manifest
	actual := PackageRef{m.Namespace, m.Name, m.Version}
	if actual != ref {
		return sp, Inspection{}, errors.New("NAMESPACE_TAKEOVER: requested reference does not match signed identity")
	}
	if err = validRef(actual); err != nil {
		return sp, Inspection{}, err
	}
	key, ok := p.Keys[m.Signer]
	if !ok || p.Retired[m.Signer] {
		return sp, Inspection{}, errors.New("UNTRUSTED_SIGNER: signer missing or rotated out")
	}
	sig, e := hex.DecodeString(mustBound(sp.Signature, 256))
	if e != nil {
		return sp, Inspection{}, errors.New("BAD_SIGNATURE: malformed signature")
	}
	b, _ := manifestBytes(m)
	if !ed25519.Verify(key, b, sig) {
		return sp, Inspection{}, errors.New("TAMPERED: signature verification failed")
	}
	digest := packageDigest(sp.Package)
	if digest != m.PackageDigest {
		return sp, Inspection{}, errors.New("TAMPERED: package digest mismatch")
	}
	now := p.Now
	if now.IsZero() {
		now = time.Now()
	}
	if m.ExpiresAt <= now.Unix() || (p.MaxMetadataAge > 0 && now.Sub(time.Unix(m.IssuedAt, 0)) > p.MaxMetadataAge) {
		return sp, Inspection{}, errors.New("STALE_METADATA: refresh registry metadata")
	}
	if m.Sequence < p.MinSequence[actual.Namespace+"/"+actual.Name] {
		return sp, Inspection{}, errors.New("REPLAY_OR_DOWNGRADE: sequence older than trusted state")
	}
	if len(p.AllowedCompatibility) > 0 {
		matched := false
		for _, c := range m.Compatibility {
			matched = matched || p.AllowedCompatibility[c]
		}
		if !matched {
			return sp, Inspection{}, errors.New("INCOMPATIBLE: no allowed runtime")
		}
	}
	in := Inspection{Reference: actual.String(), Digest: digest, Provenance: m.Provenance, License: m.License, Compatibility: m.Compatibility, Sensitivity: m.Sensitivity, Dependencies: m.Dependencies, Permissions: m.Permissions, Hooks: m.Hooks, Deprecated: m.Deprecated, Revoked: m.Revoked, Installable: m.Revoked == "" && !m.Yanked, Explanations: []string{"signature and immutable content identity verified", "dependencies and permissions shown before install", "install is inactive; activation is separate"}}
	if m.Yanked {
		return sp, in, errors.New("YANKED: version is not installable")
	}
	if m.Revoked != "" {
		return sp, in, errors.New("REVOKED: version is not installable")
	}
	return sp, in, nil
}
func mustBound(s string, n int) string {
	if len(s) > n {
		return ""
	}
	return s
}
func depKey(d Dependency) string { return d.Namespace + "/" + d.Name + "@" + d.Version }

type LockEntry struct {
	Reference   string   `json:"reference"`
	Digest      string   `json:"digest"`
	Permissions []string `json:"permissions"`
}
type Lockfile struct {
	Schema  string      `json:"schema"`
	Entries []LockEntry `json:"entries"`
}

func Resolve(reg Registry, root PackageRef, p RegistryTrust) (Lockfile, error) {
	seen := map[string]bool{}
	var entries []LockEntry
	var walk func(PackageRef, string) error
	walk = func(ref PackageRef, pin string) error {
		if seen[ref.String()] {
			return nil
		}
		sp, in, err := Inspect(reg, ref, p)
		if err != nil {
			return err
		}
		if pin != "" && pin != in.Digest {
			return fmt.Errorf("DEPENDENCY_CONFUSION: digest pin mismatch for %s", ref)
		}
		seen[ref.String()] = true
		for _, d := range sp.Manifest.Dependencies {
			if err := walk(PackageRef{d.Namespace, d.Name, d.Version}, d.Digest); err != nil {
				return err
			}
		}
		entries = append(entries, LockEntry{ref.String(), in.Digest, in.Permissions})
		return nil
	}
	if err := walk(root, ""); err != nil {
		return Lockfile{}, err
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Reference < entries[j].Reference })
	return Lockfile{"fak.lock/v1", entries}, nil
}

type UpdatePlan struct {
	From             string   `json:"from"`
	To               string   `json:"to"`
	Semantic         []string `json:"semantic_changes"`
	ObjectDiff       []string `json:"object_diff"`
	Breaking         []string `json:"breaking_changes"`
	Migration        string   `json:"migration"`
	Rollback         string   `json:"rollback"`
	RequiresApproval bool     `json:"requires_approval"`
}

func BuildUpdateDiff(old, new SignedPackage) (UpdatePlan, error) {
	if old.Manifest.Namespace != new.Manifest.Namespace || old.Manifest.Name != new.Manifest.Name {
		return UpdatePlan{}, errors.New("NAMESPACE_TAKEOVER: update identity changed")
	}
	if new.Manifest.Sequence <= old.Manifest.Sequence {
		return UpdatePlan{}, errors.New("DOWNGRADE_OR_REPLAY: update sequence did not advance")
	}
	oldVersion, err := ParseVersion(old.Manifest.Version)
	if err != nil {
		return UpdatePlan{}, fmt.Errorf("MALICIOUS_METADATA: installed version: %w", err)
	}
	newVersion, err := ParseVersion(new.Manifest.Version)
	if err != nil {
		return UpdatePlan{}, fmt.Errorf("MALICIOUS_METADATA: candidate version: %w", err)
	}
	if compareVersion(newVersion, oldVersion) <= 0 {
		return UpdatePlan{}, errors.New("DOWNGRADE_OR_REPLAY: update version did not advance")
	}
	pl := UpdatePlan{From: PackageRef{old.Manifest.Namespace, old.Manifest.Name, old.Manifest.Version}.String(), To: PackageRef{new.Manifest.Namespace, new.Manifest.Name, new.Manifest.Version}.String(), Breaking: new.Manifest.BreakingChanges, Migration: new.Manifest.Migration, Rollback: new.Manifest.Rollback, RequiresApproval: true}
	if strings.Join(old.Manifest.Permissions, ",") != strings.Join(new.Manifest.Permissions, ",") {
		pl.Semantic = append(pl.Semantic, "permissions: "+strings.Join(old.Manifest.Permissions, ",")+" -> "+strings.Join(new.Manifest.Permissions, ","))
	}
	a, b := map[string]bool{}, map[string]bool{}
	for _, o := range old.Package.Objects {
		a[o.ID] = true
	}
	for _, o := range new.Package.Objects {
		b[o.ID] = true
	}
	for id := range a {
		if !b[id] {
			pl.ObjectDiff = append(pl.ObjectDiff, "remove "+id)
		}
	}
	for id := range b {
		if !a[id] {
			pl.ObjectDiff = append(pl.ObjectDiff, "add "+id)
		}
	}
	sort.Strings(pl.ObjectDiff)
	return pl, nil
}

type InstalledState struct {
	Reference        string `json:"reference"`
	Digest           string `json:"digest"`
	Active           bool   `json:"active"`
	Quarantined      bool   `json:"quarantined"`
	EvidenceRetained bool   `json:"evidence_retained"`
}

func EnforceSyncedStatus(s InstalledState, m RegistryManifest, compromise bool) InstalledState {
	if m.Revoked != "" || m.Yanked || compromise {
		s.Active = false
		s.Quarantined = true
		s.EvidenceRetained = true
	}
	return s
}

func compareVersion(a, b [3]int) int {
	for i := range a {
		if a[i] < b[i] {
			return -1
		}
		if a[i] > b[i] {
			return 1
		}
	}
	return 0
}

func ParseVersion(v string) ([3]int, error) {
	var n [3]int
	parts := strings.Split(v, ".")
	if len(parts) != 3 {
		return n, errors.New("version must be x.y.z")
	}
	for i, s := range parts {
		x, e := strconv.Atoi(s)
		if e != nil || x < 0 {
			return n, errors.New("version must be x.y.z")
		}
		n[i] = x
	}
	return n, nil
}
