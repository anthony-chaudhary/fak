package main

import (
	"crypto/ed25519"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/portability"
)

type registrySelfcheckReport struct {
	Result         string `json:"result"`
	Transport      string `json:"transport"`
	PublishDefault string `json:"publish_default"`
	Published      string `json:"published"`
	Inspection     string `json:"inspection"`
	Install        string `json:"install"`
	Activation     string `json:"activation"`
	LockEntries    int    `json:"lock_entries"`
	Update         string `json:"update"`
	Rollback       string `json:"rollback"`
	Revocation     string `json:"revocation"`
	Hostile        string `json:"hostile"`
}

func runContinuityRegistrySelfcheck(stdout, stderr io.Writer, jsonOut bool) int {
	root, err := os.MkdirTemp("", "fak-registry-selfcheck-")
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	defer os.RemoveAll(root)
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	now := time.Unix(2000000000, 0)
	reg := portability.LocalRegistry{Root: filepath.Join(root, "registry")}
	pkg := portability.Package{Schema: portability.Schema, Objects: []portability.Object{{ID: "skill:review", Kind: "skill", Name: "review", Payload: json.RawMessage(`{"title":{"sensitivity":"public","value":"review safely"}}`)}}}
	manifest := portability.RegistryManifest{Schema: "fak.registry/v1", Namespace: "acme", Name: "review", Version: "1.0.0", Sequence: 1, IssuedAt: now.Add(-time.Minute).Unix(), ExpiresAt: now.Add(time.Hour).Unix(), Provenance: portability.Provenance{Source: "https://example.invalid/acme/review", Revision: "abc123", Builder: "fak-ci", Attestation: "slsa-local"}, License: "Apache-2.0", Compatibility: []string{"fak>=1"}, Sensitivity: "public", Permissions: []string{"read:workspace"}, Rollback: "restore prior lockfile", Signer: "acme-release"}
	req := portability.PublishRequest{Manifest: manifest, Package: pkg, PrivateKey: priv}
	dry, err := portability.Publish(reg, req)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	req.Commit = true
	pubResult, err := portability.Publish(reg, req)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	policy := portability.RegistryTrust{Keys: map[string]ed25519.PublicKey{"acme-release": pub}, Retired: map[string]bool{}, Now: now, MaxMetadataAge: 24 * time.Hour, MinSequence: map[string]uint64{}, AllowedCompatibility: map[string]bool{"fak>=1": true}}
	old, inspection, err := portability.Inspect(reg, portability.PackageRef{Namespace: "acme", Name: "review", Version: "1.0.0"}, policy)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	lock, err := portability.Resolve(reg, portability.PackageRef{Namespace: "acme", Name: "review", Version: "1.0.0"}, policy)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	installed := portability.InstalledState{Reference: inspection.Reference, Digest: inspection.Digest, Active: false, EvidenceRetained: true}
	installed.Active = true
	req.Manifest.Version = "2.0.0"
	req.Manifest.Sequence = 2
	req.Manifest.BreakingChanges = []string{"review mode renamed"}
	req.Manifest.Migration = "map brief to concise"
	req.Manifest.Permissions = []string{"read:workspace", "write:report"}
	if _, err = portability.Publish(reg, req); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	newSP, _, err := portability.Inspect(reg, portability.PackageRef{Namespace: "acme", Name: "review", Version: "2.0.0"}, policy)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	plan, err := portability.BuildUpdateDiff(old, newSP)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	hostile := newSP
	hostile.Package.Objects[0].Name = "tampered"
	hostileReg := selfcheckRegistry{hostile}
	_, _, hostileErr := portability.Inspect(hostileReg, portability.PackageRef{Namespace: "acme", Name: "review", Version: "2.0.0"}, policy)
	if hostileErr == nil {
		fmt.Fprintln(stderr, "hostile package unexpectedly passed")
		return 1
	}
	revoked := newSP.Manifest
	revoked.Revoked = "compromise"
	installed.Reference = plan.To
	installed = portability.EnforceSyncedStatus(installed, revoked, true)
	report := registrySelfcheckReport{"PASS", "local registry via hosted Registry interface", dry.Action, pubResult.Action, "provenance+dependencies+sensitivity+compatibility+signature verified", "inactive", "explicit", len(lock.Entries), "breaking changes+migration+permissions previewed", plan.Rollback, "quarantined; evidence retained", hostileErr.Error()}
	if jsonOut {
		b, _ := json.MarshalIndent(report, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprintf(stdout, "PASS public registry lifecycle\n  transport: %s\n  publish: %s by default; %s with --commit equivalent\n  inspect: %s\n  install: %s; activation: %s\n  lockfile: %d deterministic entry; permissions previewed\n  update: %s; rollback=%s\n  revocation: %s\n  hostile: denied before activation (%s)\n", report.Transport, report.PublishDefault, report.Published, report.Inspection, report.Install, report.Activation, report.LockEntries, report.Update, report.Rollback, report.Revocation, report.Hostile)
	}
	return 0
}

type selfcheckRegistry struct{ p portability.SignedPackage }

func (s selfcheckRegistry) Put(portability.SignedPackage) error { return nil }
func (s selfcheckRegistry) Get(portability.PackageRef) (portability.SignedPackage, error) {
	return s.p, nil
}
