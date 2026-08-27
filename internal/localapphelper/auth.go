// Package localapphelper binds a local-app request to one signed host install.
package localapphelper

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"strings"
)

const Schema = "fak.local-app-helper-auth/1"

// HostIdentity is supplied by the platform adapter after code-signature validation.
type HostIdentity struct {
	TeamID      string `json:"team_id"`
	BundleID    string `json:"bundle_id"`
	InstallID   string `json:"install_id"`
	HelperBuild string `json:"helper_build"`
}

// Binding is safe to persist: CapabilityDigest is a one-way digest, never the bearer secret.
type Binding struct {
	Schema           string       `json:"schema"`
	Host             HostIdentity `json:"host"`
	CapabilityDigest string       `json:"capability_digest"`
}

func Bind(host HostIdentity, capability []byte) (Binding, error) {
	if err := validateHost(host); err != nil {
		return Binding{}, err
	}
	if len(capability) < 32 {
		return Binding{}, errors.New("localapphelper: capability must contain at least 256 bits")
	}
	sum := digest(host, capability)
	return Binding{Schema: Schema, Host: host, CapabilityDigest: hex.EncodeToString(sum[:])}, nil
}

func (b Binding) Authorize(host HostIdentity, capability []byte) error {
	if b.Schema != Schema {
		return errors.New("localapphelper: unsupported binding schema")
	}
	if err := validateHost(host); err != nil {
		return err
	}
	if host != b.Host {
		return errors.New("localapphelper: host identity mismatch")
	}
	want, err := hex.DecodeString(b.CapabilityDigest)
	if err != nil || len(want) != sha256.Size {
		return errors.New("localapphelper: malformed capability digest")
	}
	got := digest(host, capability)
	if subtle.ConstantTimeCompare(got[:], want) != 1 {
		return errors.New("localapphelper: capability mismatch")
	}
	return nil
}

func validateHost(h HostIdentity) error {
	if strings.TrimSpace(h.TeamID) == "" || strings.TrimSpace(h.BundleID) == "" || strings.TrimSpace(h.InstallID) == "" || strings.TrimSpace(h.HelperBuild) == "" {
		return errors.New("localapphelper: incomplete signed-host identity")
	}
	return nil
}

func digest(h HostIdentity, capability []byte) [sha256.Size]byte {
	payload := strings.Join([]string{Schema, h.TeamID, h.BundleID, h.InstallID, h.HelperBuild}, "\x00")
	x := append([]byte(payload+"\x00"), capability...)
	return sha256.Sum256(x)
}
