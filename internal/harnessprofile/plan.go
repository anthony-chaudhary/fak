package harnessprofile

import (
	"fmt"
	"slices"
)

// Binding is the canonical, secret-free identity guard and generated products share.
// It is derived from a resolved HarnessProfile and can be compared to persisted artifacts
// without carrying credential locations into receipts.
type Binding struct {
	Schema         string             `json:"schema"`
	Host           string             `json:"host"`
	AdapterVersion string             `json:"adapter_version"`
	AdapterDigest  string             `json:"adapter_digest"`
	Wire           Wire               `json:"wire"`
	Repoint        []RepointMechanism `json:"repoint"`
}

const BindingSchema = "fak.harness-binding/v1alpha1"

// Bind derives the canonical integration identity for one already-resolved profile.
func Bind(profile HarnessProfile) (Binding, error) {
	digest, err := SemanticDigest(profile)
	if err != nil {
		return Binding{}, err
	}
	return Binding{
		Schema:         BindingSchema,
		Host:           profile.Name,
		AdapterVersion: profile.AdapterVersion,
		AdapterDigest:  digest,
		Wire:           profile.Wire,
		Repoint:        append([]RepointMechanism(nil), profile.Repoint...),
	}, nil
}

// ResolveBinding performs command detection against an explicit resolved registry. This
// keeps conformance fixtures isolated from the process-global active guard registry.
func ResolveBinding(profiles []HarnessProfile, command string) (Binding, bool, error) {
	profile, ok := lookupIn(profiles, command)
	if !ok {
		return Binding{}, false, nil
	}
	binding, err := Bind(profile)
	return binding, true, err
}

// ActiveBinding resolves the same registry guard uses at launch.
func ActiveBinding(command string) (Binding, bool, error) {
	return ResolveBinding(active(), command)
}

// VerifyFresh refuses a persisted binding when any runtime-relevant descriptor field
// changed without every derived artifact being regenerated.
func VerifyFresh(profile HarnessProfile, snapshot Binding) error {
	current, err := Bind(profile)
	if err != nil {
		return err
	}
	if current.Schema != snapshot.Schema || current.Host != snapshot.Host ||
		current.AdapterVersion != snapshot.AdapterVersion || current.AdapterDigest != snapshot.AdapterDigest ||
		current.Wire != snapshot.Wire || !slices.Equal(current.Repoint, snapshot.Repoint) {
		return fmt.Errorf("stale harness binding for %q: descriptor no longer matches persisted adapter identity", snapshot.Host)
	}
	return nil
}

// HasRepoint reports whether the binding selects mechanism.
func (b Binding) HasRepoint(mechanism RepointMechanism) bool {
	return slices.Contains(b.Repoint, mechanism)
}
