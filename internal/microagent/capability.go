package microagent

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
)

// ErrChildAuthority marks a child capability request that cannot be
// mediated against host-owned parent authority.
var ErrChildAuthority = errors.New("microagent: child capability envelope refused")

// CapabilityEnvelope is the complete authority a microagent may exercise.
// Entries are exact tokens except Paths, whose entries are lexical subtree roots.
type CapabilityEnvelope struct {
	Tools               []string
	Paths               []string
	NetworkDestinations []string
	Effects             []string
}

// Equal reports whether two normalized envelopes carry the same authority.
func (e CapabilityEnvelope) Equal(other CapabilityEnvelope) bool {
	return equalStrings(e.Tools, other.Tools) &&
		equalStrings(e.Paths, other.Paths) &&
		equalStrings(e.NetworkDestinations, other.NetworkDestinations) &&
		equalStrings(e.Effects, other.Effects)
}

// IntersectCapabilities returns only requested authority already present in
// the parent envelope. It never expands wildcard-like entries.
func IntersectCapabilities(parent, requested CapabilityEnvelope) (CapabilityEnvelope, error) {
	parent, err := normalizeEnvelope(parent)
	if err != nil {
		return CapabilityEnvelope{}, err
	}
	requested, err = normalizeEnvelope(requested)
	if err != nil {
		return CapabilityEnvelope{}, err
	}
	return CapabilityEnvelope{
		Tools:               intersectExact(parent.Tools, requested.Tools),
		Paths:               intersectPaths(parent.Paths, requested.Paths),
		NetworkDestinations: intersectExact(parent.NetworkDestinations, requested.NetworkDestinations),
		Effects:             intersectExact(parent.Effects, requested.Effects),
	}, nil
}

type stepEnvelopeKey struct{}

func withStepEnvelope(ctx context.Context, envelope CapabilityEnvelope) context.Context {
	return context.WithValue(ctx, stepEnvelopeKey{}, envelope.clone())
}

// ChildEnvelopeFromStep returns the host-mediated envelope for the current
// agent step. The returned slices do not alias host state.
func ChildEnvelopeFromStep(ctx context.Context) (CapabilityEnvelope, bool) {
	if ctx == nil {
		return CapabilityEnvelope{}, false
	}
	envelope, ok := ctx.Value(stepEnvelopeKey{}).(CapabilityEnvelope)
	return envelope.clone(), ok
}

func normalizeEnvelope(envelope CapabilityEnvelope) (CapabilityEnvelope, error) {
	tools, err := normalizeEntries("tool", envelope.Tools, false)
	if err != nil {
		return CapabilityEnvelope{}, err
	}
	paths, err := normalizeEntries("path", envelope.Paths, true)
	if err != nil {
		return CapabilityEnvelope{}, err
	}
	destinations, err := normalizeEntries("network destination", envelope.NetworkDestinations, false)
	if err != nil {
		return CapabilityEnvelope{}, err
	}
	effects, err := normalizeEntries("effect", envelope.Effects, false)
	if err != nil {
		return CapabilityEnvelope{}, err
	}
	return CapabilityEnvelope{Tools: tools, Paths: paths, NetworkDestinations: destinations, Effects: effects}, nil
}

func normalizeEntries(kind string, entries []string, cleanPath bool) ([]string, error) {
	out := make([]string, 0, len(entries))
	seen := make(map[string]struct{}, len(entries))
	for i, entry := range entries {
		entry = strings.TrimSpace(entry)
		if entry == "" || strings.ContainsRune(entry, '\x00') || strings.ContainsRune(entry, '*') {
			return nil, fmt.Errorf("%w: %s[%d] must be a non-empty exact value", ErrChildAuthority, kind, i)
		}
		if cleanPath {
			entry = filepath.Clean(entry)
		}
		if _, ok := seen[entry]; ok {
			continue
		}
		seen[entry] = struct{}{}
		out = append(out, entry)
	}
	return out, nil
}

func intersectExact(parent, requested []string) []string {
	allowed := make(map[string]struct{}, len(parent))
	for _, entry := range parent {
		allowed[entry] = struct{}{}
	}
	granted := make([]string, 0, len(requested))
	for _, entry := range requested {
		if _, ok := allowed[entry]; ok {
			granted = append(granted, entry)
		}
	}
	return granted
}

func intersectPaths(parent, requested []string) []string {
	granted := make([]string, 0, len(requested))
	for _, request := range requested {
		for _, root := range parent {
			if pathWithin(root, request) {
				granted = append(granted, request)
				break
			}
		}
	}
	return granted
}

func pathWithin(root, requested string) bool {
	rel, err := filepath.Rel(root, requested)
	if err != nil || filepath.IsAbs(rel) {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func (e CapabilityEnvelope) empty() bool {
	return len(e.Tools) == 0 && len(e.Paths) == 0 && len(e.NetworkDestinations) == 0 && len(e.Effects) == 0
}

func (e CapabilityEnvelope) clone() CapabilityEnvelope {
	return CapabilityEnvelope{
		Tools:               append([]string(nil), e.Tools...),
		Paths:               append([]string(nil), e.Paths...),
		NetworkDestinations: append([]string(nil), e.NetworkDestinations...),
		Effects:             append([]string(nil), e.Effects...),
	}
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
