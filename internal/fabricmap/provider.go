package fabricmap

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// Snapshot is one provider's atomic view of currently usable capabilities.
// Generation is provider-local and must increase; ValidUntil bounds how long a
// consumer may trust capabilities when a provider stops refreshing them.
type Snapshot struct {
	Provider   string     `json:"provider"`
	Generation uint64     `json:"generation"`
	ObservedAt time.Time  `json:"observed_at"`
	ValidUntil time.Time  `json:"valid_until,omitempty"`
	Endpoints  []Endpoint `json:"endpoints"`
	Links      []Link     `json:"links"`
}

// Provider discovers one modular source of endpoints and directed links.
type Provider interface {
	Snapshot(context.Context) (Snapshot, error)
}

// SnapshotSet retains the newest accepted snapshot per provider. It provides a
// stable read model while providers refresh independently.
type SnapshotSet struct {
	byProvider map[string]Snapshot
}

// Update accepts a provider snapshot if it is live and newer than that
// provider's current generation. Equal generations are idempotent only when the
// capability content is identical.
func (s *SnapshotSet) Update(snapshot Snapshot, now time.Time) error {
	if err := validateSnapshot(snapshot, now); err != nil {
		return err
	}
	if s.byProvider == nil {
		s.byProvider = make(map[string]Snapshot)
	}
	old, exists := s.byProvider[snapshot.Provider]
	if exists {
		if snapshot.Generation < old.Generation {
			return fmt.Errorf("provider %q: generation %d is older than accepted generation %d", snapshot.Provider, snapshot.Generation, old.Generation)
		}
		if snapshot.Generation == old.Generation {
			if snapshotsEqual(old, snapshot) {
				return nil
			}
			return fmt.Errorf("provider %q: generation %d changed content", snapshot.Provider, snapshot.Generation)
		}
	}
	s.byProvider[snapshot.Provider] = cloneSnapshot(snapshot)
	return nil
}

// Refresh reads providers in caller-independent name order and updates the set.
// It is fail-atomic: no provider update is retained unless every read and the
// resulting composed graph succeeds.
func (s *SnapshotSet) Refresh(ctx context.Context, now time.Time, providers ...Provider) (Graph, error) {
	pending := s.clone()
	snapshots := make([]Snapshot, 0, len(providers))
	for _, provider := range providers {
		if provider == nil {
			return Graph{}, errors.New("nil fabric provider")
		}
		snapshot, err := provider.Snapshot(ctx)
		if err != nil {
			return Graph{}, fmt.Errorf("read fabric provider: %w", err)
		}
		snapshots = append(snapshots, snapshot)
	}
	sort.Slice(snapshots, func(i, j int) bool { return snapshots[i].Provider < snapshots[j].Provider })
	for _, snapshot := range snapshots {
		if err := pending.Update(snapshot, now); err != nil {
			return Graph{}, err
		}
	}
	graph, err := pending.Graph(now)
	if err != nil {
		return Graph{}, err
	}
	*s = pending
	return graph, nil
}

// Graph composes all unexpired snapshots. Endpoint and link IDs are global
// identities: duplicate identical declarations coalesce, while conflicting
// declarations fail closed. No reverse link is synthesized.
func (s SnapshotSet) Graph(now time.Time) (Graph, error) {
	providers := make([]string, 0, len(s.byProvider))
	for provider := range s.byProvider {
		providers = append(providers, provider)
	}
	sort.Strings(providers)
	endpoints := make(map[string]Endpoint)
	links := make(map[string]Link)
	for _, provider := range providers {
		snapshot := s.byProvider[provider]
		if expired(snapshot, now) {
			continue
		}
		for _, endpoint := range snapshot.Endpoints {
			if old, exists := endpoints[endpoint.ID]; exists && !endpointEqual(old, endpoint) {
				return Graph{}, fmt.Errorf("endpoint %q conflicts between providers", endpoint.ID)
			}
			endpoints[endpoint.ID] = cloneEndpoint(endpoint)
		}
		for _, link := range snapshot.Links {
			if old, exists := links[link.ID]; exists && !linkEqual(old, link) {
				return Graph{}, fmt.Errorf("link %q conflicts between providers", link.ID)
			}
			links[link.ID] = cloneLink(link)
		}
	}
	graph := Graph{Endpoints: make([]Endpoint, 0, len(endpoints)), Links: make([]Link, 0, len(links))}
	for _, endpoint := range endpoints {
		graph.Endpoints = append(graph.Endpoints, endpoint)
	}
	for _, link := range links {
		graph.Links = append(graph.Links, link)
	}
	sort.Slice(graph.Endpoints, func(i, j int) bool { return graph.Endpoints[i].ID < graph.Endpoints[j].ID })
	sort.Slice(graph.Links, func(i, j int) bool { return graph.Links[i].ID < graph.Links[j].ID })
	if err := graph.Validate(); err != nil {
		return Graph{}, fmt.Errorf("composed fabric: %w", err)
	}
	return graph, nil
}

func validateSnapshot(snapshot Snapshot, now time.Time) error {
	snapshot.Provider = strings.TrimSpace(snapshot.Provider)
	if snapshot.Provider == "" {
		return errors.New("snapshot provider is required")
	}
	if snapshot.Generation == 0 {
		return fmt.Errorf("provider %q: generation must be positive", snapshot.Provider)
	}
	if snapshot.ObservedAt.IsZero() {
		return fmt.Errorf("provider %q: observed_at is required", snapshot.Provider)
	}
	if snapshot.ObservedAt.After(now) {
		return fmt.Errorf("provider %q: observed_at is in the future", snapshot.Provider)
	}
	if expired(snapshot, now) {
		return fmt.Errorf("provider %q: snapshot expired at %s", snapshot.Provider, snapshot.ValidUntil.Format(time.RFC3339Nano))
	}
	return Graph{Endpoints: snapshot.Endpoints, Links: snapshot.Links}.Validate()
}
func expired(snapshot Snapshot, now time.Time) bool {
	return !snapshot.ValidUntil.IsZero() && !now.Before(snapshot.ValidUntil)
}
func (s SnapshotSet) clone() SnapshotSet {
	out := SnapshotSet{byProvider: make(map[string]Snapshot, len(s.byProvider))}
	for k, v := range s.byProvider {
		out.byProvider[k] = cloneSnapshot(v)
	}
	return out
}
func cloneSnapshot(s Snapshot) Snapshot {
	out := s
	out.Endpoints = make([]Endpoint, len(s.Endpoints))
	for i, e := range s.Endpoints {
		out.Endpoints[i] = cloneEndpoint(e)
	}
	out.Links = make([]Link, len(s.Links))
	for i, l := range s.Links {
		out.Links[i] = cloneLink(l)
	}
	return out
}
func cloneEndpoint(e Endpoint) Endpoint { e.Labels = cloneLabels(e.Labels); return e }
func cloneLink(l Link) Link             { l.Labels = cloneLabels(l.Labels); return l }
func cloneLabels(labels map[string]string) map[string]string {
	if labels == nil {
		return nil
	}
	out := make(map[string]string, len(labels))
	for k, v := range labels {
		out[k] = v
	}
	return out
}
func snapshotsEqual(a, b Snapshot) bool {
	if a.Provider != b.Provider || a.Generation != b.Generation || !a.ObservedAt.Equal(b.ObservedAt) || !a.ValidUntil.Equal(b.ValidUntil) || len(a.Endpoints) != len(b.Endpoints) || len(a.Links) != len(b.Links) {
		return false
	}
	ga := Graph{Endpoints: a.Endpoints, Links: a.Links}
	gb := Graph{Endpoints: b.Endpoints, Links: b.Links}
	sort.Slice(ga.Endpoints, func(i, j int) bool { return ga.Endpoints[i].ID < ga.Endpoints[j].ID })
	sort.Slice(gb.Endpoints, func(i, j int) bool { return gb.Endpoints[i].ID < gb.Endpoints[j].ID })
	sort.Slice(ga.Links, func(i, j int) bool { return ga.Links[i].ID < ga.Links[j].ID })
	sort.Slice(gb.Links, func(i, j int) bool { return gb.Links[i].ID < gb.Links[j].ID })
	for i := range ga.Endpoints {
		if !endpointEqual(ga.Endpoints[i], gb.Endpoints[i]) {
			return false
		}
	}
	for i := range ga.Links {
		if !linkEqual(ga.Links[i], gb.Links[i]) {
			return false
		}
	}
	return true
}
func endpointEqual(a, b Endpoint) bool {
	return a.ID == b.ID && a.Kind == b.Kind && labelsEqual(a.Labels, b.Labels)
}
func linkEqual(a, b Link) bool {
	return a.ID == b.ID && a.From == b.From && a.To == b.To && a.Transport == b.Transport && a.Cost == b.Cost && a.LatencyNanos == b.LatencyNanos && a.BandwidthBytesPerSecond == b.BandwidthBytesPerSecond && a.CPUPath == b.CPUPath && labelsEqual(a.Labels, b.Labels)
}
func labelsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}
