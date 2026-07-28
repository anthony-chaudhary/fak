package egresslist

// manifest.go — the PROVENANCE half of the bundled registry: where each shipped list
// came from, the checksum that pins the bytes we actually compiled, and when it was last
// refreshed. registry.go answers "what text ships for this name?"; this file answers
// "where did that text come from, is it still the text we pinned, and how stale is it?"
//
// WHY THE MANIFEST IS THE SOURCE OF TRUTH, NOT THE NETWORK. The decide path is offline
// and deterministic by construction (see the package doc): a policy that names a
// block_list compiles embedded bytes, never a fetch. But bundled tracker/malware lists go
// stale — upstream churns daily. The reconciliation is that refreshing is an EXPLICIT,
// reviewable step that rewrites the checked-in artifact and re-pins its checksum here,
// producing a diff a human reads. This file is the pure, stdlib-only contract that step
// reads and writes; the network-touching engine that drives it lives in
// internal/egressrefresh, so this leaf keeps its tier-1 "no net calls" discipline.
//
// THE CHECKSUM IS LOAD-BEARING, NOT DECORATION. TestBundledListsMatchPinnedChecksum reds
// the trunk when a list's bytes and its pinned sha256 disagree — so a hand-edit that
// quietly widens (or empties) a block list cannot land claiming to be the refreshed
// upstream. An empty block list is all-permissive, which is exactly the silent failure a
// security layer must never have.

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// manifestJSON is the checked-in provenance record for every bundled list. It sits beside
// the lists it describes so a refresh diff shows the artifact and its re-pinned checksum
// in one reviewable change.
//
//go:embed lists/manifest.json
var manifestJSON []byte

// Source is the provenance of one bundled filter list.
//
// SHA256 pins the checked-in artifact — not the upstream response — because the artifact
// is what actually compiles into a decision. Rules is the compiled rule count at pin
// time, which gives a refresh a truncation guard (a feed that collapses from 100k rules
// to 3 is a broken fetch, not a quiet upstream). LastRefreshed is the staleness witness an
// operator reads; it is deliberately NOT stamped into the artifact, so an unchanged
// upstream re-renders byte-identical and produces an empty diff.
//
// License records the UPSTREAM's terms under which we redistribute the derived artifact.
// It is provenance, not decoration: bundling a community list means shipping someone
// else's curation inside our binary, and several of the most popular feeds (EasyList and
// EasyPrivacy among them) are copyleft in a way that is a real decision for a downstream
// packager rather than a detail. Recording it per-list is what lets that decision be
// audited from the checked-in manifest instead of rediscovered from a URL, and what keeps
// an attribution-requiring upstream (MIT, CC BY-SA) actually attributed where we ship it.
type Source struct {
	Name          string `json:"name"`
	URL           string `json:"url"`
	Format        string `json:"format"`
	License       string `json:"license"`
	SHA256        string `json:"sha256"`
	Rules         int    `json:"rules"`
	LastRefreshed string `json:"last_refreshed"`
	Description   string `json:"description"`
}

// Refreshable reports whether this list records an upstream to re-fetch. A bundled list
// with no provenance URL (a hand-authored sample or an operator's own pinned set) is not
// refreshable, and a refresh run reports it as skipped rather than inventing a source.
func (s Source) Refreshable() bool { return strings.TrimSpace(s.URL) != "" }

// Manifest is the parsed provenance record for the whole bundled registry.
type Manifest struct {
	Version int      `json:"version"`
	Lists   []Source `json:"lists"`
}

// ManifestVersion is the schema version this leaf writes and understands.
const ManifestVersion = 1

// ParseManifest decodes manifest bytes. It rejects an unknown schema version and a
// duplicate list name rather than silently letting the last entry win — a duplicated name
// would make "which checksum pins this list?" ambiguous, and ambiguity in a security
// artifact resolves to a refusal.
func ParseManifest(b []byte) (Manifest, error) {
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return Manifest{}, fmt.Errorf("egresslist: parse manifest: %w", err)
	}
	if m.Version != ManifestVersion {
		return Manifest{}, fmt.Errorf("egresslist: manifest version %d, want %d", m.Version, ManifestVersion)
	}
	seen := map[string]bool{}
	for _, s := range m.Lists {
		if s.Name == "" {
			return Manifest{}, fmt.Errorf("egresslist: manifest holds a list with an empty name")
		}
		if seen[s.Name] {
			return Manifest{}, fmt.Errorf("egresslist: manifest names %q twice", s.Name)
		}
		seen[s.Name] = true
	}
	sort.Slice(m.Lists, func(i, j int) bool { return m.Lists[i].Name < m.Lists[j].Name })
	return m, nil
}

// Render encodes a manifest back to canonical bytes: name-sorted, two-space indented, one
// trailing newline. Canonical form means a refresh that changes one checksum produces a
// one-line diff instead of a reordered file.
func (m Manifest) Render() ([]byte, error) {
	out := m
	out.Version = ManifestVersion
	out.Lists = append([]Source(nil), m.Lists...)
	sort.Slice(out.Lists, func(i, j int) bool { return out.Lists[i].Name < out.Lists[j].Name })
	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("egresslist: render manifest: %w", err)
	}
	return append(b, '\n'), nil
}

// Get returns the source recorded for a name.
func (m Manifest) Get(name string) (Source, bool) {
	for _, s := range m.Lists {
		if s.Name == name {
			return s, true
		}
	}
	return Source{}, false
}

// Set replaces (or appends) the entry for s.Name.
func (m *Manifest) Set(s Source) {
	for i := range m.Lists {
		if m.Lists[i].Name == s.Name {
			m.Lists[i] = s
			return
		}
	}
	m.Lists = append(m.Lists, s)
}

// BundledManifest returns the parsed provenance record that ships in the binary.
func BundledManifest() (Manifest, error) { return ParseManifest(manifestJSON) }

// Sources returns the provenance of every bundled list, name-sorted.
func Sources() ([]Source, error) {
	m, err := BundledManifest()
	if err != nil {
		return nil, err
	}
	return m.Lists, nil
}

// SourceFor returns the provenance recorded for one bundled list name.
func SourceFor(name string) (Source, bool) {
	m, err := BundledManifest()
	if err != nil {
		return Source{}, false
	}
	return m.Get(name)
}

// Checksum is the pinning hash: hex(sha256) of the list text with line endings normalized
// to LF and any UTF-8 BOM dropped.
//
// The normalization is what makes the pin portable. This repo is developed on Windows and
// Linux against the same checked-in artifact; a checkout that materializes CRLF would
// otherwise hash differently from the LF bytes the pin was computed on and red the
// checksum gate on one platform only — a false alarm on a security guard, which is worse
// than no guard because it teaches people to ignore it. Hashing the canonical LF form
// pins the CONTENT, which is what actually decides an egress verdict.
func Checksum(text string) string {
	t := strings.TrimPrefix(text, "\xef\xbb\xbf") // UTF-8 BOM
	t = strings.ReplaceAll(t, "\r\n", "\n")
	sum := sha256.Sum256([]byte(t))
	return hex.EncodeToString(sum[:])
}

// VerifyBundled reports whether a bundled list's shipped bytes still hash to the checksum
// the manifest pins. It fails closed: an unknown name, an unparseable manifest, or a
// mismatch is an error, never a silent pass.
func VerifyBundled(name string) error {
	s, ok := SourceFor(name)
	if !ok {
		return fmt.Errorf("egresslist: no manifest entry for list %q", name)
	}
	text, ok := BundledList(name)
	if !ok {
		return fmt.Errorf("egresslist: manifest names list %q but no such list ships", name)
	}
	if got := Checksum(text); got != s.SHA256 {
		return fmt.Errorf("egresslist: list %q checksum mismatch: artifact hashes to %s, manifest pins %s "+
			"(hand-edited? re-pin with `fak egresslist refresh`)", name, got, s.SHA256)
	}
	return nil
}

// Rule is one compiled rule surfaced back out of a List, so a refresh can re-emit the
// list it just parsed in canonical form and an inspection surface can enumerate what a
// list actually decides.
type Rule struct {
	Domain string
	Kind   Kind
	Source string
}

// Rules returns every compiled rule, sorted deterministically (all Block rules by domain,
// then all Allow rules by domain). Determinism is the point: the refresh path renders this
// order straight to the checked-in artifact, so re-rendering an unchanged upstream yields
// byte-identical text and an empty diff.
func (l *List) Rules() []Rule {
	if l == nil {
		return nil
	}
	out := make([]Rule, 0, len(l.block)+len(l.allow))
	for _, e := range l.block {
		out = append(out, Rule{Domain: e.rule, Kind: Block, Source: e.source})
	}
	for _, e := range l.allow {
		out = append(out, Rule{Domain: e.rule, Kind: Allow, Source: e.source})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind // Block (1) before Allow (2)
		}
		return out[i].Domain < out[j].Domain
	})
	return out
}

// RenderArtifact renders a compiled List back to canonical filter-list text: a provenance
// header plus one adblock domain-anchor line per rule, block rules then allow rules, each
// group domain-sorted.
//
// It is the exact inverse of AddFilterText for the grammar this leaf models, so a refresh
// round-trips (TestRenderArtifactRoundTrips pins it) and the checked-in artifact is always
// the NORMALIZED form — the messy upstream (hosts lines, element-hiding rules, options we
// deliberately do not model) is reduced to the rules that actually decide, which is what a
// reviewer should be reading in the diff.
//
// It carries NO timestamp: staleness lives in the manifest's last_refreshed. That keeps
// the artifact a pure function of the upstream rules, so re-refreshing an unchanged
// upstream re-renders identical bytes and shows an empty diff instead of timestamp noise.
func RenderArtifact(s Source, l *List) string {
	block, allow := l.Counts()
	var b strings.Builder
	b.WriteString("! fak bundled egress filter list - GENERATED by `fak egresslist refresh`; do not hand-edit.\n")
	b.WriteString("! Name: " + s.Name + "\n")
	if s.URL != "" {
		b.WriteString("! Source: " + s.URL + "\n")
	}
	if s.License != "" {
		// Carried in the artifact as well as the manifest: an MIT/CC BY-SA upstream
		// requires attribution to travel WITH the copy, and the artifact is the copy.
		b.WriteString("! Upstream-License: " + s.License + "\n")
	}
	if s.Description != "" {
		b.WriteString("! Description: " + s.Description + "\n")
	}
	fmt.Fprintf(&b, "! Rules: %d block, %d allow\n", block, allow)
	b.WriteString("! Last-refreshed is recorded in lists/manifest.json, not here, so an unchanged\n")
	b.WriteString("!   upstream re-renders byte-identical (empty diff).\n")
	b.WriteString("!\n")
	for _, r := range l.Rules() {
		if r.Kind == Allow {
			b.WriteString("@@")
		}
		b.WriteString("||" + r.Domain + "^\n")
	}
	return b.String()
}
