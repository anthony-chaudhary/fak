package memview

import (
	"fmt"
	"sort"
	"strconv"
)

const KindSkillManifest ViewKind = "skill-manifest"

type SkillManifestEntry struct {
	Name, Version, Provenance                        string
	Value                                            float64
	Active, Witnessed, Admitted, Sealed, Quarantined bool
}

type SkillManifest struct {
	Surface                     Surface
	Format                      Format
	Bytes                       []byte
	SourceDigest, ContentDigest string
	Included, Dropped           int
}

func (m SkillManifest) IsValid(entries []SkillManifestEntry) bool {
	return m.SourceDigest != "" && m.SourceDigest == skillManifestSourceDigest(entries)
}

func BuildSkillManifest(entries []SkillManifestEntry, format Format, budget int) (SkillManifest, error) {
	if budget <= 0 {
		return SkillManifest{}, fmt.Errorf("skill-manifest budget must be positive")
	}
	eligible := make([]SkillManifestEntry, 0, len(entries))
	for _, e := range entries {
		if e.Active && e.Witnessed && e.Admitted && !e.Sealed && !e.Quarantined && e.Name != "" {
			eligible = append(eligible, e)
		}
	}
	sort.SliceStable(eligible, func(i, j int) bool {
		if eligible[i].Value == eligible[j].Value {
			return eligible[i].Name < eligible[j].Name
		}
		return eligible[i].Value > eligible[j].Value
	})
	sourceDigest := skillManifestSourceDigest(entries)
	for keep := len(eligible); keep >= 0; keep-- {
		rows := make([]Row, 0, keep+1)
		for _, e := range eligible[:keep] {
			rows = append(rows, Row{e.Name, e.Version, e.Provenance, strconv.FormatFloat(e.Value, 'f', -1, 64)})
		}
		dropped := len(eligible) - keep
		if dropped > 0 {
			rows = append(rows, Row{"OVERFLOW", "", fmt.Sprintf("dropped=%d reason=byte_budget", dropped), ""})
		}
		s, err := NewSurface("skill-manifest", []string{"skill", "version", "provenance", "value"}, rows)
		if err != nil {
			return SkillManifest{}, err
		}
		b, err := Encode(format, s)
		if err != nil {
			return SkillManifest{}, err
		}
		if len(b) <= budget {
			return SkillManifest{Surface: s, Format: format, Bytes: b, SourceDigest: sourceDigest, ContentDigest: Digest(b), Included: keep, Dropped: dropped}, nil
		}
	}
	return SkillManifest{}, fmt.Errorf("skill-manifest budget %d cannot fit typed overflow view", budget)
}

func skillManifestSourceDigest(entries []SkillManifestEntry) string {
	cp := append([]SkillManifestEntry(nil), entries...)
	sort.Slice(cp, func(i, j int) bool {
		if cp[i].Name == cp[j].Name {
			return cp[i].Version < cp[j].Version
		}
		return cp[i].Name < cp[j].Name
	})
	var b []byte
	for _, e := range cp {
		b = append(b, fmt.Sprintf("%s\x00%s\x00%s\x00%g\x00%t%t%t%t%t\n", e.Name, e.Version, e.Provenance, e.Value, e.Active, e.Witnessed, e.Admitted, e.Sealed, e.Quarantined)...)
	}
	return Digest(b)
}
