package studyclass

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/studyforge"
)

// Classify deterministically projects every corpus row into exactly one primary
// disposition and zero or more evidence-bearing mechanism matches.
func Classify(c studyforge.Corpus, rawSHA256 string) (Output, error) {
	rawSHA256 = normalizeDigest(rawSHA256)
	if !validDigest(rawSHA256) {
		return Output{}, fmt.Errorf("raw SHA-256 is invalid")
	}
	if !validDigest(c.Receipt.IndexChecksum) {
		return Output{}, fmt.Errorf("corpus index checksum is invalid")
	}
	if c.Receipt.Repository == "" || c.Receipt.Revision == "" || c.Receipt.Cutoff == "" {
		return Output{}, fmt.Errorf("corpus receipt repository, revision, and cutoff are required")
	}

	records := make([]Classification, len(c.Records))
	seen := make(map[string]bool, len(c.Records))
	for i, record := range c.Records {
		key := identity(record.Source, record.ID)
		if record.Source == "" || record.ID == 0 {
			return Output{}, fmt.Errorf("record %d has incomplete identity", i)
		}
		if seen[key] {
			return Output{}, fmt.Errorf("duplicate identity %s", key)
		}
		seen[key] = true
		records[i] = classifyRecord(record)
	}
	sortClassifications(records)
	clusters := buildClusters(records)
	out := Output{
		Schema:             FullSchema,
		Rules:              RulesSchema,
		RelationshipPolicy: "explicit_field_or_text_evidence_only",
		Input: InputBinding{
			RawSHA256:                rawSHA256,
			IndexChecksum:            c.Receipt.IndexChecksum,
			Repository:               c.Receipt.Repository,
			Revision:                 c.Receipt.Revision,
			Cutoff:                   c.Receipt.Cutoff,
			CutoffMode:               "created_at_inclusive_non_atomic_observation",
			PostCutoffUpdatedRecords: postCutoffUpdateCount(c),
			RecordCount:              len(records),
		},
		Records:  records,
		Clusters: clusters,
	}
	out.Summary = summarize(records, clusters)
	out.RecordsChecksum = digestJSON(records)
	out.ClustersChecksum = digestJSON(clusters)
	if err := Validate(out); err != nil {
		return Output{}, fmt.Errorf("classify: %w", err)
	}
	return out, nil
}

// ValidateAgainst verifies semantic output plus exact corpus identity coverage
// and immutable input binding.
func ValidateAgainst(out Output, corpus studyforge.Corpus, rawSHA256 string) error {
	if err := Validate(out); err != nil {
		return err
	}
	wantRaw := normalizeDigest(rawSHA256)
	if out.Input.RawSHA256 != wantRaw || out.Input.IndexChecksum != corpus.Receipt.IndexChecksum ||
		out.Input.Repository != corpus.Receipt.Repository || out.Input.Revision != corpus.Receipt.Revision ||
		out.Input.Cutoff != corpus.Receipt.Cutoff ||
		out.Input.CutoffMode != "created_at_inclusive_non_atomic_observation" ||
		out.Input.PostCutoffUpdatedRecords != postCutoffUpdateCount(corpus) ||
		out.Input.RecordCount != len(corpus.Records) {
		return fmt.Errorf("input binding does not match corpus")
	}
	want := make(map[string]bool, len(corpus.Records))
	for _, r := range corpus.Records {
		key := identity(r.Source, r.ID)
		if want[key] {
			return fmt.Errorf("corpus has duplicate identity %s", key)
		}
		want[key] = true
	}
	for _, r := range out.Records {
		if !want[r.Identity] {
			return fmt.Errorf("classification identity %s is absent from corpus", r.Identity)
		}
		delete(want, r.Identity)
	}
	if len(want) != 0 {
		return fmt.Errorf("classification is missing %d corpus identities", len(want))
	}
	return nil
}

func postCutoffUpdateCount(c studyforge.Corpus) int {
	cutoff, err := time.Parse(time.RFC3339Nano, c.Receipt.Cutoff)
	if err != nil {
		return 0
	}
	count := 0
	for _, record := range c.Records {
		if record.UpdatedAt == "" {
			continue
		}
		updated, err := time.Parse(time.RFC3339Nano, record.UpdatedAt)
		if err == nil && updated.After(cutoff) {
			count++
		}
	}
	return count
}

func buildClusters(records []Classification) []Cluster {
	type bucket struct {
		mechanism   Mechanism
		rule        string
		signal      string
		confidence  Confidence
		members     []IdentityRef
		memberIndex map[string]int
	}
	buckets := map[string]*bucket{}
	for _, record := range records {
		for _, match := range record.Mechanisms {
			for _, evidence := range match.Evidence {
				key := clusterKey(match.Name, evidence.Rule, evidence.Signal)
				b := buckets[key]
				if b == nil {
					b = &bucket{
						mechanism: match.Name, rule: evidence.Rule, signal: evidence.Signal,
						confidence: match.Confidence, memberIndex: map[string]int{},
					}
					buckets[key] = b
				}
				b.confidence = higherConfidence(b.confidence, evidenceConfidence(evidence.Field))
				ref := IdentityRef{
					Identity: record.Identity, Source: record.Source, Kind: record.Kind, ID: record.ID,
					NodeID: record.NodeID, Number: record.Number, URL: record.URL,
					Labels: append([]string(nil), record.Labels...), State: record.State, Dates: record.Dates,
					Disposition: record.Disposition,
					Confidence:  evidenceConfidence(evidence.Field), Evidence: []Evidence{evidence},
				}
				if memberAt, ok := b.memberIndex[record.Identity]; ok {
					b.members[memberAt].Confidence = higherConfidence(b.members[memberAt].Confidence, ref.Confidence)
					b.members[memberAt].Evidence = append(b.members[memberAt].Evidence, evidence)
					sortEvidence(b.members[memberAt].Evidence)
				} else {
					b.memberIndex[record.Identity] = len(b.members)
					b.members = append(b.members, ref)
				}
			}
		}
	}
	keys := make([]string, 0, len(buckets))
	for key := range buckets {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	clusters := make([]Cluster, 0, len(keys))
	for _, key := range keys {
		b := buckets[key]
		sortIdentityRefs(b.members)
		clusters = append(clusters, Cluster{
			Key: key, Mechanism: b.mechanism, Rule: b.rule, Signal: b.signal, Confidence: b.confidence,
			Actionable:     clusterActionable(b.members),
			Representative: b.members[0], Related: append([]IdentityRef(nil), b.members[1:]...),
		})
	}
	return clusters
}

func clusterKey(mechanism Mechanism, rule, signal string) string {
	return string(mechanism) + ":" + strings.TrimPrefix(rule, "mechanism."+string(mechanism)+".") + ":" + slug(signal)
}

func slug(s string) string {
	var b strings.Builder
	lastDash := false
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastDash = false
		} else if !lastDash && b.Len() > 0 {
			b.WriteByte('-')
			lastDash = true
		}
	}
	return strings.Trim(b.String(), "-")
}

func evidenceConfidence(field string) Confidence {
	switch field {
	case "labels", "disposition":
		return ConfidenceHigh
	case "title":
		return ConfidenceMedium
	default:
		return ConfidenceLow
	}
}

func summarize(records []Classification, clusters []Cluster) Summary {
	s := Summary{
		RecordCount: len(records), ClusterCount: len(clusters),
		BySource: map[string]int{}, ByDisposition: map[string]int{}, ByMechanism: map[string]int{},
		ByState: map[string]int{}, ByConfidence: map[string]int{},
	}
	for _, r := range records {
		s.BySource[r.Source]++
		s.ByDisposition[string(r.Disposition)]++
		s.ByState[r.State]++
		s.ByConfidence[string(r.Confidence)]++
		for _, mechanism := range r.Mechanisms {
			s.ByMechanism[string(mechanism.Name)]++
		}
	}
	return s
}

func sortClassifications(records []Classification) {
	sort.Slice(records, func(i, j int) bool {
		if records[i].Source != records[j].Source {
			return records[i].Source < records[j].Source
		}
		return records[i].ID < records[j].ID
	})
}

func sortIdentityRefs(refs []IdentityRef) {
	sort.Slice(refs, func(i, j int) bool { return identityRefLess(refs[i], refs[j]) })
}

func clusterActionable(refs []IdentityRef) bool {
	for _, ref := range refs {
		if ref.State == "open" && (ref.Disposition == DispositionOpenProposal || ref.Disposition == DispositionRegressionBug) {
			return true
		}
	}
	return false
}

func identityRefLess(a, b IdentityRef) bool {
	actionableA := a.State == "open" && (a.Disposition == DispositionOpenProposal || a.Disposition == DispositionRegressionBug)
	actionableB := b.State == "open" && (b.Disposition == DispositionOpenProposal || b.Disposition == DispositionRegressionBug)
	if actionableA != actionableB {
		return actionableA
	}
	mergedA, mergedB := a.Disposition == DispositionMergedLanded, b.Disposition == DispositionMergedLanded
	if mergedA != mergedB {
		return mergedA
	}
	if confidenceRank(a.Confidence) != confidenceRank(b.Confidence) {
		return confidenceRank(a.Confidence) < confidenceRank(b.Confidence)
	}
	if a.Dates.Updated != b.Dates.Updated {
		return a.Dates.Updated > b.Dates.Updated
	}
	if a.Source != b.Source {
		return a.Source < b.Source
	}
	return a.ID < b.ID
}

func digestJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("studyclass: marshal digest input: %v", err))
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func normalizeDigest(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if len(s) == 64 {
		return "sha256:" + s
	}
	return s
}
