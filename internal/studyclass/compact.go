package studyclass

import (
	"errors"
	"fmt"
	"reflect"
)

// Compact projects the validated full output into a bounded committed index.
// Omitted related identities remain bound by count and members checksum.
func Compact(out Output, relatedSampleLimit int) (CompactIndex, error) {
	if err := Validate(out); err != nil {
		return CompactIndex{}, err
	}
	if relatedSampleLimit < 0 {
		return CompactIndex{}, fmt.Errorf("related sample limit cannot be negative")
	}
	if relatedSampleLimit == 0 {
		relatedSampleLimit = DefaultRelatedSampleLimit
	}
	clusters := make([]CompactCluster, len(out.Clusters))
	for i, cluster := range out.Clusters {
		n := relatedSampleLimit
		if len(cluster.Related) < n {
			n = len(cluster.Related)
		}
		members := append([]IdentityRef{cluster.Representative}, cluster.Related...)
		clusters[i] = CompactCluster{
			Key: cluster.Key, Mechanism: cluster.Mechanism, Rule: cluster.Rule, Signal: cluster.Signal,
			Actionable: cluster.Actionable, Confidence: cluster.Confidence, MemberCount: len(members), RelatedCount: len(cluster.Related),
			MembersChecksum: digestJSON(members), Representative: cluster.Representative,
			RelatedSamples: append([]IdentityRef(nil), cluster.Related[:n]...),
		}
	}
	index := CompactIndex{
		Schema: CompactSchema, Rules: out.Rules, RelationshipPolicy: out.RelationshipPolicy,
		Input: out.Input, FullOutputChecksum: canonicalOutputChecksum(out),
		RecordsChecksum: out.RecordsChecksum, ClustersChecksum: out.ClustersChecksum,
		RelatedSampleLimit: relatedSampleLimit, Summary: out.Summary, Clusters: clusters,
	}
	index.CompactClustersChecksum = digestJSON(index.Clusters)
	if err := ValidateCompact(index); err != nil {
		return CompactIndex{}, err
	}
	return index, nil
}

func ValidateCompact(index CompactIndex) error {
	var es []error
	if index.Schema != CompactSchema {
		es = append(es, fmt.Errorf("schema must be %q", CompactSchema))
	}
	if index.Rules != RulesSchema || index.RelationshipPolicy != "explicit_field_or_text_evidence_only" {
		es = append(es, errors.New("compact rules or relationship policy is invalid"))
	}
	validateBinding(index.Input, &es)
	if index.RelatedSampleLimit <= 0 {
		es = append(es, errors.New("related_sample_limit must be positive"))
	}
	for name, digest := range map[string]string{
		"full output": index.FullOutputChecksum, "records": index.RecordsChecksum,
		"clusters": index.ClustersChecksum, "compact clusters": index.CompactClustersChecksum,
	} {
		if !validDigest(digest) {
			es = append(es, fmt.Errorf("%s checksum is invalid", name))
		}
	}
	if index.CompactClustersChecksum != digestJSON(index.Clusters) {
		es = append(es, errors.New("compact clusters checksum drift"))
	}
	if index.Summary.RecordCount != index.Input.RecordCount || index.Summary.ClusterCount != len(index.Clusters) {
		es = append(es, errors.New("compact summary count drift"))
	}
	if !summaryMapsInitialized(index.Summary) {
		es = append(es, errors.New("compact summary maps must be present"))
	} else {
		validateCompactSummary(index.Summary, &es)
	}
	for _, cluster := range index.Clusters {
		if cluster.Key != clusterKey(cluster.Mechanism, cluster.Rule, cluster.Signal) || !validMechanism(cluster.Mechanism) || !validConfidence(cluster.Confidence) {
			es = append(es, fmt.Errorf("compact cluster %q has invalid vocabulary or key", cluster.Key))
		}
		if cluster.MemberCount != cluster.RelatedCount+1 || cluster.MemberCount <= 0 {
			es = append(es, fmt.Errorf("compact cluster %q count drift", cluster.Key))
		}
		if len(cluster.RelatedSamples) > index.RelatedSampleLimit || len(cluster.RelatedSamples) > cluster.RelatedCount {
			es = append(es, fmt.Errorf("compact cluster %q exceeds sample bound", cluster.Key))
		}
		if !validDigest(cluster.MembersChecksum) {
			es = append(es, fmt.Errorf("compact cluster %q has invalid members checksum", cluster.Key))
		}
		members := append([]IdentityRef{cluster.Representative}, cluster.RelatedSamples...)
		if !identityRefsSorted(members) {
			es = append(es, fmt.Errorf("compact cluster %q samples are not stable", cluster.Key))
		}
		if cluster.MemberCount == len(members) && cluster.MembersChecksum != digestJSON(members) {
			es = append(es, fmt.Errorf("compact cluster %q members checksum drift", cluster.Key))
		}
		for _, ref := range members {
			if ref.Identity != identity(ref.Source, ref.ID) || !validState(ref.State) || !validConfidence(ref.Confidence) {
				es = append(es, fmt.Errorf("compact cluster %q has invalid member", cluster.Key))
			}
			if cluster.Actionable && len(ref.Evidence) == 0 {
				es = append(es, fmt.Errorf("actionable compact cluster %q has member without evidence", cluster.Key))
			}
		}
	}
	if len(index.Clusters) > 1 {
		for i := 1; i < len(index.Clusters); i++ {
			if index.Clusters[i-1].Key >= index.Clusters[i].Key {
				es = append(es, errors.New("compact clusters are not in stable key order"))
				break
			}
		}
	}
	return errors.Join(es...)
}

// ValidateCompactAgainst binds the commit-sized projection to the validated
// full classification. Digests that intentionally summarize omitted rows are
// references in the compact file; this join makes every one recomputable.
func ValidateCompactAgainst(index CompactIndex, full Output) error {
	if err := Validate(full); err != nil {
		return fmt.Errorf("full classification: %w", err)
	}
	if err := ValidateCompact(index); err != nil {
		return err
	}
	want, err := Compact(full, index.RelatedSampleLimit)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(index, want) {
		return errors.New("compact index does not match full classification")
	}
	return nil
}

func validateCompactSummary(summary Summary, es *[]error) {
	for _, group := range []struct {
		name   string
		counts map[string]int
		total  int
	}{
		{"source", summary.BySource, summary.RecordCount},
		{"disposition", summary.ByDisposition, summary.RecordCount},
		{"state", summary.ByState, summary.RecordCount},
		{"confidence", summary.ByConfidence, summary.RecordCount},
	} {
		total := 0
		for key, count := range group.counts {
			if key == "" || count < 0 {
				*es = append(*es, fmt.Errorf("compact %s summary has invalid row", group.name))
			}
			total += count
		}
		if total != group.total {
			*es = append(*es, fmt.Errorf("compact %s summary total drift", group.name))
		}
	}
	for key, count := range summary.ByMechanism {
		if !validMechanism(Mechanism(key)) || count < 0 {
			*es = append(*es, errors.New("compact mechanism summary has invalid row"))
		}
	}
	for key := range summary.ByDisposition {
		if !validDisposition(Disposition(key)) {
			*es = append(*es, errors.New("compact disposition summary has invalid row"))
		}
	}
	for key := range summary.ByConfidence {
		if !validConfidence(Confidence(key)) {
			*es = append(*es, errors.New("compact confidence summary has invalid row"))
		}
	}
	for key := range summary.ByState {
		if !validState(key) {
			*es = append(*es, errors.New("compact state summary has invalid row"))
		}
	}
}

func summaryMapsInitialized(s Summary) bool {
	return !reflect.ValueOf(s.BySource).IsNil() && !reflect.ValueOf(s.ByDisposition).IsNil() &&
		!reflect.ValueOf(s.ByMechanism).IsNil() && !reflect.ValueOf(s.ByState).IsNil() &&
		!reflect.ValueOf(s.ByConfidence).IsNil()
}
