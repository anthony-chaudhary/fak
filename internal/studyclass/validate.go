package studyclass

import (
	"encoding/hex"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"
)

// Validate checks the full semantic contract, all derived projections, and
// both content checksums. It never trusts precomputed summaries or clusters.
func Validate(out Output) error {
	var es []error
	if out.Schema != FullSchema {
		es = append(es, fmt.Errorf("schema must be %q", FullSchema))
	}
	if out.Rules != RulesSchema {
		es = append(es, fmt.Errorf("rules must be %q", RulesSchema))
	}
	if out.RelationshipPolicy != "explicit_field_or_text_evidence_only" {
		es = append(es, errors.New("relationship_policy is invalid"))
	}
	validateBinding(out.Input, &es)
	if out.Input.RecordCount != len(out.Records) {
		es = append(es, errors.New("input record count does not match records"))
	}
	seen := map[string]bool{}
	for i, r := range out.Records {
		where := fmt.Sprintf("record %d", i)
		if r.Identity == "" || r.Source == "" || r.Kind == "" || r.ID == 0 || r.Identity != identity(r.Source, r.ID) {
			es = append(es, fmt.Errorf("%s has invalid identity", where))
		}
		if seen[r.Identity] {
			es = append(es, fmt.Errorf("duplicate identity %s", r.Identity))
		}
		seen[r.Identity] = true
		if !validState(r.State) {
			es = append(es, fmt.Errorf("%s has invalid state %q", r.Identity, r.State))
		}
		if !validDisposition(r.Disposition) {
			es = append(es, fmt.Errorf("%s has invalid or unclassified disposition %q", r.Identity, r.Disposition))
		}
		if !validConfidence(r.Confidence) {
			es = append(es, fmt.Errorf("%s has invalid confidence %q", r.Identity, r.Confidence))
		}
		if len(r.DispositionEvidence) == 0 {
			es = append(es, fmt.Errorf("%s disposition has no evidence", r.Identity))
		}
		validateEvidence(r.Identity+" disposition", r.DispositionEvidence, &es)
		if r.State == "open" && (r.Dates.Closed != "" || r.Dates.Merged != "" || r.Merged) {
			es = append(es, fmt.Errorf("%s open state conflicts with closed or merged evidence", r.Identity))
		}
		if (r.Merged || r.Dates.Merged != "") && r.Source != "pulls" {
			es = append(es, fmt.Errorf("%s non-pull carries merged evidence", r.Identity))
		}
		if r.State == "none" && r.Source != "releases" && r.Source != "labels" && r.Source != "milestones" {
			es = append(es, fmt.Errorf("%s actionable source has no state", r.Identity))
		}
		validateDispositionCoherence(r, &es)
		seenMechanisms := map[Mechanism]bool{}
		for _, match := range r.Mechanisms {
			if !validMechanism(match.Name) {
				es = append(es, fmt.Errorf("%s has invalid mechanism %q", r.Identity, match.Name))
			}
			if seenMechanisms[match.Name] {
				es = append(es, fmt.Errorf("%s repeats mechanism %q", r.Identity, match.Name))
			}
			seenMechanisms[match.Name] = true
			if !validConfidence(match.Confidence) {
				es = append(es, fmt.Errorf("%s mechanism %q has invalid confidence", r.Identity, match.Name))
			}
			if len(match.Evidence) == 0 {
				es = append(es, fmt.Errorf("%s mechanism %q has no evidence", r.Identity, match.Name))
			}
			validateEvidence(r.Identity+" mechanism", match.Evidence, &es)
		}
		if r.Disposition == DispositionReleaseMetadataNoncandidate && !seenMechanisms[MechanismExplicitNonCandidate] {
			es = append(es, fmt.Errorf("%s non-candidate disposition lacks explicit_non_candidate mechanism", r.Identity))
		}
	}
	if !classificationsSorted(out.Records) {
		es = append(es, errors.New("records are not in stable source/id order"))
	}
	wantSummary := summarize(out.Records, out.Clusters)
	if !reflect.DeepEqual(out.Summary, wantSummary) {
		es = append(es, errors.New("summary drift"))
	}
	if out.RecordsChecksum != digestJSON(out.Records) {
		es = append(es, errors.New("records checksum drift"))
	}
	wantClusters := buildClusters(out.Records)
	if !reflect.DeepEqual(out.Clusters, wantClusters) {
		es = append(es, errors.New("cluster projection drift"))
	}
	if out.ClustersChecksum != digestJSON(out.Clusters) {
		es = append(es, errors.New("clusters checksum drift"))
	}
	for _, cluster := range out.Clusters {
		validateCluster(cluster, &es)
	}
	return errors.Join(es...)
}

func validateBinding(binding InputBinding, es *[]error) {
	if !validDigest(binding.RawSHA256) {
		*es = append(*es, errors.New("input raw_sha256 is invalid"))
	}
	if !validDigest(binding.IndexChecksum) {
		*es = append(*es, errors.New("input index_checksum is invalid"))
	}
	if binding.Repository == "" || binding.Revision == "" {
		*es = append(*es, errors.New("input repository and revision are required"))
	}
	if _, err := time.Parse(time.RFC3339Nano, binding.Cutoff); err != nil {
		*es = append(*es, errors.New("input cutoff must be RFC3339"))
	}
	if binding.CutoffMode != "created_at_inclusive_non_atomic_observation" {
		*es = append(*es, errors.New("input cutoff_mode is invalid"))
	}
	if binding.PostCutoffUpdatedRecords < 0 || binding.PostCutoffUpdatedRecords > binding.RecordCount {
		*es = append(*es, errors.New("input post_cutoff_updated_records is out of range"))
	}
	if binding.RecordCount < 0 {
		*es = append(*es, errors.New("input record_count cannot be negative"))
	}
}

func validateDispositionCoherence(r Classification, es *[]error) {
	switch r.Disposition {
	case DispositionMergedLanded:
		if r.Source != "pulls" || r.State != "closed" || (!r.Merged && r.Dates.Merged == "") {
			*es = append(*es, fmt.Errorf("%s merged_landed is incoherent with source/state/merge evidence", r.Identity))
		}
	case DispositionOpenProposal:
		if r.State != "open" {
			*es = append(*es, fmt.Errorf("%s open_proposal is not open", r.Identity))
		}
	case DispositionClosedUnmerged:
		if r.State != "closed" || r.Merged || r.Dates.Merged != "" {
			*es = append(*es, fmt.Errorf("%s closed_unmerged is incoherent with state/merge evidence", r.Identity))
		}
	case DispositionReleaseMetadataNoncandidate:
		if r.Source != "releases" && r.Source != "labels" && r.Source != "milestones" {
			*es = append(*es, fmt.Errorf("%s metadata non-candidate has actionable source", r.Identity))
		}
	case DispositionRegressionBug, DispositionDuplicate, DispositionSupportQuestion, DispositionStaleSuperseded:
		// These evidence-based dispositions are coherent for either open or
		// closed records; their required typed evidence is validated separately.
	}
}

func validateEvidence(where string, evidence []Evidence, es *[]error) {
	seen := map[string]bool{}
	for _, e := range evidence {
		if e.Rule == "" || e.Field == "" || e.Signal == "" {
			*es = append(*es, fmt.Errorf("%s has incomplete evidence", where))
		}
		key := e.Rule + "\x00" + e.Field + "\x00" + e.Signal
		if seen[key] {
			*es = append(*es, fmt.Errorf("%s has duplicate evidence", where))
		}
		seen[key] = true
	}
}

func validateCluster(c Cluster, es *[]error) {
	if c.Key != clusterKey(c.Mechanism, c.Rule, c.Signal) {
		*es = append(*es, fmt.Errorf("cluster %q key drift", c.Key))
	}
	if !validMechanism(c.Mechanism) || !validConfidence(c.Confidence) || c.Rule == "" || c.Signal == "" {
		*es = append(*es, fmt.Errorf("cluster %q has invalid vocabulary", c.Key))
	}
	members := append([]IdentityRef{c.Representative}, c.Related...)
	seen := map[string]bool{}
	for _, member := range members {
		if member.Identity == "" || member.Kind == "" || member.Identity != identity(member.Source, member.ID) || seen[member.Identity] {
			*es = append(*es, fmt.Errorf("cluster %q has invalid or duplicate member", c.Key))
		}
		seen[member.Identity] = true
		if !validState(member.State) || !validConfidence(member.Confidence) {
			*es = append(*es, fmt.Errorf("cluster %q member %q has invalid state/confidence", c.Key, member.Identity))
		}
		if c.Actionable && len(member.Evidence) == 0 {
			*es = append(*es, fmt.Errorf("actionable cluster %q member %q has no evidence", c.Key, member.Identity))
		}
		validateEvidence("cluster "+c.Key+" member", member.Evidence, es)
	}
	if !identityRefsSorted(members) {
		*es = append(*es, fmt.Errorf("cluster %q members are not stable", c.Key))
	}
	if c.Actionable != clusterActionable(members) {
		*es = append(*es, fmt.Errorf("cluster %q actionable flag drift", c.Key))
	}
}

func validDigest(s string) bool {
	if !strings.HasPrefix(s, "sha256:") || len(s) != len("sha256:")+64 {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(s, "sha256:"))
	return err == nil
}

func validDisposition(d Disposition) bool {
	for _, candidate := range Dispositions {
		if candidate == d {
			return true
		}
	}
	return false
}

func validMechanism(m Mechanism) bool   { return mechanismRank(m) < len(Mechanisms) }
func validConfidence(c Confidence) bool { return confidenceRank(c) < len(Confidences) }
func validState(s string) bool          { return s == "open" || s == "closed" || s == "none" }

func classificationsSorted(records []Classification) bool {
	for i := 1; i < len(records); i++ {
		if records[i-1].Source > records[i].Source || (records[i-1].Source == records[i].Source && records[i-1].ID >= records[i].ID) {
			return false
		}
	}
	return true
}

func identityRefsSorted(refs []IdentityRef) bool {
	for i := 1; i < len(refs); i++ {
		if !identityRefLess(refs[i-1], refs[i]) {
			return false
		}
	}
	return true
}
