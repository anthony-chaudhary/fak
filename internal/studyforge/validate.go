package studyforge

import (
	"encoding/hex"
	"errors"
	"fmt"
	"net/url"
	"sort"
	"strings"
	"time"
)

// Validate verifies that a corpus is terminally complete as well as internally sound.
func Validate(c Corpus) error { return validateCorpus(c, true) }

// validateCheckpoint accepts explicit partial/failed progress, but verifies every durable
// fact already claimed by the checkpoint. Resume must never build on a weak partial file.
func validateCheckpoint(c Corpus) error { return validateCorpus(c, false, false) }

// validateResumeCheckpoint recognizes only the historical shapes with a
// one-way Capture migration. The exception is private to resume loading;
// ordinary reads, writes, and validation remain strict.
func validateResumeCheckpoint(c Corpus) error { return validateCorpus(c, false, true) }

func validateCorpus(c Corpus, requireComplete bool, allowLegacyResume ...bool) error {
	allowLegacy := len(allowLegacyResume) == 1 && allowLegacyResume[0]
	es := validateCorpusHeader(c)
	r := c.Receipt

	seenSources := map[string]bool{}
	seenNodeIDs := map[string]bool{}
	complete := true
	var issuesSource, pullsSource *SourceReceipt
	for sourceIndex, s := range r.Sources {
		if sourceIndex >= len(SourceNames) || s.Name != SourceNames[sourceIndex] {
			es = append(es, fmt.Errorf("source order mismatch at position %d", sourceIndex+1))
		}
		if sourceRank(s.Name) >= len(SourceNames) {
			es = append(es, fmt.Errorf("unknown source %q", s.Name))
			continue
		}
		if seenSources[s.Name] {
			es = append(es, fmt.Errorf("duplicate source %q", s.Name))
		}
		seenSources[s.Name] = true
		if s.Endpoint == "" {
			es = append(es, fmt.Errorf("%s endpoint is required", s.Name))
		}
		if s.Status != StatusComplete && s.Status != StatusPartial && s.Status != StatusFailed {
			es = append(es, fmt.Errorf("%s has invalid status %q", s.Name, s.Status))
		}
		if s.Status != StatusComplete {
			complete = false
		}
		if s.Status == StatusComplete && s.Failure != "" {
			es = append(es, fmt.Errorf("%s complete with failure evidence", s.Name))
		}
		if s.CrawlStartedAt != "" {
			if _, e := time.Parse(time.RFC3339Nano, s.CrawlStartedAt); e != nil {
				es = append(es, fmt.Errorf("%s crawl_started_at must be RFC3339", s.Name))
			}
		}
		if s.CrawlEndedAt != "" {
			if _, e := time.Parse(time.RFC3339Nano, s.CrawlEndedAt); e != nil {
				es = append(es, fmt.Errorf("%s crawl_ended_at must be RFC3339", s.Name))
			}
		}
		if s.Status == StatusComplete && (s.CrawlStartedAt == "" || s.CrawlEndedAt == "") && r.Status == StatusComplete {
			// Old partial checkpoints may predate per-source windows. Capture
			// backfills them from accepted page receipts before completion.
			es = append(es, fmt.Errorf("%s complete source requires crawl start and end", s.Name))
		}
		if s.CrawlStartedAt != "" && s.CrawlEndedAt != "" {
			started, startErr := time.Parse(time.RFC3339Nano, s.CrawlStartedAt)
			ended, endErr := time.Parse(time.RFC3339Nano, s.CrawlEndedAt)
			if startErr == nil && endErr == nil && ended.Before(started) {
				es = append(es, fmt.Errorf("%s crawl ends before it starts", s.Name))
			}
		}

		seenPageURLs := map[string]bool{}
		for i, p := range s.Pages {
			if p.Number != i+1 {
				es = append(es, fmt.Errorf("%s missing page %d", s.Name, i+1))
			}
			if p.URL == "" || seenPageURLs[p.URL] {
				es = append(es, fmt.Errorf("%s page %d has empty or repeated URL", s.Name, p.Number))
			}
			seenPageURLs[p.URL] = true
			if !validDigest(p.Checksum) {
				es = append(es, fmt.Errorf("%s page %d checksum is invalid", s.Name, p.Number))
			}
			if _, e := time.Parse(time.RFC3339Nano, p.FetchedAt); e != nil {
				es = append(es, fmt.Errorf("%s page %d fetched_at must be RFC3339", s.Name, p.Number))
			}
			disabledDiscussions := s.Name == "discussions" && s.Status == StatusComplete && len(s.Pages) == 1 && i == 0 &&
				p.URL == s.Endpoint && p.StatusCode == 410 && p.ItemCount == 0 && p.Next == "" &&
				p.Checksum == digest([]byte(discussionsDisabledPayload))
			if (p.StatusCode < 200 || p.StatusCode >= 300) && !disabledDiscussions {
				es = append(es, fmt.Errorf("%s page %d status code is not successful", s.Name, p.Number))
			}
			if i < len(s.Pages)-1 && p.Next != s.Pages[i+1].URL {
				es = append(es, fmt.Errorf("%s page %d next cursor does not match page %d URL", s.Name, p.Number, p.Number+1))
			}
			if p.Next != "" && seenPageURLs[p.Next] {
				es = append(es, fmt.Errorf("%s page %d repeats an earlier next cursor", s.Name, p.Number))
			}
			if i == len(s.Pages)-1 && s.Status == StatusComplete && p.Next != "" {
				es = append(es, fmt.Errorf("%s complete with unfetched next page", s.Name))
			}
		}
		if s.Status == StatusComplete && len(s.Pages) == 0 {
			es = append(es, fmt.Errorf("%s complete without a terminal page", s.Name))
		}
		if s.PageChecksum != "" && s.PageChecksum != pageDigest(s.Pages) {
			es = append(es, fmt.Errorf("%s page chain checksum mismatch", s.Name))
		}

		rs := recordsForSource(c.Records, s.Name)
		pageItems := 0
		for _, p := range s.Pages {
			pageItems += p.ItemCount
		}
		if pageItems != s.FetchedCount {
			es = append(es, fmt.Errorf("%s page item count mismatch", s.Name))
		}
		ids := map[int64]bool{}
		for _, x := range rs {
			if ids[x.ID] {
				es = append(es, fmt.Errorf("%s duplicate id %d", s.Name, x.ID))
			}
			ids[x.ID] = true
			if x.NodeID != "" {
				if seenNodeIDs[x.NodeID] {
					es = append(es, fmt.Errorf("duplicate node_id %s", x.NodeID))
				}
				seenNodeIDs[x.NodeID] = true
			}
			want := map[string]string{"issues": "issue", "pulls": "pull", "discussions": "discussion", "releases": "release", "labels": "label", "milestones": "milestone"}[s.Name]
			if x.Kind != want {
				es = append(es, fmt.Errorf("%s id %d has mixed or unclassified kind %q", s.Name, x.ID, x.Kind))
			}
			if x.CreatedAt != "" {
				if created, e := time.Parse(time.RFC3339, x.CreatedAt); e != nil || created.After(mustParseCutoff(r.Cutoff)) {
					es = append(es, fmt.Errorf("%s id %d violates cutoff", s.Name, x.ID))
				}
			}
		}
		if s.NormalizedCount != len(rs) || s.UniqueCount != len(ids) {
			es = append(es, fmt.Errorf("%s count mismatch", s.Name))
		}
		if s.FetchedCount != s.NormalizedCount+s.ClassifiedPullCount+s.CutoffExcludedCount {
			es = append(es, fmt.Errorf("%s fetched count mismatch", s.Name))
		}
		if s.Checksum != recordDigest(rs) {
			es = append(es, fmt.Errorf("%s checksum mismatch", s.Name))
		}
		if s.Name == "issues" {
			legacy := allowLegacy && isLegacyMixedPullCheckpoint(r, s)
			countOnly := r.NonAtomicDelta != nil && r.NonAtomicDelta.EvidenceMode == NonAtomicDeltaEvidenceModeLegacyCountOnly
			if countOnly {
				if s.ClassifiedPullIdentities != nil || s.ClassifiedPullChecksum != "" {
					es = append(es, errors.New("legacy count-only issues evidence must keep mixed identities explicitly unavailable"))
				}
			} else if !legacy {
				if len(s.ClassifiedPullIdentities) != s.ClassifiedPullCount {
					es = append(es, fmt.Errorf("issues classified pull identity count %d does not match classified_pull_count %d", len(s.ClassifiedPullIdentities), s.ClassifiedPullCount))
				}
				if !validDigest(s.ClassifiedPullChecksum) || s.ClassifiedPullChecksum != identityDigest(s.ClassifiedPullIdentities) {
					es = append(es, errors.New("issues classified pull identity checksum mismatch"))
				}
				if err := validateIdentityList("issues classified pulls", s.ClassifiedPullIdentities); err != nil {
					es = append(es, err)
				}
			}
			copy := s
			issuesSource = &copy
		}
		if s.Name == "pulls" {
			copy := s
			pullsSource = &copy
		}
	}
	if r.NonAtomicDelta != nil && (issuesSource == nil || pullsSource == nil || issuesSource.Status != StatusComplete || pullsSource.Status != StatusComplete) {
		es = append(es, errors.New("non_atomic_delta evidence requires completed issues and pulls sources"))
	}
	if issuesSource != nil && pullsSource != nil && issuesSource.Status == StatusComplete && pullsSource.Status == StatusComplete {
		legacy := allowLegacy && isLegacyMixedPullCheckpoint(r, *issuesSource)
		if !legacy {
			if err := validateNonAtomicDelta(r, *issuesSource, *pullsSource, recordsForSource(c.Records, "pulls"), requireComplete, allowLegacy); err != nil {
				es = append(es, err)
			}
		}
	}
	es = append(es, validateRecordSourcesAndOrder(c.Records, seenSources)...)
	if r.Status == StatusComplete && (!complete || len(seenSources) != len(SourceNames)) {
		es = append(es, errors.New("partial receipt marked complete"))
	}
	if requireComplete && r.Status != StatusComplete {
		es = append(es, fmt.Errorf("receipt status must be complete, got %q", r.Status))
	}
	if r.IndexChecksum != recordDigest(c.Records) {
		es = append(es, errors.New("index checksum mismatch"))
	}
	return errors.Join(es...)
}

func validateCorpusHeader(c Corpus) []error {
	var es []error
	if c.Schema != CorpusSchema {
		es = append(es, fmt.Errorf("schema must be %q", CorpusSchema))
	}
	r := c.Receipt
	if r.Schema != ReceiptSchema {
		es = append(es, fmt.Errorf("receipt schema must be %q", ReceiptSchema))
	}
	if r.Repository == "" {
		es = append(es, errors.New("repository is required"))
	}
	if r.Revision == "" && (r.Status != StatusFailed || len(r.Sources) != 0) {
		es = append(es, errors.New("revision is required once source capture starts"))
	}
	if _, e := time.Parse(time.RFC3339Nano, r.Cutoff); e != nil {
		es = append(es, errors.New("cutoff must be RFC3339"))
	}
	if r.APIBase != "" {
		base, e := url.Parse(r.APIBase)
		if e != nil || base.Scheme == "" || base.Host == "" {
			es = append(es, errors.New("api_base must be an absolute URL"))
		}
	}
	if r.StartedAt != "" {
		if _, e := time.Parse(time.RFC3339Nano, r.StartedAt); e != nil {
			es = append(es, errors.New("started_at must be RFC3339"))
		}
	}
	if r.CompletedAt != "" {
		if _, e := time.Parse(time.RFC3339Nano, r.CompletedAt); e != nil {
			es = append(es, errors.New("completed_at must be RFC3339"))
		}
	}
	if r.Status != StatusComplete && r.Status != StatusPartial && r.Status != StatusFailed {
		es = append(es, fmt.Errorf("invalid receipt status %q", r.Status))
	}
	for _, api := range r.API {
		if api.URL == "" || !validDigest(api.Checksum) {
			es = append(es, fmt.Errorf("%s API receipt has invalid URL or checksum", api.Purpose))
		}
	}
	return es
}

func validateRecordSourcesAndOrder(records []Record, seenSources map[string]bool) []error {
	var es []error
	for _, record := range records {
		if !seenSources[record.Source] {
			es = append(es, fmt.Errorf("record source %q has no receipt", record.Source))
		}
	}
	ordered := append([]Record(nil), records...)
	sort.Slice(ordered, func(i, j int) bool {
		a, b := ordered[i], ordered[j]
		if sourceRank(a.Source) != sourceRank(b.Source) {
			return sourceRank(a.Source) < sourceRank(b.Source)
		}
		if a.ID != b.ID {
			return a.ID < b.ID
		}
		if a.Number != b.Number {
			return a.Number < b.Number
		}
		return a.Name < b.Name
	})
	for i := range ordered {
		if ordered[i].Source != records[i].Source || ordered[i].ID != records[i].ID || ordered[i].Number != records[i].Number || ordered[i].Name != records[i].Name {
			es = append(es, errors.New("records are not in canonical order"))
			break
		}
	}
	return es
}

func validateNonAtomicDelta(r Receipt, issues, pulls SourceReceipt, pullRecords []Record, requireComplete bool, allowLegacyResume ...bool) error {
	if r.NonAtomicDelta == nil {
		return errors.New("completed issues and pulls sources require non_atomic_delta evidence")
	}

	delta := *r.NonAtomicDelta
	if delta.Type != NonAtomicDeltaType || delta.MixedSource != "issues" || delta.DedicatedSource != "pulls" {
		return errors.New("non_atomic_delta has invalid type or sources")
	}
	if delta.MixedCrawl != (CrawlWindow{StartedAt: issues.CrawlStartedAt, EndedAt: issues.CrawlEndedAt}) || delta.DedicatedCrawl != (CrawlWindow{StartedAt: pulls.CrawlStartedAt, EndedAt: pulls.CrawlEndedAt}) {
		return errors.New("non_atomic_delta crawl windows contradict source receipts")
	}
	for name, window := range map[string]CrawlWindow{"mixed": delta.MixedCrawl, "dedicated": delta.DedicatedCrawl} {
		started, startErr := time.Parse(time.RFC3339Nano, window.StartedAt)
		ended, endErr := time.Parse(time.RFC3339Nano, window.EndedAt)
		if startErr != nil || endErr != nil || ended.Before(started) {
			return fmt.Errorf("non_atomic_delta %s crawl window is invalid", name)
		}
	}
	legacyExactShape := delta.EvidenceMode == "" &&
		(delta.IdentityBasis == identityBasisCaptured || delta.IdentityBasis == identityBasisLegacyProjection) &&
		delta.Overlap != nil && delta.OnlyInMixed != nil && delta.OnlyInDedicated != nil &&
		delta.OverlapCount != nil && delta.OnlyInMixedCount != nil && delta.OnlyInDedicatedCount != nil
	if legacyExactShape {
		delta.EvidenceMode = NonAtomicDeltaEvidenceModeExactIdentity
	}
	switch delta.EvidenceMode {
	case NonAtomicDeltaEvidenceModeExactIdentity:
		if delta.Policy.Metric == "" {
			if err := validateNonAtomicDeltaPolicy(delta.Policy, NonAtomicDeltaPolicyMetricExactIdentitySymmetricDifference, true); err != nil {
				return err
			}
		} else {
			if err := validateExactNonAtomicDeltaPolicy(delta.Policy, len(issues.ClassifiedPullIdentities), len(pullRecords)); err != nil {
				return err
			}
		}
		return validateExactNonAtomicDelta(r, issues, pullRecords, delta, requireComplete, legacyExactShape)
	case NonAtomicDeltaEvidenceModeLegacyCountOnly:
		if len(allowLegacyResume) == 1 && allowLegacyResume[0] && delta.Policy.Metric == "" {
			return validateLegacyPreMetricCountOnlyNonAtomicDelta(r, issues, pullRecords, delta)
		}
		if err := validateNonAtomicDeltaPolicy(delta.Policy, NonAtomicDeltaPolicyMetricEndpointCardinalityDelta, false); err != nil {
			return err
		}
		return validateLegacyCountOnlyNonAtomicDelta(r, issues, pullRecords, delta, requireComplete)
	default:
		return fmt.Errorf("non_atomic_delta has invalid evidence_mode %q", delta.EvidenceMode)
	}
}

func validateExactNonAtomicDelta(r Receipt, issues SourceReceipt, pullRecords []Record, delta NonAtomicDeltaEvidence, requireComplete, legacyShape bool) error {
	validBasis := delta.IdentityBasis == identityBasisCaptured || delta.IdentityBasis == identityBasisLegacyProjection
	identityAvailabilityConsistent := delta.IdentitySetsAvailable == nil || *delta.IdentitySetsAvailable
	explicitProvenance := identityAvailabilityConsistent && delta.EvidenceReason == "" &&
		delta.MixedEvidence == CrossEndpointEvidenceExactIdentities &&
		delta.DedicatedEvidence == CrossEndpointEvidenceExactIdentities &&
		delta.RelationEvidence == CrossEndpointEvidenceExactIdentities
	legacyProvenance := legacyShape && delta.EvidenceReason == "" && delta.MixedEvidence == "" && delta.DedicatedEvidence == "" && delta.RelationEvidence == ""
	if !validBasis || (!explicitProvenance && !legacyProvenance) {
		return errors.New("exact non_atomic_delta has contradictory provenance")
	}
	if len(issues.ClassifiedPullIdentities) != issues.ClassifiedPullCount {
		return fmt.Errorf("issues classified pull identity count %d does not match classified_pull_count %d", len(issues.ClassifiedPullIdentities), issues.ClassifiedPullCount)
	}
	if !validDigest(issues.ClassifiedPullChecksum) || issues.ClassifiedPullChecksum != identityDigest(issues.ClassifiedPullIdentities) {
		return errors.New("issues classified pull identity checksum mismatch")
	}
	if err := validateIdentityList("issues classified pulls", issues.ClassifiedPullIdentities); err != nil {
		return err
	}
	if delta.Overlap == nil || delta.OnlyInMixed == nil || delta.OnlyInDedicated == nil || delta.OverlapCount == nil || delta.OnlyInMixedCount == nil || delta.OnlyInDedicatedCount == nil {
		return errors.New("exact non_atomic_delta requires available identity sets and counts")
	}
	if err := validateIdentityList("non_atomic_delta overlap", delta.Overlap); err != nil {
		return err
	}
	if err := validateIdentityList("non_atomic_delta only_in_mixed", delta.OnlyInMixed); err != nil {
		return err
	}
	if err := validateIdentityList("non_atomic_delta only_in_dedicated", delta.OnlyInDedicated); err != nil {
		return err
	}

	mixedByID := make(map[int64]CrossEndpointIdentity, len(issues.ClassifiedPullIdentities))
	for _, identity := range issues.ClassifiedPullIdentities {
		mixedByID[identity.ID] = identity
	}
	dedicated := pullRecordIdentities(pullRecords)
	dedicatedByID := make(map[int64]CrossEndpointIdentity, len(dedicated))
	for _, identity := range dedicated {
		dedicatedByID[identity.ID] = identity
	}
	wantOverlap := make([]CrossEndpointIdentity, 0, min(len(mixedByID), len(dedicatedByID)))
	wantOnlyMixed := make([]CrossEndpointIdentity, 0)
	wantOnlyDedicated := make([]CrossEndpointIdentity, 0)
	for _, identity := range issues.ClassifiedPullIdentities {
		other, exists := dedicatedByID[identity.ID]
		if !exists {
			wantOnlyMixed = append(wantOnlyMixed, identity)
			continue
		}
		if identity.Number != other.Number || (identity.NodeID != "" && other.NodeID != "" && identity.NodeID != other.NodeID) {
			return fmt.Errorf("pull identity %d contradicts across issues and pulls endpoints", identity.ID)
		}
		wantOverlap = append(wantOverlap, identity)
	}
	for _, identity := range dedicated {
		if _, exists := mixedByID[identity.ID]; !exists {
			wantOnlyDedicated = append(wantOnlyDedicated, identity)
		}
	}
	if !sameIdentityList(delta.Overlap, wantOverlap) || !sameIdentityList(delta.OnlyInMixed, wantOnlyMixed) || !sameIdentityList(delta.OnlyInDedicated, wantOnlyDedicated) {
		return errors.New("non_atomic_delta identity sets contradict endpoint evidence")
	}
	total := len(delta.OnlyInMixed) + len(delta.OnlyInDedicated)
	validBounds := delta.SymmetricDifferenceLowerBound == total && delta.SymmetricDifferenceUpperBound == total
	if legacyShape && delta.SymmetricDifferenceLowerBound == 0 && delta.SymmetricDifferenceUpperBound == 0 {
		validBounds = true
	}
	if delta.MixedCount != len(issues.ClassifiedPullIdentities) || delta.DedicatedCount != len(dedicated) || *delta.OverlapCount != len(delta.Overlap) || *delta.OnlyInMixedCount != len(delta.OnlyInMixed) || *delta.OnlyInDedicatedCount != len(delta.OnlyInDedicated) || !validBounds {
		return errors.New("non_atomic_delta counts contradict identity sets")
	}
	wantCardinalityDelta := absNonNegativeDifference(delta.MixedCount, delta.DedicatedCount)
	if delta.ObservedEndpointCardinalityDelta != nil && *delta.ObservedEndpointCardinalityDelta != wantCardinalityDelta {
		return errors.New("non_atomic_delta observed endpoint-cardinality delta contradicts counts")
	}
	accepted := len(delta.OnlyInMixed) <= delta.Policy.MaxOnlyInMixed && len(delta.OnlyInDedicated) <= delta.Policy.MaxOnlyInDedicated && total <= delta.Policy.MaxTotal
	wantVerdict := NonAtomicDeltaVerdictRejected
	if accepted {
		wantVerdict = NonAtomicDeltaVerdictAccepted
	}
	validVerdict := delta.Verdict == wantVerdict
	if legacyShape && delta.Verdict == "" {
		validVerdict = true
	}
	if delta.Accepted != accepted || !validVerdict {
		return errors.New("non_atomic_delta accepted verdict contradicts its policy")
	}
	if !accepted && (requireComplete || r.Status == StatusComplete) {
		return fmt.Errorf("non_atomic_delta policy overflow: only_in_mixed=%d only_in_dedicated=%d total=%d", len(delta.OnlyInMixed), len(delta.OnlyInDedicated), total)
	}
	return nil
}

func validateLegacyCountOnlyNonAtomicDelta(r Receipt, issues SourceReceipt, pullRecords []Record, delta NonAtomicDeltaEvidence, requireComplete bool) error {
	lower, err := validateLegacyCountOnlyNonAtomicDeltaEvidence(issues, pullRecords, delta)
	if err != nil {
		return err
	}
	if delta.ObservedEndpointCardinalityDelta == nil {
		return errors.New("legacy count-only non_atomic_delta observed endpoint-cardinality delta is missing")
	}
	if *delta.ObservedEndpointCardinalityDelta != lower {
		return errors.New("legacy count-only non_atomic_delta observed endpoint-cardinality delta contradicts counts")
	}
	wantVerdict := NonAtomicDeltaVerdictRejected
	if lower <= delta.Policy.MaxTotal {
		wantVerdict = NonAtomicDeltaVerdictAccepted
	}
	if delta.Verdict != wantVerdict || delta.Accepted != (wantVerdict == NonAtomicDeltaVerdictAccepted) {
		return errors.New("legacy count-only non_atomic_delta verdict contradicts its observed endpoint-cardinality delta and policy")
	}
	if wantVerdict != NonAtomicDeltaVerdictAccepted && (requireComplete || r.Status == StatusComplete) {
		return fmt.Errorf("legacy count-only non_atomic_delta is rejected: endpoint_cardinality_delta=%d limit=%d", lower, delta.Policy.MaxTotal)
	}
	return nil
}

// validateLegacyPreMetricCountOnlyNonAtomicDelta admits only the checkpoint
// shape written between the count-only migration and its typed metric. It is
// called exclusively by resume validation; ordinary reads remain strict.
func validateLegacyPreMetricCountOnlyNonAtomicDelta(r Receipt, issues SourceReceipt, pullRecords []Record, delta NonAtomicDeltaEvidence) error {
	if r.Status != StatusPartial {
		return errors.New("legacy pre-metric count-only non_atomic_delta requires a partial receipt")
	}
	if delta.EvidenceMode != NonAtomicDeltaEvidenceModeLegacyCountOnly || delta.Policy.Metric != "" {
		return errors.New("legacy pre-metric count-only non_atomic_delta has a non-legacy metric")
	}
	// The first live count-only migration predated the explicit
	// identity_sets_available=false field. Admit that omission only through
	// this resume-only pre-metric path; strict current receipt validation still
	// requires the field.
	if delta.IdentitySetsAvailable == nil {
		delta.IdentitySetsAvailable = boolPointer(false)
	}
	lower, err := validateLegacyCountOnlyNonAtomicDeltaEvidence(issues, pullRecords, delta)
	if err != nil {
		return err
	}
	if delta.Policy.Type != NonAtomicDeltaPolicyType ||
		delta.Policy.MaxOnlyInMixed != DefaultNonAtomicDeltaLimit ||
		delta.Policy.MaxOnlyInDedicated != DefaultNonAtomicDeltaLimit ||
		delta.Policy.MaxTotal != DefaultNonAtomicDeltaLimit {
		return errors.New("legacy pre-metric count-only non_atomic_delta policy does not match the historical bound")
	}
	if delta.ObservedEndpointCardinalityDelta != nil {
		return errors.New("legacy pre-metric count-only non_atomic_delta already has an observed endpoint-cardinality delta")
	}
	if delta.Verdict != NonAtomicDeltaVerdictCompatibleUnproven || delta.Accepted {
		return errors.New("legacy pre-metric count-only non_atomic_delta verdict is contradictory")
	}
	minimumOnlyMixed := max(delta.MixedCount-delta.DedicatedCount, 0)
	minimumOnlyDedicated := max(delta.DedicatedCount-delta.MixedCount, 0)
	upper := delta.MixedCount + delta.DedicatedCount
	if lower > delta.Policy.MaxTotal || minimumOnlyMixed > delta.Policy.MaxOnlyInMixed || minimumOnlyDedicated > delta.Policy.MaxOnlyInDedicated ||
		(upper <= delta.Policy.MaxTotal && delta.MixedCount <= delta.Policy.MaxOnlyInMixed && delta.DedicatedCount <= delta.Policy.MaxOnlyInDedicated) {
		return errors.New("legacy pre-metric count-only non_atomic_delta compatible_unproven verdict contradicts its historical policy")
	}
	return nil
}

func validateLegacyCountOnlyNonAtomicDeltaEvidence(issues SourceReceipt, pullRecords []Record, delta NonAtomicDeltaEvidence) (int, error) {
	if delta.IdentitySetsAvailable == nil || *delta.IdentitySetsAvailable || delta.IdentityBasis != identityBasisLegacyCountOnly || delta.EvidenceReason != legacyCountOnlyReason ||
		delta.MixedEvidence != CrossEndpointEvidenceExactCountOnly ||
		delta.DedicatedEvidence != CrossEndpointEvidenceExactIdentities ||
		delta.RelationEvidence != CrossEndpointEvidenceUnavailable {
		return 0, errors.New("legacy count-only non_atomic_delta has contradictory provenance")
	}
	if issues.ClassifiedPullIdentities != nil || issues.ClassifiedPullChecksum != "" {
		return 0, errors.New("legacy count-only non_atomic_delta contradicts unavailable mixed identities")
	}
	if delta.Overlap != nil || delta.OnlyInMixed != nil || delta.OnlyInDedicated != nil || delta.OverlapCount != nil || delta.OnlyInMixedCount != nil || delta.OnlyInDedicatedCount != nil {
		return 0, errors.New("legacy count-only non_atomic_delta must keep relation identity sets and counts unavailable")
	}
	dedicated := pullRecordIdentities(pullRecords)
	if err := validateIdentityList("legacy count-only dedicated pulls", dedicated); err != nil {
		return 0, err
	}
	if delta.MixedCount != issues.ClassifiedPullCount || delta.DedicatedCount != len(dedicated) {
		return 0, errors.New("legacy count-only non_atomic_delta counts contradict endpoint evidence")
	}
	if delta.MixedCount < 0 || delta.DedicatedCount < 0 {
		return 0, errors.New("legacy count-only non_atomic_delta endpoint counts must be non-negative")
	}
	lower := absNonNegativeDifference(delta.MixedCount, delta.DedicatedCount)
	if delta.MixedCount > int(^uint(0)>>1)-delta.DedicatedCount {
		return 0, errors.New("legacy count-only non_atomic_delta possible symmetric-difference upper bound overflows")
	}
	upper := delta.MixedCount + delta.DedicatedCount
	if delta.SymmetricDifferenceLowerBound != lower || delta.SymmetricDifferenceUpperBound != upper {
		return 0, errors.New("legacy count-only non_atomic_delta symmetric-difference bounds contradict counts")
	}
	return lower, nil
}

func validateNonAtomicDeltaPolicy(policy NonAtomicDeltaPolicy, metric NonAtomicDeltaPolicyMetric, allowLegacyMissingMetric bool) error {
	if policy.Type != NonAtomicDeltaPolicyType || policy.MaxOnlyInMixed < 0 || policy.MaxOnlyInDedicated < 0 || policy.MaxTotal < 0 || policy.MaxOnlyInMixed > DefaultNonAtomicDeltaLimit || policy.MaxOnlyInDedicated > DefaultNonAtomicDeltaLimit || policy.MaxTotal > DefaultNonAtomicDeltaLimit {
		return errors.New("non_atomic_delta acceptance policy is missing or exceeds the repository bound")
	}
	if policy.Metric != metric && !(allowLegacyMissingMetric && policy.Metric == "") {
		return fmt.Errorf("non_atomic_delta policy metric %q does not match evidence mode", policy.Metric)
	}
	return nil
}

func validateExactNonAtomicDeltaPolicy(policy NonAtomicDeltaPolicy, mixedCount, dedicatedCount int) error {
	if policy.Type != NonAtomicDeltaPolicyType {
		return errors.New("non_atomic_delta acceptance policy is missing")
	}
	if policy.Metric != NonAtomicDeltaPolicyMetricExactIdentitySymmetricDifference {
		return fmt.Errorf("non_atomic_delta policy metric %q does not match evidence mode", policy.Metric)
	}
	want, err := exactNonAtomicDeltaPolicy(mixedCount, dedicatedCount)
	if err != nil {
		return err
	}
	if policy != want {
		return errors.New("exact non_atomic_delta policy does not match complete endpoint identity counts")
	}
	return nil
}

func isLegacyMixedPullCheckpoint(r Receipt, issues SourceReceipt) bool {
	if r.Status != StatusPartial || issues.Name != "issues" {
		return false
	}
	if issues.Status != StatusComplete || issues.ClassifiedPullCount <= 0 || issues.ClassifiedPullIdentities != nil || issues.ClassifiedPullChecksum != "" || r.NonAtomicDelta != nil {
		return false
	}
	if issues.FetchedCount != issues.NormalizedCount+issues.ClassifiedPullCount+issues.CutoffExcludedCount {
		return false
	}
	for _, source := range r.Sources {
		if source.Name == "pulls" {
			return source.Status == StatusPartial || source.Status == StatusComplete
		}
	}
	return false
}

func validateIdentityList(name string, identities []CrossEndpointIdentity) error {
	seen := make(map[int64]bool, len(identities))
	for i, identity := range identities {
		if identity.ID <= 0 {
			return fmt.Errorf("%s contains invalid id %d", name, identity.ID)
		}
		if seen[identity.ID] {
			return fmt.Errorf("%s contains duplicate identity %d", name, identity.ID)
		}
		seen[identity.ID] = true
		if i > 0 {
			previous := identities[i-1]
			if previous.ID > identity.ID || (previous.ID == identity.ID && (previous.Number > identity.Number || (previous.Number == identity.Number && previous.NodeID > identity.NodeID))) {
				return fmt.Errorf("%s is not in canonical order", name)
			}
		}
	}
	return nil
}

func sameIdentityList(a, b []CrossEndpointIdentity) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func validDigest(value string) bool {
	const prefix = "sha256:"
	if !strings.HasPrefix(value, prefix) || len(value) != len(prefix)+64 {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, prefix))
	return err == nil
}

func mustParseCutoff(v string) time.Time { t, _ := time.Parse(time.RFC3339Nano, v); return t }

func validateResume(c Corpus, repo string, cutoff time.Time, baseURL string, expectedEndpoints map[string]string, allowLegacy bool) error {
	validator := validateCheckpoint
	if allowLegacy {
		validator = validateResumeCheckpoint
	}
	if err := validator(c); err != nil {
		return fmt.Errorf("invalid resume: %w", err)
	}
	if c.Receipt.Repository != repo {
		return errors.New("resume repository mismatch")
	}
	t, e := time.Parse(time.RFC3339Nano, c.Receipt.Cutoff)
	if e != nil || !t.Equal(cutoff) {
		return errors.New("resume cutoff mismatch")
	}
	if c.Receipt.Revision == "" {
		return errors.New("resume revision is required")
	}
	baseURL = strings.TrimRight(baseURL, "/")
	if c.Receipt.APIBase != "" && strings.TrimRight(c.Receipt.APIBase, "/") != baseURL {
		return errors.New("resume API base mismatch")
	}
	for _, source := range c.Receipt.Sources {
		if source.Endpoint != expectedEndpoints[source.Name] {
			return fmt.Errorf("resume %s endpoint does not match API base", source.Name)
		}
		for _, page := range source.Pages {
			if !sameAPIBase(page.URL, baseURL) || (page.Next != "" && !sameAPIBase(page.Next, baseURL)) {
				return fmt.Errorf("resume %s page chain leaves API base", source.Name)
			}
		}
	}
	for _, api := range c.Receipt.API {
		if !sameAPIBase(api.URL, baseURL) {
			return errors.New("resume API receipt base mismatch")
		}
	}
	return nil
}

func sameAPIBase(rawURL, baseURL string) bool {
	u, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	base, err := url.Parse(baseURL)
	if err != nil {
		return false
	}
	return u.Scheme == base.Scheme && u.Host == base.Host && (base.Path == "" || strings.HasPrefix(u.Path, strings.TrimRight(base.Path, "/")+"/"))
}
