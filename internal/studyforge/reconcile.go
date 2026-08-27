package studyforge

import (
	"encoding/json"
	"fmt"
	"sort"
)

const (
	identityBasisCaptured         = "captured_endpoint_rows"
	identityBasisLegacyProjection = "legacy_checkpoint_projection"
	identityBasisLegacyCountOnly  = "legacy_checkpoint_counts"
	legacyCountOnlyReason         = "legacy_mixed_pull_identities_not_retained"
)

func intPointer(value int) *int { return &value }

func boolPointer(value bool) *bool { return &value }

func identityFromRecord(record Record) CrossEndpointIdentity {
	return CrossEndpointIdentity{ID: record.ID, Number: record.Number, NodeID: record.NodeID}
}

func identityDigest(identities []CrossEndpointIdentity) string {
	b, _ := json.Marshal(identities)
	return digest(b)
}

func sortIdentities(identities []CrossEndpointIdentity) {
	sort.Slice(identities, func(i, j int) bool {
		if identities[i].ID != identities[j].ID {
			return identities[i].ID < identities[j].ID
		}
		if identities[i].Number != identities[j].Number {
			return identities[i].Number < identities[j].Number
		}
		return identities[i].NodeID < identities[j].NodeID
	})
}

func pullRecordIdentities(records []Record) []CrossEndpointIdentity {
	identities := make([]CrossEndpointIdentity, 0, len(records))
	for _, record := range records {
		identities = append(identities, identityFromRecord(record))
	}
	sortIdentities(identities)
	return identities
}

func backfillCrawlWindow(source *SourceReceipt) {
	if len(source.Pages) == 0 {
		return
	}
	if source.CrawlStartedAt == "" {
		source.CrawlStartedAt = source.Pages[0].FetchedAt
	}
	if source.Status == StatusComplete && source.CrawlEndedAt == "" {
		source.CrawlEndedAt = source.Pages[len(source.Pages)-1].FetchedAt
	}
}

func defaultNonAtomicDeltaPolicy(metric NonAtomicDeltaPolicyMetric) NonAtomicDeltaPolicy {
	return NonAtomicDeltaPolicy{
		Type:               NonAtomicDeltaPolicyType,
		Metric:             metric,
		MaxOnlyInMixed:     DefaultNonAtomicDeltaLimit,
		MaxOnlyInDedicated: DefaultNonAtomicDeltaLimit,
		MaxTotal:           DefaultNonAtomicDeltaLimit,
	}
}

// reconcileNonAtomicDelta derives the set relation once both endpoint
// traversals are terminal. New captures use exact identities. A typed legacy
// count-only marker stays count-only forever; dedicated identities must never
// be projected into the missing mixed endpoint set.
func reconcileNonAtomicDelta(corpus *Corpus) error {
	issuesIndex, pullsIndex := -1, -1
	for i := range corpus.Receipt.Sources {
		backfillCrawlWindow(&corpus.Receipt.Sources[i])
		switch corpus.Receipt.Sources[i].Name {
		case "issues":
			issuesIndex = i
		case "pulls":
			pullsIndex = i
		}
	}
	if issuesIndex < 0 || pullsIndex < 0 {
		return nil
	}
	issues := &corpus.Receipt.Sources[issuesIndex]
	pulls := &corpus.Receipt.Sources[pullsIndex]
	if issues.Status != StatusComplete || pulls.Status != StatusComplete {
		corpus.Receipt.NonAtomicDelta = nil
		return nil
	}

	if corpus.Receipt.NonAtomicDelta != nil && corpus.Receipt.NonAtomicDelta.EvidenceMode == NonAtomicDeltaEvidenceModeLegacyCountOnly {
		if err := reconcileLegacyCountOnlyDelta(corpus, *issues, *pulls); err != nil {
			return err
		}
		if !corpus.Receipt.NonAtomicDelta.Accepted {
			return fmt.Errorf("legacy count-only non_atomic_delta rejected: endpoint_cardinality_delta=%d limit=%d", *corpus.Receipt.NonAtomicDelta.ObservedEndpointCardinalityDelta, corpus.Receipt.NonAtomicDelta.Policy.MaxTotal)
		}
		return nil
	}
	basis := identityBasisCaptured
	if corpus.Receipt.NonAtomicDelta != nil && corpus.Receipt.NonAtomicDelta.IdentityBasis == identityBasisLegacyProjection {
		basis = identityBasisLegacyProjection
	}
	mixed := append([]CrossEndpointIdentity(nil), issues.ClassifiedPullIdentities...)
	dedicated := pullRecordIdentities(recordsForSource(corpus.Records, "pulls"))
	sortIdentities(mixed)
	sortIdentities(dedicated)

	mixedByID := make(map[int64]CrossEndpointIdentity, len(mixed))
	for _, identity := range mixed {
		if _, exists := mixedByID[identity.ID]; exists {
			return fmt.Errorf("issues duplicate classified pull identity %d", identity.ID)
		}
		mixedByID[identity.ID] = identity
	}
	dedicatedByID := make(map[int64]CrossEndpointIdentity, len(dedicated))
	for _, identity := range dedicated {
		dedicatedByID[identity.ID] = identity
	}

	overlap := make([]CrossEndpointIdentity, 0, min(len(mixed), len(dedicated)))
	onlyMixed := make([]CrossEndpointIdentity, 0)
	onlyDedicated := make([]CrossEndpointIdentity, 0)
	for _, identity := range mixed {
		other, exists := dedicatedByID[identity.ID]
		if !exists {
			onlyMixed = append(onlyMixed, identity)
			continue
		}
		if identity.Number != other.Number || (identity.NodeID != "" && other.NodeID != "" && identity.NodeID != other.NodeID) {
			return fmt.Errorf("pull identity %d contradicts across issues and pulls endpoints", identity.ID)
		}
		overlap = append(overlap, identity)
	}
	for _, identity := range dedicated {
		if _, exists := mixedByID[identity.ID]; !exists {
			onlyDedicated = append(onlyDedicated, identity)
		}
	}
	// Exact receipts retain their established bounded_identity_delta policy
	// shape. The typed metric is required only for the legacy count-only mode.
	policy := defaultNonAtomicDeltaPolicy("")
	accepted := len(onlyMixed) <= policy.MaxOnlyInMixed && len(onlyDedicated) <= policy.MaxOnlyInDedicated && len(onlyMixed)+len(onlyDedicated) <= policy.MaxTotal
	corpus.Receipt.NonAtomicDelta = &NonAtomicDeltaEvidence{
		Type:                          NonAtomicDeltaType,
		MixedSource:                   "issues",
		DedicatedSource:               "pulls",
		EvidenceMode:                  NonAtomicDeltaEvidenceModeExactIdentity,
		IdentityBasis:                 basis,
		MixedEvidence:                 CrossEndpointEvidenceExactIdentities,
		DedicatedEvidence:             CrossEndpointEvidenceExactIdentities,
		RelationEvidence:              CrossEndpointEvidenceExactIdentities,
		MixedCrawl:                    CrawlWindow{StartedAt: issues.CrawlStartedAt, EndedAt: issues.CrawlEndedAt},
		DedicatedCrawl:                CrawlWindow{StartedAt: pulls.CrawlStartedAt, EndedAt: pulls.CrawlEndedAt},
		MixedCount:                    len(mixed),
		DedicatedCount:                len(dedicated),
		OverlapCount:                  intPointer(len(overlap)),
		OnlyInMixedCount:              intPointer(len(onlyMixed)),
		OnlyInDedicatedCount:          intPointer(len(onlyDedicated)),
		Overlap:                       overlap,
		OnlyInMixed:                   onlyMixed,
		OnlyInDedicated:               onlyDedicated,
		SymmetricDifferenceLowerBound: len(onlyMixed) + len(onlyDedicated),
		SymmetricDifferenceUpperBound: len(onlyMixed) + len(onlyDedicated),
		Policy:                        policy,
		Verdict:                       map[bool]string{true: NonAtomicDeltaVerdictAccepted, false: NonAtomicDeltaVerdictRejected}[accepted],
		Accepted:                      accepted,
	}
	if !accepted {
		return fmt.Errorf("non_atomic_delta exceeds policy: only_in_mixed=%d only_in_dedicated=%d total=%d limit=%d", len(onlyMixed), len(onlyDedicated), len(onlyMixed)+len(onlyDedicated), DefaultNonAtomicDeltaLimit)
	}
	return nil
}

func reconcileLegacyCountOnlyDelta(corpus *Corpus, issues, pulls SourceReceipt) error {
	mixedCount := issues.ClassifiedPullCount
	dedicatedCount := len(recordsForSource(corpus.Records, "pulls"))
	if mixedCount < 0 || dedicatedCount < 0 {
		return fmt.Errorf("legacy count-only non_atomic_delta endpoint counts must be non-negative: mixed=%d dedicated=%d", mixedCount, dedicatedCount)
	}
	lower := absNonNegativeDifference(mixedCount, dedicatedCount)
	if mixedCount > int(^uint(0)>>1)-dedicatedCount {
		return fmt.Errorf("legacy count-only non_atomic_delta possible symmetric-difference upper bound overflows: mixed=%d dedicated=%d", mixedCount, dedicatedCount)
	}
	upper := mixedCount + dedicatedCount
	policy := defaultNonAtomicDeltaPolicy(NonAtomicDeltaPolicyMetricEndpointCardinalityDelta)
	accepted := lower <= policy.MaxTotal
	verdict := map[bool]string{true: NonAtomicDeltaVerdictAccepted, false: NonAtomicDeltaVerdictRejected}[accepted]
	corpus.Receipt.NonAtomicDelta = &NonAtomicDeltaEvidence{
		Type:                             NonAtomicDeltaType,
		MixedSource:                      "issues",
		DedicatedSource:                  "pulls",
		EvidenceMode:                     NonAtomicDeltaEvidenceModeLegacyCountOnly,
		EvidenceReason:                   legacyCountOnlyReason,
		IdentitySetsAvailable:            boolPointer(false),
		IdentityBasis:                    identityBasisLegacyCountOnly,
		MixedEvidence:                    CrossEndpointEvidenceExactCountOnly,
		DedicatedEvidence:                CrossEndpointEvidenceExactIdentities,
		RelationEvidence:                 CrossEndpointEvidenceUnavailable,
		MixedCrawl:                       CrawlWindow{StartedAt: issues.CrawlStartedAt, EndedAt: issues.CrawlEndedAt},
		DedicatedCrawl:                   CrawlWindow{StartedAt: pulls.CrawlStartedAt, EndedAt: pulls.CrawlEndedAt},
		MixedCount:                       mixedCount,
		DedicatedCount:                   dedicatedCount,
		ObservedEndpointCardinalityDelta: intPointer(lower),
		SymmetricDifferenceLowerBound:    lower,
		SymmetricDifferenceUpperBound:    upper,
		Policy:                           policy,
		Verdict:                          verdict,
		Accepted:                         accepted,
	}
	return nil
}

func absNonNegativeDifference(a, b int) int {
	if a >= b {
		return a - b
	}
	return b - a
}

// upgradeLegacyPreMetricCountOnlyEvidence finishes the one-way migration of
// the short-lived count-only checkpoint shape that predates a typed policy
// metric. Resume validation must prove that exact shape before this is called.
func upgradeLegacyPreMetricCountOnlyEvidence(corpus *Corpus) bool {
	delta := corpus.Receipt.NonAtomicDelta
	if delta == nil || delta.EvidenceMode != NonAtomicDeltaEvidenceModeLegacyCountOnly || delta.Policy.Metric != "" {
		return false
	}
	observed := absNonNegativeDifference(delta.MixedCount, delta.DedicatedCount)
	delta.Policy.Metric = NonAtomicDeltaPolicyMetricEndpointCardinalityDelta
	delta.ObservedEndpointCardinalityDelta = intPointer(observed)
	delta.Accepted = observed <= delta.Policy.MaxTotal
	delta.Verdict = map[bool]string{true: NonAtomicDeltaVerdictAccepted, false: NonAtomicDeltaVerdictRejected}[delta.Accepted]
	return true
}

// upgradeLegacyMixedPullEvidence performs the one-way pre-#9314 migration.
// The old issues receipt retained an exact classified count but no identities.
// Once the pulls census is terminal, this records that asymmetry explicitly;
// it never copies dedicated identities into the missing mixed set.
func upgradeLegacyMixedPullEvidence(corpus *Corpus) (bool, error) {
	issuesIndex, pullsIndex := -1, -1
	for i := range corpus.Receipt.Sources {
		switch corpus.Receipt.Sources[i].Name {
		case "issues":
			issuesIndex = i
		case "pulls":
			pullsIndex = i
		}
	}
	if issuesIndex < 0 || pullsIndex < 0 || !isLegacyMixedPullCheckpoint(corpus.Receipt, corpus.Receipt.Sources[issuesIndex]) {
		return false, nil
	}
	pulls := corpus.Receipt.Sources[pullsIndex]
	if pulls.Status != StatusComplete {
		return false, nil
	}

	pullRecords := recordsForSource(corpus.Records, "pulls")
	for _, record := range pullRecords {
		if record.ID <= 0 || record.Number <= 0 || record.NodeID == "" || record.Kind != "pull" || record.Source != "pulls" {
			return false, fmt.Errorf("legacy checkpoint pull %d lacks an exact dedicated identity", record.ID)
		}
	}

	upgraded := cloneCorpus(*corpus)
	for i := range upgraded.Receipt.Sources {
		backfillCrawlWindow(&upgraded.Receipt.Sources[i])
	}
	if err := reconcileLegacyCountOnlyDelta(&upgraded, upgraded.Receipt.Sources[issuesIndex], upgraded.Receipt.Sources[pullsIndex]); err != nil {
		return false, err
	}
	*corpus = upgraded
	return true, nil
}
