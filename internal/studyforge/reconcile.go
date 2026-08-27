package studyforge

import (
	"encoding/json"
	"fmt"
	"sort"
)

const (
	identityBasisCaptured         = "captured_endpoint_rows"
	identityBasisLegacyProjection = "legacy_checkpoint_projection"
)

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

func defaultNonAtomicDeltaPolicy() NonAtomicDeltaPolicy {
	return NonAtomicDeltaPolicy{
		Type:               NonAtomicDeltaPolicyType,
		MaxOnlyInMixed:     DefaultNonAtomicDeltaLimit,
		MaxOnlyInDedicated: DefaultNonAtomicDeltaLimit,
		MaxTotal:           DefaultNonAtomicDeltaLimit,
	}
}

// reconcileNonAtomicDelta derives the exact set relation once both endpoint
// traversals are terminal. Legacy projection is deliberately handled by the
// resume-only upgrade before this strict reconciler runs.
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
	policy := defaultNonAtomicDeltaPolicy()
	accepted := len(onlyMixed) <= policy.MaxOnlyInMixed && len(onlyDedicated) <= policy.MaxOnlyInDedicated && len(onlyMixed)+len(onlyDedicated) <= policy.MaxTotal
	corpus.Receipt.NonAtomicDelta = &NonAtomicDeltaEvidence{
		Type:                 NonAtomicDeltaType,
		MixedSource:          "issues",
		DedicatedSource:      "pulls",
		IdentityBasis:        basis,
		MixedCrawl:           CrawlWindow{StartedAt: issues.CrawlStartedAt, EndedAt: issues.CrawlEndedAt},
		DedicatedCrawl:       CrawlWindow{StartedAt: pulls.CrawlStartedAt, EndedAt: pulls.CrawlEndedAt},
		MixedCount:           len(mixed),
		DedicatedCount:       len(dedicated),
		OverlapCount:         len(overlap),
		OnlyInMixedCount:     len(onlyMixed),
		OnlyInDedicatedCount: len(onlyDedicated),
		Overlap:              overlap,
		OnlyInMixed:          onlyMixed,
		OnlyInDedicated:      onlyDedicated,
		Policy:               policy,
		Accepted:             accepted,
	}
	if !accepted {
		return fmt.Errorf("non_atomic_delta exceeds policy: only_in_mixed=%d only_in_dedicated=%d total=%d limit=%d", len(onlyMixed), len(onlyDedicated), len(onlyMixed)+len(onlyDedicated), DefaultNonAtomicDeltaLimit)
	}
	return nil
}

// upgradeLegacyMixedPullEvidence performs the one-way pre-#9314 migration.
// The old issues receipt retained the exact classified count while the pulls
// source retained typed PR identities. Equality is the only uniquely
// projectable case; a subset or superset would require guessing which IDs were
// observed by the mixed endpoint and therefore fails closed.
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
	issues := corpus.Receipt.Sources[issuesIndex]
	pulls := corpus.Receipt.Sources[pullsIndex]
	if pulls.Status != StatusComplete {
		return false, nil
	}

	pullRecords := recordsForSource(corpus.Records, "pulls")
	if len(pullRecords) != issues.ClassifiedPullCount {
		return false, fmt.Errorf("legacy checkpoint mixed pull evidence is ambiguous: classified_pull_count=%d dedicated_identity_count=%d", issues.ClassifiedPullCount, len(pullRecords))
	}
	identities := make([]CrossEndpointIdentity, 0, len(pullRecords))
	for _, record := range pullRecords {
		if record.ID <= 0 || record.Number <= 0 || record.NodeID == "" || record.Kind != "pull" || record.Source != "pulls" {
			return false, fmt.Errorf("legacy checkpoint pull %d lacks an exact dedicated identity", record.ID)
		}
		identities = append(identities, identityFromRecord(record))
	}
	sortIdentities(identities)
	if err := validateIdentityList("legacy checkpoint dedicated pulls", identities); err != nil {
		return false, err
	}

	upgraded := cloneCorpus(*corpus)
	for i := range upgraded.Receipt.Sources {
		backfillCrawlWindow(&upgraded.Receipt.Sources[i])
	}
	upgradedIssues := &upgraded.Receipt.Sources[issuesIndex]
	upgradedIssues.ClassifiedPullIdentities = identities
	upgradedIssues.ClassifiedPullChecksum = identityDigest(identities)
	upgraded.Receipt.NonAtomicDelta = &NonAtomicDeltaEvidence{IdentityBasis: identityBasisLegacyProjection}
	if err := reconcileNonAtomicDelta(&upgraded); err != nil {
		return false, fmt.Errorf("upgrade legacy mixed pull evidence: %w", err)
	}
	*corpus = upgraded
	return true, nil
}
