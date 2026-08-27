package studyforge

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"
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
// traversals are terminal. A legacy partial checkpoint may not contain the
// mixed identities introduced by #9314; in that one compatibility case the
// already-normalized dedicated rows provide a declared count-compatible
// projection. The identity_basis field keeps that migration evidence distinct
// from identities captured directly from endpoint rows.
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
	if len(mixed) == 0 && issues.ClassifiedPullCount > 0 {
		var err error
		mixed, err = projectLegacyMixedIdentities(*issues, dedicated, recordsForSource(corpus.Records, "pulls"))
		if err != nil {
			return err
		}
		basis = identityBasisLegacyProjection
		issues.ClassifiedPullIdentities = append([]CrossEndpointIdentity(nil), mixed...)
		issues.ClassifiedPullChecksum = identityDigest(mixed)
	}
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

func projectLegacyMixedIdentities(issues SourceReceipt, dedicated []CrossEndpointIdentity, pullRecords []Record) ([]CrossEndpointIdentity, error) {
	want := issues.ClassifiedPullCount
	if want == len(dedicated) {
		return append([]CrossEndpointIdentity(nil), dedicated...), nil
	}
	if want < 0 || want > len(dedicated) {
		return nil, errors.New("legacy checkpoint lacks enough mixed pull identity evidence")
	}
	endedAt, err := time.Parse(time.RFC3339Nano, issues.CrawlEndedAt)
	if err != nil {
		return nil, errors.New("legacy checkpoint mixed crawl end is required for identity projection")
	}
	projected := make([]CrossEndpointIdentity, 0, want)
	for _, record := range pullRecords {
		createdAt, parseErr := time.Parse(time.RFC3339, record.CreatedAt)
		if parseErr != nil || !createdAt.After(endedAt) {
			projected = append(projected, identityFromRecord(record))
		}
	}
	if len(projected) != want {
		return nil, fmt.Errorf("legacy checkpoint mixed identities are not uniquely projectable: count=%d temporal_candidates=%d", want, len(projected))
	}
	sortIdentities(projected)
	return projected, nil
}
