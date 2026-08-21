package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

// ObservationSnapshotSchemaV1 is the compatibility contract for the aggregate
// gateway read. Each source remains a guardvars.ObservationEnvelope so source
// evolution is bounded independently of the outer snapshot.
const ObservationSnapshotSchemaV1 = "fak-observation-snapshot/1"

const (
	observationReasonSessionNotConfigured = "SESSION_SOURCE_NOT_CONFIGURED"
	observationReasonSessionReadFailed    = "SESSION_SOURCE_READ_FAILED"
	observationReasonHarnessNotConfigured = "HARNESS_SOURCE_NOT_CONFIGURED"
	observationReasonHarnessReadFailed    = "HARNESS_SOURCE_READ_FAILED"
	observationReasonHarnessUnavailable   = "HARNESS_SOURCE_UNAVAILABLE"
	observationReasonHarnessStale         = "HARNESS_SAMPLE_STALE"
	observationReasonHarnessNotApplicable = "HARNESS_NOT_APPLICABLE"
	observationReasonHarnessInvalidState  = "HARNESS_SOURCE_INVALID_STATE"
	observationReasonManagedCacheDisabled = "MANAGED_CACHE_DISABLED"
	observationReasonEncodingFailed       = "OBSERVATION_ENCODING_FAILED"
)

const (
	observationProvenanceSessions     = "gateway.session_registry"
	observationProvenanceCache        = "gateway.cache_accounting"
	observationProvenanceManagedCache = "gateway.managed_cache"
	observationProvenanceHarness      = "host.harness_sampler"
)

// ObservationSnapshot is one point-in-time, payload-free gateway view. ObservedAt
// is the request boundary shared by every source captured at that instant; a
// source envelope may carry an older source-specific timestamp when it is stale.
type ObservationSnapshot struct {
	Schema     string                     `json:"schema"`
	ObservedAt string                     `json:"observed_at"`
	Sources    ObservationSnapshotSources `json:"sources"`
}

// ObservationSnapshotSources is the stable set of source families required by
// the first observation contract.
type ObservationSnapshotSources struct {
	Sessions         guardvars.ObservationEnvelope `json:"sessions"`
	CacheAttribution guardvars.ObservationEnvelope `json:"cache_attribution"`
	ManagedCache     guardvars.ObservationEnvelope `json:"managed_cache"`
	Harness          guardvars.ObservationEnvelope `json:"harness"`
}

// ObservationSessionData is the payload-free live-session projection shared
// with /debug/vars, plus an explicit count so a clean zero is data rather than
// an omitted or null collection.
type ObservationSessionData struct {
	Sessions []guardvars.SessionVars `json:"sessions"`
	Count    int                     `json:"count"`
}

type observationSnapshotBuild struct {
	Snapshot         ObservationSnapshot
	Sessions         []debugSessionVars
	CacheAttribution *debugCacheAttributionVars
	ManagedCache     *debugManagedCacheVars
	Harness          *SessionHarness
}

func (s *Server) handleFakObservation(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	writeJSON(w, http.StatusOK, s.observationSnapshotContext(r.Context(), time.Now()))
}

func (s *Server) observationSnapshotContext(ctx context.Context, now time.Time) ObservationSnapshot {
	m := s.metrics
	if m == nil {
		m = newGatewayMetrics(now)
	}
	var vdsoHits int64
	if s.k != nil {
		vdsoHits = s.k.Counters().VDSOHits
	}
	return s.buildObservationSnapshot(ctx, now, m.adjudicationSummary(), vdsoHits, m.servedInlineSnapshot()).Snapshot
}

func (s *Server) buildObservationSnapshot(
	ctx context.Context,
	now time.Time,
	sum AdjudicationSummary,
	vdsoHits int64,
	servedInline uint64,
) observationSnapshotBuild {
	boundary := now.UTC().Format(time.RFC3339Nano)

	sessions, sessionAvailability, sessionReason := s.captureObservationSessions(ctx, now)
	sessionData := ObservationSessionData{
		Sessions: append([]guardvars.SessionVars(nil), sessions...),
		Count:    len(sessions),
	}
	if sessionData.Sessions == nil {
		sessionData.Sessions = []guardvars.SessionVars{}
	}

	cache := cacheAttributionVars(sum, vdsoHits, servedInline)
	cacheData := guardvars.CacheAttributionVars{}
	if cache != nil {
		cacheData = *cache
	}

	managed := managedCacheVars(s.cacheTTL1H, s.provider, sum)
	managedAvailability := guardvars.AvailabilityObserved
	managedReason := ""
	var managedData any
	if managed == nil {
		managedAvailability = guardvars.AvailabilityNotApplicable
		managedReason = observationReasonManagedCacheDisabled
	} else {
		managedData = *managed
	}

	harness, harnessEnvelope := s.captureObservationHarness(now, boundary)

	return observationSnapshotBuild{
		Snapshot: ObservationSnapshot{
			Schema:     ObservationSnapshotSchemaV1,
			ObservedAt: boundary,
			Sources: ObservationSnapshotSources{
				Sessions: newObservationEnvelope(
					"sessions",
					observationProvenanceSessions,
					boundary,
					"",
					sessionAvailability,
					sessionData,
					sessionReason,
				),
				CacheAttribution: newObservationEnvelope(
					"cache_attribution",
					observationProvenanceCache,
					boundary,
					"",
					guardvars.AvailabilityObserved,
					cacheData,
					"",
				),
				ManagedCache: newObservationEnvelope(
					"managed_cache",
					observationProvenanceManagedCache,
					boundary,
					"",
					managedAvailability,
					managedData,
					managedReason,
				),
				Harness: harnessEnvelope,
			},
		},
		Sessions:         sessions,
		CacheAttribution: cache,
		ManagedCache:     managed,
		Harness:          harness,
	}
}

func (s *Server) captureObservationSessions(
	ctx context.Context,
	now time.Time,
) (sessions []debugSessionVars, availability guardvars.Availability, reason string) {
	if s == nil || s.listSessions == nil {
		return nil, guardvars.AvailabilityUnavailable, observationReasonSessionNotConfigured
	}
	availability = guardvars.AvailabilityObserved
	defer func() {
		if recover() != nil {
			sessions = nil
			availability = guardvars.AvailabilityUnavailable
			reason = observationReasonSessionReadFailed
		}
	}()
	sessions = s.debugSessions(ctx, now)
	if sessions == nil {
		sessions = []debugSessionVars{}
	}
	return sessions, availability, ""
}

func (s *Server) captureObservationHarness(now time.Time, boundary string) (harness *SessionHarness, envelope guardvars.ObservationEnvelope) {
	if s == nil {
		return nil, newObservationEnvelope(
			"harness", observationProvenanceHarness, boundary, "",
			guardvars.AvailabilityUnavailable, nil, observationReasonHarnessNotConfigured,
		)
	}

	var observation SessionHarnessObservation
	var configured bool
	readFailed := false
	func() {
		defer func() {
			if recover() != nil {
				readFailed = true
			}
		}()
		observation, configured = s.sessionHarnessObservation()
	}()
	if readFailed {
		return nil, newObservationEnvelope(
			"harness", observationProvenanceHarness, boundary, "",
			guardvars.AvailabilityUnavailable, nil, observationReasonHarnessReadFailed,
		)
	}
	if !configured {
		return nil, newObservationEnvelope(
			"harness", observationProvenanceHarness, boundary, "",
			guardvars.AvailabilityUnavailable, nil, observationReasonHarnessNotConfigured,
		)
	}

	availability := observation.Availability
	if availability == "" {
		availability = guardvars.AvailabilityObserved
		if observation.Snapshot.Samples <= 0 {
			availability = guardvars.AvailabilityEmpty
		}
	}
	observedAt := boundary
	if !observation.ObservedAt.IsZero() {
		observedAt = observation.ObservedAt.UTC().Format(time.RFC3339Nano)
	}
	revision := strings.TrimSpace(observation.Revision)

	switch availability {
	case guardvars.AvailabilityObserved:
		if observation.Snapshot.Samples <= 0 {
			return nil, newObservationEnvelope(
				"harness", observationProvenanceHarness, observedAt, revision,
				guardvars.AvailabilityEmpty, nil, "",
			)
		}
		snapshot := observation.Snapshot
		return &snapshot, newObservationEnvelope(
			"harness", observationProvenanceHarness, observedAt, revision,
			guardvars.AvailabilityObserved, snapshot, "",
		)
	case guardvars.AvailabilityEmpty:
		return nil, newObservationEnvelope(
			"harness", observationProvenanceHarness, observedAt, revision,
			guardvars.AvailabilityEmpty, nil, "",
		)
	case guardvars.AvailabilityUnavailable:
		return nil, newObservationEnvelope(
			"harness", observationProvenanceHarness, observedAt, revision,
			guardvars.AvailabilityUnavailable, nil, observationReasonHarnessUnavailable,
		)
	case guardvars.AvailabilityStale:
		return nil, newObservationEnvelope(
			"harness", observationProvenanceHarness, observedAt, revision,
			guardvars.AvailabilityStale, nil, observationReasonHarnessStale,
		)
	case guardvars.AvailabilityNotApplicable:
		return nil, newObservationEnvelope(
			"harness", observationProvenanceHarness, observedAt, revision,
			guardvars.AvailabilityNotApplicable, nil, observationReasonHarnessNotApplicable,
		)
	default:
		return nil, newObservationEnvelope(
			"harness", observationProvenanceHarness, observedAt, revision,
			guardvars.AvailabilityUnavailable, nil, observationReasonHarnessInvalidState,
		)
	}
}

func newObservationEnvelope(
	source string,
	provenance string,
	observedAt string,
	revision string,
	availability guardvars.Availability,
	data any,
	reason string,
) guardvars.ObservationEnvelope {
	envelope := guardvars.ObservationEnvelope{
		Schema:       guardvars.ObservationSchemaV1,
		Source:       source,
		ObservedAt:   observedAt,
		Revision:     revision,
		Provenance:   provenance,
		Availability: availability,
		Reason:       reason,
	}
	if availability == guardvars.AvailabilityObserved {
		raw, err := json.Marshal(data)
		if err != nil {
			envelope.Availability = guardvars.AvailabilityUnavailable
			envelope.Reason = observationReasonEncodingFailed
			return envelope
		}
		envelope.Data = raw
	}
	return envelope
}
