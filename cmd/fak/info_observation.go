package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

const guardInfoObservationViewSchema = "fak-info-observation-view/1"

const (
	guardInfoObservationProbe uint8 = iota
	guardInfoObservationTyped
	guardInfoObservationLegacy
)

const (
	guardInfoObservationTransportTyped  = "versioned_snapshot"
	guardInfoObservationTransportLegacy = "legacy_debug_vars"
)

// guardInfoObservationMetric is the renderer-ready meaning of one gateway
// observation. Value remains present for a measured zero and absent for every
// non-observed state, so no renderer has to reinterpret raw omission.
type guardInfoObservationMetric struct {
	Availability guardvars.Availability `json:"availability"`
	Value        *float64               `json:"value,omitempty"`
	Unit         string                 `json:"unit"`
	ObservedAt   string                 `json:"observed_at,omitempty"`
	Revision     string                 `json:"revision,omitempty"`
	Provenance   string                 `json:"provenance"`
	Reason       string                 `json:"reason,omitempty"`
	NextCheck    string                 `json:"next_check,omitempty"`
}

// guardInfoObservationView is the one semantic model consumed by compact,
// visual, tabbed, and JSON output. Raw legacy fields remain in guardInfoVars for
// compatibility, but source state and recovery copy come only from this view.
type guardInfoObservationView struct {
	Schema           string                     `json:"schema"`
	Transport        string                     `json:"transport"`
	SnapshotSchema   string                     `json:"snapshot_schema,omitempty"`
	FallbackReason   string                     `json:"fallback_reason,omitempty"`
	Sessions         guardInfoObservationMetric `json:"sessions"`
	CacheAttribution guardInfoObservationMetric `json:"cache_attribution"`
	ManagedCache     guardInfoObservationMetric `json:"managed_cache"`
	Harness          guardInfoObservationMetric `json:"harness"`
}

// readGuardInfoVars prefers the versioned observation snapshot for the four
// overlapping source families and uses /debug/vars for the rest of the info
// surface. An older gateway gets one compatibility probe, then stays on the
// explicit legacy/unknown model rather than paying a permanent 404 per tick.
func readGuardInfoVars(c *claudeMacDebugClient) (guardInfoVars, error) {
	legacy, present, legacyErr := readLegacyGuardInfoVars(c)

	var snapshot gateway.ObservationSnapshot
	var snapshotErr error
	if c.infoObservationMode != guardInfoObservationLegacy {
		snapshotErr = c.get("/v1/fak/observation", &snapshot)
		if snapshotErr == nil && snapshot.Schema != gateway.ObservationSnapshotSchemaV1 {
			snapshotErr = fmt.Errorf("unsupported observation snapshot schema %q", snapshot.Schema)
			c.infoObservationMode = guardInfoObservationLegacy
		}
		if snapshotErr == nil {
			c.infoObservationMode = guardInfoObservationTyped
			v := legacy
			applyGuardInfoObservationSnapshot(&v, snapshot)
			return v, nil
		}
		if guardInfoObservationCompatibilityMiss(snapshotErr) {
			c.infoObservationMode = guardInfoObservationLegacy
		}
	}

	if legacyErr == nil {
		fallback := "VERSIONED_SNAPSHOT_UNAVAILABLE"
		if c.infoObservationMode == guardInfoObservationLegacy {
			fallback = "VERSIONED_SNAPSHOT_UNSUPPORTED"
		}
		applyLegacyGuardInfoObservation(&legacy, present, fallback)
		return legacy, nil
	}
	if snapshotErr != nil {
		return guardInfoVars{}, fmt.Errorf("%v; typed observation fallback: %v", legacyErr, snapshotErr)
	}
	return guardInfoVars{}, legacyErr
}

func readLegacyGuardInfoVars(c *claudeMacDebugClient) (guardInfoVars, map[string]json.RawMessage, error) {
	raw, err := c.getRaw("/debug/vars")
	if err != nil {
		return guardInfoVars{}, nil, err
	}
	var v guardInfoVars
	if err := json.Unmarshal([]byte(raw), &v); err != nil {
		return guardInfoVars{}, nil, fmt.Errorf("decode /debug/vars: %w", err)
	}
	present := map[string]json.RawMessage{}
	if err := json.Unmarshal([]byte(raw), &present); err != nil {
		return guardInfoVars{}, nil, fmt.Errorf("decode /debug/vars field presence: %w", err)
	}
	return v, present, nil
}

func guardInfoObservationCompatibilityMiss(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "status 404") || strings.Contains(s, "status 405") ||
		strings.Contains(s, "unsupported observation snapshot schema")
}

func applyGuardInfoObservationSnapshot(v *guardInfoVars, snapshot gateway.ObservationSnapshot) {
	view := &guardInfoObservationView{
		Schema:         guardInfoObservationViewSchema,
		Transport:      guardInfoObservationTransportTyped,
		SnapshotSchema: snapshot.Schema,
	}

	view.Sessions = guardInfoMetricFromEnvelope(snapshot.Sources.Sessions, "live sessions", "sessions")
	if view.Sessions.Availability == guardvars.AvailabilityObserved {
		var data gateway.ObservationSessionData
		if err := json.Unmarshal(snapshot.Sources.Sessions.Data, &data); err != nil || data.Count < 0 || data.Count != len(data.Sessions) {
			view.Sessions = guardInfoInvalidTypedMetric(view.Sessions, "live sessions", "sessions")
			v.Sessions = nil
		} else {
			v.Sessions = append([]guardInfoSession(nil), data.Sessions...)
			view.Sessions.Value = guardInfoObservationNumber(float64(data.Count))
		}
	} else {
		v.Sessions = nil
	}

	view.CacheAttribution = guardInfoMetricFromEnvelope(snapshot.Sources.CacheAttribution, "token-equivalent", "cache")
	if view.CacheAttribution.Availability == guardvars.AvailabilityObserved {
		var data guardvars.CacheAttributionVars
		if err := json.Unmarshal(snapshot.Sources.CacheAttribution.Data, &data); err != nil {
			view.CacheAttribution = guardInfoInvalidTypedMetric(view.CacheAttribution, "token-equivalent", "cache")
			v.CacheAttribution = nil
		} else {
			v.CacheAttribution = &data
			view.CacheAttribution.Value = guardInfoObservationNumber(data.TotalTokenEquiv)
		}
	} else {
		v.CacheAttribution = nil
	}

	view.ManagedCache = guardInfoMetricFromEnvelope(snapshot.Sources.ManagedCache, "upgraded heads", "managed_cache")
	if view.ManagedCache.Availability == guardvars.AvailabilityObserved {
		var data guardvars.ManagedCacheVars
		if err := json.Unmarshal(snapshot.Sources.ManagedCache.Data, &data); err != nil {
			view.ManagedCache = guardInfoInvalidTypedMetric(view.ManagedCache, "upgraded heads", "managed_cache")
			v.ManagedCache = nil
		} else {
			v.ManagedCache = &data
			view.ManagedCache.Value = guardInfoObservationNumber(float64(data.Upgraded))
		}
	} else {
		v.ManagedCache = nil
	}

	view.Harness = guardInfoMetricFromEnvelope(snapshot.Sources.Harness, "samples", "harness")
	if view.Harness.Availability == guardvars.AvailabilityObserved {
		var data gateway.SessionHarness
		if err := json.Unmarshal(snapshot.Sources.Harness.Data, &data); err != nil {
			view.Harness = guardInfoInvalidTypedMetric(view.Harness, "samples", "harness")
			v.Harness = nil
		} else {
			v.Harness = &data
			view.Harness.Value = guardInfoObservationNumber(float64(data.Samples))
		}
	} else {
		v.Harness = nil
	}

	v.Observation = view
}

func guardInfoMetricFromEnvelope(envelope guardvars.ObservationEnvelope, unit, source string) guardInfoObservationMetric {
	metric := guardInfoObservationMetric{
		Availability: envelope.Availability,
		Unit:         unit,
		ObservedAt:   envelope.ObservedAt,
		Revision:     envelope.Revision,
		Provenance:   envelope.Provenance,
		Reason:       envelope.Reason,
	}
	if err := envelope.Validate(); err != nil {
		return guardInfoInvalidTypedMetric(metric, unit, source)
	}
	metric.NextCheck = guardInfoObservationNextCheck(source, metric.Availability)
	return metric
}

func guardInfoInvalidTypedMetric(metric guardInfoObservationMetric, unit, source string) guardInfoObservationMetric {
	provenance := strings.TrimSpace(metric.Provenance)
	if provenance == "" {
		provenance = "typed.unknown"
	}
	return guardInfoObservationMetric{
		Availability: guardvars.AvailabilityUnavailable,
		Unit:         unit,
		ObservedAt:   metric.ObservedAt,
		Revision:     metric.Revision,
		Provenance:   provenance,
		Reason:       "INVALID_TYPED_ENVELOPE",
		NextCheck:    guardInfoObservationNextCheck(source, guardvars.AvailabilityUnavailable),
	}
}

func applyLegacyGuardInfoObservation(v *guardInfoVars, present map[string]json.RawMessage, fallback string) {
	const provenance = "legacy.debug_vars/unknown"
	view := &guardInfoObservationView{
		Schema:         guardInfoObservationViewSchema,
		Transport:      guardInfoObservationTransportLegacy,
		FallbackReason: fallback,
	}

	if raw, ok := present["sessions"]; ok && string(bytes.TrimSpace(raw)) != "null" {
		if len(v.Sessions) == 0 {
			view.Sessions = guardInfoLegacyMetric(guardvars.AvailabilityEmpty, nil, "live sessions", provenance, "", "sessions")
		} else {
			view.Sessions = guardInfoLegacyMetric(guardvars.AvailabilityObserved, guardInfoObservationNumber(float64(len(v.Sessions))), "live sessions", provenance, "", "sessions")
		}
	} else {
		view.Sessions = guardInfoLegacyMetric(guardvars.AvailabilityUnavailable, nil, "live sessions", provenance, "LEGACY_STATE_UNKNOWN", "sessions")
	}

	if _, ok := present["cache_attribution"]; ok && v.CacheAttribution != nil {
		view.CacheAttribution = guardInfoLegacyMetric(guardvars.AvailabilityObserved, guardInfoObservationNumber(v.CacheAttribution.TotalTokenEquiv), "token-equivalent", provenance, "", "cache")
	} else {
		view.CacheAttribution = guardInfoLegacyMetric(guardvars.AvailabilityUnavailable, nil, "token-equivalent", provenance, "LEGACY_STATE_UNKNOWN", "cache")
	}

	if _, ok := present["managed_cache"]; ok && v.ManagedCache != nil {
		view.ManagedCache = guardInfoLegacyMetric(guardvars.AvailabilityObserved, guardInfoObservationNumber(float64(v.ManagedCache.Upgraded)), "upgraded heads", provenance, "", "managed_cache")
	} else {
		view.ManagedCache = guardInfoLegacyMetric(guardvars.AvailabilityUnavailable, nil, "upgraded heads", provenance, "LEGACY_STATE_UNKNOWN", "managed_cache")
	}

	if _, ok := present["harness"]; ok && v.Harness != nil {
		view.Harness = guardInfoLegacyMetric(guardvars.AvailabilityObserved, guardInfoObservationNumber(float64(v.Harness.Samples)), "samples", provenance, "", "harness")
	} else {
		view.Harness = guardInfoLegacyMetric(guardvars.AvailabilityUnavailable, nil, "samples", provenance, "LEGACY_STATE_UNKNOWN", "harness")
	}
	v.Observation = view
}

func guardInfoLegacyMetric(availability guardvars.Availability, value *float64, unit, provenance, reason, source string) guardInfoObservationMetric {
	return guardInfoObservationMetric{
		Availability: availability,
		Value:        value,
		Unit:         unit,
		Provenance:   provenance,
		Reason:       reason,
		NextCheck:    guardInfoObservationNextCheck(source, availability),
	}
}

func guardInfoObservationNumber(value float64) *float64 { return &value }

func guardInfoObservationNextCheck(source string, availability guardvars.Availability) string {
	switch availability {
	case guardvars.AvailabilityEmpty:
		if source == "sessions" {
			return "start or attach a managed session"
		}
		return "wait for the first source sample"
	case guardvars.AvailabilityUnavailable:
		return "check /v1/fak/observation.sources." + source + " and gateway logs"
	case guardvars.AvailabilityStale:
		return "refresh the gateway " + source + " source"
	case guardvars.AvailabilityNotApplicable:
		return "check the gateway wire/capability boundary"
	default:
		return ""
	}
}

func guardInfoObservationSummary(view *guardInfoObservationView) string {
	if view == nil {
		return ""
	}
	transport := "typed"
	if view.Transport == guardInfoObservationTransportLegacy {
		transport = "legacy/unknown"
	}
	return transport + " · " +
		guardInfoObservationMetricText("sessions", view.Sessions) + " · " +
		guardInfoObservationMetricText("cache", view.CacheAttribution)
}

func guardInfoObservationRows(view *guardInfoObservationView) []string {
	if view == nil {
		return nil
	}
	transport := "typed"
	if view.Transport == guardInfoObservationTransportLegacy {
		transport = "legacy/unknown"
	}
	return []string{
		" source   " + transport,
		" " + guardInfoObservationMetricText("sessions", view.Sessions),
		" " + guardInfoObservationMetricText("cache", view.CacheAttribution),
	}
}

// guardInfoCacheSourceWord returns the semantic cache posture when the typed
// source view, rather than provider-cache detail, must lead the renderer.
func guardInfoCacheSourceWord(v guardInfoVars) string {
	if v.Observation == nil {
		return ""
	}
	if v.Observation.Transport == guardInfoObservationTransportLegacy && v.VCache != nil {
		return ""
	}
	metric := v.Observation.CacheAttribution
	if metric.Availability == guardvars.AvailabilityObserved {
		if v.VCache != nil {
			return ""
		}
		if metric.Value == nil {
			return "source unavailable (invalid observed value)"
		}
		if *metric.Value == 0 {
			return "cold (observed zero)"
		}
		return fmt.Sprintf("observed %s %s", strconv.FormatFloat(*metric.Value, 'f', -1, 64), metric.Unit)
	}
	reason := strings.TrimSpace(metric.Reason)
	if reason == "" {
		reason = string(metric.Availability)
	}
	state := strings.ToLower(strings.ReplaceAll(string(metric.Availability), "_", " "))
	word := "source " + state + " (" + reason
	if metric.NextCheck != "" {
		word += "; next: " + metric.NextCheck
	}
	return word + ")"
}

func guardInfoCacheSourceObserved(v guardInfoVars) bool {
	return v.Observation == nil ||
		(v.Observation.Transport == guardInfoObservationTransportLegacy && v.VCache != nil) ||
		v.Observation.CacheAttribution.Availability == guardvars.AvailabilityObserved
}

func guardInfoObservationMetricText(name string, metric guardInfoObservationMetric) string {
	provenance := strings.TrimSpace(metric.Provenance)
	if provenance == "" {
		provenance = "unknown"
	}
	freshness := guardInfoObservationFreshnessText(metric)
	if metric.Availability == guardvars.AvailabilityObserved && metric.Value != nil {
		value := strconv.FormatFloat(*metric.Value, 'f', -1, 64)
		cold := ""
		if *metric.Value == 0 {
			cold = " (cold)"
		}
		return fmt.Sprintf("%s %s %s OBSERVED%s%s [%s]", name, value, metric.Unit, cold, freshness, provenance)
	}
	reason := strings.TrimSpace(metric.Reason)
	if reason == "" {
		reason = string(metric.Availability)
	}
	text := fmt.Sprintf("%s %s reason=%s", name, metric.Availability, reason)
	if metric.NextCheck != "" {
		text += " next=" + metric.NextCheck
	}
	return text + freshness + " [" + provenance + "]"
}

func guardInfoObservationFreshnessText(metric guardInfoObservationMetric) string {
	parts := make([]string, 0, 2)
	if observedAt := strings.TrimSpace(metric.ObservedAt); observedAt != "" {
		parts = append(parts, "at="+observedAt)
	}
	if revision := strings.TrimSpace(metric.Revision); revision != "" {
		parts = append(parts, "rev="+revision)
	}
	if len(parts) == 0 {
		return ""
	}
	return " " + strings.Join(parts, " ")
}
