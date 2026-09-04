package studydrift

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

// RefreshReceiptSchema defines the schema identifier for studydrift refresh receipt payloads.
const RefreshReceiptSchema = "fak-study-refresh-receipt/1"

// RefreshStatus classifies the outcome of validating and refreshing a pinned source against new evidence.
type RefreshStatus string

const (
	// RefreshUnchanged indicates the observed source content and metadata match the pinned source.
	RefreshUnchanged RefreshStatus = "unchanged"
	// RefreshMoved indicates content digest is identical but repository location or revision changed.
	RefreshMoved RefreshStatus = "moved"
	// RefreshChanged indicates content digest modified, requiring superseded observation and decision tracking.
	RefreshChanged RefreshStatus = "changed"
	// RefreshUnavailable indicates observation was explicitly marked unavailable upstream.
	RefreshUnavailable RefreshStatus = "unavailable"
	// RefreshUnverifiable indicates content digest mismatch, corrupted timestamps, or missing tracking IDs.
	RefreshUnverifiable RefreshStatus = "unverifiable"
)

// PinnedSource is the immutable source material and decisions a study used.
type PinnedSource struct {
	Repository    string `json:"repository"`
	URL           string `json:"url"`
	Revision      string `json:"revision"`
	Bytes         []byte `json:"bytes"`
	Digest        string `json:"digest"`
	ObservationID string `json:"observation_id"`
	DecisionID    string `json:"decision_id"`
}

// SourceObservation is caller-provided evidence. RefreshSource performs no network access.
type SourceObservation struct {
	ObservedAt    string `json:"observed_at"`
	URL           string `json:"url,omitempty"`
	Revision      string `json:"revision,omitempty"`
	Bytes         []byte `json:"bytes,omitempty"`
	Digest        string `json:"digest,omitempty"`
	ObservationID string `json:"observation_id,omitempty"`
	DecisionID    string `json:"decision_id,omitempty"`
	Unavailable   bool   `json:"unavailable,omitempty"`
}

// Supersession records prior observation and decision IDs replaced by an updated source.
type Supersession struct {
	Observation string `json:"observation"`
	Decision    string `json:"decision"`
}

// RefreshReceipt preserves both the prior pin and the new evidence. Canonical is
// nil for failure states, so an unavailable or unverifiable observation cannot
// be mistaken for an update to the caller's canonical record.
type RefreshReceipt struct {
	Schema      string            `json:"schema"`
	Status      RefreshStatus     `json:"status"`
	ObservedAt  string            `json:"observed_at"`
	Prior       PinnedSource      `json:"prior"`
	Observation SourceObservation `json:"observation"`
	Canonical   *PinnedSource     `json:"canonical,omitempty"`
	Supersedes  *Supersession     `json:"supersedes,omitempty"`
}

// DigestSource computes a deterministic sha256 hex digest prefixed with "sha256:" for byte slices.
func DigestSource(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// RefreshSource evaluates an observed source against an immutable pinned source pin.
// Contract:
// - Invariant: inputs are cloned deeply so caller slices and pinned sources are never mutated in place.
// - Fail-closed guard: missing timestamps, digest mismatches, or missing supersede IDs immediately yield RefreshUnverifiable with nil Canonical.
func RefreshSource(pin PinnedSource, observation SourceObservation) RefreshReceipt {
	pin = clonePinnedSource(pin)
	observation = cloneSourceObservation(observation)
	receipt := RefreshReceipt{Schema: RefreshReceiptSchema, ObservedAt: observation.ObservedAt, Prior: pin, Observation: observation}
	if _, err := time.Parse(time.RFC3339, observation.ObservedAt); err != nil || observation.ObservedAt == "" {
		receipt.Status = RefreshUnverifiable
		return receipt
	}
	if observation.Unavailable {
		receipt.Status = RefreshUnavailable
		return receipt
	}
	actual := DigestSource(observation.Bytes)
	if observation.Digest == "" || !strings.EqualFold(observation.Digest, actual) || pin.Digest == "" || !strings.EqualFold(pin.Digest, DigestSource(pin.Bytes)) {
		receipt.Status = RefreshUnverifiable
		return receipt
	}
	next := PinnedSource{Repository: pin.Repository, URL: observation.URL, Revision: observation.Revision, Bytes: append([]byte(nil), observation.Bytes...), Digest: actual, ObservationID: observation.ObservationID, DecisionID: observation.DecisionID}
	switch {
	case !strings.EqualFold(actual, pin.Digest):
		if observation.ObservationID == "" || observation.DecisionID == "" {
			receipt.Status = RefreshUnverifiable
			return receipt
		}
		receipt.Status = RefreshChanged
		receipt.Supersedes = &Supersession{Observation: pin.ObservationID, Decision: pin.DecisionID}
	case observation.URL != pin.URL || observation.Revision != pin.Revision:
		receipt.Status = RefreshMoved
	default:
		receipt.Status = RefreshUnchanged
	}
	receipt.Canonical = &next
	return receipt
}

func clonePinnedSource(source PinnedSource) PinnedSource {
	source.Bytes = append([]byte(nil), source.Bytes...)
	return source
}

func cloneSourceObservation(observation SourceObservation) SourceObservation {
	observation.Bytes = append([]byte(nil), observation.Bytes...)
	return observation
}

// SelectDue returns at most limit active repositories needing refresh, ordered
// by monitored-repository priority and then oldest check. A non-positive limit
// deliberately selects nothing.
func SelectDue(registry studymonitor.Registry, now time.Time, dueDays, limit int) []studymonitor.Repository {
	if limit <= 0 {
		return nil
	}
	cutoff := now.AddDate(0, 0, -dueDays)
	var due []studymonitor.Repository
	for _, repository := range registry.Repositories {
		if repository.Status != "active" {
			continue
		}
		checked, err := time.Parse("2006-01-02", repository.LastChecked)
		if err != nil || !checked.After(cutoff) {
			due = append(due, repository)
		}
	}
	sort.Slice(due, func(i, j int) bool {
		if due[i].Priority != due[j].Priority {
			return due[i].Priority < due[j].Priority
		}
		if due[i].LastChecked != due[j].LastChecked {
			return due[i].LastChecked < due[j].LastChecked
		}
		return due[i].Repository < due[j].Repository
	})
	if len(due) > limit {
		due = due[:limit]
	}
	return due
}
