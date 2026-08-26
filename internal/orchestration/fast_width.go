package orchestration

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"
)

const WidthEvidenceSchema = "fak-orchestration-width-evidence/1"

type WidthVerdict string

const (
	WidthGain    WidthVerdict = "GAIN"
	WidthNoGain  WidthVerdict = "NO_GAIN"
	WidthAbstain WidthVerdict = "ABSTAIN"
)

type WidthEvidenceKey struct {
	TaskClass           WorkClass `json:"task_class"`
	AccessMix           string    `json:"access_mix"`
	ModelProvider       string    `json:"model_provider"`
	CachePosture        string    `json:"cache_posture"`
	HarnessCapabilities []string  `json:"harness_capabilities"`
	BenchmarkRevision   string    `json:"benchmark_revision"`
}

type WidthEvidenceCell struct {
	Width                int          `json:"width"`
	Verdict              WidthVerdict `json:"verdict"`
	AcceptedOutcomeEqual bool         `json:"accepted_outcome_equal"`
	TelemetryComplete    bool         `json:"telemetry_complete"`
	CriticalPathMillis   int64        `json:"critical_path_millis"`
	TotalWorkerMillis    int64        `json:"total_worker_millis"`
	TotalTokens          int64        `json:"total_tokens"`
	CacheCostTokens      int64        `json:"cache_cost_tokens"`
	LeaseWaitMillis      int64        `json:"lease_wait_millis"`
	ReconcileMillis      int64        `json:"reconcile_millis"`
}

type WidthEvidence struct {
	Schema        string              `json:"schema"`
	Digest        string              `json:"digest"`
	GeneratedAt   string              `json:"generated_at"`
	MaxAgeSeconds int64               `json:"max_age_seconds"`
	Key           WidthEvidenceKey    `json:"key"`
	Cells         []WidthEvidenceCell `json:"cells"`
}

type WidthRequest struct {
	ObjectiveMillis int64            `json:"objective_millis"`
	AsOf            string           `json:"as_of"`
	Key             WidthEvidenceKey `json:"key"`
	Evidence        *WidthEvidence   `json:"evidence,omitempty"`
}

type WidthHold string

const (
	WidthHoldMissing        WidthHold = "missing_evidence"
	WidthHoldStale          WidthHold = "stale_evidence"
	WidthHoldIncomparable   WidthHold = "incomparable_evidence"
	WidthHoldUnequalOutcome WidthHold = "unequal_outcome"
	WidthHoldIncomplete     WidthHold = "incomplete_telemetry"
	WidthHoldNonGain        WidthHold = "non_gain"
	WidthHoldNoQualifying   WidthHold = "no_qualifying_width"
)

type WidthConsideration struct {
	Width     int          `json:"width"`
	Verdict   WidthVerdict `json:"verdict,omitempty"`
	Qualifies bool         `json:"qualifies"`
	Reason    string       `json:"reason"`
}

type WidthSelection struct {
	Adaptive           bool                 `json:"adaptive"`
	Considered         []WidthConsideration `json:"considered,omitempty"`
	Selected           int                  `json:"selected"`
	Realized           int                  `json:"realized"`
	EvidenceDigest     string               `json:"evidence_digest,omitempty"`
	EvidenceAgeSeconds int64                `json:"evidence_age_seconds,omitempty"`
	Reason             string               `json:"reason"`
	Hold               WidthHold            `json:"hold,omitempty"`
	Cap                int                  `json:"cap"`
}

func WidthEvidenceDigest(e WidthEvidence) (string, error) {
	e.Digest = ""
	raw, err := json.Marshal(e)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func validateWidthEvidence(e WidthEvidence) error {
	if e.Schema != WidthEvidenceSchema {
		return fmt.Errorf("width evidence schema must be %q", WidthEvidenceSchema)
	}
	if e.Digest == "" {
		return errors.New("width evidence digest is required")
	}
	got, err := WidthEvidenceDigest(e)
	if err != nil {
		return err
	}
	if got != e.Digest {
		return fmt.Errorf("width evidence digest mismatch: got %s want %s", e.Digest, got)
	}
	if e.Key.BenchmarkRevision == "" || e.Key.AccessMix == "" || e.Key.ModelProvider == "" || e.Key.CachePosture == "" {
		return errors.New("width evidence comparison key is incomplete")
	}
	if e.MaxAgeSeconds <= 0 {
		return errors.New("width evidence max_age_seconds must be positive")
	}
	if _, err := time.Parse(time.RFC3339, e.GeneratedAt); err != nil {
		return fmt.Errorf("width evidence generated_at: %w", err)
	}
	seen := map[int]bool{}
	for _, c := range e.Cells {
		if c.Width < 1 || seen[c.Width] {
			return errors.New("width evidence widths must be positive and unique")
		}
		seen[c.Width] = true
	}
	return nil
}

func equalWidthKey(a, b WidthEvidenceKey) bool {
	aa, bb := append([]string(nil), a.HarnessCapabilities...), append([]string(nil), b.HarnessCapabilities...)
	sort.Strings(aa)
	sort.Strings(bb)
	a.HarnessCapabilities, b.HarnessCapabilities = aa, bb
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y)
}

func SelectFastWidth(req *WidthRequest, cap int, tokenCap int64) WidthSelection {
	out := WidthSelection{Adaptive: true, Selected: 1, Cap: cap, Reason: "conservative width one", Hold: WidthHoldMissing}
	if cap < 1 {
		cap = 1
		out.Cap = cap
	}
	if req == nil || req.Evidence == nil {
		return out
	}
	e := *req.Evidence
	out.EvidenceDigest = e.Digest
	if err := validateWidthEvidence(e); err != nil {
		out.Hold = WidthHoldIncomparable
		out.Reason = err.Error()
		return out
	}
	if !equalWidthKey(req.Key, e.Key) {
		out.Hold = WidthHoldIncomparable
		out.Reason = "evidence key does not match current task/access/capability envelope"
		return out
	}
	asOf, err := time.Parse(time.RFC3339, req.AsOf)
	if err != nil {
		out.Hold = WidthHoldIncomparable
		out.Reason = "invalid selection as_of"
		return out
	}
	generated, _ := time.Parse(time.RFC3339, e.GeneratedAt)
	age := asOf.Sub(generated)
	if age < 0 {
		out.Hold = WidthHoldIncomparable
		out.Reason = "evidence is from the future"
		return out
	}
	out.EvidenceAgeSeconds = int64(age / time.Second)
	if out.EvidenceAgeSeconds > e.MaxAgeSeconds {
		out.Hold = WidthHoldStale
		out.Reason = "evidence exceeds max age"
		return out
	}
	cells := append([]WidthEvidenceCell(nil), e.Cells...)
	sort.Slice(cells, func(i, j int) bool { return cells[i].Width < cells[j].Width })
	lastHold := WidthHoldNoQualifying
	selected := 0
	for _, c := range cells {
		item := WidthConsideration{Width: c.Width, Verdict: c.Verdict}
		switch {
		case c.Width > cap:
			item.Reason = "rejected by worker cap"
		case c.TotalTokens > tokenCap:
			item.Reason = "rejected by token cap"
		case !c.TelemetryComplete:
			item.Reason = "incomplete telemetry"
			lastHold = WidthHoldIncomplete
		case !c.AcceptedOutcomeEqual:
			item.Reason = "accepted outcome is unequal"
			lastHold = WidthHoldUnequalOutcome
		case c.Verdict != WidthGain:
			item.Reason = "net-value verdict is not GAIN"
			lastHold = WidthHoldNonGain
		case req.ObjectiveMillis <= 0 || c.CriticalPathMillis > req.ObjectiveMillis:
			item.Reason = "latency objective not met"
			lastHold = WidthHoldNoQualifying
		case selected != 0:
			item.Reason = "larger than smallest qualifying width"
		default:
			item.Qualifies = true
			item.Reason = "equal accepted outcome, positive net value, and latency objective met"
			selected = c.Width
		}
		out.Considered = append(out.Considered, item)
	}
	if selected != 0 {
		out.Selected, out.Hold, out.Reason = selected, "", "smallest qualifying witnessed width"
		return out
	}
	out.Hold = lastHold
	out.Reason = "no qualifying witnessed width; selected conservative width one"
	return out
}
