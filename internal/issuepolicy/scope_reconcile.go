package issuepolicy

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

const ScopeReconcileSchema = "fak/issue-scope-reconcile/v1"

const (
	ScopeAligned    = "aligned"
	ScopeGap        = "gap"
	ScopeStale      = "stale"
	ScopeExpanded   = "expanded"
	ScopeContracted = "contracted"
	ScopeUnknown    = "unknown"
)

type ScopeSnapshot struct {
	Key               string          `json:"key"`
	PreviousTarget    []EnvelopeValue `json:"previous_target,omitempty"`
	Target            []EnvelopeValue `json:"target"`
	Witnessed         []EnvelopeValue `json:"witnessed"`
	Observed          []EnvelopeValue `json:"observed,omitempty"`
	WitnessedAt       time.Time       `json:"witnessed_at"`
	MaxAge            string          `json:"max_age"`
	ContractionReason string          `json:"contraction_reason,omitempty"`
}

type ScopeChange struct {
	Dimension string         `json:"dimension"`
	Kind      string         `json:"kind"`
	Previous  *EnvelopeValue `json:"previous,omitempty"`
	Current   *EnvelopeValue `json:"current,omitempty"`
	Detail    string         `json:"detail"`
}

type ScopeReconciliation struct {
	Schema                  string        `json:"schema"`
	Key                     string        `json:"key,omitempty"`
	Status                  string        `json:"status"`
	ProductionCreditCurrent bool          `json:"production_credit_current"`
	Action                  string        `json:"action"`
	WitnessedAt             time.Time     `json:"witnessed_at,omitempty"`
	FreshUntil              time.Time     `json:"fresh_until,omitempty"`
	Changes                 []ScopeChange `json:"changes,omitempty"`
	Gaps                    []EnvelopeGap `json:"gaps,omitempty"`
	Unknown                 []string      `json:"unknown,omitempty"`
}

// ReconcileScope compares contractual target, direct witness, observed demand,
// and the prior target. It never shrinks a target from observations and never
// guesses unit conversion. The output is stable JSON suitable for a periodic
// status loop or an on-demand operator check.
func ReconcileScope(snapshot ScopeSnapshot, now time.Time) ScopeReconciliation {
	out := ScopeReconciliation{Schema: ScopeReconcileSchema, Key: snapshot.Key, Status: ScopeUnknown, Action: "declare a target, witness time, and freshness policy", WitnessedAt: snapshot.WitnessedAt}
	if now.IsZero() {
		now = time.Now().UTC()
	}
	if len(snapshot.Target) == 0 || snapshot.WitnessedAt.IsZero() {
		out.Unknown = append(out.Unknown, "target and witnessed_at are required")
		return out
	}
	maxAge, err := time.ParseDuration(strings.TrimSpace(snapshot.MaxAge))
	if err != nil || maxAge <= 0 {
		out.Unknown = append(out.Unknown, "max_age must be a positive Go duration such as 24h or 168h")
		return out
	}
	out.FreshUntil = snapshot.WitnessedAt.Add(maxAge)
	out.Changes, out.Unknown = scopeChanges(snapshot.PreviousTarget, snapshot.Target, snapshot.Observed)
	if len(out.Unknown) > 0 {
		out.Action = "resolve incompatible or unknown dimensions/units; do not guess a conversion"
		return out
	}
	var expanded, contracted bool
	for _, change := range out.Changes {
		expanded = expanded || change.Kind == ScopeExpanded
		contracted = contracted || change.Kind == ScopeContracted
	}
	if expanded {
		out.Status = ScopeExpanded
		out.Gaps = compareEnvelopes(snapshot.Target, snapshot.Witnessed)
		out.Action = "re-test the expanded target envelope and refresh production evidence"
		return out
	}
	if contracted {
		out.Status = ScopeContracted
		if strings.TrimSpace(snapshot.ContractionReason) == "" {
			out.Action = "record an operator-approved contraction reason and denominator audit; target is not auto-shrunk"
		} else {
			out.Action = "audit the recorded contraction and rebaseline the parent denominator before restoring credit"
		}
		return out
	}
	if now.After(out.FreshUntil) {
		out.Status = ScopeStale
		out.Action = "re-run the declared evidence stages; the prior witness is retained but no longer current"
		return out
	}
	out.Gaps = compareEnvelopes(snapshot.Target, snapshot.Witnessed)
	if len(out.Gaps) > 0 {
		out.Status = ScopeGap
		out.Action = "re-test the named gap dimensions at the declared target; keep production credit withheld"
		return out
	}
	out.Status = ScopeAligned
	out.ProductionCreditCurrent = true
	out.Action = "no action; reconcile again before the freshness deadline or when target/observed scope changes"
	return out
}

func scopeChanges(previous, current, observed []EnvelopeValue) ([]ScopeChange, []string) {
	prev := envelopeMap(previous)
	cur := envelopeMap(current)
	obs := envelopeMap(observed)
	var changes []ScopeChange
	var unknown []string
	for dim, c := range cur {
		if p, ok := prev[dim]; ok {
			kind, detail, bad := compareTargetChange(p, c)
			if bad != "" {
				unknown = append(unknown, bad)
			} else if kind != "" {
				p2, c2 := p, c
				changes = append(changes, ScopeChange{Dimension: dim, Kind: kind, Previous: &p2, Current: &c2, Detail: detail})
			}
		} else if len(previous) > 0 {
			c2 := c
			changes = append(changes, ScopeChange{Dimension: dim, Kind: ScopeExpanded, Current: &c2, Detail: "new target dimension"})
		}
		if o, ok := obs[dim]; ok && !c.NotApplicable && !o.NotApplicable {
			if o.Unit != c.Unit {
				unknown = append(unknown, fmt.Sprintf("%s: target unit %s, observed unit %s", dim, c.Unit, o.Unit))
				continue
			}
			if c.Operator == ">=" && o.Value > c.Value {
				c2, o2 := c, o
				changes = append(changes, ScopeChange{Dimension: dim, Kind: ScopeExpanded, Previous: &c2, Current: &o2, Detail: "observed demand exceeds declared target"})
			}
			if c.Operator == "<=" && o.Value > c.Value {
				c2, o2 := c, o
				changes = append(changes, ScopeChange{Dimension: dim, Kind: ScopeGap, Previous: &c2, Current: &o2, Detail: "observed value violates declared maximum"})
			}
		}
	}
	for dim, p := range prev {
		if _, ok := cur[dim]; !ok {
			p2 := p
			changes = append(changes, ScopeChange{Dimension: dim, Kind: ScopeContracted, Previous: &p2, Detail: "target dimension removed"})
		}
	}
	sort.Slice(changes, func(i, j int) bool {
		if changes[i].Dimension == changes[j].Dimension {
			return changes[i].Kind < changes[j].Kind
		}
		return changes[i].Dimension < changes[j].Dimension
	})
	sort.Strings(unknown)
	return changes, unknown
}

func envelopeMap(values []EnvelopeValue) map[string]EnvelopeValue {
	out := map[string]EnvelopeValue{}
	for _, v := range values {
		out[normalizeEnvelopeToken(v.Dimension)] = v
	}
	return out
}

func compareTargetChange(previous, current EnvelopeValue) (kind, detail, unknown string) {
	if previous.NotApplicable != current.NotApplicable {
		return "", "", fmt.Sprintf("%s: applicability changed and requires operator review", current.Dimension)
	}
	if current.NotApplicable {
		return "", "", ""
	}
	if previous.Unit != current.Unit || previous.Operator != current.Operator {
		return "", "", fmt.Sprintf("%s: target operator/unit changed (%s %s -> %s %s)", current.Dimension, previous.Operator, previous.Unit, current.Operator, current.Unit)
	}
	if previous.Value == current.Value {
		return "", "", ""
	}
	expanded := (current.Operator == ">=" && current.Value > previous.Value) || (current.Operator == "<=" && current.Value < previous.Value)
	if expanded {
		return ScopeExpanded, fmt.Sprintf("target changed from %g to %g %s", previous.Value, current.Value, current.Unit), ""
	}
	return ScopeContracted, fmt.Sprintf("target changed from %g to %g %s", previous.Value, current.Value, current.Unit), ""
}
