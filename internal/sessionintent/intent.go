// Package sessionintent defines provider-neutral session-level operator intent.
package sessionintent

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// Clock names the time that an effort bound measures.
type Clock string

const (
	ClockElapsed Clock = "elapsed"
	ClockActive  Clock = "active"
)

// BoundKind distinguishes eligibility, planning, and enforcement semantics.
type BoundKind string

const (
	BoundMinimum BoundKind = "minimum"
	BoundTarget  BoundKind = "target"
	BoundMaximum BoundKind = "maximum"
)

// EffortBound is one independently measured constraint on a session.
type EffortBound struct {
	Kind     BoundKind     `json:"kind"`
	Clock    Clock         `json:"clock"`
	Duration time.Duration `json:"-"`
}

// MarshalJSON renders durations in Go's stable, human-readable duration syntax.
func (b EffortBound) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Kind     BoundKind `json:"kind"`
		Clock    Clock     `json:"clock"`
		Duration string    `json:"duration"`
	}{b.Kind, b.Clock, b.Duration.String()})
}

// TriggerKind names how a session becomes eligible to start.
type TriggerKind string

const (
	TriggerImmediate TriggerKind = "immediate"
	TriggerAt        TriggerKind = "at"
	TriggerEvent     TriggerKind = "event"
)

// Trigger is an immediate, temporal, or external-event start condition.
type Trigger struct {
	Kind  TriggerKind `json:"kind"`
	At    *time.Time  `json:"at,omitempty"`
	Event string      `json:"event,omitempty"`
}

// Recurrence describes repeated activation without embedding an executor.
type Recurrence struct {
	Every         time.Duration `json:"-"`
	Cron          string        `json:"cron,omitempty"`
	Timezone      string        `json:"timezone,omitempty"`
	OverlapPolicy string        `json:"overlap_policy,omitempty"`
	MisfirePolicy string        `json:"misfire_policy,omitempty"`
}

// MarshalJSON renders a recurrence's interval as text and omits it for cron schedules.
func (r Recurrence) MarshalJSON() ([]byte, error) {
	var every string
	if r.Every > 0 {
		every = r.Every.String()
	}
	return json.Marshal(struct {
		Every         string `json:"every,omitempty"`
		Cron          string `json:"cron,omitempty"`
		Timezone      string `json:"timezone,omitempty"`
		OverlapPolicy string `json:"overlap_policy"`
		MisfirePolicy string `json:"misfire_policy"`
	}{every, r.Cron, r.Timezone, r.OverlapPolicy, r.MisfirePolicy})
}

// Hook binds a lifecycle event to a named, separately authorized action.
type Hook struct {
	Event         string        `json:"event"`
	Action        string        `json:"action"`
	Timeout       time.Duration `json:"-"`
	FailurePolicy string        `json:"failure_policy"`
}

// MarshalJSON renders a hook timeout as text.
func (h Hook) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Event         string `json:"event"`
		Action        string `json:"action"`
		Timeout       string `json:"timeout"`
		FailurePolicy string `json:"failure_policy"`
	}{h.Event, h.Action, h.Timeout.String(), h.FailurePolicy})
}

// Intent is the portable declaration. It deliberately does not grant capabilities or launch work.
type Intent struct {
	Version    string        `json:"version"`
	Objective  string        `json:"objective"`
	Trigger    Trigger       `json:"trigger"`
	Effort     []EffortBound `json:"effort,omitempty"`
	Deadline   *time.Time    `json:"deadline,omitempty"`
	Recurrence *Recurrence   `json:"recurrence,omitempty"`
	Hooks      []Hook        `json:"hooks,omitempty"`
}

var hookEvents = map[string]bool{
	"on_start": true, "before_tool": true, "after_tool": true, "before_stop": true,
	"on_complete": true, "on_timeout": true, "on_failure": true,
}

// Validate rejects ambiguous or contradictory intent before an executor sees it.
func (i Intent) Validate() error {
	if i.Version != "fak.session-intent/v1alpha1" {
		return fmt.Errorf("version must be fak.session-intent/v1alpha1")
	}
	if strings.TrimSpace(i.Objective) == "" {
		return errors.New("objective is required")
	}
	if err := validateTrigger(i.Trigger); err != nil {
		return err
	}
	seen := map[BoundKind]map[Clock]time.Duration{}
	for _, b := range i.Effort {
		if b.Kind != BoundMinimum && b.Kind != BoundTarget && b.Kind != BoundMaximum {
			return fmt.Errorf("unknown effort kind %q", b.Kind)
		}
		if b.Clock != ClockElapsed && b.Clock != ClockActive {
			return fmt.Errorf("unknown effort clock %q", b.Clock)
		}
		if b.Duration <= 0 {
			return fmt.Errorf("%s %s duration must be positive", b.Clock, b.Kind)
		}
		if seen[b.Kind] == nil {
			seen[b.Kind] = map[Clock]time.Duration{}
		}
		if _, ok := seen[b.Kind][b.Clock]; ok {
			return fmt.Errorf("duplicate %s %s effort bound", b.Clock, b.Kind)
		}
		seen[b.Kind][b.Clock] = b.Duration
	}
	for _, clock := range []Clock{ClockElapsed, ClockActive} {
		min, hasMin := seen[BoundMinimum][clock]
		target, hasTarget := seen[BoundTarget][clock]
		max, hasMax := seen[BoundMaximum][clock]
		if hasMin && hasTarget && min > target {
			return fmt.Errorf("%s minimum exceeds target", clock)
		}
		if hasTarget && hasMax && target > max {
			return fmt.Errorf("%s target exceeds maximum", clock)
		}
		if hasMin && hasMax && min > max {
			return fmt.Errorf("%s minimum exceeds maximum", clock)
		}
	}
	if i.Deadline != nil && i.Trigger.Kind == TriggerAt && i.Trigger.At != nil && !i.Deadline.After(*i.Trigger.At) {
		return errors.New("deadline must be after start time")
	}
	if i.Recurrence != nil {
		r := i.Recurrence
		if (r.Every > 0) == (strings.TrimSpace(r.Cron) != "") {
			return errors.New("recurrence requires exactly one of every or cron")
		}
		if r.Every < 0 {
			return errors.New("recurrence interval must be positive")
		}
		if r.Cron != "" && r.Timezone == "" {
			return errors.New("cron recurrence requires timezone")
		}
		if r.OverlapPolicy != "allow" && r.OverlapPolicy != "forbid" && r.OverlapPolicy != "replace" {
			return errors.New("overlap policy must be allow, forbid, or replace")
		}
		if r.MisfirePolicy != "skip" && r.MisfirePolicy != "catch_up_one" {
			return errors.New("misfire policy must be skip or catch_up_one")
		}
	}
	for _, h := range i.Hooks {
		if !hookEvents[h.Event] {
			return fmt.Errorf("unknown hook event %q", h.Event)
		}
		if strings.TrimSpace(h.Action) == "" {
			return errors.New("hook action is required")
		}
		if h.Timeout <= 0 {
			return fmt.Errorf("hook %s timeout must be positive", h.Event)
		}
		if h.FailurePolicy != "continue" && h.FailurePolicy != "block" {
			return fmt.Errorf("hook %s failure policy must be continue or block", h.Event)
		}
	}
	return nil
}

func validateTrigger(t Trigger) error {
	switch t.Kind {
	case TriggerImmediate:
		if t.At != nil || t.Event != "" {
			return errors.New("immediate trigger cannot carry at or event")
		}
	case TriggerAt:
		if t.At == nil || t.Event != "" {
			return errors.New("at trigger requires only at")
		}
	case TriggerEvent:
		if t.At != nil || strings.TrimSpace(t.Event) == "" {
			return errors.New("event trigger requires only event")
		}
	default:
		return fmt.Errorf("unknown trigger kind %q", t.Kind)
	}
	return nil
}

// SortedEffort returns a deterministic display order without mutating the intent.
func (i Intent) SortedEffort() []EffortBound {
	out := append([]EffortBound(nil), i.Effort...)
	rank := map[BoundKind]int{BoundMinimum: 0, BoundTarget: 1, BoundMaximum: 2}
	sort.Slice(out, func(a, b int) bool {
		if out[a].Clock != out[b].Clock {
			return out[a].Clock < out[b].Clock
		}
		return rank[out[a].Kind] < rank[out[b].Kind]
	})
	return out
}
