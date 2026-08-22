package ultracodebench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
)

const ActivationSchema = "fak.ultracode_activation.v1"

type ActivationSetting string

const (
	SettingAuto ActivationSetting = "auto"
	SettingOn   ActivationSetting = "on"
	SettingOff  ActivationSetting = "off"
)

type ActivationObservable string

const (
	ObservableUnknown  ActivationObservable = "unknown"
	ObservableActive   ActivationObservable = "active"
	ObservableInactive ActivationObservable = "inactive"
	ObservableDegraded ActivationObservable = "degraded"
)

type ActivationObservationSource string

const (
	SourceExplicitAcknowledgement ActivationObservationSource = "explicit_harness_ack"
	SourceRuntimeObservation      ActivationObservationSource = "runtime_observation"
)

type ActivationState string

const (
	ActivationActive   ActivationState = "active"
	ActivationInactive ActivationState = "inactive"
	ActivationDegraded ActivationState = "degraded"
	ActivationUnknown  ActivationState = "unknown"
)

// ActivationReceipt contains only join identities and typed posture evidence.
// Harness-private argv, settings, paths, prompts, accounts, and hosts are never fields.
type ActivationReceipt struct {
	Schema            string                      `json:"schema"`
	RunID             string                      `json:"run_id"`
	ChildID           string                      `json:"child_id"`
	Harness           string                      `json:"harness"`
	Requested         ActivationSetting           `json:"requested"`
	Resolved          ActivationSetting           `json:"resolved"`
	Injected          bool                        `json:"injected"`
	Observable        ActivationObservable        `json:"observable"`
	ObservationSource ActivationObservationSource `json:"observation_source,omitempty"`
	Degradations      []string                    `json:"degradations,omitempty"`
}

type BeforeSpawnInput struct {
	RunID        string
	ChildID      string
	Harness      string
	Requested    ActivationSetting
	Resolved     ActivationSetting
	Injected     bool
	Degradations []string
}

// BeforeSpawn creates the durable pre-spawn fact. It never treats injection or
// process start as evidence that the child accepted the requested posture.
func BeforeSpawn(in BeforeSpawnInput) (ActivationReceipt, error) {
	r := ActivationReceipt{
		Schema: ActivationSchema, RunID: strings.TrimSpace(in.RunID), ChildID: strings.TrimSpace(in.ChildID),
		Harness: strings.ToLower(strings.TrimSpace(in.Harness)), Requested: in.Requested, Resolved: in.Resolved,
		Injected: in.Injected, Observable: ObservableUnknown, Degradations: normalizeDegradations(in.Degradations),
	}
	return r, r.Validate()
}

// Acknowledge returns an updated receipt only for an explicit harness
// acknowledgement or a declared supported observable.
func Acknowledge(r ActivationReceipt, observable ActivationObservable, source ActivationObservationSource, degradations ...string) (ActivationReceipt, error) {
	if source != SourceExplicitAcknowledgement && source != SourceRuntimeObservation {
		return ActivationReceipt{}, fmt.Errorf("activation acknowledgement source %q is unsupported", source)
	}
	if observable == ObservableUnknown || observable == "" {
		return ActivationReceipt{}, fmt.Errorf("activation acknowledgement must report active, inactive, or degraded")
	}
	r.Observable = observable
	r.ObservationSource = source
	r.Degradations = normalizeDegradations(append(r.Degradations, degradations...))
	if observable == ObservableDegraded && len(r.Degradations) == 0 {
		r.Degradations = []string{"activation_degraded"}
	}
	return r, r.Validate()
}

func DecodeActivation(data []byte) (ActivationReceipt, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var r ActivationReceipt
	if err := dec.Decode(&r); err != nil {
		return ActivationReceipt{}, fmt.Errorf("decode activation receipt: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return ActivationReceipt{}, fmt.Errorf("decode activation receipt: trailing JSON value")
		}
		return ActivationReceipt{}, fmt.Errorf("decode activation receipt: %w", err)
	}
	r.Degradations = normalizeDegradations(r.Degradations)
	return r, r.Validate()
}

func (r ActivationReceipt) Validate() error {
	if r.Schema != ActivationSchema {
		return fmt.Errorf("activation schema must be %q", ActivationSchema)
	}
	for name, value := range map[string]string{"run_id": r.RunID, "child_id": r.ChildID, "harness": r.Harness} {
		if !activationToken(value) {
			return fmt.Errorf("activation %s must be a non-empty opaque token", name)
		}
	}
	if r.Requested != SettingAuto && r.Requested != SettingOn && r.Requested != SettingOff {
		return fmt.Errorf("activation requested posture %q is invalid", r.Requested)
	}
	if r.Resolved != SettingOn && r.Resolved != SettingOff {
		return fmt.Errorf("activation resolved posture %q is invalid", r.Resolved)
	}
	if r.Resolved == SettingOff && r.Injected {
		return fmt.Errorf("activation resolved off cannot be injected")
	}
	if r.Resolved == SettingOn && !r.Injected && len(r.Degradations) == 0 {
		return fmt.Errorf("activation resolved on without injection requires a degradation")
	}
	if r.Observable != ObservableUnknown && r.Observable != ObservableActive && r.Observable != ObservableInactive && r.Observable != ObservableDegraded {
		return fmt.Errorf("activation observable %q is invalid", r.Observable)
	}
	if r.Observable == ObservableUnknown && r.ObservationSource != "" {
		return fmt.Errorf("unknown activation cannot carry an observation source")
	}
	if r.Observable != ObservableUnknown && r.ObservationSource != SourceExplicitAcknowledgement && r.ObservationSource != SourceRuntimeObservation {
		return fmt.Errorf("observed activation requires an explicit acknowledgement or supported observable")
	}
	if r.Observable == ObservableDegraded && len(r.Degradations) == 0 {
		return fmt.Errorf("degraded activation requires a degradation code")
	}
	for _, d := range r.Degradations {
		if !activationToken(d) {
			return fmt.Errorf("activation degradation %q must be an opaque token", d)
		}
	}
	return nil
}

func (r ActivationReceipt) State() ActivationState {
	if len(r.Degradations) > 0 || r.Observable == ObservableDegraded {
		return ActivationDegraded
	}
	if r.Resolved == SettingOff && !r.Injected {
		return ActivationInactive
	}
	switch r.Observable {
	case ObservableActive:
		return ActivationActive
	case ObservableInactive:
		return ActivationInactive
	default:
		return ActivationUnknown
	}
}

func (r ActivationReceipt) key() string { return r.RunID + "\x00" + r.ChildID }

type ChildActivationStatus struct {
	RunID   string          `json:"run_id"`
	ChildID string          `json:"child_id"`
	Harness string          `json:"harness"`
	State   ActivationState `json:"state"`
}

type ActivationSummary struct {
	Schema   string                  `json:"schema"`
	Total    int                     `json:"total"`
	Active   int                     `json:"active"`
	Inactive int                     `json:"inactive"`
	Degraded int                     `json:"degraded"`
	Unknown  int                     `json:"unknown"`
	Verified int                     `json:"verified"`
	Ratio    float64                 `json:"ratio"`
	Children []ChildActivationStatus `json:"children"`
}

func SummarizeActivation(receipts []ActivationReceipt) (ActivationSummary, error) {
	c := ActivationSummary{Schema: ActivationSchema, Children: make([]ChildActivationStatus, 0, len(receipts))}
	seen := make(map[string]struct{}, len(receipts))
	for _, r := range receipts {
		if err := r.Validate(); err != nil {
			return ActivationSummary{}, err
		}
		if _, ok := seen[r.key()]; ok {
			return ActivationSummary{}, fmt.Errorf("duplicate activation identity %s/%s", r.RunID, r.ChildID)
		}
		seen[r.key()] = struct{}{}
		state := r.State()
		c.Total++
		switch state {
		case ActivationActive:
			c.Active++
		case ActivationInactive:
			c.Inactive++
		case ActivationDegraded:
			c.Degraded++
		default:
			c.Unknown++
		}
		c.Children = append(c.Children, ChildActivationStatus{RunID: r.RunID, ChildID: r.ChildID, Harness: r.Harness, State: state})
	}
	sort.Slice(c.Children, func(i, j int) bool {
		if c.Children[i].RunID != c.Children[j].RunID {
			return c.Children[i].RunID < c.Children[j].RunID
		}
		return c.Children[i].ChildID < c.Children[j].ChildID
	})
	c.Verified = c.Active + c.Inactive
	if c.Total > 0 {
		c.Ratio = round(float64(c.Verified) / float64(c.Total))
	}
	return c, nil
}

func activationToken(s string) bool {
	if s == "" || s == "." || s == ".." || len(s) > 160 {
		return false
	}
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.' {
			continue
		}
		return false
	}
	return true
}

func normalizeDegradations(items []string) []string {
	seen := map[string]struct{}{}
	for _, item := range items {
		item = strings.ToLower(strings.TrimSpace(item))
		if item != "" {
			seen[item] = struct{}{}
		}
	}
	out := make([]string, 0, len(seen))
	for item := range seen {
		out = append(out, item)
	}
	sort.Strings(out)
	return out
}
