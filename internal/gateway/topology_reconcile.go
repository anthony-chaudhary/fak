package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/inferencetopology"
)

const TopologyShadowSchema = "fak-topology-shadow/1"

// TopologyShadowInput is an explicitly supplied evidence document. It is not a
// live inventory collector or authority to launch/drain capacity. Only queue/KV
// objectives are accepted here; measured curves remain owned by RuntimeSearch.
type TopologyShadowInput struct {
	Schema          string                               `json:"schema"`
	Search          inferencetopology.RuntimeSearchInput `json:"search"`
	Workers         []TopologyPressureObservation        `json:"workers"`
	Objective       PoolObjective                        `json:"objective"`
	Incumbent       int                                  `json:"incumbent"`
	HysteresisTicks int                                  `json:"hysteresis_ticks"`
	CooldownSeconds int                                  `json:"cooldown_seconds"`
}

type TopologyShadowReport struct {
	Schema       string                                `json:"schema"`
	Evidence     string                                `json:"evidence"`
	Backpressure bool                                  `json:"backpressure"`
	Reason       string                                `json:"reason"`
	Search       inferencetopology.RuntimeSearchReport `json:"search"`
	Decision     ScaleDecision                         `json:"decision"`
}

func DecodeTopologyShadow(r io.Reader) (TopologyShadowInput, error) {
	var in TopologyShadowInput
	raw, err := io.ReadAll(io.LimitReader(r, (16<<20)+1))
	if err != nil || len(raw) > 16<<20 {
		return in, errors.New("topology shadow: input read or size limit")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err = d.Decode(&in); err != nil {
		return in, err
	}
	var extra any
	if d.Decode(&extra) != io.EOF {
		return in, errors.New("topology shadow: trailing JSON")
	}
	return in, validateTopologyShadow(in)
}

func validateTopologyShadow(in TopologyShadowInput) error {
	o := in.Objective
	finite := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
	if in.Schema != TopologyShadowSchema || len(in.Workers) > 8192 || in.Incumbent < 0 || in.Incumbent > 64 || in.HysteresisTicks < 1 || in.HysteresisTicks > 1000 || in.CooldownSeconds < 0 || in.CooldownSeconds > 86400 || o.MinReplicas < 0 || o.MaxReplicas < 1 || o.MaxReplicas > 64 || o.MinReplicas > o.MaxReplicas || in.Incumbent > o.MaxReplicas || o.ScaleStep < 1 || o.ScaleStep > 64 || o.QueueHigh < 1 || o.QueueLow < 0 || o.QueueLow >= o.QueueHigh || !finite(o.KVUtilHigh) || !finite(o.KVUtilLow) || o.KVUtilHigh <= 0 || o.KVUtilHigh > 1 || o.KVUtilLow < 0 || o.KVUtilLow >= o.KVUtilHigh || o.GoodputTarget != 0 || o.TTFTSLO != 0 || o.TPOTSLO != 0 || !finite(o.ScaleDownSensitivity) || o.ScaleDownSensitivity < 0 || o.ScaleDownSensitivity > 1 {
		return errors.New("topology shadow: invalid bounds or unsupported objective")
	}
	seen := map[string]bool{}
	for _, w := range in.Workers {
		key := w.TopologyID + "\x00" + w.RankID
		if w.TopologyID == "" || w.RankID == "" || w.WorkerID == "" || seen[key] {
			return errors.New("topology shadow: duplicate or empty binding")
		}
		seen[key] = true
	}
	// Validate structural/search budgets even when all curves are expired. The
	// search owns identity and capability validation; this never upgrades evidence.
	_, err := inferencetopology.SearchRuntimeTopology(context.Background(), in.Search, time.Now())
	return err
}

type topologyShadowController struct {
	mu        sync.Mutex
	input     TopologyShadowInput
	scaler    *ServingAutoscaler
	incumbent int
	observe   func(TopologyShadowReport)
}

// SetTopologyShadow installs bounded supplied observations before serving. The
// deep copy prevents caller mutation. No new membership registry is created.
// The observer runs only on the existing background health loop, never dispatch.
func (s *Server) SetTopologyShadow(in TopologyShadowInput, observe func(TopologyShadowReport)) error {
	if s == nil {
		return errors.New("topology shadow: nil server")
	}
	if err := validateTopologyShadow(in); err != nil {
		return err
	}
	raw, err := json.Marshal(in)
	if err != nil {
		return err
	}
	frozen, err := DecodeTopologyShadow(bytes.NewReader(raw))
	if err != nil {
		return err
	}
	s.fleetMu.Lock()
	defer s.fleetMu.Unlock()
	if s.fleet == nil {
		return errors.New("topology shadow: existing replica fleet required")
	}
	s.topologyShadow = &topologyShadowController{input: frozen, incumbent: in.Incumbent, observe: observe, scaler: NewServingAutoscaler(nil, nil, ServingAutoscalerConfig{Mode: ScaleMode("shadow"), HysteresisTicks: in.HysteresisTicks, Cooldown: time.Duration(in.CooldownSeconds) * time.Second})}
	return nil
}

// ReconcileTopologyShadow uses the same server membership as ordinary dispatch.
// It is safe for explicit deterministic ticks and the existing serve lifecycle.
func (s *Server) ReconcileTopologyShadow(ctx context.Context, now time.Time) (TopologyShadowReport, bool) {
	s.fleetMu.Lock()
	c, m := s.topologyShadow, s.fleet
	s.fleetMu.Unlock()
	if c == nil || m == nil {
		return TopologyShadowReport{}, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	report := TopologyShadowReport{Schema: "fak-topology-shadow-report/1", Evidence: "SUPPLIED_OBSERVATIONS_SHADOW_ONLY_NO_CAPACITY_ACTION", Backpressure: true}
	search, err := inferencetopology.SearchRuntimeTopology(ctx, c.input.Search, now)
	report.Search = search
	known := err == nil && ctx.Err() == nil && search.Chosen != nil
	current := 0
	var sig PoolSignals
	obj := c.input.Objective
	if known {
		chosen := search.Chosen
		var option *inferencetopology.RuntimeSearchOption
		for i := range c.input.Search.Options {
			if c.input.Search.Options[i].Topology.ID == chosen.Config.TopologyID {
				option = &c.input.Search.Options[i]
				break
			}
		}
		if option == nil {
			known = false
		} else {
			// Slot capacity is accepted measured evidence, not an upstream tuning
			// constant. Explicit safety policy cannot request above measured bounds.
			if obj.QueueHigh > chosen.Config.Slots || obj.MaxReplicas > len(option.Topology.Configured.Groups) {
				known = false
				report.Reason = "policy_exceeds_measured_capacity"
			}
			statuses := map[string]WorkerStatus{}
			for _, st := range m.Snapshot() {
				statuses[st.Spec.ID] = st
			}
			bindings := map[string]TopologyPressureObservation{}
			used := map[string]bool{}
			for _, w := range c.input.Workers {
				if w.TopologyID == chosen.Config.TopologyID {
					bindings[w.RankID] = w
					if used[w.WorkerID] {
						known = false
					}
					used[w.WorkerID] = true
				}
			}
			for _, group := range option.Topology.Configured.Groups {
				ready := true
				groupKnown := true
				groupQueue := 0
				groupKV := 0.0
				for _, rank := range group.RankIDs {
					w, ok := bindings[rank]
					st, present := statuses[w.WorkerID]
					if !ok || !present || st.Health == HealthUnknown || w.Incarnation != st.Incarnation {
						ready = false
						known = false
						continue
					}
					if st.Health != HealthHealthy || st.Draining {
						ready = false
						continue
					}
					if w.ModelRevision != search.ModelRevision || w.Backend != search.Backend || w.Kernel != search.Kernel || w.LayoutDigest != chosen.LayoutDigest || w.LinkDigest != chosen.LinkDigest {
						groupKnown = false
					}
					p := topologyPressure(now, chosen.ObservationEpoch, st, w)
					if !p.Known {
						groupKnown = false
					}
					// PP/EP ranks can describe the same logical request. Use bottleneck
					// pressure within a group, never sum replicated stage observations.
					groupQueue = max(groupQueue, p.QueueDepth)
					groupKV = max(groupKV, p.KVUtilization)
				}
				if ready {
					current++
					known = known && groupKnown
					sig.QueueDepth = saturatingNonnegativeAdd(sig.QueueDepth, groupQueue)
					sig.KVUtilization = max(sig.KVUtilization, groupKV)
				}
			}
			if current > 0 && known {
				report.Backpressure = sig.QueueDepth >= obj.QueueHigh || sig.KVUtilization >= obj.KVUtilHigh
			}
		}
	}
	if ctx.Err() != nil {
		known = false
		report.Reason = "canceled"
	}
	report.Decision = c.scaler.shadowDecision(now, current, c.incumbent, known, sig, obj)
	if known && current > 0 {
		c.incumbent = report.Decision.DesiredReplicas
	}
	if report.Reason == "" {
		report.Reason = report.Decision.Reason
	}
	if err != nil {
		report.Reason = fmt.Sprintf("search_refused: %v", err)
	}
	if !known || current == 0 {
		report.Backpressure = true
	}
	return report, true
}

func (s *Server) observeTopologyShadow(ctx context.Context, now time.Time) {
	r, ok := s.ReconcileTopologyShadow(ctx, now)
	if !ok {
		return
	}
	s.fleetMu.Lock()
	c := s.topologyShadow
	s.fleetMu.Unlock()
	if c != nil && c.observe != nil {
		c.observe(r)
	}
}
