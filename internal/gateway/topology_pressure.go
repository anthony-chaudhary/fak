/*
Copyright 2025 The Kubernetes Authors.
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
// Modified for fak: explicit source/incarnation/freshness checks and existing
// membership bookings instead of an upstream ledger. See third_party/resource-topology/NOTICE.
package gateway

import (
	"math"
	"time"
)

// TopologyPressureObservation is supplied telemetry, not a live probe receipt.
// Nil counters mean unknown; a measured zero must be explicitly present.
type TopologyPressureObservation struct {
	TopologyID    string    `json:"topology_id"`
	RankID        string    `json:"rank_id"`
	WorkerID      string    `json:"worker_id"`
	Incarnation   uint64    `json:"incarnation"`
	Epoch         uint64    `json:"epoch"`
	Source        string    `json:"source"`
	ModelRevision string    `json:"model_revision"`
	Backend       string    `json:"backend"`
	Kernel        string    `json:"kernel"`
	LayoutDigest  string    `json:"layout_digest"`
	LinkDigest    string    `json:"link_digest"`
	ObservedAt    time.Time `json:"observed_at"`
	TTLSeconds    int64     `json:"ttl_seconds"`
	Waiting       *float64  `json:"waiting"`
	Running       *float64  `json:"running"`
	KVUtilization *float64  `json:"kv_utilization"`
}

type TopologyPressure struct {
	Known         bool    `json:"known"`
	QueueDepth    int     `json:"queue_depth"`
	KVUtilization float64 `json:"kv_utilization"`
	Reason        string  `json:"reason"`
}

func topologyPressure(now time.Time, epoch uint64, st WorkerStatus, o TopologyPressureObservation) TopologyPressure {
	unknown := func(reason string) TopologyPressure { return TopologyPressure{Reason: reason} }
	if st.Health != HealthHealthy || st.Draining {
		return unknown("worker_unavailable")
	}
	if o.Source == "" || o.WorkerID != st.Spec.ID || o.Incarnation == 0 || o.Incarnation != st.Incarnation || o.Epoch != epoch {
		return unknown("observation_identity_unknown")
	}
	if o.ObservedAt.IsZero() || o.ObservedAt.After(now) || o.TTLSeconds <= 0 || o.TTLSeconds > 86400 || now.Sub(o.ObservedAt) >= time.Duration(o.TTLSeconds)*time.Second {
		return unknown("observation_stale")
	}
	// The bound also keeps float-to-int conversion and queue arithmetic safe on
	// every supported architecture; fractional request counts are not observations.
	validCount := func(v *float64) bool {
		return v != nil && !math.IsNaN(*v) && !math.IsInf(*v, 0) && *v >= 0 && *v <= 1e8 && math.Trunc(*v) == *v
	}
	if !validCount(o.Waiting) || !validCount(o.Running) || st.Inflight < 0 || st.Inflight > 1e8 || o.KVUtilization == nil || math.IsNaN(*o.KVUtilization) || math.IsInf(*o.KVUtilization, 0) || *o.KVUtilization < 0 || *o.KVUtilization > 1 {
		return unknown("observation_unknown")
	}
	waiting, running := int(*o.Waiting), int(*o.Running)
	// llm-d Detector.queueDepth: only locally booked work absent from the scrape
	// is added. Scraped running is NOT queued, and KV remains measured, not summed.
	queue := waiting + max(0, st.Inflight-waiting-running)
	return TopologyPressure{Known: true, QueueDepth: queue, KVUtilization: *o.KVUtilization, Reason: "supplied_observation"}
}
