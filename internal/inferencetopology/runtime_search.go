// Copyright 2024 DistServe Authors
// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified for fak: observed-only finite search, strict evidence identity, and
// fail-closed eligibility. See third_party/resource-topology/NOTICE.

package inferencetopology

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fitplan"
)

const RuntimeSearchSchema = "fak-runtime-topology-search/1"

// RuntimeSearchInput searches declared supported layouts; it does not invent
// device assignments or predict performance for unmeasured configurations.
type RuntimeSearchInput struct {
	Schema        string                     `json:"schema"`
	ModelRevision string                     `json:"model_revision"`
	Backend       string                     `json:"backend"`
	Kernel        string                     `json:"kernel"`
	Workload      fitplan.SearchWorkload     `json:"workload"`
	SLO           fitplan.SearchSLO          `json:"slo"`
	Options       []RuntimeSearchOption      `json:"options"`
	Observations  []RuntimeSearchObservation `json:"observations"`
	Pins          RuntimeSearchConfig        `json:"pins"`
	Incumbent     *RuntimeSearchConfig       `json:"incumbent,omitempty"`
	MaxCandidates int                        `json:"max_candidates"`
	MaxDurationMS int                        `json:"max_duration_ms"`
}

// Each topology fixes replica, P:D, TP, PP and EP placement. The axes describe
// configurations supported by that layout, not assumed backend capabilities.
type RuntimeSearchOption struct {
	Topology RuntimeTopology `json:"topology"`
	Batches  []int           `json:"batches"`
	Chunks   []int           `json:"chunks"`
	Slots    []int           `json:"slots"`
	// Reservations are outstanding external bookings, additional to the measured
	// candidate peak. Every configured rank must explicitly supply even zero.
	Reservations map[string]float64 `json:"reservations"`
}

type RuntimeSearchConfig struct {
	TopologyID string `json:"topology_id"`
	Batch      int    `json:"batch"`
	Chunk      int    `json:"chunk"`
	Slots      int    `json:"slots"`
}

// RuntimeSearchObservation is an exact measured point, not a fitted hardware
// constant. Decode uses fitplan's existing receipt validation; no second
// regression or partition estimator is introduced. TTFT and useful throughput
// are end-to-end measurements, already including the transfer critical path.
type RuntimeSearchObservation struct {
	Config                      RuntimeSearchConfig        `json:"config"`
	ModelRevision               string                     `json:"model_revision"`
	Backend                     string                     `json:"backend"`
	Kernel                      string                     `json:"kernel"`
	LayoutDigest                string                     `json:"layout_digest"`
	LinkDigest                  string                     `json:"link_digest"`
	ObservationEpoch            uint64                     `json:"observation_epoch"`
	Workload                    fitplan.SearchWorkload     `json:"workload"`
	ExpiresAt                   string                     `json:"expires_at"`
	Decode                      fitplan.CalibrationReceipt `json:"decode"`
	TTFTSeconds                 float64                    `json:"ttft_seconds"`
	UsefulTokensPerSecond       float64                    `json:"useful_tokens_per_second"`
	TransferCriticalPathSeconds float64                    `json:"transfer_critical_path_seconds"`
	PeakBytes                   map[string]float64         `json:"peak_bytes"`
}

type RuntimeSearchReason string

const (
	SearchEligible              RuntimeSearchReason = "eligible"
	SearchObservationUnknown    RuntimeSearchReason = "observation_unknown"
	SearchObservationStale      RuntimeSearchReason = "observation_stale"
	SearchIdentityMismatch      RuntimeSearchReason = "identity_mismatch"
	SearchTopologyUnavailable   RuntimeSearchReason = "topology_unavailable"
	SearchTransferUnknown       RuntimeSearchReason = "transfer_unknown"
	SearchMemoryUnknown         RuntimeSearchReason = "memory_unknown"
	SearchMemoryInfeasible      RuntimeSearchReason = "memory_infeasible"
	SearchPinMismatch           RuntimeSearchReason = "pin_mismatch"
	SearchConcurrencyInfeasible RuntimeSearchReason = "concurrency_infeasible"
	SearchSLOInfeasible         RuntimeSearchReason = "slo_infeasible"
)

type RuntimeSearchRow struct {
	Config                      RuntimeSearchConfig `json:"config"`
	Reason                      RuntimeSearchReason `json:"reason"`
	TTFTSeconds                 float64             `json:"ttft_seconds,omitempty"`
	TPOTSeconds                 float64             `json:"tpot_seconds,omitempty"`
	UsefulTokensPerSecond       float64             `json:"useful_tokens_per_second,omitempty"`
	TransferCriticalPathSeconds float64             `json:"transfer_critical_path_seconds,omitempty"`
	ReceiptSHA256               string              `json:"receipt_sha256,omitempty"`
	ObservationEpoch            uint64              `json:"observation_epoch,omitempty"`
	LayoutDigest                string              `json:"layout_digest,omitempty"`
	LinkDigest                  string              `json:"link_digest,omitempty"`
}

type RuntimeSearchReport struct {
	Schema        string                 `json:"schema"`
	Status        string                 `json:"status"`
	Evidence      string                 `json:"evidence"`
	EvaluatedAt   string                 `json:"evaluated_at"`
	ModelRevision string                 `json:"model_revision"`
	Backend       string                 `json:"backend"`
	Kernel        string                 `json:"kernel"`
	Workload      fitplan.SearchWorkload `json:"workload"`
	SLO           fitplan.SearchSLO      `json:"slo"`
	Chosen        *RuntimeSearchRow      `json:"chosen,omitempty"`
	Incumbent     *RuntimeSearchRow      `json:"incumbent,omitempty"`
	Rows          []RuntimeSearchRow     `json:"rows"`
	Enumerated    int                    `json:"enumerated"`
	Evaluated     int                    `json:"evaluated"`
	StopReason    string                 `json:"stop_reason"`
}

func DecodeRuntimeSearch(r io.Reader) (RuntimeSearchInput, error) {
	var in RuntimeSearchInput
	raw, err := io.ReadAll(io.LimitReader(r, (16<<20)+1))
	if err != nil || len(raw) > 16<<20 {
		return in, invalid("search input read or size limit")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&in); err != nil {
		return in, invalid("search decode: %v", err)
	}
	var extra any
	if d.Decode(&extra) != io.EOF {
		return in, invalid("search trailing JSON")
	}
	return in, nil
}

// RuntimeSearchIdentity canonically binds measured points to rank/device
// incarnations, ownership, group layout and observed links. Capacity changes
// are checked separately; a changed link requires a matching new measurement,
// rather than extrapolating physical performance from an old curve.
func RuntimeSearchIdentity(t RuntimeTopology) (layout, links string, err error) {
	c := t.Configured
	c.Groups = append([]RuntimeTopologyGroup(nil), c.Groups...)
	c.Ranks = append([]RuntimeTopologyRank(nil), c.Ranks...)
	for i := range c.Groups {
		c.Groups[i].RankIDs = append([]string(nil), c.Groups[i].RankIDs...)
		sort.Strings(c.Groups[i].RankIDs)
	}
	for i := range c.Ranks {
		c.Ranks[i].KVOwnership = append([]string(nil), c.Ranks[i].KVOwnership...)
		sort.Strings(c.Ranks[i].KVOwnership)
	}
	sort.Slice(c.Groups, func(i, j int) bool { return c.Groups[i].ID < c.Groups[j].ID })
	sort.Slice(c.Ranks, func(i, j int) bool { return c.Ranks[i].ID < c.Ranks[j].ID })
	ls := append([]RuntimeObservedLink(nil), t.Observed.Links...)
	// Sort full serialized records, including parallel links of different kinds.
	sort.Slice(ls, func(i, j int) bool {
		a, _ := json.Marshal(ls[i])
		b, _ := json.Marshal(ls[j])
		return bytes.Compare(a, b) < 0
	})
	digest := func(v any) (string, error) {
		b, e := json.Marshal(v)
		if e != nil {
			return "", e
		}
		h := sha256.Sum256(b)
		return hex.EncodeToString(h[:]), nil
	}
	if layout, err = digest(c); err != nil {
		return "", "", invalid("layout identity")
	}
	if links, err = digest(ls); err != nil {
		return "", "", invalid("link identity")
	}
	return layout, links, nil
}

func searchConfigKey(c RuntimeSearchConfig) string {
	return fmt.Sprintf("%s:b%d:c%d:s%d", c.TopologyID, c.Batch, c.Chunk, c.Slots)
}

func searchAxis(axis []int) bool {
	if len(axis) == 0 || len(axis) > 64 {
		return false
	}
	seen := map[int]bool{}
	for _, n := range axis {
		if n <= 0 || n > 1<<20 || seen[n] {
			return false
		}
		seen[n] = true
	}
	return true
}

// SearchRuntimeTopology adapts DistServe's finite product/prune loop and
// Dynamo's bounded SLA filter. Unlike the source simulator/controller, unknown
// capacity never receives a fallback limit and SLO violations never win.
// now evaluates observation freshness; the monotonic duration budget and
// caller context independently bound execution. Inputs are not mutated.
func SearchRuntimeTopology(ctx context.Context, in RuntimeSearchInput, now time.Time) (RuntimeSearchReport, error) {
	report := RuntimeSearchReport{Schema: "fak-runtime-topology-search-report/1", Status: "unknown", Evidence: "SUPPLIED_OBSERVATIONS_NOT_HARDWARE_QUALIFICATION", Rows: []RuntimeSearchRow{}, StopReason: "complete"}
	report.EvaluatedAt = now.UTC().Format(time.RFC3339Nano)
	report.ModelRevision = in.ModelRevision
	report.Backend = in.Backend
	report.Kernel = in.Kernel
	report.Workload = in.Workload
	report.SLO = in.SLO
	if in.Schema != RuntimeSearchSchema || in.ModelRevision == "" || in.Backend == "" || in.Kernel == "" || in.Workload.PromptTokens <= 0 || in.Workload.OutputTokens <= 0 || in.Workload.Concurrency <= 0 || !nonnegative(in.SLO.TPOTMS) || !nonnegative(in.SLO.TTFTMS) || len(in.Options) == 0 || len(in.Options) > 128 || len(in.Observations) > 8192 || in.MaxCandidates <= 0 || in.MaxCandidates > 4096 || in.MaxDurationMS <= 0 || in.MaxDurationMS > 2000 || in.Pins.Batch < 0 || in.Pins.Chunk < 0 || in.Pins.Slots < 0 {
		return report, invalid("search identity, workload, axes or budget")
	}
	deadline := time.Now().Add(time.Duration(in.MaxDurationMS) * time.Millisecond)
	options := map[string]RuntimeSearchOption{}
	configs := []RuntimeSearchConfig{}
	for _, o := range in.Options {
		id := o.Topology.ID
		if id == "" || options[id].Topology.ID != "" || !searchAxis(o.Batches) || !searchAxis(o.Chunks) || !searchAxis(o.Slots) || len(o.Topology.Configured.Ranks) > 64 || len(o.Topology.Configured.Groups) > 64 || len(o.Topology.Observed.Links) > 4096 {
			return report, invalid("search option identity, size or axes")
		}
		count := len(o.Batches) * len(o.Chunks) * len(o.Slots)
		if len(configs)+count > 4096 {
			return report, invalid("search space exceeds 4096 configurations")
		}
		options[id] = o
		for _, b := range o.Batches {
			for _, c := range o.Chunks {
				for _, s := range o.Slots {
					configs = append(configs, RuntimeSearchConfig{TopologyID: id, Batch: b, Chunk: c, Slots: s})
				}
			}
		}
	}
	report.Enumerated = len(configs)
	points := map[RuntimeSearchConfig][]RuntimeSearchObservation{}
	for _, p := range in.Observations {
		points[p.Config] = append(points[p.Config], p)
	}
	incumbentFound := in.Incumbent == nil
	for _, c := range configs {
		if in.Incumbent != nil && c == *in.Incumbent {
			incumbentFound = true
		}
	}
	if !incumbentFound {
		return report, invalid("incumbent is outside the supported search space")
	}
	sort.Slice(configs, func(i, j int) bool {
		if in.Incumbent != nil {
			if configs[i] == *in.Incumbent {
				return true
			}
			if configs[j] == *in.Incumbent {
				return false
			}
		}
		return searchConfigKey(configs[i]) < searchConfigKey(configs[j])
	})
	unknown := false
	for _, c := range configs {
		if ctx.Err() != nil {
			report.StopReason = "cancelled"
			break
		}
		if !time.Now().Before(deadline) {
			report.StopReason = "time_budget"
			break
		}
		if report.Evaluated >= in.MaxCandidates {
			report.StopReason = "candidate_budget"
			break
		}
		row := evaluateRuntimeSearch(in, options[c.TopologyID], c, points[c], now)
		report.Rows = append(report.Rows, row)
		report.Evaluated++
		if in.Incumbent != nil && c == *in.Incumbent {
			copy := row
			report.Incumbent = &copy
		}
		switch row.Reason {
		case SearchEligible:
			if report.Chosen == nil || betterRuntimeSearch(row, *report.Chosen) {
				copy := row
				report.Chosen = &copy
			}
		case SearchObservationUnknown, SearchObservationStale, SearchIdentityMismatch, SearchTopologyUnavailable, SearchMemoryUnknown, SearchTransferUnknown:
			unknown = true
		}
	}
	sort.Slice(report.Rows, func(i, j int) bool {
		return searchConfigKey(report.Rows[i].Config) < searchConfigKey(report.Rows[j].Config)
	})
	if report.Chosen != nil {
		report.Status = "feasible"
	} else if report.StopReason == "complete" && !unknown {
		report.Status = "infeasible"
	}
	return report, nil
}

func betterRuntimeSearch(a, b RuntimeSearchRow) bool {
	if a.UsefulTokensPerSecond != b.UsefulTokensPerSecond {
		return a.UsefulTokensPerSecond > b.UsefulTokensPerSecond
	}
	if a.TTFTSeconds != b.TTFTSeconds {
		return a.TTFTSeconds < b.TTFTSeconds
	}
	if a.TPOTSeconds != b.TPOTSeconds {
		return a.TPOTSeconds < b.TPOTSeconds
	}
	return searchConfigKey(a.Config) < searchConfigKey(b.Config)
}

func evaluateRuntimeSearch(in RuntimeSearchInput, o RuntimeSearchOption, c RuntimeSearchConfig, points []RuntimeSearchObservation, now time.Time) RuntimeSearchRow {
	r := RuntimeSearchRow{Config: c, Reason: SearchObservationUnknown}
	pins := in.Pins
	if (pins.TopologyID != "" && pins.TopologyID != c.TopologyID) || (pins.Batch != 0 && pins.Batch != c.Batch) || (pins.Chunk != 0 && pins.Chunk != c.Chunk) || (pins.Slots != 0 && pins.Slots != c.Slots) {
		r.Reason = SearchPinMismatch
		return r
	}
	if c.Slots < in.Workload.Concurrency || c.Batch > c.Slots {
		r.Reason = SearchConcurrencyInfeasible
		return r
	}
	if _, err := ValidateRuntimeTopology(o.Topology, now); err != nil {
		r.Reason = SearchTopologyUnavailable
		return r
	}
	split, connected := runtimeSearchTransfers(o.Topology)
	if !connected {
		r.Reason = SearchTransferUnknown
		return r
	}
	layout, links, err := RuntimeSearchIdentity(o.Topology)
	if err != nil {
		r.Reason = SearchIdentityMismatch
		return r
	}
	// Exactly one point must match this workload; duplicate/conflicting evidence
	// is unknown rather than whichever item happened to arrive last.
	var matches []RuntimeSearchObservation
	for _, p := range points {
		if p.Workload == in.Workload {
			matches = append(matches, p)
		}
	}
	if len(matches) != 1 {
		return r
	}
	p := matches[0]
	if p.ModelRevision != in.ModelRevision || p.Backend != in.Backend || p.Kernel != in.Kernel || p.LayoutDigest != layout || p.LinkDigest != links || p.ObservationEpoch != o.Topology.Observed.Epoch || p.Decode.Key.Epoch != strconv.FormatUint(p.ObservationEpoch, 10) || p.Decode.Key.Backend != in.Backend || p.Decode.Key.KernelPath != in.Kernel || p.Decode.Key.BatchBucket != fitplan.BatchBucket(c.Batch) {
		r.Reason = SearchIdentityMismatch
		return r
	}
	at, e1 := time.Parse(time.RFC3339Nano, p.Decode.MeasuredAt)
	expires, e2 := time.Parse(time.RFC3339Nano, p.ExpiresAt)
	snapshotAt, _ := time.Parse(time.RFC3339Nano, o.Topology.Observed.At)
	if e1 != nil || e2 != nil || at.Before(snapshotAt) || now.Before(at) || !at.Before(expires) || !now.Before(expires) {
		r.Reason = SearchObservationStale
		return r
	}
	if p.Decode.Kind != fitplan.ReceiptDecode || p.Decode.Validate() != nil || p.Decode.Provenance.Path == "" || !searchDigest(p.Decode.Provenance.SHA256) || !nonnegative(p.TTFTSeconds) || p.TTFTSeconds == 0 || !nonnegative(p.UsefulTokensPerSecond) || p.UsefulTokensPerSecond == 0 || !nonnegative(p.TransferCriticalPathSeconds) || p.TransferCriticalPathSeconds > p.TTFTSeconds {
		return r
	}
	if split && p.TransferCriticalPathSeconds == 0 {
		r.Reason = SearchTransferUnknown
		return r
	}
	devices := map[string]RuntimeObservedDevice{}
	for _, d := range o.Topology.Observed.Devices {
		devices[d.ID] = d
	}
	if len(p.PeakBytes) != len(o.Topology.Configured.Ranks) || len(o.Reservations) != len(o.Topology.Configured.Ranks) {
		r.Reason = SearchMemoryUnknown
		return r
	}
	for _, rank := range o.Topology.Configured.Ranks {
		peak, known := p.PeakBytes[rank.ID]
		reserved, booked := o.Reservations[rank.ID]
		if !known || !booked || !nonnegative(peak) || peak == 0 || !nonnegative(reserved) || peak < rank.Commitment.ResidentWeightBytes+rank.Commitment.KVResidentBytes {
			r.Reason = SearchMemoryUnknown
			return r
		}
		need := peak + reserved
		if !nonnegative(need) || need > devices[rank.DeviceID].FreeBytes {
			r.Reason = SearchMemoryInfeasible
			return r
		}
	}
	r.TTFTSeconds = p.TTFTSeconds
	r.TPOTSeconds = p.Decode.StepSeconds
	r.UsefulTokensPerSecond = p.UsefulTokensPerSecond
	r.TransferCriticalPathSeconds = p.TransferCriticalPathSeconds
	r.ReceiptSHA256 = p.Decode.Provenance.SHA256
	r.ObservationEpoch = p.ObservationEpoch
	r.LayoutDigest = layout
	r.LinkDigest = links
	if (in.SLO.TPOTMS > 0 && r.TPOTSeconds > in.SLO.TPOTMS/1000) || (in.SLO.TTFTMS > 0 && r.TTFTSeconds > in.SLO.TTFTMS/1000) {
		r.Reason = SearchSLOInfeasible
		return r
	}
	r.Reason = SearchEligible
	return r
}

func searchDigest(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size
}

// Separate roles need observed connectivity between their groups. Group-local
// TP/PP/EP edges remain the snapshot validator's authority. This only verifies
// reachability, not an executable transfer protocol or a new link cost model.
func runtimeSearchTransfers(t RuntimeTopology) (split, ok bool) {
	ranks := map[runtimeDeviceKey]RuntimeTopologyRank{}
	roles := map[string]string{}
	adj := map[string][]string{}
	for _, r := range t.Configured.Ranks {
		ranks[runtimeDeviceKey{r.DeviceID, r.DeviceIncarnation}] = r
	}
	for _, g := range t.Configured.Groups {
		roles[g.ID] = g.Role
		if g.Role != RuntimeRoleBoth {
			split = true
		}
	}
	if !split {
		return false, true
	}
	for _, l := range t.Observed.Links {
		a, aok := ranks[runtimeDeviceKey{l.FromDeviceID, l.FromDeviceIncarnation}]
		b, bok := ranks[runtimeDeviceKey{l.ToDeviceID, l.ToDeviceIncarnation}]
		if aok && bok && l.State == "available" && l.Kind != "" && nonnegative(l.BytesPerSecond) && l.BytesPerSecond > 0 {
			adj[a.GroupID] = append(adj[a.GroupID], b.GroupID)
			adj[b.GroupID] = append(adj[b.GroupID], a.GroupID)
		}
	}
	for id, role := range roles {
		if role == RuntimeRoleBoth {
			continue
		}
		seen := map[string]bool{id: true}
		queue := []string{id}
		found := false
		for len(queue) > 0 {
			current := queue[0]
			queue = queue[1:]
			if roles[current] != role {
				found = true
			}
			for _, next := range adj[current] {
				if !seen[next] {
					seen[next] = true
					queue = append(queue, next)
				}
			}
		}
		if !found {
			return true, false
		}
	}
	return true, true
}
