package inferencetopology

import (
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
)

func runtimeTopologyFixture(prefill, decode, decodePP int) RuntimeTopology {
	groups := []RuntimeTopologyGroup{}
	ranks := []RuntimeTopologyRank{}
	devices := []RuntimeObservedDevice{}
	support := []RuntimeTopologySupport{}
	available := []RuntimeTopologyAvailable{}
	visible := []string{}
	links := []RuntimeObservedLink{}
	addGroup := func(id, role string, count, pp int, capacityBase float64) {
		g := RuntimeTopologyGroup{ID: id, Role: role, TP: 1, PP: pp, EP: 1, Layers: pp * 4, KVFormat: "paged-v1"}
		for n := 0; n < count; n++ {
			for stage := 0; stage < pp; stage++ {
				rankID := id + "-r" + string(rune('0'+n)) + "-s" + string(rune('0'+stage))
				deviceID := "gpu-" + rankID
				g.RankIDs = append(g.RankIDs, rankID)
				ranks = append(ranks, RuntimeTopologyRank{ID: rankID, Incarnation: "rank-inc-1", GroupID: id, Role: role, DeviceID: deviceID, DeviceIncarnation: "device-inc-1", PPStage: stage, LayerStart: stage * 4, LayerEnd: (stage + 1) * 4, KVOwnership: []string{"kv-" + rankID}, Commitment: Resources{ResidentWeightBytes: 40 + float64(stage), KVResidentBytes: 5}})
				devices = append(devices, RuntimeObservedDevice{ID: deviceID, Incarnation: "device-inc-1", Lifecycle: "active", Health: "healthy", CapacityBytes: capacityBase + float64(stage)*20, FreeBytes: capacityBase - 5 + float64(stage)*20, FreeBytesKnown: true})
				support = append(support, RuntimeTopologySupport{DeviceID: deviceID, Incarnation: "device-inc-1", State: "supported"})
				available = append(available, RuntimeTopologyAvailable{DeviceID: deviceID, Incarnation: "device-inc-1", State: "available"})
				visible = append(visible, deviceID)
			}
		}
		groups = append(groups, g)
	}
	for n := 0; n < prefill; n++ {
		addGroup("prefill-"+string(rune('0'+n)), RuntimeRolePrefill, 1, 1, 80)
	}
	for n := 0; n < decode; n++ {
		addGroup("decode-"+string(rune('0'+n)), RuntimeRoleDecode, 1, decodePP, 100)
	}
	for _, g := range groups {
		for i, left := range g.RankIDs {
			for _, right := range g.RankIDs[i+1:] {
				var leftRank, rightRank RuntimeTopologyRank
				for _, r := range ranks {
					if r.ID == left {
						leftRank = r
					}
					if r.ID == right {
						rightRank = r
					}
				}
				required := leftRank.PPStage == rightRank.PPStage || (absRuntimeInt(leftRank.PPStage-rightRank.PPStage) == 1 && leftRank.TPRank == rightRank.TPRank && leftRank.EPRank == rightRank.EPRank)
				if required {
					links = append(links, RuntimeObservedLink{FromDeviceID: leftRank.DeviceID, FromDeviceIncarnation: leftRank.DeviceIncarnation, ToDeviceID: rightRank.DeviceID, ToDeviceIncarnation: rightRank.DeviceIncarnation, Kind: "pcie", State: "available", BytesPerSecond: 1e9})
				}
			}
		}
	}
	return RuntimeTopology{Schema: RuntimeTopologySchema, ID: "fixture", Configured: RuntimeTopologyConfigured{Groups: groups, Ranks: ranks}, Observed: RuntimeTopologyObserved{At: "2026-10-26T10:00:00Z", TTLSeconds: 60, Epoch: 7, Provenance: "synthetic-test-observation", Visibility: visible, Devices: devices, Links: links}, Supported: support, Available: available}
}

// fak-test:runtime fast est=20ms lane=default
func TestRuntimeTopologyObservedAvailability(t *testing.T) {
	now := time.Date(2026, 10, 26, 10, 0, 30, 0, time.UTC)
	for _, tc := range []struct {
		name string
		top  RuntimeTopology
	}{
		{name: "1P1D", top: runtimeTopologyFixture(1, 1, 1)},
		{name: "1P2D", top: runtimeTopologyFixture(1, 2, 1)},
		{name: "1P+D(PP2)_mixed_capacity_disjoint_ownership", top: runtimeTopologyFixture(1, 1, 2)},
		{name: "PP3_requires_adjacent_edges_not_full_clique", top: runtimeTopologyFixture(1, 1, 3)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			receipt, err := ValidateRuntimeTopology(tc.top, now)
			if err != nil {
				t.Fatal(err)
			}
			if len(receipt.Configured) != len(tc.top.Configured.Ranks) || !reflect.DeepEqual(receipt.Configured, receipt.Observed) || !reflect.DeepEqual(receipt.Observed, receipt.Supported) || !reflect.DeepEqual(receipt.Supported, receipt.Available) {
				t.Fatalf("configured/observed/supported/available evidence was collapsed or lost: %+v", receipt)
			}
		})
	}

	t.Run("role_both", func(t *testing.T) {
		top := runtimeTopologyFixture(1, 0, 1)
		top.Configured.Groups[0].Role = RuntimeRoleBoth
		top.Configured.Ranks[0].Role = RuntimeRoleBoth
		if _, err := ValidateRuntimeTopology(top, now); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("unconfigured_unknown_device_and_unavailable_link_do_not_poison", func(t *testing.T) {
		top := runtimeTopologyFixture(1, 1, 1)
		top.Observed.Devices = append(top.Observed.Devices, RuntimeObservedDevice{ID: "gpu-unconfigured", Incarnation: "device-inc-unknown"})
		top.Observed.Visibility = append(top.Observed.Visibility, "gpu-unconfigured")
		top.Observed.Links = append(top.Observed.Links, RuntimeObservedLink{FromDeviceID: "gpu-unconfigured", FromDeviceIncarnation: "device-inc-unknown", ToDeviceID: top.Configured.Ranks[0].DeviceID, ToDeviceIncarnation: top.Configured.Ranks[0].DeviceIncarnation, State: "unavailable"})
		top.Supported = append(top.Supported, RuntimeTopologySupport{DeviceID: "gpu-unconfigured", Incarnation: "device-inc-unknown"})
		top.Available = append(top.Available, RuntimeTopologyAvailable{DeviceID: "gpu-unconfigured", Incarnation: "device-inc-unknown"})
		if _, err := ValidateRuntimeTopology(top, now); err != nil {
			t.Fatal(err)
		}
	})

	base := runtimeTopologyFixture(1, 1, 2)
	want, err := ValidateRuntimeTopology(base, now)
	if err != nil {
		t.Fatal(err)
	}
	permuted := base
	permuted.Configured.Groups = slices.Clone(base.Configured.Groups)
	permuted.Configured.Ranks = slices.Clone(base.Configured.Ranks)
	permuted.Observed.Visibility = slices.Clone(base.Observed.Visibility)
	permuted.Observed.Devices = slices.Clone(base.Observed.Devices)
	permuted.Observed.Links = slices.Clone(base.Observed.Links)
	permuted.Supported = slices.Clone(base.Supported)
	permuted.Available = slices.Clone(base.Available)
	slices.Reverse(permuted.Configured.Groups)
	slices.Reverse(permuted.Configured.Ranks)
	slices.Reverse(permuted.Observed.Visibility)
	slices.Reverse(permuted.Observed.Devices)
	slices.Reverse(permuted.Observed.Links)
	slices.Reverse(permuted.Supported)
	slices.Reverse(permuted.Available)
	got, err := ValidateRuntimeTopology(permuted, now)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("discovery order changed eligibility: got=%+v want=%+v err=%v", got, want, err)
	}

	for _, tc := range []struct {
		name   string
		change func(*RuntimeTopology)
	}{
		{name: "unknown_visibility", change: func(top *RuntimeTopology) { top.Observed.Visibility[0] = "missing" }},
		{name: "empty_visibility", change: func(top *RuntimeTopology) { top.Observed.Visibility = nil }},
		{name: "retired", change: func(top *RuntimeTopology) { top.Observed.Devices[0].Lifecycle = "retired" }},
		{name: "unknown_health", change: func(top *RuntimeTopology) { top.Observed.Devices[0].Health = "" }},
		{name: "unknown_headroom", change: func(top *RuntimeTopology) { top.Observed.Devices[0].FreeBytesKnown = false }},
		{name: "duplicate_stable_device_incarnation", change: func(top *RuntimeTopology) {
			duplicate := top.Observed.Devices[0]
			duplicate.Incarnation = "new-incarnation-in-same-epoch"
			top.Observed.Devices = append(top.Observed.Devices, duplicate)
		}},
		{name: "unavailable", change: func(top *RuntimeTopology) { top.Available[0].State = "unavailable" }},
		{name: "stale", change: func(top *RuntimeTopology) { top.Observed.TTLSeconds = 30 }},
		{name: "ttl_overflow", change: func(top *RuntimeTopology) { top.Observed.TTLSeconds = int64(^uint64(0) >> 1) }},
		{name: "incomplete_tp", change: func(top *RuntimeTopology) { top.Configured.Groups[0].TP = 2 }},
		{name: "parallelism_overflow", change: func(top *RuntimeTopology) { top.Configured.Groups[0].TP = int(^uint(0) >> 1) }},
		{name: "incomplete_ep", change: func(top *RuntimeTopology) { top.Configured.Groups[0].EP = 2 }},
		{name: "noncontiguous_pp", change: func(top *RuntimeTopology) { top.Configured.Ranks[len(top.Configured.Ranks)-1].PPStage = 0 }},
		{name: "missing_required_link", change: func(top *RuntimeTopology) { top.Observed.Links = nil }},
		{name: "stale_link_incarnation", change: func(top *RuntimeTopology) { top.Observed.Links[0].ToDeviceIncarnation = "old-incarnation" }},
		{name: "pp_layer_gap", change: func(top *RuntimeTopology) { top.Configured.Ranks[2].LayerStart++ }},
		{name: "pp_layer_overlap", change: func(top *RuntimeTopology) {
			top.Configured.Ranks[1].LayerEnd++
		}},
		{name: "overlapping_device_ownership", change: func(top *RuntimeTopology) {
			top.Configured.Ranks[1].DeviceID = top.Configured.Ranks[0].DeviceID
			top.Configured.Ranks[1].DeviceIncarnation = top.Configured.Ranks[0].DeviceIncarnation
		}},
		{name: "overlapping_kv_ownership", change: func(top *RuntimeTopology) {
			top.Configured.Ranks[1].KVOwnership[0] = top.Configured.Ranks[0].KVOwnership[0]
		}},
		{name: "unknown_support", change: func(top *RuntimeTopology) { top.Supported = top.Supported[1:] }},
	} {
		t.Run("reject_"+tc.name, func(t *testing.T) {
			top := runtimeTopologyFixture(1, 1, 2)
			tc.change(&top)
			if _, err := ValidateRuntimeTopology(top, now); !errors.Is(err, ErrInvalid) {
				t.Fatalf("wanted typed fail-closed refusal, got %v", err)
			}
		})
	}

	t.Run("reject_unknown_json", func(t *testing.T) {
		top := runtimeTopologyFixture(1, 1, 1)
		raw, err := json.Marshal(top)
		if err != nil {
			t.Fatal(err)
		}
		input := strings.TrimSuffix(string(raw), "}") + `,"unknown":true}`
		if _, err := DecodeRuntimeTopology(strings.NewReader(input)); !errors.Is(err, ErrInvalid) {
			t.Fatalf("wanted typed unknown-field refusal, got %v", err)
		}
	})

	t.Run("reject_trailing_json", func(t *testing.T) {
		raw, err := json.Marshal(runtimeTopologyFixture(1, 1, 1))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := DecodeRuntimeTopology(strings.NewReader(string(raw) + `{}`)); !errors.Is(err, ErrInvalid) {
			t.Fatalf("wanted typed trailing-JSON refusal, got %v", err)
		}
	})
}
