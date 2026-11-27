package modelengine

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/inferencetopology"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func testRoleGraphTopology(now time.Time, prefillGroups, prefillTP, decodeGroups, decodePP, decodeTP, decodeEP, layers int) inferencetopology.RuntimeTopology {
	var groups []inferencetopology.RuntimeTopologyGroup
	var ranks []inferencetopology.RuntimeTopologyRank
	var devices []inferencetopology.RuntimeObservedDevice
	var support []inferencetopology.RuntimeTopologySupport
	var available []inferencetopology.RuntimeTopologyAvailable
	var visible []string
	var links []inferencetopology.RuntimeObservedLink

	addGroup := func(id, role string, tp, pp, ep int, capBytes float64) {
		if layers%pp != 0 {
			panic("layers must divide pp")
		}
		layerPerStage := layers / pp
		g := inferencetopology.RuntimeTopologyGroup{
			ID:       id,
			Role:     role,
			TP:       tp,
			PP:       pp,
			EP:       ep,
			Layers:   layers,
			KVFormat: "paged-v1",
		}
		for ppStage := 0; ppStage < pp; ppStage++ {
			for tpRank := 0; tpRank < tp; tpRank++ {
				for epRank := 0; epRank < ep; epRank++ {
					rankID := id + "-s" + string(rune('0'+ppStage)) + "-t" + string(rune('0'+tpRank)) + "-e" + string(rune('0'+epRank))
					devID := "gpu-" + rankID
					g.RankIDs = append(g.RankIDs, rankID)
					ranks = append(ranks, inferencetopology.RuntimeTopologyRank{
						ID:                rankID,
						Incarnation:       "rank-inc-1",
						GroupID:           id,
						Role:              role,
						DeviceID:          devID,
						DeviceIncarnation: "dev-inc-1",
						TPRank:            tpRank,
						PPStage:           ppStage,
						EPRank:            epRank,
						LayerStart:        ppStage * layerPerStage,
						LayerEnd:          (ppStage + 1) * layerPerStage,
						KVOwnership:       []string{"kv-" + rankID},
						Commitment: inferencetopology.Resources{
							ResidentWeightBytes: 40,
							KVResidentBytes:     5,
						},
					})
					devices = append(devices, inferencetopology.RuntimeObservedDevice{
						ID:             devID,
						Incarnation:    "dev-inc-1",
						Lifecycle:      "active",
						Health:         "healthy",
						CapacityBytes:  capBytes,
						FreeBytes:      capBytes - 5,
						FreeBytesKnown: true,
					})
					support = append(support, inferencetopology.RuntimeTopologySupport{
						DeviceID:    devID,
						Incarnation: "dev-inc-1",
						State:       "supported",
					})
					available = append(available, inferencetopology.RuntimeTopologyAvailable{
						DeviceID:    devID,
						Incarnation: "dev-inc-1",
						State:       "available",
					})
					visible = append(visible, devID)
				}
			}
		}
		groups = append(groups, g)
	}

	for i := 0; i < prefillGroups; i++ {
		addGroup("prefill-"+string(rune('0'+i)), inferencetopology.RuntimeRolePrefill, prefillTP, 1, 1, 120)
	}
	for i := 0; i < decodeGroups; i++ {
		addGroup("decode-"+string(rune('0'+i)), inferencetopology.RuntimeRoleDecode, decodeTP, decodePP, decodeEP, 140)
	}

	rankByID := make(map[string]inferencetopology.RuntimeTopologyRank, len(ranks))
	for _, r := range ranks {
		rankByID[r.ID] = r
	}
	for _, g := range groups {
		for i, leftID := range g.RankIDs {
			for _, rightID := range g.RankIDs[i+1:] {
				l := rankByID[leftID]
				r := rankByID[rightID]
				diffPP := l.PPStage - r.PPStage
				if diffPP < 0 {
					diffPP = -diffPP
				}
				required := l.PPStage == r.PPStage || (diffPP == 1 && l.TPRank == r.TPRank && l.EPRank == r.EPRank)
				if required {
					links = append(links, inferencetopology.RuntimeObservedLink{
						FromDeviceID:          l.DeviceID,
						FromDeviceIncarnation: l.DeviceIncarnation,
						ToDeviceID:            r.DeviceID,
						ToDeviceIncarnation:   r.DeviceIncarnation,
						Kind:                  "nvlink",
						State:                 "available",
						BytesPerSecond:        1e11,
					})
				}
			}
		}
	}

	return inferencetopology.RuntimeTopology{
		Schema: inferencetopology.RuntimeTopologySchema,
		ID:     "pd-role-graph-fixture",
		Configured: inferencetopology.RuntimeTopologyConfigured{
			Groups: groups,
			Ranks:  ranks,
		},
		Observed: inferencetopology.RuntimeTopologyObserved{
			At:         now.Add(-5 * time.Second).UTC().Format(time.RFC3339Nano),
			TTLSeconds: 60,
			Epoch:      11,
			Provenance: "sw-verified-role-graph-fixture",
			Visibility: visible,
			Devices:    devices,
			Links:      links,
		},
		Supported: support,
		Available: available,
	}
}

func testIntPtr(v int) *int { return &v }

// fak-test:runtime fast est=200ms lane=default
func TestNativePDRoleGraphStageKV(t *testing.T) {
	now := time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC)

	t.Run("valid_stage_rank_kv_grouping_and_seams", func(t *testing.T) {
		top := testRoleGraphTopology(now, 1, 1, 1, 2, 1, 2, 8)
		kvTransfers := 0
		ledger := NewRoleGraphReservationLedger()

		sameGeo := &RoleGraphKVGeometry{
			LayerStart:  0,
			LayerEnd:    4,
			KVFormat:    "paged-v1",
			KVPrecision: "f32",
			BlockTokens: 16,
			NumKVHeads:  2,
			HeadDim:     16,
			VHeadDim:    16,
			LayerGroup:  model.KVGroupFull.String(),
		}
		diffGeo := &RoleGraphKVGeometry{
			LayerStart:  0,
			LayerEnd:    4,
			KVFormat:    "paged-v1",
			KVPrecision: "f32",
			BlockTokens: 32,
			NumKVHeads:  2,
			HeadDim:     16,
			VHeadDim:    16,
			LayerGroup:  model.KVGroupSlidingWindow.String(),
		}
		stage1Geo := &RoleGraphKVGeometry{
			LayerStart:  4,
			LayerEnd:    8,
			KVFormat:    "paged-v1",
			KVPrecision: "f32",
			BlockTokens: 16,
			NumKVHeads:  2,
			HeadDim:     16,
			VHeadDim:    16,
			LayerGroup:  model.KVGroupFull.String(),
		}

		spec := PDRoleGraphSpec{
			Schema:   RoleGraphSchema,
			Topology: top,
			DecodeLanes: []RoleGraphDecodeLane{
				{LaneID: "lane-s0-e0-a", DecodeGroupID: "decode-0", PPStage: testIntPtr(0), EPRank: testIntPtr(0), RequestedBlocks: 2, KVGeometry: sameGeo},
				{LaneID: "lane-s0-e0-b", DecodeGroupID: "decode-0", PPStage: testIntPtr(0), EPRank: testIntPtr(0), RequestedBlocks: 2, KVGeometry: sameGeo},
				{LaneID: "lane-s0-e0-swa", DecodeGroupID: "decode-0", PPStage: testIntPtr(0), EPRank: testIntPtr(0), RequestedBlocks: 1, KVGeometry: diffGeo},
				{LaneID: "lane-s0-e1", DecodeGroupID: "decode-0", PPStage: testIntPtr(0), EPRank: testIntPtr(1), RequestedBlocks: 1, KVGeometry: sameGeo},
				{LaneID: "lane-s1-e0", DecodeGroupID: "decode-0", PPStage: testIntPtr(1), EPRank: testIntPtr(0), RequestedBlocks: 1, KVGeometry: stage1Geo},
			},
		}

		cfg := SyntheticConfig()
		cfg.NumLayers = 8
		compiled, err := CompilePDRoleGraph(context.Background(), spec, RoleGraphCompileOptions{
			Now:          now,
			ModelConfig:  &cfg,
			Ledger:       ledger,
			OnKVTransfer: func() { kvTransfers++ },
		})
		if err != nil {
			t.Fatalf("CompilePDRoleGraph: %v", err)
		}
		if kvTransfers != 0 || compiled.KVTransfersTriggered != 0 {
			t.Fatalf("dry-run compile must trigger zero KV transfers, got hook=%d field=%d", kvTransfers, compiled.KVTransfersTriggered)
		}
		if len(compiled.DecodeGroups) != 4 {
			t.Fatalf("DecodeGroups count = %d, want 4 distinct (stage, rank, KVGeometry) groups: %+v", len(compiled.DecodeGroups), compiled.DecodeGroups)
		}
		if compiled.EPDriver != "EPDecodeCoordinator" {
			t.Fatalf("EPDriver = %q, want EPDecodeCoordinator", compiled.EPDriver)
		}
		if err := compiled.ValidatePipelineStages(cfg); err != nil {
			t.Fatalf("ValidatePipelineStages: %v", err)
		}

		groupByLane := make(map[string]string, len(compiled.Assignments))
		for _, a := range compiled.Assignments {
			groupByLane[a.LaneID] = a.DecodeKVGroupKey
			if a.WaitState != RoleGraphWaitRemoteKV || a.WaitState.IsComputedProgress() {
				t.Fatalf("lane %q wait state = %q (progress=%v), want waiting_for_remote_kv non-progress", a.LaneID, a.WaitState, a.WaitState.IsComputedProgress())
			}
		}
		if groupByLane["lane-s0-e0-a"] != groupByLane["lane-s0-e0-b"] {
			t.Fatalf("identical (stage, rank, KVGeometry) lanes did not share a group: %q != %q", groupByLane["lane-s0-e0-a"], groupByLane["lane-s0-e0-b"])
		}
		for _, distinct := range [][2]string{
			{"lane-s0-e0-a", "lane-s0-e0-swa"},
			{"lane-s0-e0-a", "lane-s0-e1"},
			{"lane-s0-e0-a", "lane-s1-e0"},
			{"lane-s0-e0-swa", "lane-s0-e1"},
		} {
			if groupByLane[distinct[0]] == groupByLane[distinct[1]] {
				t.Fatalf("lanes %q and %q with distinct stage/rank/KV geometry shared group %q", distinct[0], distinct[1], groupByLane[distinct[0]])
			}
		}

		// Verify PipelineEngine seam for stage 0 and NativeScheduler.AdmitImported gating.
		var stage0Group RoleGraphDecodeGroup
		for _, dg := range compiled.DecodeGroups {
			if dg.PPStage == 0 && dg.EPRank == 0 && len(dg.LaneIDs) == 2 {
				stage0Group = dg
			}
		}
		if !slices.Equal(stage0Group.LaneIDs, []string{"lane-s0-e0-a", "lane-s0-e0-b"}) {
			t.Fatalf("stage0 shared group lanes = %v, want [lane-s0-e0-a lane-s0-e0-b]", stage0Group.LaneIDs)
		}
		synth := model.NewSynthetic(SyntheticConfig())
		stage, pipeEng, err := BuildDryRunPipelineEngineForGroup(stage0Group, synth, model.LocalTransport{})
		if err != nil || !stage.Spec.First || pipeEng == nil {
			t.Fatalf("BuildDryRunPipelineEngineForGroup: stage=%+v eng=%v err=%v", stage.Spec, pipeEng, err)
		}

		sched := NewNativeScheduler(synth)
		defer sched.Close()
		sess := synth.NewSession()
		prompt := []int{3, 7, 11}
		logits := sess.Prefill(prompt)

		// Waiting KV cannot be admitted into NativeScheduler.
		if _, err := AdmitCompiledDecodeLane(context.Background(), sched, nil, ImportedSequence{Session: sess, Logits: logits, Prompt: prompt}, compiled, "lane-s0-e0-a", *sameGeo, false, nil); !errors.Is(err, ErrRoleGraphLayoutUnsupported) {
			t.Fatalf("waiting KV admitted early: err=%v, want ErrRoleGraphLayoutUnsupported", err)
		}
		// Mismatched KV geometry is refused before AdmitImported.
		if _, err := AdmitCompiledDecodeLane(context.Background(), sched, nil, ImportedSequence{Session: sess, Logits: logits, Prompt: prompt}, compiled, "lane-s0-e0-a", *diffGeo, true, nil); !errors.Is(err, ErrRoleGraphLayoutUnsupported) {
			t.Fatalf("mismatched KV geometry admitted: err=%v, want ErrRoleGraphLayoutUnsupported", err)
		}
		// Complete compatible KV admits through NativeScheduler.AdmitImported.
		req, err := AdmitCompiledDecodeLane(context.Background(), sched, nil, ImportedSequence{Session: sess, Logits: logits, Prompt: prompt, Tool: "rg_test"}, compiled, "lane-s0-e0-a", *sameGeo, true, ledger)
		if err != nil {
			t.Fatalf("AdmitCompiledDecodeLane(ready): %v", err)
		}
		if got, want := drainPDRequest(t, req), serialGreedyTokens(synth, prompt); !reflect.DeepEqual(got, want) {
			t.Fatalf("AdmitCompiledDecodeLane tokens = %v, want %v", got, want)
		}
	})

	t.Run("mixed_unsupported_layouts_fail_before_transfer", func(t *testing.T) {
		cases := []struct {
			name string
			spec func() PDRoleGraphSpec
		}{
			{
				name: "tp_mismatch_between_p_and_d",
				spec: func() PDRoleGraphSpec {
					return PDRoleGraphSpec{Schema: RoleGraphSchema, Topology: testRoleGraphTopology(now, 1, 2, 1, 1, 1, 1, 8)}
				},
			},
			{
				name: "kv_format_mismatch",
				spec: func() PDRoleGraphSpec {
					top := testRoleGraphTopology(now, 1, 1, 1, 1, 1, 1, 8)
					top.Configured.Groups[1].KVFormat = "mla-v1"
					return PDRoleGraphSpec{Schema: RoleGraphSchema, Topology: top}
				},
			},
			{
				name: "layer_count_mismatch",
				spec: func() PDRoleGraphSpec {
					top := testRoleGraphTopology(now, 1, 1, 1, 1, 1, 1, 8)
					top.Configured.Groups[1].Layers = 12
					top.Configured.Ranks[1].LayerEnd = 12
					return PDRoleGraphSpec{Schema: RoleGraphSchema, Topology: top}
				},
			},
			{
				name: "second_ep_coordinator_prohibited",
				spec: func() PDRoleGraphSpec {
					return PDRoleGraphSpec{
						Schema:   RoleGraphSchema,
						Topology: testRoleGraphTopology(now, 1, 1, 1, 1, 1, 2, 8),
						EPDriver: "second-coordinator",
					}
				},
			},
			{
				name: "missing_decode_role",
				spec: func() PDRoleGraphSpec {
					return PDRoleGraphSpec{Schema: RoleGraphSchema, Topology: testRoleGraphTopology(now, 1, 1, 0, 1, 1, 1, 8)}
				},
			},
			{
				name: "stage_kv_head_dim_mismatch_p_vs_d",
				spec: func() PDRoleGraphSpec {
					top := testRoleGraphTopology(now, 1, 1, 1, 1, 1, 1, 8)
					return PDRoleGraphSpec{
						Schema:   RoleGraphSchema,
						Topology: top,
						StageKV: []RoleGraphRankKVSpec{
							{RankID: top.Configured.Ranks[0].ID, CapacityBlocks: 8, KVGeometry: RoleGraphKVGeometry{HeadDim: 16, NumKVHeads: 2}},
							{RankID: top.Configured.Ranks[1].ID, CapacityBlocks: 8, KVGeometry: RoleGraphKVGeometry{HeadDim: 32, NumKVHeads: 2}},
						},
					}
				},
			},
			{
				name: "stale_observation_epoch",
				spec: func() PDRoleGraphSpec {
					top := testRoleGraphTopology(now, 1, 1, 1, 1, 1, 1, 8)
					top.Observed.TTLSeconds = 1
					return PDRoleGraphSpec{Schema: RoleGraphSchema, Topology: top}
				},
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				kvTransfers := 0
				ledger := NewRoleGraphReservationLedger()
				_, err := CompilePDRoleGraph(context.Background(), tc.spec(), RoleGraphCompileOptions{
					Now:          now,
					Ledger:       ledger,
					OnKVTransfer: func() { kvTransfers++ },
				})
				if !errors.Is(err, ErrRoleGraphLayoutUnsupported) {
					t.Fatalf("%s: err=%v, want errors.Is(ErrRoleGraphLayoutUnsupported)", tc.name, err)
				}
				if kvTransfers != 0 {
					t.Fatalf("%s: triggered %d KV transfers on rejected layout, want 0", tc.name, kvTransfers)
				}
				if ledger.ReservationCount() != 0 || ledger.UsedBlocks() != 0 {
					t.Fatalf("%s: leaked reservations count=%d blocks=%d", tc.name, ledger.ReservationCount(), ledger.UsedBlocks())
				}
			})
		}
	})

	t.Run("rejection_cleanup_returns_reservations_to_baseline", func(t *testing.T) {
		top := testRoleGraphTopology(now, 1, 1, 1, 2, 1, 1, 8)
		rank0 := "decode-0-s0-t0-e0"
		rank1 := "decode-0-s1-t0-e0"

		ledger := NewRoleGraphReservationLedger()
		mgr0 := SelectNativeKVManager(10, NativeKVLayoutV1)
		ledger.SetRankManager(rank0, mgr0)
		// Seed a baseline reservation before compiling so cleanup must restore baseline, not zero blindly.
		if _, err := ledger.Acquire("baseline-owner", "baseline-lane", "decode-0", rank0, "baseline-group", 3, NativeKVLayoutV1); err != nil {
			t.Fatalf("seed baseline reservation: %v", err)
		}
		baselineCount := ledger.ReservationCount()
		baselineUsed := ledger.UsedBlocks()
		if baselineCount != 1 || baselineUsed != 3 {
			t.Fatalf("unexpected seeded baseline count=%d used=%d", baselineCount, baselineUsed)
		}

		spec := PDRoleGraphSpec{
			Schema:   RoleGraphSchema,
			Topology: top,
			StageKV: []RoleGraphRankKVSpec{
				{RankID: rank0, CapacityBlocks: 10},
				{RankID: rank1, CapacityBlocks: 2},
			},
			DecodeLanes: []RoleGraphDecodeLane{
				{LaneID: "lane-a-ok", DecodeGroupID: "decode-0", RankID: rank0, RequestedBlocks: 4},
				{LaneID: "lane-b-overflow", DecodeGroupID: "decode-0", RankID: rank1, RequestedBlocks: 5},
			},
		}

		_, err := CompilePDRoleGraph(context.Background(), spec, RoleGraphCompileOptions{
			Now:    now,
			Ledger: ledger,
		})
		if !errors.Is(err, ErrRoleGraphLayoutUnsupported) {
			t.Fatalf("expected capacity exhaustion rejection, got %v", err)
		}
		if ledger.ReservationCount() != baselineCount || ledger.UsedBlocks() != baselineUsed || mgr0.Used() != baselineUsed {
			t.Fatalf("rejection leaked reservations: count=%d (want %d), used=%d (want %d), mgr0=%d",
				ledger.ReservationCount(), baselineCount, ledger.UsedBlocks(), baselineUsed, mgr0.Used())
		}
	})

	t.Run("incumbent_retention_preserves_pinned_placements_deterministically", func(t *testing.T) {
		top := testRoleGraphTopology(now, 1, 1, 2, 1, 1, 1, 8)
		initial := PDRoleGraphSpec{
			Schema:   RoleGraphSchema,
			Topology: top,
			DecodeLanes: []RoleGraphDecodeLane{
				{LaneID: "lane-pinned", DecodeGroupID: "decode-1", RankID: "decode-1-s0-t0-e0", Pinned: true, RequestedBlocks: 1},
				{LaneID: "lane-flex", RequestedBlocks: 1},
			},
		}
		incumbent, err := CompilePDRoleGraph(context.Background(), initial, RoleGraphCompileOptions{Now: now})
		if err != nil {
			t.Fatalf("initial compile: %v", err)
		}

		// Permute discovery order in topology and recompile against incumbent.
		permuted := top
		permuted.Configured.Groups = slices.Clone(top.Configured.Groups)
		permuted.Configured.Ranks = slices.Clone(top.Configured.Ranks)
		slices.Reverse(permuted.Configured.Groups)
		slices.Reverse(permuted.Configured.Ranks)

		recompiledSpec := PDRoleGraphSpec{
			Schema:    RoleGraphSchema,
			Topology:  permuted,
			Incumbent: &incumbent,
			DecodeLanes: []RoleGraphDecodeLane{
				// Omit explicit DecodeGroupID/RankID on lane-pinned: incumbent retention must pin it back to decode-1.
				{LaneID: "lane-pinned", Pinned: true, RequestedBlocks: 1},
				{LaneID: "lane-flex", RequestedBlocks: 1},
			},
		}
		recompiledA, err := CompilePDRoleGraph(context.Background(), recompiledSpec, RoleGraphCompileOptions{Now: now})
		if err != nil {
			t.Fatalf("recompiledA: %v", err)
		}
		recompiledB, err := CompilePDRoleGraph(context.Background(), recompiledSpec, RoleGraphCompileOptions{Now: now})
		if err != nil {
			t.Fatalf("recompiledB: %v", err)
		}
		if !reflect.DeepEqual(recompiledA, recompiledB) {
			t.Fatalf("incumbent recompile is non-deterministic:\nA=%+v\nB=%+v", recompiledA, recompiledB)
		}
		var pinnedAssign RoleGraphLaneAssignment
		for _, a := range recompiledA.Assignments {
			if a.LaneID == "lane-pinned" {
				pinnedAssign = a
			}
		}
		if pinnedAssign.DecodeGroupID != "decode-1" || pinnedAssign.RankID != "decode-1-s0-t0-e0" || !pinnedAssign.Pinned || !pinnedAssign.IncumbentPreserved {
			t.Fatalf("pinned lane lost incumbent placement: %+v", pinnedAssign)
		}

		// If the pinned rank's device retires, compilation must fail closed rather than silently unpinning.
		retired := top
		retired.Observed.Devices = slices.Clone(top.Observed.Devices)
		for i := range retired.Observed.Devices {
			if retired.Observed.Devices[i].ID == "gpu-decode-1-s0-t0-e0" {
				retired.Observed.Devices[i].Lifecycle = "retired"
			}
		}
		ledger := NewRoleGraphReservationLedger()
		_, err = CompilePDRoleGraph(context.Background(), PDRoleGraphSpec{
			Schema:      RoleGraphSchema,
			Topology:    retired,
			Incumbent:   &incumbent,
			DecodeLanes: recompiledSpec.DecodeLanes,
		}, RoleGraphCompileOptions{Now: now, Ledger: ledger})
		if !errors.Is(err, ErrRoleGraphLayoutUnsupported) {
			t.Fatalf("retired pinned placement should fail closed with ErrRoleGraphLayoutUnsupported, got %v", err)
		}
		if ledger.ReservationCount() != 0 {
			t.Fatalf("failed pinned recompile leaked %d reservations", ledger.ReservationCount())
		}
	})
}
