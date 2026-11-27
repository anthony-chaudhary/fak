package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/inferencetopology"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

func buildServeRoleGraphFixture(now time.Time, prefillTP, decodePP, decodeTP, decodeEP int, layers int) inferencetopology.RuntimeTopology {
	var groups []inferencetopology.RuntimeTopologyGroup
	var ranks []inferencetopology.RuntimeTopologyRank
	var devices []inferencetopology.RuntimeObservedDevice
	var support []inferencetopology.RuntimeTopologySupport
	var available []inferencetopology.RuntimeTopologyAvailable
	var visible []string
	var links []inferencetopology.RuntimeObservedLink

	addGroup := func(id, role string, tp, pp, ep int, capBytes float64) {
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

	addGroup("prefill-0", inferencetopology.RuntimeRolePrefill, prefillTP, 1, 1, 120)
	addGroup("decode-0", inferencetopology.RuntimeRoleDecode, decodeTP, decodePP, decodeEP, 140)

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
				if l.PPStage == r.PPStage || (diffPP == 1 && l.TPRank == r.TPRank && l.EPRank == r.EPRank) {
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
		ID:     "serve-role-graph-fixture",
		Configured: inferencetopology.RuntimeTopologyConfigured{
			Groups: groups,
			Ranks:  ranks,
		},
		Observed: inferencetopology.RuntimeTopologyObserved{
			At:         now.Add(-5 * time.Second).UTC().Format(time.RFC3339Nano),
			TTLSeconds: 120,
			Epoch:      21,
			Provenance: "sw-verified-serve-fixture",
			Visibility: visible,
			Devices:    devices,
			Links:      links,
		},
		Supported: support,
		Available: available,
	}
}

func writeServeRoleGraphFixtureFile(t *testing.T, name string, spec modelengine.PDRoleGraphSpec) string {
	t.Helper()
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("marshal fixture %s: %v", name, err)
	}
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatalf("write fixture %s: %v", name, err)
	}
	return path
}

func serveIntPtr(v int) *int { return &v }

// fak-test:runtime fast est=2s lane=default
func TestServeRoleGraphPDClusteredDecode(t *testing.T) {
	if os.Getenv("FAK_TEST_SERVE_ROLE_GRAPH_HELPER") == "run" {
		cmdServe([]string{"--" + serveRoleGraphDryRunFlag, os.Getenv("FAK_TEST_SERVE_ROLE_GRAPH_PATH")})
		os.Exit(0)
	}

	now := time.Now().UTC()
	sameGeo := &modelengine.RoleGraphKVGeometry{
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
	swaGeo := &modelengine.RoleGraphKVGeometry{
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
	stage1Geo := &modelengine.RoleGraphKVGeometry{
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

	t.Run("valid_dry_run_prints_table_and_exits_0", func(t *testing.T) {
		validSpec := modelengine.PDRoleGraphSpec{
			Schema:   modelengine.RoleGraphSchema,
			Topology: buildServeRoleGraphFixture(now, 1, 2, 1, 2, 8),
			DecodeLanes: []modelengine.RoleGraphDecodeLane{
				{LaneID: "lane-0a", DecodeGroupID: "decode-0", PPStage: serveIntPtr(0), EPRank: serveIntPtr(0), RequestedBlocks: 2, KVGeometry: sameGeo},
				{LaneID: "lane-0b", DecodeGroupID: "decode-0", PPStage: serveIntPtr(0), EPRank: serveIntPtr(0), RequestedBlocks: 2, KVGeometry: sameGeo},
				{LaneID: "lane-0-swa", DecodeGroupID: "decode-0", PPStage: serveIntPtr(0), EPRank: serveIntPtr(0), RequestedBlocks: 1, KVGeometry: swaGeo},
				{LaneID: "lane-1", DecodeGroupID: "decode-0", PPStage: serveIntPtr(1), EPRank: serveIntPtr(0), RequestedBlocks: 1, KVGeometry: stage1Geo},
			},
		}
		path := writeServeRoleGraphFixtureFile(t, "valid-role-graph.json", validSpec)

		cmd := exec.Command(os.Args[0], "-test.run=^TestServeRoleGraphPDClusteredDecode$")
		cmd.Env = append(os.Environ(),
			"FAK_TEST_SERVE_ROLE_GRAPH_HELPER=run",
			"FAK_TEST_SERVE_ROLE_GRAPH_PATH="+path,
		)
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			t.Fatalf("fak serve --%s failed on valid fixture: %v\nstdout:\n%s\nstderr:\n%s",
				serveRoleGraphDryRunFlag, err, stdout.String(), stderr.String())
		}
		out := stdout.String()
		for _, want := range []string{
			"ROLE_GRAPH_DRY_RUN",
			"topology=serve-role-graph-fixture",
			"decode_kv_groups=3",
			"kv_transfers=0",
			"engine_started=false",
			"STAGE_KV_GROUPS",
			"LANE_ASSIGNMENTS",
			"lane-0a,lane-0b",
			"NativeScheduler.AdmitImported+NewPipelineEngine",
			"EPDecodeCoordinator:coordinator",
			"waiting_for_remote_kv",
		} {
			if !strings.Contains(out, want) {
				t.Fatalf("dry-run table missing %q:\n%s", want, out)
			}
		}

		kvTransfers := 0
		engineStarted := false
		var directOut bytes.Buffer
		compiled, err := executeServeRoleGraphDryRun(context.Background(), path, &directOut, nil, serveRoleGraphDryRunHooks{
			Now:           now,
			OnKVTransfer:  func() { kvTransfers++ },
			OnEngineStart: func() { engineStarted = true },
		})
		if err != nil {
			t.Fatalf("executeServeRoleGraphDryRun: %v", err)
		}
		if kvTransfers != 0 || engineStarted {
			t.Fatalf("dry-run triggered transfer=%d or engineStarted=%v", kvTransfers, engineStarted)
		}
		if len(compiled.DecodeGroups) != 3 {
			t.Fatalf("DecodeGroups=%d, want 3", len(compiled.DecodeGroups))
		}
	})

	t.Run("mixed_unsupported_layout_exits_nonzero_zero_transfer_no_engine", func(t *testing.T) {
		mixedSpec := modelengine.PDRoleGraphSpec{
			Schema:   modelengine.RoleGraphSchema,
			Topology: buildServeRoleGraphFixture(now, 2, 1, 1, 1, 8), // TP=2 prefill vs TP=1 decode
		}
		path := writeServeRoleGraphFixtureFile(t, "mixed-tp-role-graph.json", mixedSpec)

		cmd := exec.Command(os.Args[0], "-test.run=^TestServeRoleGraphPDClusteredDecode$")
		cmd.Env = append(os.Environ(),
			"FAK_TEST_SERVE_ROLE_GRAPH_HELPER=run",
			"FAK_TEST_SERVE_ROLE_GRAPH_PATH="+path,
		)
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		err := cmd.Run()
		if err == nil {
			t.Fatalf("fak serve --%s unexpectedly exited 0 on mixed TP layout\nstdout:\n%s", serveRoleGraphDryRunFlag, stdout.String())
		}
		if stdout.Len() != 0 {
			t.Fatalf("failed dry-run printed assignment table to stdout:\n%s", stdout.String())
		}
		if !strings.Contains(stderr.String(), modelengine.ErrRoleGraphLayoutUnsupported.Error()) {
			t.Fatalf("stderr missing typed sentinel %q:\n%s", modelengine.ErrRoleGraphLayoutUnsupported.Error(), stderr.String())
		}

		kvTransfers := 0
		engineStarted := false
		_, directErr := executeServeRoleGraphDryRun(context.Background(), path, &stdout, &stderr, serveRoleGraphDryRunHooks{
			Now:           now,
			OnKVTransfer:  func() { kvTransfers++ },
			OnEngineStart: func() { engineStarted = true },
		})
		if !errors.Is(directErr, modelengine.ErrRoleGraphLayoutUnsupported) {
			t.Fatalf("err=%v, want errors.Is(ErrRoleGraphLayoutUnsupported)", directErr)
		}
		if kvTransfers != 0 || engineStarted {
			t.Fatalf("rejected fixture must not transfer KV (%d) or start engine (%v)", kvTransfers, engineStarted)
		}
	})

	t.Run("cleanup_fixture_zero_residual_reservations", func(t *testing.T) {
		top := buildServeRoleGraphFixture(now, 1, 2, 1, 1, 8)
		rank0 := "decode-0-s0-t0-e0"
		rank1 := "decode-0-s1-t0-e0"
		spec := modelengine.PDRoleGraphSpec{
			Schema:   modelengine.RoleGraphSchema,
			Topology: top,
			StageKV: []modelengine.RoleGraphRankKVSpec{
				{RankID: rank0, CapacityBlocks: 8},
				{RankID: rank1, CapacityBlocks: 2},
			},
			DecodeLanes: []modelengine.RoleGraphDecodeLane{
				{LaneID: "lane-0-ok", DecodeGroupID: "decode-0", RankID: rank0, RequestedBlocks: 4},
				{LaneID: "lane-1-reject", DecodeGroupID: "decode-0", RankID: rank1, RequestedBlocks: 5},
			},
		}
		path := writeServeRoleGraphFixtureFile(t, "cleanup-role-graph.json", spec)

		ledger := modelengine.NewRoleGraphReservationLedger()
		kvTransfers := 0
		engineStarted := false
		var out, errBuf bytes.Buffer
		_, err := executeServeRoleGraphDryRun(context.Background(), path, &out, &errBuf, serveRoleGraphDryRunHooks{
			Now:           now,
			Ledger:        ledger,
			OnKVTransfer:  func() { kvTransfers++ },
			OnEngineStart: func() { engineStarted = true },
		})
		if !errors.Is(err, modelengine.ErrRoleGraphLayoutUnsupported) {
			t.Fatalf("expected ErrRoleGraphLayoutUnsupported, got %v", err)
		}
		if kvTransfers != 0 || engineStarted {
			t.Fatalf("rejected cleanup fixture triggered transfer=%d or engineStarted=%v", kvTransfers, engineStarted)
		}
		if ledger.ReservationCount() != 0 || ledger.UsedBlocks() != 0 {
			t.Fatalf("post-rejection residual reservations: count=%d used=%d, want 0/0",
				ledger.ReservationCount(), ledger.UsedBlocks())
		}
	})

	t.Run("default_serve_path_unchanged", func(t *testing.T) {
		fs, sf := newServeFlagSet()
		if err := fs.Parse(nil); err != nil {
			t.Fatalf("newServeFlagSet default Parse: %v", err)
		}
		if sf.roleGraphDryRun == nil || *sf.roleGraphDryRun != "" {
			t.Fatalf("default roleGraphDryRun = %v, want empty string", sf.roleGraphDryRun)
		}
		if *sf.engineID != "mock" || *sf.pdRole != "" {
			t.Fatalf("default serve flags drifted: engine=%q role=%q", *sf.engineID, *sf.pdRole)
		}
		var out, errBuf bytes.Buffer
		handled, err := runServeRoleGraphDryRunIfRequested(sf, &out, &errBuf)
		if handled || err != nil || out.Len() != 0 || errBuf.Len() != 0 {
			t.Fatalf("default path was altered: handled=%v err=%v out=%q err=%q",
				handled, err, out.String(), errBuf.String())
		}
	})
}
