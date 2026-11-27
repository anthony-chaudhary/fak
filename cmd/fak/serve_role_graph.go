package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

const serveRoleGraphDryRunFlag = "role-graph-dry-run"

const maxServeRoleGraphInputBytes = 16 << 20

type serveRoleGraphDryRunHooks struct {
	Now           time.Time
	Ledger        *modelengine.RoleGraphReservationLedger
	OnKVTransfer  func()
	OnEngineStart func()
}

func runServeRoleGraphDryRunIfRequested(sf *serveFlags, stdout, stderr io.Writer) (bool, error) {
	if sf == nil || sf.roleGraphDryRun == nil {
		return false, nil
	}
	path := strings.TrimSpace(*sf.roleGraphDryRun)
	if path == "" {
		return false, nil
	}
	if _, err := executeServeRoleGraphDryRun(context.Background(), path, stdout, stderr, serveRoleGraphDryRunHooks{}); err != nil {
		if stderr != nil {
			fmt.Fprintf(stderr, "fak serve: --%s: %v\n", serveRoleGraphDryRunFlag, err)
		}
		return true, err
	}
	return true, nil
}

func executeServeRoleGraphDryRun(ctx context.Context, path string, stdout, stderr io.Writer, hooks serveRoleGraphDryRunHooks) (modelengine.CompiledPDRoleGraph, error) {
	_ = stderr
	if path == "" {
		return modelengine.CompiledPDRoleGraph{}, fmt.Errorf("%w: empty role-graph path", modelengine.ErrRoleGraphLayoutUnsupported)
	}
	info, err := os.Stat(path)
	if err != nil {
		return modelengine.CompiledPDRoleGraph{}, fmt.Errorf("role-graph dry-run: stat %q: %w", path, err)
	}
	if !info.Mode().IsRegular() || info.Size() > maxServeRoleGraphInputBytes {
		return modelengine.CompiledPDRoleGraph{}, fmt.Errorf("role-graph dry-run: %q must be a regular file of at most 16 MiB", path)
	}
	f, err := os.Open(path)
	if err != nil {
		return modelengine.CompiledPDRoleGraph{}, fmt.Errorf("role-graph dry-run: open %q: %w", path, err)
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil {
		return modelengine.CompiledPDRoleGraph{}, fmt.Errorf("role-graph dry-run: stat opened %q: %w", path, err)
	}
	if !opened.Mode().IsRegular() || opened.Size() > maxServeRoleGraphInputBytes {
		return modelengine.CompiledPDRoleGraph{}, fmt.Errorf("role-graph dry-run: opened %q is not a bounded regular file", path)
	}
	spec, err := modelengine.DecodePDRoleGraph(f)
	if err != nil {
		return modelengine.CompiledPDRoleGraph{}, err
	}
	ledger := hooks.Ledger
	if ledger == nil {
		ledger = modelengine.NewRoleGraphReservationLedger()
	}
	compiled, err := modelengine.CompilePDRoleGraph(ctx, spec, modelengine.RoleGraphCompileOptions{
		Now:          hooks.Now,
		Ledger:       ledger,
		OnKVTransfer: hooks.OnKVTransfer,
	})
	if err != nil {
		return modelengine.CompiledPDRoleGraph{}, err
	}
	// Dry-run compile never transfers KV or starts a serving engine; release
	// dry-run reservations after printing the compiled assignment table.
	defer func() {
		_ = compiled.ReleaseReservations(ledger)
	}()
	if stdout != nil {
		if err := writeServeRoleGraphTable(stdout, compiled); err != nil {
			return modelengine.CompiledPDRoleGraph{}, err
		}
	}
	return compiled, nil
}

func writeServeRoleGraphTable(w io.Writer, compiled modelengine.CompiledPDRoleGraph) error {
	if _, err := fmt.Fprintf(
		w,
		"ROLE_GRAPH_DRY_RUN topology=%s epoch=%d evidence=%s prefill_groups=%d decode_kv_groups=%d lanes=%d ep_driver=%s kv_transfers=%d engine_started=false\n",
		compiled.TopologyID,
		compiled.ObservationEpoch,
		compiled.Evidence,
		len(compiled.PrefillGroups),
		len(compiled.DecodeGroups),
		len(compiled.Assignments),
		compiled.EPDriver,
		compiled.KVTransfersTriggered,
	); err != nil {
		return err
	}
	if _, err := fmt.Fprintln(w, "\nSTAGE_KV_GROUPS"); err != nil {
		return err
	}
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "GROUP_KEY\tDECODE_GROUP\tSTAGE\tLAYERS\tEP_RANK\tRANK_ID\tDEVICE\tKV_FORMAT\tLAYER_GROUP\tBLK_TOKS\tWAIT_STATE\tRESERVED_BLOCKS\tLANES\tENTRYPOINT\tEP_DRIVER")
	for _, g := range compiled.DecodeGroups {
		fmt.Fprintf(
			tw,
			"%s\t%s\t%d\t[%d,%d)\t%d\t%s\t%s\t%s\t%s\t%d\t%s\t%d\t%s\t%s+%s\t%s\n",
			g.GroupKey,
			g.DecodeGroupID,
			g.PPStage,
			g.StageSpec.Lo,
			g.StageSpec.Hi,
			g.EPRank,
			g.RankID,
			g.DeviceID,
			g.KVGeometry.KVFormat,
			g.KVGeometry.LayerGroup,
			g.KVGeometry.BlockTokens,
			g.WaitState,
			g.ReservedBlocks,
			strings.Join(g.LaneIDs, ","),
			g.AdmitEntrypoint,
			g.PipelineEntrypoint,
			g.EPDriver,
		)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	if _, err := fmt.Fprintln(w, "\nLANE_ASSIGNMENTS"); err != nil {
		return err
	}
	twLanes := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(twLanes, "LANE_ID\tPREFILL_GROUP\tDECODE_GROUP\tGROUP_KEY\tSTAGE\tRANK_ID\tWAIT_STATE\tRESERVED_BLOCKS\tPINNED\tINCUMBENT_PRESERVED")
	for _, a := range compiled.Assignments {
		fmt.Fprintf(
			twLanes,
			"%s\t%s\t%s\t%s\t%d\t%s\t%s\t%d\t%t\t%t\n",
			a.LaneID,
			a.PrefillGroupID,
			a.DecodeGroupID,
			a.DecodeKVGroupKey,
			a.PPStage,
			a.RankID,
			a.WaitState,
			a.ReservedBlocks,
			a.Pinned,
			a.IncumbentPreserved,
		)
	}
	return twLanes.Flush()
}
