package main

import (
	"fmt"
	"net"
	"os"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serve_ep.go — the SHARDED expert-parallel serve topology (#971/#1728). A sharded EP serve runs
// N SEPARATE processes of `fak serve --gguf X --expert-parallel N`, each holding ONLY its expert
// band (ggufload.WithExpertShard) so no single process holds the 466 GB GLM-5.2 expert set. The
// processes are differentiated by env (the torchrun/NCCL convention — the launch command is
// identical across ranks, only the env differs):
//
//   - FAK_EP_RANK       this process's rank in [0,N).            (unset => rank 0)
//   - FAK_EP_COORD_ADDR host:port rank 0 binds / ranks>0 dial.   (unset with N>1 => the existing
//                       single-process all-band path — no sharding, byte-identical to today)
//
// The per-rank [H] routed partials are reduced across the group through a real cross-process
// collective (model.DistComm over a TCP star rooted at rank 0), reduced in rank order — the HOST
// rung the device-NCCL tensor rung stands on. This proves per-band residency + correct distributed
// tokens; it does NOT claim multi-GPU (DistComm reduces host float32).

// epRankConfig is this process's resolved place in a sharded EP serve. sharded is true only when a
// real multi-process group is requested (ranks>1 AND a coordinator address): only then does the
// serve shard the load, join the group, and take the rank-local forward. Otherwise the serve keeps
// its existing single-process behavior exactly.
type epRankConfig struct {
	ranks     int    // world size (--expert-parallel N)
	rank      int    // this process's rank in [0,ranks)
	coordAddr string // rank 0 binds it; ranks>0 dial it
	sharded   bool   // ranks>1 && coordAddr != ""
}

// resolveEPRankConfig reads FAK_EP_RANK / FAK_EP_COORD_ADDR against the --expert-parallel world
// size and returns this process's place, failing closed on a misconfig. It is deliberately
// conservative: a rank index or a coordinator address supplied WITHOUT a real world (ranks<=1) is
// an error (a rank with no group is meaningless), and an out-of-range rank is an error. When it
// returns sharded=false, the serve is byte-identical to a non-EP serve (the setter is never
// called, so the forward stays on the all-band path).
func resolveEPRankConfig(ranks int) (epRankConfig, error) {
	cfg := epRankConfig{ranks: ranks}
	rankStr := os.Getenv("FAK_EP_RANK")
	cfg.coordAddr = os.Getenv("FAK_EP_COORD_ADDR")
	if rankStr != "" {
		r, err := strconv.Atoi(rankStr)
		if err != nil {
			return epRankConfig{}, fmt.Errorf("FAK_EP_RANK=%q is not an integer: %w", rankStr, err)
		}
		cfg.rank = r
	}
	// A rank index or a coordinator address only means something inside a real world.
	if ranks <= 1 {
		if rankStr != "" && cfg.rank != 0 {
			return epRankConfig{}, fmt.Errorf("FAK_EP_RANK=%d set but --expert-parallel=%d (a rank index needs a world of >1)", cfg.rank, ranks)
		}
		if cfg.coordAddr != "" {
			return epRankConfig{}, fmt.Errorf("FAK_EP_COORD_ADDR=%q set but --expert-parallel=%d (a process group needs a world of >1)", cfg.coordAddr, ranks)
		}
		return cfg, nil // ranks<=1, no group — the plain single-process serve
	}
	if cfg.rank < 0 || cfg.rank >= ranks {
		return epRankConfig{}, fmt.Errorf("FAK_EP_RANK=%d outside [0,%d) for --expert-parallel=%d", cfg.rank, ranks, ranks)
	}
	cfg.sharded = cfg.coordAddr != ""
	return cfg, nil
}

// expertShardForConfig returns this rank's routed-expert band [Lo,Hi) for the sharded load, or nil
// when the serve is not sharded (the full model loads, as today). numExperts comes from the GGUF
// header (ggufNumExperts) so the shard is known BEFORE the load reads a tensor byte.
func expertShardForConfig(cfg epRankConfig, numExperts int) (*ggufload.ExpertShard, error) {
	if !cfg.sharded {
		return nil, nil
	}
	shard, err := ggufload.ExpertShardForRank(numExperts, cfg.ranks, cfg.rank)
	if err != nil {
		return nil, fmt.Errorf("expert-parallel shard for rank %d of %d over %d experts: %w", cfg.rank, cfg.ranks, numExperts, err)
	}
	return &shard, nil
}

// ggufNumExperts reads NumExperts from a GGUF's header WITHOUT loading a tensor (OpenWeights parses
// only the header). It is how the sharded serve sizes the per-rank band before the load.
func ggufNumExperts(ggufPath string) (int, error) {
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		return 0, err
	}
	defer ws.Close()
	fileCfg, err := ws.File.Config()
	if err != nil {
		return 0, err
	}
	return fileCfg.NumExperts, nil
}

// dialEPGroup forms this rank's handle to the DistComm process group: rank 0 listens on coordAddr
// and coordinates; ranks>0 dial it and join. It MUST be called AFTER a successful model load and
// BEFORE binding the HTTP listener — Coordinate/Join have no accept timeout, so a load that failed
// on one rank must not leave its peers blocked on a group that never completes. The caller owns the
// returned group's lifecycle (Close on serve shutdown). It is only called when cfg.sharded.
func dialEPGroup(cfg epRankConfig) (*fakmodel.DistComm, error) {
	if !cfg.sharded {
		return nil, fmt.Errorf("dialEPGroup called on a non-sharded config")
	}
	if cfg.rank == 0 {
		ln, err := net.Listen("tcp", cfg.coordAddr)
		if err != nil {
			return nil, fmt.Errorf("expert-parallel coordinator listen on %q: %w", cfg.coordAddr, err)
		}
		g, err := fakmodel.Coordinate(ln, cfg.ranks)
		if err != nil {
			ln.Close()
			return nil, fmt.Errorf("expert-parallel coordinate %d ranks: %w", cfg.ranks, err)
		}
		// Coordinate has accepted every worker; the listener is no longer needed.
		ln.Close()
		return g, nil
	}
	conn, err := net.Dial("tcp", cfg.coordAddr)
	if err != nil {
		return nil, fmt.Errorf("expert-parallel rank %d dial coordinator %q: %w", cfg.rank, cfg.coordAddr, err)
	}
	g, err := fakmodel.Join(conn, cfg.rank, cfg.ranks)
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("expert-parallel rank %d join: %w", cfg.rank, err)
	}
	return g, nil
}

// joinDevicePGIfSupported is the opt-in upgrade over the host DistComm reduce: on a backend that
// implements compute.ProcessGroupBackend (the multi-process NCCL bootstrap, -tags cuda,nccl on a
// real device), form the process-group communicator and return a model.Collective that reduces
// through it. On any other build (be is nil, or does not implement the seam) it returns
// (nil, nil) — NOT an error — so the caller's existing NewDistCommCollective(group) path is
// unchanged; every non-cuda/nccl build takes this branch today.
//
// The unique-ID rendezvous reuses `group`, the ALREADY-OPEN DistComm this rank joined via
// dialEPGroup — no new listener/socket. Only rank 0 mints the ID (ProcessGroupUniqueID); every
// rank (including rank 0) then calls BroadcastFromRoot together, exactly like every other
// DistComm collective, and joins with its own rank/device via InitProcessGroup. device is fixed
// at 0: the sharded-EP topology is one GPU visible per process (CUDA_VISIBLE_DEVICES), so device
// 0 is always this process's own GPU, never a peer's.
//
// STATUS: unverified end-to-end — compute.ProcessGroupBackend has no implementation reachable
// on a GPU-free host (cuda_collective_pg.go builds only under -tags cuda,nccl), so this path has
// never run against real GPUs. See dist_device_collective.go's honesty note.
func joinDevicePGIfSupported(be compute.Backend, group *fakmodel.DistComm, cfg epRankConfig) (fakmodel.Collective, error) {
	if be == nil {
		return nil, nil
	}
	pg, ok := be.(compute.ProcessGroupBackend)
	if !ok {
		return nil, nil
	}
	var id []byte
	var err error
	if cfg.rank == 0 {
		id, err = pg.ProcessGroupUniqueID()
		if err != nil {
			return nil, fmt.Errorf("mint NCCL process-group unique id: %w", err)
		}
	}
	id, err = group.BroadcastFromRoot(id)
	if err != nil {
		return nil, fmt.Errorf("broadcast NCCL process-group unique id to %d ranks: %w", cfg.ranks, err)
	}
	const device = 0
	if err := pg.InitProcessGroup(id, cfg.ranks, cfg.rank, device); err != nil {
		return nil, fmt.Errorf("join %d-rank NCCL process group as rank %d: %w", cfg.ranks, cfg.rank, err)
	}
	coll, err := fakmodel.NewDevicePGCollective(be)
	if err != nil {
		return nil, err
	}
	return coll, nil
}
