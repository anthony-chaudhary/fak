package modelengine

// native_pd_role_graph.go — dry-run role-graph compiler composing native P/D
// (NativePDRole / NativeScheduler.AdmitImported), pipeline stages
// (model.PartitionPlan / NewPipelineEngine), and the existing EP decode
// coordinator over an epoch-bound inferencetopology.RuntimeTopology.
//
// Concept adaptation (Apache-2.0):
//   - vllm-project/vllm (vllm/v1/core/kv_cache_planning.py:1337-1383,1589-1634
//     and vllm/v1/core/sched/scheduler.py:1383-1492 @e7473bf6862289f0647d1f4b01498288b181e063):
//     group decode lanes by actual (stage, rank) KV geometry, never merge incompatible
//     KV geometries, reserve receiver capacity before remote KV load, and treat
//     waiting-for-remote-KV as non-progress until complete compatible stage/rank KV arrives.
//   - deepseek-ai/DeepInfer (csrc/engine/scheduler.hpp:288-340
//     @9bd9cf243fc53ef9afbf8f04d391d989117d6ab9):
//     separate prefill/decode role branches.
//
// Clean-room Go implementation; no upstream code copied. Does not implement or
// duplicate native-pd-kv-handshake-fence; dry-run / [SW-VERIFIED] fixtures only.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/inferencetopology"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	RoleGraphSchema         = "fak-pd-role-graph/1"
	RoleGraphCompiledSchema = "fak-pd-role-graph-compiled/1"
)

// RoleGraphWaitState mirrors vLLM's explicit remote-KV-wait state: waiting for
// remote KV reserves receiver capacity but is never counted as computed-token progress.
type RoleGraphWaitState string

const (
	RoleGraphWaitRemoteKV RoleGraphWaitState = "waiting_for_remote_kv"
	RoleGraphWaitReady    RoleGraphWaitState = "ready"
)

// IsComputedProgress reports whether a lane in this wait state has made
// computed-token progress. Waiting KV is never computed-token progress.
func (s RoleGraphWaitState) IsComputedProgress() bool {
	return s == RoleGraphWaitReady
}

// ErrRoleGraphLayoutUnsupported marks a mixed or unsupported P/D, TP/PP/EP,
// coordinator, capacity, pin, or stage/rank KV geometry layout refused before
// any KV transfer. Callers may use errors.Is.
var ErrRoleGraphLayoutUnsupported = errors.New("modelengine: role graph layout unsupported")

// RoleGraphKVGeometry captures the exact (stage, rank) KV memory geometry used
// to partition decode lanes into compatible groups.
type RoleGraphKVGeometry struct {
	PPStage     int    `json:"pp_stage"`
	TPRank      int    `json:"tp_rank"`
	EPRank      int    `json:"ep_rank"`
	LayerStart  int    `json:"layer_start"`
	LayerEnd    int    `json:"layer_end"`
	KVFormat    string `json:"kv_format"`
	KVPrecision string `json:"kv_precision,omitempty"`
	BlockTokens int    `json:"block_tokens,omitempty"`
	NumKVHeads  int    `json:"num_kv_heads,omitempty"`
	HeadDim     int    `json:"head_dim,omitempty"`
	VHeadDim    int    `json:"v_head_dim,omitempty"`
	KVLoraRank  int    `json:"kv_lora_rank,omitempty"`
	LayerGroup  string `json:"layer_group,omitempty"`
}

// GeometryKey returns a canonical identity string for this KV geometry.
func (g RoleGraphKVGeometry) GeometryKey() string {
	return fmt.Sprintf(
		"fmt=%s|prec=%s|blk=%d|kvh=%d|hd=%d|vhd=%d|lora=%d|lg=%s|layers=[%d,%d)",
		g.KVFormat, g.KVPrecision, g.BlockTokens, g.NumKVHeads, g.HeadDim, g.VHeadDim, g.KVLoraRank, g.LayerGroup, g.LayerStart, g.LayerEnd,
	)
}

// DecodeGroupKey returns the canonical grouping key combining (group, stage, rank)
// and KV geometry so two lanes with different KV geometry or stage/rank never share a group.
func (g RoleGraphKVGeometry) DecodeGroupKey(decodeGroupID string) string {
	raw := fmt.Sprintf("dg=%s|pp=%d|tp=%d|ep=%d|%s", decodeGroupID, g.PPStage, g.TPRank, g.EPRank, g.GeometryKey())
	sum := sha256.Sum256([]byte(raw))
	return fmt.Sprintf("%s:s%d:t%d:e%d:%s", decodeGroupID, g.PPStage, g.TPRank, g.EPRank, hex.EncodeToString(sum[:6]))
}

// RoleGraphRankKVSpec supplies optional per-rank KV geometry and block capacity.
type RoleGraphRankKVSpec struct {
	RankID         string              `json:"rank_id"`
	CapacityBlocks int                 `json:"capacity_blocks,omitempty"`
	KVGeometry     RoleGraphKVGeometry `json:"kv_geometry"`
}

// RoleGraphDecodeLane describes one requested decode lane in the role graph.
type RoleGraphDecodeLane struct {
	LaneID          string               `json:"lane_id"`
	PrefillGroupID  string               `json:"prefill_group_id,omitempty"`
	DecodeGroupID   string               `json:"decode_group_id,omitempty"`
	RankID          string               `json:"rank_id,omitempty"`
	PPStage         *int                 `json:"pp_stage,omitempty"`
	TPRank          *int                 `json:"tp_rank,omitempty"`
	EPRank          *int                 `json:"ep_rank,omitempty"`
	RequestedBlocks int                  `json:"requested_blocks,omitempty"`
	Pinned          bool                 `json:"pinned,omitempty"`
	KVGeometry      *RoleGraphKVGeometry `json:"kv_geometry,omitempty"`
}

// RoleGraphPinnedPlace pins a lane to a concrete incumbent decode rank/stage.
type RoleGraphPinnedPlace struct {
	DecodeGroupID string `json:"decode_group_id"`
	RankID        string `json:"rank_id"`
	PPStage       int    `json:"pp_stage"`
	TPRank        int    `json:"tp_rank"`
	EPRank        int    `json:"ep_rank"`
}

// PDRoleGraphSpec is the bounded input document compiled by CompilePDRoleGraph.
type PDRoleGraphSpec struct {
	Schema           string                            `json:"schema"`
	Topology         inferencetopology.RuntimeTopology `json:"topology"`
	EvaluatedAt      string                            `json:"evaluated_at,omitempty"`
	StageKV          []RoleGraphRankKVSpec             `json:"stage_kv,omitempty"`
	DecodeLanes      []RoleGraphDecodeLane             `json:"decode_lanes,omitempty"`
	PinnedPlacements map[string]RoleGraphPinnedPlace   `json:"pinned_placements,omitempty"`
	Incumbent        *CompiledPDRoleGraph              `json:"incumbent,omitempty"`
	EPDriver         string                            `json:"ep_driver,omitempty"`
}

// RoleGraphPrefillBinding describes an eligible prefill group in the compiled graph.
type RoleGraphPrefillBinding struct {
	GroupID   string       `json:"group_id"`
	Role      NativePDRole `json:"role"`
	TP        int          `json:"tp"`
	PP        int          `json:"pp"`
	EP        int          `json:"ep"`
	Layers    int          `json:"layers"`
	KVFormat  string       `json:"kv_format"`
	RankIDs   []string     `json:"rank_ids"`
	DeviceIDs []string     `json:"device_ids"`
}

// RoleGraphDecodeGroup holds decode lanes grouped by identical (stage, rank) KV geometry.
type RoleGraphDecodeGroup struct {
	GroupKey           string              `json:"group_key"`
	DecodeGroupID      string              `json:"decode_group_id"`
	Role               NativePDRole        `json:"role"`
	PPStage            int                 `json:"pp_stage"`
	TPRank             int                 `json:"tp_rank"`
	EPRank             int                 `json:"ep_rank"`
	RankID             string              `json:"rank_id"`
	DeviceID           string              `json:"device_id"`
	StageSpec          model.StageSpec     `json:"stage_spec"`
	KVGeometry         RoleGraphKVGeometry `json:"kv_geometry"`
	CapacityBlocks     int                 `json:"capacity_blocks"`
	ReservedBlocks     int                 `json:"reserved_blocks"`
	LaneIDs            []string            `json:"lane_ids"`
	WaitState          RoleGraphWaitState  `json:"wait_state"`
	AdmitEntrypoint    string              `json:"admit_entrypoint"`
	PipelineEntrypoint string              `json:"pipeline_entrypoint"`
	EPDriver           string              `json:"ep_driver,omitempty"`
}

// RoleGraphLaneAssignment is one lane's compiled P->D placement and reservation.
type RoleGraphLaneAssignment struct {
	LaneID             string              `json:"lane_id"`
	PrefillGroupID     string              `json:"prefill_group_id"`
	DecodeGroupID      string              `json:"decode_group_id"`
	DecodeKVGroupKey   string              `json:"decode_kv_group_key"`
	PPStage            int                 `json:"pp_stage"`
	TPRank             int                 `json:"tp_rank"`
	EPRank             int                 `json:"ep_rank"`
	RankID             string              `json:"rank_id"`
	DeviceID           string              `json:"device_id"`
	StageSpec          model.StageSpec     `json:"stage_spec"`
	KVGeometry         RoleGraphKVGeometry `json:"kv_geometry"`
	ReservedBlocks     int                 `json:"reserved_blocks"`
	WaitState          RoleGraphWaitState  `json:"wait_state"`
	Pinned             bool                `json:"pinned"`
	IncumbentPreserved bool                `json:"incumbent_preserved"`
}

// RoleGraphReservation records one active receiver-capacity reservation in the ledger.
type RoleGraphReservation struct {
	Owner         string                `json:"owner"`
	LaneID        string                `json:"lane_id"`
	DecodeGroupID string                `json:"decode_group_id"`
	RankID        string                `json:"rank_id"`
	GroupKey      string                `json:"group_key"`
	Blocks        int                   `json:"blocks"`
	LayoutVersion NativeKVLayoutVersion `json:"layout_version"`
	State         RoleGraphWaitState    `json:"state"`
}

// CompiledPDRoleGraph is the deterministic output of dry-run role-graph compilation.
type CompiledPDRoleGraph struct {
	Schema               string                         `json:"schema"`
	Evidence             string                         `json:"evidence"`
	TopologyID           string                         `json:"topology_id"`
	ObservationEpoch     uint64                         `json:"observation_epoch"`
	PrefillGroups        []RoleGraphPrefillBinding      `json:"prefill_groups"`
	DecodeGroups         []RoleGraphDecodeGroup         `json:"decode_groups"`
	PipelinePlans        map[string]model.PartitionPlan `json:"pipeline_plans"`
	Assignments          []RoleGraphLaneAssignment      `json:"assignments"`
	Reservations         []RoleGraphReservation         `json:"reservations"`
	EPDriver             string                         `json:"ep_driver"`
	KVTransfersTriggered int                            `json:"kv_transfers_triggered"`
}

// ReleaseReservations releases every reservation held by this compiled graph from ledger.
func (c *CompiledPDRoleGraph) ReleaseReservations(ledger *RoleGraphReservationLedger) error {
	if c == nil || ledger == nil {
		return nil
	}
	var firstErr error
	for _, res := range c.Reservations {
		if err := ledger.Release(res.Owner); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// ValidatePipelineStages checks all compiled decode pipeline plans against cfg.
func (c *CompiledPDRoleGraph) ValidatePipelineStages(cfg model.Config) error {
	if c == nil {
		return fmt.Errorf("%w: nil compiled role graph", ErrRoleGraphLayoutUnsupported)
	}
	ids := make([]string, 0, len(c.PipelinePlans))
	for id := range c.PipelinePlans {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		plan := c.PipelinePlans[id]
		if err := plan.Validate(cfg); err != nil {
			return fmt.Errorf("%w: pipeline group %q: %v", ErrRoleGraphLayoutUnsupported, id, err)
		}
	}
	return nil
}

// BuildStageAssignments converts a decode group's compiled stage spec into a
// model.PipelineStage, and for stage 0 constructs a dry-run PipelineEngine handle
// against downstream without starting workers or network I/O.
func BuildDryRunPipelineEngineForGroup(group RoleGraphDecodeGroup, stageModel *model.Model, downstream model.StageTransport) (model.PipelineStage, *PipelineEngine, error) {
	stage := model.PipelineStage{Spec: group.StageSpec, Model: stageModel}
	if !group.StageSpec.First {
		return stage, nil, nil
	}
	if stageModel == nil {
		return stage, nil, fmt.Errorf("%w: stage 0 requires a model to attach PipelineEngine", ErrRoleGraphLayoutUnsupported)
	}
	return stage, NewPipelineEngine(stage, downstream), nil
}

// RoleGraphReservationLedger coordinates per-rank NativeKVManager block pools
// and tracks active scheduler reservations so failed compiles roll back cleanly.
type RoleGraphReservationLedger struct {
	mu           sync.Mutex
	managers     map[string]NativeKVManager
	reservations map[string]RoleGraphReservation
}

// NewRoleGraphReservationLedger returns an empty reservation ledger.
func NewRoleGraphReservationLedger() *RoleGraphReservationLedger {
	return &RoleGraphReservationLedger{
		managers:     make(map[string]NativeKVManager),
		reservations: make(map[string]RoleGraphReservation),
	}
}

// SetRankManager binds an explicit NativeKVManager for rankID.
func (l *RoleGraphReservationLedger) SetRankManager(rankID string, mgr NativeKVManager) {
	if l == nil || rankID == "" || mgr == nil {
		return
	}
	l.mu.Lock()
	l.managers[rankID] = mgr
	l.mu.Unlock()
}

func (l *RoleGraphReservationLedger) ensureRankCapacity(rankID string, capacity int, layout NativeKVLayoutVersion) (NativeKVManager, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if mgr, ok := l.managers[rankID]; ok {
		if mgr.Layout() != layout {
			return nil, fmt.Errorf("%w: rank %q layout mismatch", ErrRoleGraphLayoutUnsupported, rankID)
		}
		return mgr, nil
	}
	if capacity < 0 {
		capacity = 0
	}
	mgr := SelectNativeKVManager(capacity, layout)
	l.managers[rankID] = mgr
	return mgr, nil
}

// Acquire reserves blocks for owner on rankID under layout.
func (l *RoleGraphReservationLedger) Acquire(owner, laneID, decodeGroupID, rankID, groupKey string, blocks int, layout NativeKVLayoutVersion) (RoleGraphReservation, error) {
	if l == nil {
		return RoleGraphReservation{}, errors.New("modelengine: nil role-graph reservation ledger")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if owner == "" {
		return RoleGraphReservation{}, fmt.Errorf("%w: empty reservation owner", ErrRoleGraphLayoutUnsupported)
	}
	if _, dup := l.reservations[owner]; dup {
		return RoleGraphReservation{}, fmt.Errorf("%w: duplicate reservation owner %q", ErrRoleGraphLayoutUnsupported, owner)
	}
	mgr, ok := l.managers[rankID]
	if !ok {
		return RoleGraphReservation{}, fmt.Errorf("%w: rank %q has no KV manager", ErrRoleGraphLayoutUnsupported, rankID)
	}
	if _, gotLayout, err := mgr.Acquire(owner, blocks, layout); err != nil {
		return RoleGraphReservation{}, fmt.Errorf("%w: reserve %d blocks on rank %q (layout=%d): %v", ErrRoleGraphLayoutUnsupported, blocks, rankID, gotLayout, err)
	}
	res := RoleGraphReservation{
		Owner:         owner,
		LaneID:        laneID,
		DecodeGroupID: decodeGroupID,
		RankID:        rankID,
		GroupKey:      groupKey,
		Blocks:        blocks,
		LayoutVersion: layout,
		State:         RoleGraphWaitRemoteKV,
	}
	l.reservations[owner] = res
	return res, nil
}

// Release releases owner's reservation back to its rank KV manager. Releasing
// an unknown owner is an idempotent no-op.
func (l *RoleGraphReservationLedger) Release(owner string) error {
	if l == nil || owner == "" {
		return nil
	}
	l.mu.Lock()
	res, ok := l.reservations[owner]
	if !ok {
		l.mu.Unlock()
		return nil
	}
	mgr := l.managers[res.RankID]
	if mgr != nil {
		if err := mgr.Release(owner, res.LayoutVersion); err != nil {
			l.mu.Unlock()
			return err
		}
	}
	delete(l.reservations, owner)
	l.mu.Unlock()
	return nil
}

// ReservationCount returns the number of active role-graph reservations.
func (l *RoleGraphReservationLedger) ReservationCount() int {
	if l == nil {
		return 0
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.reservations)
}

// UsedBlocks returns total used KV blocks across all managed ranks.
func (l *RoleGraphReservationLedger) UsedBlocks() int {
	if l == nil {
		return 0
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	total := 0
	for _, mgr := range l.managers {
		total += mgr.Used()
	}
	return total
}

// RoleGraphCompileOptions controls dry-run compilation and injection hooks.
type RoleGraphCompileOptions struct {
	Now          time.Time
	ModelConfig  *model.Config
	Ledger       *RoleGraphReservationLedger
	Incumbent    *CompiledPDRoleGraph
	OnKVTransfer func()
	// PreCommitHook runs after capacity reservations are acquired and before commit;
	// returning an error triggers atomic rollback of all reservations.
	PreCommitHook func(*CompiledPDRoleGraph) error
}

// DecodePDRoleGraph decodes one bounded, strict role-graph specification JSON document.
func DecodePDRoleGraph(r io.Reader) (PDRoleGraphSpec, error) {
	var spec PDRoleGraphSpec
	raw, err := io.ReadAll(io.LimitReader(r, (16<<20)+1))
	if err != nil || len(raw) > 16<<20 {
		return spec, fmt.Errorf("%w: role graph read or size limit exceeded", ErrRoleGraphLayoutUnsupported)
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&spec); err != nil {
		return spec, fmt.Errorf("%w: decode role graph: %v", ErrRoleGraphLayoutUnsupported, err)
	}
	var extra any
	if d.Decode(&extra) != io.EOF {
		return spec, fmt.Errorf("%w: trailing JSON in role graph", ErrRoleGraphLayoutUnsupported)
	}
	return spec, nil
}

// CompilePDRoleGraph validates topology eligibility, checks P/D/PP/TP/EP and
// stage/rank KV geometry compatibility before any KV transfer, reserves capacity,
// preserves pinned incumbent placements deterministically, and rolls back all
// admitted reservations on any rejection.
func CompilePDRoleGraph(ctx context.Context, spec PDRoleGraphSpec, opts RoleGraphCompileOptions) (CompiledPDRoleGraph, error) {
	if err := ctx.Err(); err != nil {
		return CompiledPDRoleGraph{}, fmt.Errorf("%w: %v", ErrRoleGraphLayoutUnsupported, err)
	}
	if spec.Schema != RoleGraphSchema {
		return CompiledPDRoleGraph{}, fmt.Errorf("%w: schema %q != %q", ErrRoleGraphLayoutUnsupported, spec.Schema, RoleGraphSchema)
	}

	epDriver, err := normalizeEPDriver(spec.EPDriver)
	if err != nil {
		return CompiledPDRoleGraph{}, err
	}

	now := opts.Now
	if now.IsZero() && spec.EvaluatedAt != "" {
		parsed, perr := time.Parse(time.RFC3339Nano, spec.EvaluatedAt)
		if perr != nil {
			return CompiledPDRoleGraph{}, fmt.Errorf("%w: invalid evaluated_at %q: %v", ErrRoleGraphLayoutUnsupported, spec.EvaluatedAt, perr)
		}
		now = parsed
	}
	if now.IsZero() {
		now = time.Now()
	}

	if _, err := inferencetopology.ValidateRuntimeTopology(spec.Topology, now); err != nil {
		return CompiledPDRoleGraph{}, wrapLayoutAndCause(err)
	}

	deviceFreeMap := make(map[string]float64, len(spec.Topology.Observed.Devices))
	for _, d := range spec.Topology.Observed.Devices {
		deviceFreeMap[d.ID] = d.FreeBytes
	}

	stageKVByRank := make(map[string]RoleGraphRankKVSpec, len(spec.StageKV))
	for _, skv := range spec.StageKV {
		if skv.RankID == "" {
			return CompiledPDRoleGraph{}, fmt.Errorf("%w: stage_kv entry missing rank_id", ErrRoleGraphLayoutUnsupported)
		}
		if _, dup := stageKVByRank[skv.RankID]; dup {
			return CompiledPDRoleGraph{}, fmt.Errorf("%w: duplicate stage_kv for rank %q", ErrRoleGraphLayoutUnsupported, skv.RankID)
		}
		stageKVByRank[skv.RankID] = skv
	}

	groupsByID := make(map[string]inferencetopology.RuntimeTopologyGroup, len(spec.Topology.Configured.Groups))
	ranksByID := make(map[string]inferencetopology.RuntimeTopologyRank, len(spec.Topology.Configured.Ranks))
	ranksByGroup := make(map[string][]inferencetopology.RuntimeTopologyRank, len(spec.Topology.Configured.Groups))
	for _, g := range spec.Topology.Configured.Groups {
		groupsByID[g.ID] = g
	}
	for _, r := range spec.Topology.Configured.Ranks {
		ranksByID[r.ID] = r
		ranksByGroup[r.GroupID] = append(ranksByGroup[r.GroupID], r)
	}
	for _, skv := range spec.StageKV {
		if _, ok := ranksByID[skv.RankID]; !ok {
			return CompiledPDRoleGraph{}, fmt.Errorf("%w: stage_kv references unknown rank %q", ErrRoleGraphLayoutUnsupported, skv.RankID)
		}
	}

	var prefillGroups []inferencetopology.RuntimeTopologyGroup
	var decodeGroups []inferencetopology.RuntimeTopologyGroup
	for _, g := range spec.Topology.Configured.Groups {
		switch g.Role {
		case inferencetopology.RuntimeRolePrefill:
			prefillGroups = append(prefillGroups, g)
		case inferencetopology.RuntimeRoleDecode:
			decodeGroups = append(decodeGroups, g)
		case inferencetopology.RuntimeRoleBoth:
			prefillGroups = append(prefillGroups, g)
			decodeGroups = append(decodeGroups, g)
		}
	}
	sort.Slice(prefillGroups, func(i, j int) bool { return prefillGroups[i].ID < prefillGroups[j].ID })
	sort.Slice(decodeGroups, func(i, j int) bool { return decodeGroups[i].ID < decodeGroups[j].ID })
	if len(prefillGroups) == 0 || len(decodeGroups) == 0 {
		return CompiledPDRoleGraph{}, fmt.Errorf("%w: require at least one prefill and one decode group", ErrRoleGraphLayoutUnsupported)
	}

	pipelinePlans := make(map[string]model.PartitionPlan, len(decodeGroups))
	stageSpecByGroupStage := make(map[string]map[int]model.StageSpec, len(decodeGroups))
	for _, dg := range decodeGroups {
		plan, specsByStage, err := buildAndValidateDecodePipelinePlan(dg, ranksByGroup[dg.ID], opts.ModelConfig)
		if err != nil {
			return CompiledPDRoleGraph{}, err
		}
		pipelinePlans[dg.ID] = plan
		stageSpecByGroupStage[dg.ID] = specsByStage
	}

	rankGeo := make(map[string]RoleGraphKVGeometry, len(ranksByID))
	rankCap := make(map[string]int, len(ranksByID))
	for id, r := range ranksByID {
		g := groupsByID[r.GroupID]
		specKV, hasSpec := stageKVByRank[id]
		var rawGeo RoleGraphKVGeometry
		if hasSpec {
			rawGeo = specKV.KVGeometry
		}
		geo, err := normalizeAndValidateRankKVGeometry(rawGeo, hasSpec, r, g, opts.ModelConfig)
		if err != nil {
			return CompiledPDRoleGraph{}, err
		}
		rankGeo[id] = geo
		capBlocks := 0
		if hasSpec && specKV.CapacityBlocks > 0 {
			capBlocks = specKV.CapacityBlocks
		} else if hasSpec && specKV.CapacityBlocks < 0 {
			return CompiledPDRoleGraph{}, fmt.Errorf("%w: rank %q negative capacity_blocks %d", ErrRoleGraphLayoutUnsupported, id, specKV.CapacityBlocks)
		} else {
			capBlocks = deriveRankBlockCapacity(r, deviceFreeMap[r.DeviceID])
		}
		rankCap[id] = capBlocks
	}

	// Validate cross-role P->D compatibility if there are no lane-scoped filters,
	// and build compatible prefill options per decode group.
	compatPrefillByDecode := make(map[string][]string, len(decodeGroups))
	for _, dg := range decodeGroups {
		for _, pg := range prefillGroups {
			if pg.ID == dg.ID && len(prefillGroups) > 1 && len(decodeGroups) > 1 {
				continue
			}
			if err := checkPrefillDecodeGroupCompatibility(pg, dg, ranksByGroup[pg.ID], ranksByGroup[dg.ID], rankGeo); err == nil {
				compatPrefillByDecode[dg.ID] = append(compatPrefillByDecode[dg.ID], pg.ID)
			}
		}
		sort.Strings(compatPrefillByDecode[dg.ID])
	}
	if len(spec.DecodeLanes) == 0 {
		for _, dg := range decodeGroups {
			if len(compatPrefillByDecode[dg.ID]) == 0 {
				return CompiledPDRoleGraph{}, fmt.Errorf("%w: no compatible prefill group for decode group %q (check TP/Layers/KVFormat/KVGeometry)", ErrRoleGraphLayoutUnsupported, dg.ID)
			}
		}
	}

	incumbent := opts.Incumbent
	if incumbent == nil {
		incumbent = spec.Incumbent
	}
	pinnedByLane := make(map[string]RoleGraphPinnedPlace)
	incumbentPrefillByLane := make(map[string]string)
	if incumbent != nil {
		for _, a := range incumbent.Assignments {
			incumbentPrefillByLane[a.LaneID] = a.PrefillGroupID
			if a.Pinned {
				pinnedByLane[a.LaneID] = RoleGraphPinnedPlace{
					DecodeGroupID: a.DecodeGroupID,
					RankID:        a.RankID,
					PPStage:       a.PPStage,
					TPRank:        a.TPRank,
					EPRank:        a.EPRank,
				}
			}
		}
	}
	for laneID, pin := range spec.PinnedPlacements {
		if existing, ok := pinnedByLane[laneID]; ok && existing != pin {
			return CompiledPDRoleGraph{}, fmt.Errorf("%w: conflicting pin for lane %q", ErrRoleGraphLayoutUnsupported, laneID)
		}
		pinnedByLane[laneID] = pin
	}

	lanes := append([]RoleGraphDecodeLane(nil), spec.DecodeLanes...)
	if len(lanes) == 0 {
		for _, dg := range decodeGroups {
			dgRanks := append([]inferencetopology.RuntimeTopologyRank(nil), ranksByGroup[dg.ID]...)
			sortRanksCanonical(dgRanks)
			for _, r := range dgRanks {
				lanes = append(lanes, RoleGraphDecodeLane{
					LaneID:          "lane-" + r.ID,
					DecodeGroupID:   dg.ID,
					RankID:          r.ID,
					RequestedBlocks: 1,
				})
			}
		}
	}

	seenLanes := make(map[string]bool, len(lanes))
	for _, ln := range lanes {
		if ln.LaneID == "" || seenLanes[ln.LaneID] {
			return CompiledPDRoleGraph{}, fmt.Errorf("%w: empty or duplicate decode lane id %q", ErrRoleGraphLayoutUnsupported, ln.LaneID)
		}
		seenLanes[ln.LaneID] = true
		if ln.RequestedBlocks < 0 {
			return CompiledPDRoleGraph{}, fmt.Errorf("%w: lane %q negative requested_blocks %d", ErrRoleGraphLayoutUnsupported, ln.LaneID, ln.RequestedBlocks)
		}
	}

	// Deterministic lane resolution order: pinned lanes first, then lexicographical by LaneID.
	orderedLanes := append([]RoleGraphDecodeLane(nil), lanes...)
	sort.Slice(orderedLanes, func(i, j int) bool {
		pi := orderedLanes[i].Pinned || hasKey(pinnedByLane, orderedLanes[i].LaneID)
		pj := orderedLanes[j].Pinned || hasKey(pinnedByLane, orderedLanes[j].LaneID)
		if pi != pj {
			return pi
		}
		return orderedLanes[i].LaneID < orderedLanes[j].LaneID
	})

	ledger := opts.Ledger
	if ledger == nil {
		ledger = NewRoleGraphReservationLedger()
	}
	for _, dg := range decodeGroups {
		for _, r := range ranksByGroup[dg.ID] {
			if _, err := ledger.ensureRankCapacity(r.ID, rankCap[r.ID], NativeKVLayoutV1); err != nil {
				return CompiledPDRoleGraph{}, err
			}
		}
	}

	var acquiredOwners []string
	committed := false
	defer func() {
		if !committed {
			for _, owner := range acquiredOwners {
				_ = ledger.Release(owner)
			}
		}
	}()

	groupAccum := make(map[string]*RoleGraphDecodeGroup)
	assignments := make([]RoleGraphLaneAssignment, 0, len(orderedLanes))
	reservations := make([]RoleGraphReservation, 0, len(orderedLanes))

	for _, ln := range orderedLanes {
		if err := ctx.Err(); err != nil {
			return CompiledPDRoleGraph{}, fmt.Errorf("%w: %v", ErrRoleGraphLayoutUnsupported, err)
		}

		pin, hasPin := pinnedByLane[ln.LaneID]
		rank, dg, pinnedFlag, incumbentPreserved, err := resolveLaneTargetRank(ln, pin, hasPin, decodeGroups, groupsByID, ranksByID, ranksByGroup, compatPrefillByDecode)
		if err != nil {
			return CompiledPDRoleGraph{}, err
		}

		pgID, err := resolveLanePrefillGroup(ln, dg, incumbentPrefillByLane[ln.LaneID], groupsByID, ranksByGroup, rankGeo, compatPrefillByDecode)
		if err != nil {
			return CompiledPDRoleGraph{}, err
		}

		laneGeo, err := resolveLaneKVGeometry(ln, rank, dg, rankGeo[rank.ID])
		if err != nil {
			return CompiledPDRoleGraph{}, err
		}

		groupKey := laneGeo.DecodeGroupKey(dg.ID)
		blocks := ln.RequestedBlocks
		if blocks == 0 {
			blocks = 1
		}
		owner := fmt.Sprintf("role-graph:%s:%s:%s", spec.Topology.ID, rank.ID, ln.LaneID)
		res, err := ledger.Acquire(owner, ln.LaneID, dg.ID, rank.ID, groupKey, blocks, NativeKVLayoutV1)
		if err != nil {
			return CompiledPDRoleGraph{}, err
		}
		acquiredOwners = append(acquiredOwners, owner)
		reservations = append(reservations, res)

		stageSpec := stageSpecByGroupStage[dg.ID][rank.PPStage]
		assignments = append(assignments, RoleGraphLaneAssignment{
			LaneID:             ln.LaneID,
			PrefillGroupID:     pgID,
			DecodeGroupID:      dg.ID,
			DecodeKVGroupKey:   groupKey,
			PPStage:            rank.PPStage,
			TPRank:             rank.TPRank,
			EPRank:             rank.EPRank,
			RankID:             rank.ID,
			DeviceID:           rank.DeviceID,
			StageSpec:          stageSpec,
			KVGeometry:         laneGeo,
			ReservedBlocks:     blocks,
			WaitState:          RoleGraphWaitRemoteKV,
			Pinned:             pinnedFlag,
			IncumbentPreserved: incumbentPreserved,
		})

		grp, ok := groupAccum[groupKey]
		if !ok {
			pipeEntry := "NativeScheduler"
			if dg.PP > 1 {
				pipeEntry = "NewPipelineEngine"
			}
			epDesc := ""
			if dg.EP > 1 {
				if rank.EPRank == 0 {
					epDesc = epDriver + ":coordinator"
				} else {
					epDesc = epDriver + ":follower"
				}
			}
			grp = &RoleGraphDecodeGroup{
				GroupKey:           groupKey,
				DecodeGroupID:      dg.ID,
				Role:               NativePDRoleDecode,
				PPStage:            rank.PPStage,
				TPRank:             rank.TPRank,
				EPRank:             rank.EPRank,
				RankID:             rank.ID,
				DeviceID:           rank.DeviceID,
				StageSpec:          stageSpec,
				KVGeometry:         laneGeo,
				CapacityBlocks:     rankCap[rank.ID],
				WaitState:          RoleGraphWaitRemoteKV,
				AdmitEntrypoint:    "NativeScheduler.AdmitImported",
				PipelineEntrypoint: pipeEntry,
				EPDriver:           epDesc,
			}
			groupAccum[groupKey] = grp
		}
		grp.ReservedBlocks += blocks
		grp.LaneIDs = append(grp.LaneIDs, ln.LaneID)
	}

	sort.Slice(assignments, func(i, j int) bool { return assignments[i].LaneID < assignments[j].LaneID })
	sort.Slice(reservations, func(i, j int) bool { return reservations[i].LaneID < reservations[j].LaneID })

	compiledDecodeGroups := make([]RoleGraphDecodeGroup, 0, len(groupAccum))
	for _, grp := range groupAccum {
		sort.Strings(grp.LaneIDs)
		compiledDecodeGroups = append(compiledDecodeGroups, *grp)
	}
	sort.Slice(compiledDecodeGroups, func(i, j int) bool {
		return compiledDecodeGroups[i].GroupKey < compiledDecodeGroups[j].GroupKey
	})

	prefillBindings := make([]RoleGraphPrefillBinding, 0, len(prefillGroups))
	for _, pg := range prefillGroups {
		pgRanks := append([]inferencetopology.RuntimeTopologyRank(nil), ranksByGroup[pg.ID]...)
		sortRanksCanonical(pgRanks)
		rankIDs := make([]string, len(pgRanks))
		devIDs := make([]string, len(pgRanks))
		for i, r := range pgRanks {
			rankIDs[i] = r.ID
			devIDs[i] = r.DeviceID
		}
		prefillBindings = append(prefillBindings, RoleGraphPrefillBinding{
			GroupID:   pg.ID,
			Role:      NativePDRolePrefill,
			TP:        pg.TP,
			PP:        pg.PP,
			EP:        pg.EP,
			Layers:    pg.Layers,
			KVFormat:  pg.KVFormat,
			RankIDs:   rankIDs,
			DeviceIDs: devIDs,
		})
	}

	compiled := CompiledPDRoleGraph{
		Schema:               RoleGraphCompiledSchema,
		Evidence:             "SW_VERIFIED_DRY_RUN_ROLE_GRAPH_NO_TRANSFER",
		TopologyID:           spec.Topology.ID,
		ObservationEpoch:     spec.Topology.Observed.Epoch,
		PrefillGroups:        prefillBindings,
		DecodeGroups:         compiledDecodeGroups,
		PipelinePlans:        maps.Clone(pipelinePlans),
		Assignments:          assignments,
		Reservations:         reservations,
		EPDriver:             epDriver,
		KVTransfersTriggered: 0,
	}

	if opts.PreCommitHook != nil {
		if err := opts.PreCommitHook(&compiled); err != nil {
			return CompiledPDRoleGraph{}, fmt.Errorf("%w: %v", ErrRoleGraphLayoutUnsupported, err)
		}
	}
	committed = true
	return compiled, nil
}

// AdmitCompiledDecodeLane gates NativeScheduler.AdmitImported on a complete,
// geometry-compatible compiled role-graph assignment. Waiting-for-remote-KV lanes
// or incomplete/mismatched KV caches are refused before AdmitImported and release
// their reservation cleanly.
func AdmitCompiledDecodeLane(ctx context.Context, sched *NativeScheduler, call *abi.ToolCall, seq ImportedSequence, compiled CompiledPDRoleGraph, laneID string, actualKV RoleGraphKVGeometry, kvReady bool, ledger *RoleGraphReservationLedger) (abi.EngineRequest, error) {
	var found *RoleGraphLaneAssignment
	for i := range compiled.Assignments {
		if compiled.Assignments[i].LaneID == laneID {
			found = &compiled.Assignments[i]
			break
		}
	}
	releaseLane := func() {
		if ledger == nil {
			return
		}
		for _, res := range compiled.Reservations {
			if res.LaneID == laneID {
				_ = ledger.Release(res.Owner)
			}
		}
	}
	if found == nil {
		releaseLane()
		return nil, fmt.Errorf("%w: lane %q absent from compiled role graph", ErrRoleGraphLayoutUnsupported, laneID)
	}
	if !kvReady {
		releaseLane()
		return nil, fmt.Errorf("%w: lane %q still %s (waiting KV is not computed progress)", ErrRoleGraphLayoutUnsupported, laneID, RoleGraphWaitRemoteKV)
	}
	if actualKV.GeometryKey() != found.KVGeometry.GeometryKey() ||
		actualKV.PPStage != found.PPStage ||
		actualKV.TPRank != found.TPRank ||
		actualKV.EPRank != found.EPRank {
		releaseLane()
		return nil, fmt.Errorf("%w: lane %q KV geometry mismatch (got %s, want %s)", ErrRoleGraphLayoutUnsupported, laneID, actualKV.GeometryKey(), found.KVGeometry.GeometryKey())
	}
	req, err := sched.AdmitImported(ctx, call, seq)
	if err != nil {
		releaseLane()
		return nil, err
	}
	return req, nil
}

func normalizeEPDriver(raw string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "ep_decode_coordinator", "epdecodecoordinator", "existing":
		return "EPDecodeCoordinator", nil
	default:
		return "", fmt.Errorf("%w: ep_driver %q prohibited (must reuse existing EPDecodeCoordinator)", ErrRoleGraphLayoutUnsupported, raw)
	}
}

func buildAndValidateDecodePipelinePlan(dg inferencetopology.RuntimeTopologyGroup, dgRanks []inferencetopology.RuntimeTopologyRank, cfg *model.Config) (model.PartitionPlan, map[int]model.StageSpec, error) {
	specsByStage := make(map[int]model.StageSpec, dg.PP)
	stages := make([]model.StageSpec, dg.PP)
	for pp := 0; pp < dg.PP; pp++ {
		lo, hi := -1, -1
		for _, r := range dgRanks {
			if r.PPStage == pp {
				lo, hi = r.LayerStart, r.LayerEnd
				break
			}
		}
		if lo < 0 || hi <= lo {
			return model.PartitionPlan{}, nil, fmt.Errorf("%w: decode group %q missing valid layer bounds for stage %d", ErrRoleGraphLayoutUnsupported, dg.ID, pp)
		}
		spec := model.StageSpec{Lo: lo, Hi: hi, First: pp == 0, Last: pp == dg.PP-1}
		stages[pp] = spec
		specsByStage[pp] = spec
	}
	plan := model.PartitionPlan{NumLayers: dg.Layers, Stages: stages}
	checkCfg := model.Config{NumLayers: dg.Layers}
	if cfg != nil {
		if cfg.NumLayers > 0 && cfg.NumLayers != dg.Layers {
			return model.PartitionPlan{}, nil, fmt.Errorf("%w: decode group %q layers=%d != model layers=%d", ErrRoleGraphLayoutUnsupported, dg.ID, dg.Layers, cfg.NumLayers)
		}
		checkCfg = *cfg
		if checkCfg.NumLayers <= 0 {
			checkCfg.NumLayers = dg.Layers
		}
	}
	if err := plan.Validate(checkCfg); err != nil {
		return model.PartitionPlan{}, nil, fmt.Errorf("%w: decode group %q pipeline validation: %v", ErrRoleGraphLayoutUnsupported, dg.ID, err)
	}
	return plan, specsByStage, nil
}

func normalizeAndValidateRankKVGeometry(raw RoleGraphKVGeometry, explicit bool, r inferencetopology.RuntimeTopologyRank, g inferencetopology.RuntimeTopologyGroup, cfg *model.Config) (RoleGraphKVGeometry, error) {
	out := raw
	if explicit {
		if (raw.PPStage != 0 || raw.TPRank != 0 || raw.EPRank != 0) &&
			(raw.PPStage != r.PPStage || raw.TPRank != r.TPRank || raw.EPRank != r.EPRank) {
			return RoleGraphKVGeometry{}, fmt.Errorf("%w: stage_kv rank %q coordinates (%d,%d,%d) != (%d,%d,%d)",
				ErrRoleGraphLayoutUnsupported, r.ID, raw.PPStage, raw.TPRank, raw.EPRank, r.PPStage, r.TPRank, r.EPRank)
		}
		if raw.KVFormat != "" && raw.KVFormat != g.KVFormat {
			return RoleGraphKVGeometry{}, fmt.Errorf("%w: stage_kv rank %q kv_format %q != group %q %q",
				ErrRoleGraphLayoutUnsupported, r.ID, raw.KVFormat, g.ID, g.KVFormat)
		}
		if raw.BlockTokens < 0 || raw.NumKVHeads < 0 || raw.HeadDim < 0 || raw.VHeadDim < 0 || raw.KVLoraRank < 0 {
			return RoleGraphKVGeometry{}, fmt.Errorf("%w: stage_kv rank %q has negative KV dimension", ErrRoleGraphLayoutUnsupported, r.ID)
		}
		if (raw.LayerStart != 0 || raw.LayerEnd != 0) && (raw.LayerStart != r.LayerStart || raw.LayerEnd != r.LayerEnd) {
			return RoleGraphKVGeometry{}, fmt.Errorf("%w: stage_kv rank %q layers [%d,%d) != [%d,%d)",
				ErrRoleGraphLayoutUnsupported, r.ID, raw.LayerStart, raw.LayerEnd, r.LayerStart, r.LayerEnd)
		}
	}
	out.PPStage = r.PPStage
	out.TPRank = r.TPRank
	out.EPRank = r.EPRank
	out.LayerStart = r.LayerStart
	out.LayerEnd = r.LayerEnd
	if out.KVFormat == "" {
		out.KVFormat = g.KVFormat
	}
	if out.KVPrecision == "" {
		out.KVPrecision = "f32"
	}
	if out.BlockTokens <= 0 {
		out.BlockTokens = nativePDDefaultBlockTokens
	}
	if cfg != nil {
		if cfg.NumKVHeads > 0 {
			if cfg.NumKVHeads%g.TP != 0 {
				return RoleGraphKVGeometry{}, fmt.Errorf("%w: group %q TP=%d does not divide NumKVHeads=%d", ErrRoleGraphLayoutUnsupported, g.ID, g.TP, cfg.NumKVHeads)
			}
			if out.NumKVHeads == 0 {
				out.NumKVHeads = cfg.NumKVHeads / g.TP
			}
		}
		if out.HeadDim == 0 && cfg.HeadDim > 0 {
			out.HeadDim = cfg.HeadDim
		}
		if out.VHeadDim == 0 {
			if cfg.VHeadDim > 0 {
				out.VHeadDim = cfg.VHeadDim
			} else {
				out.VHeadDim = out.HeadDim
			}
		}
		if out.KVLoraRank == 0 && cfg.KVLoraRank > 0 {
			out.KVLoraRank = cfg.KVLoraRank
		}
		if out.LayerGroup == "" && r.LayerStart >= 0 && r.LayerEnd <= cfg.NumLayers {
			out.LayerGroup = uniformLayerGroup(*cfg, r.LayerStart, r.LayerEnd)
		}
	}
	if out.VHeadDim == 0 {
		out.VHeadDim = out.HeadDim
	}
	if out.LayerGroup == "" {
		out.LayerGroup = model.KVGroupFull.String()
	}
	return out, nil
}

func uniformLayerGroup(cfg model.Config, lo, hi int) string {
	if lo >= hi {
		return model.KVGroupFull.String()
	}
	first := cfg.KVLayerGroupOf(lo).String()
	for l := lo + 1; l < hi; l++ {
		if cfg.KVLayerGroupOf(l).String() != first {
			return "hybrid"
		}
	}
	return first
}

func deriveRankBlockCapacity(r inferencetopology.RuntimeTopologyRank, freeBytes float64) int {
	headroom := freeBytes - r.Commitment.ResidentWeightBytes
	if headroom <= 0 {
		return 0
	}
	if r.Commitment.KVResidentBytes > 0 {
		blocks := int(headroom / r.Commitment.KVResidentBytes)
		if blocks >= 1 {
			return blocks
		}
	}
	if headroom >= 1 {
		return int(headroom)
	}
	return 1
}

func checkPrefillDecodeGroupCompatibility(pg, dg inferencetopology.RuntimeTopologyGroup, pgRanks, dgRanks []inferencetopology.RuntimeTopologyRank, rankGeo map[string]RoleGraphKVGeometry) error {
	if pg.Layers != dg.Layers {
		return fmt.Errorf("%w: prefill %q layers=%d != decode %q layers=%d", ErrRoleGraphLayoutUnsupported, pg.ID, pg.Layers, dg.ID, dg.Layers)
	}
	if pg.KVFormat != dg.KVFormat {
		return fmt.Errorf("%w: prefill %q kv_format=%q != decode %q kv_format=%q", ErrRoleGraphLayoutUnsupported, pg.ID, pg.KVFormat, dg.ID, dg.KVFormat)
	}
	if pg.TP != dg.TP {
		return fmt.Errorf("%w: prefill %q TP=%d != decode %q TP=%d", ErrRoleGraphLayoutUnsupported, pg.ID, pg.TP, dg.ID, dg.TP)
	}
	for _, dr := range dgRanks {
		dGeo := rankGeo[dr.ID]
		for _, pr := range pgRanks {
			if pr.TPRank != dr.TPRank {
				continue
			}
			if pr.LayerEnd <= dr.LayerStart || dr.LayerEnd <= pr.LayerStart {
				continue
			}
			pGeo := rankGeo[pr.ID]
			if pGeo.KVFormat != dGeo.KVFormat ||
				pGeo.KVPrecision != dGeo.KVPrecision ||
				pGeo.BlockTokens != dGeo.BlockTokens ||
				pGeo.NumKVHeads != dGeo.NumKVHeads ||
				pGeo.HeadDim != dGeo.HeadDim ||
				pGeo.VHeadDim != dGeo.VHeadDim ||
				pGeo.KVLoraRank != dGeo.KVLoraRank {
				return fmt.Errorf("%w: prefill rank %q KV geometry (%s) incompatible with decode rank %q (%s)",
					ErrRoleGraphLayoutUnsupported, pr.ID, pGeo.GeometryKey(), dr.ID, dGeo.GeometryKey())
			}
		}
	}
	return nil
}

func resolveLaneTargetRank(
	ln RoleGraphDecodeLane,
	pin RoleGraphPinnedPlace,
	hasPin bool,
	decodeGroups []inferencetopology.RuntimeTopologyGroup,
	groupsByID map[string]inferencetopology.RuntimeTopologyGroup,
	ranksByID map[string]inferencetopology.RuntimeTopologyRank,
	ranksByGroup map[string][]inferencetopology.RuntimeTopologyRank,
	compatPrefillByDecode map[string][]string,
) (inferencetopology.RuntimeTopologyRank, inferencetopology.RuntimeTopologyGroup, bool, bool, error) {
	if hasPin {
		r, ok := ranksByID[pin.RankID]
		dg, dgOK := groupsByID[pin.DecodeGroupID]
		if !ok || !dgOK || r.GroupID != dg.ID || !isDecodeCapableRole(dg.Role) ||
			r.PPStage != pin.PPStage || r.TPRank != pin.TPRank || r.EPRank != pin.EPRank {
			return inferencetopology.RuntimeTopologyRank{}, inferencetopology.RuntimeTopologyGroup{}, false, false,
				fmt.Errorf("%w: incumbent pinned placement for lane %q (%+v) is no longer eligible", ErrRoleGraphLayoutUnsupported, ln.LaneID, pin)
		}
		if (ln.DecodeGroupID != "" && ln.DecodeGroupID != pin.DecodeGroupID) ||
			(ln.RankID != "" && ln.RankID != pin.RankID) ||
			(ln.PPStage != nil && *ln.PPStage != pin.PPStage) ||
			(ln.TPRank != nil && *ln.TPRank != pin.TPRank) ||
			(ln.EPRank != nil && *ln.EPRank != pin.EPRank) {
			return inferencetopology.RuntimeTopologyRank{}, inferencetopology.RuntimeTopologyGroup{}, false, false,
				fmt.Errorf("%w: lane %q constraints conflict with incumbent pinned placement", ErrRoleGraphLayoutUnsupported, ln.LaneID)
		}
		return r, dg, true, true, nil
	}

	if ln.RankID != "" {
		r, ok := ranksByID[ln.RankID]
		if !ok {
			return inferencetopology.RuntimeTopologyRank{}, inferencetopology.RuntimeTopologyGroup{}, false, false,
				fmt.Errorf("%w: lane %q references unknown rank %q", ErrRoleGraphLayoutUnsupported, ln.LaneID, ln.RankID)
		}
		dg := groupsByID[r.GroupID]
		if !isDecodeCapableRole(dg.Role) {
			return inferencetopology.RuntimeTopologyRank{}, inferencetopology.RuntimeTopologyGroup{}, false, false,
				fmt.Errorf("%w: lane %q rank %q belongs to non-decode group %q", ErrRoleGraphLayoutUnsupported, ln.LaneID, r.ID, dg.ID)
		}
		if ln.DecodeGroupID != "" && ln.DecodeGroupID != dg.ID {
			return inferencetopology.RuntimeTopologyRank{}, inferencetopology.RuntimeTopologyGroup{}, false, false,
				fmt.Errorf("%w: lane %q decode_group_id %q != rank group %q", ErrRoleGraphLayoutUnsupported, ln.LaneID, ln.DecodeGroupID, dg.ID)
		}
		if !laneMatchesCoords(ln, r) {
			return inferencetopology.RuntimeTopologyRank{}, inferencetopology.RuntimeTopologyGroup{}, false, false,
				fmt.Errorf("%w: lane %q stage/rank coordinates mismatch rank %q", ErrRoleGraphLayoutUnsupported, ln.LaneID, r.ID)
		}
		return r, dg, ln.Pinned, false, nil
	}

	if ln.Pinned {
		return inferencetopology.RuntimeTopologyRank{}, inferencetopology.RuntimeTopologyGroup{}, false, false,
			fmt.Errorf("%w: pinned lane %q requires explicit rank_id or incumbent pin", ErrRoleGraphLayoutUnsupported, ln.LaneID)
	}

	var candidateGroups []inferencetopology.RuntimeTopologyGroup
	if ln.DecodeGroupID != "" {
		dg, ok := groupsByID[ln.DecodeGroupID]
		if !ok || !isDecodeCapableRole(dg.Role) {
			return inferencetopology.RuntimeTopologyRank{}, inferencetopology.RuntimeTopologyGroup{}, false, false,
				fmt.Errorf("%w: lane %q unknown or non-decode group %q", ErrRoleGraphLayoutUnsupported, ln.LaneID, ln.DecodeGroupID)
		}
		candidateGroups = []inferencetopology.RuntimeTopologyGroup{dg}
	} else {
		for _, dg := range decodeGroups {
			if ln.PrefillGroupID != "" || len(compatPrefillByDecode[dg.ID]) > 0 {
				candidateGroups = append(candidateGroups, dg)
			}
		}
	}

	for _, dg := range candidateGroups {
		ranks := append([]inferencetopology.RuntimeTopologyRank(nil), ranksByGroup[dg.ID]...)
		sortRanksCanonical(ranks)
		for _, r := range ranks {
			if laneMatchesCoords(ln, r) {
				return r, dg, false, false, nil
			}
		}
	}
	return inferencetopology.RuntimeTopologyRank{}, inferencetopology.RuntimeTopologyGroup{}, false, false,
		fmt.Errorf("%w: no eligible decode rank matches lane %q", ErrRoleGraphLayoutUnsupported, ln.LaneID)
}

func resolveLanePrefillGroup(
	ln RoleGraphDecodeLane,
	dg inferencetopology.RuntimeTopologyGroup,
	incumbentPrefill string,
	groupsByID map[string]inferencetopology.RuntimeTopologyGroup,
	ranksByGroup map[string][]inferencetopology.RuntimeTopologyRank,
	rankGeo map[string]RoleGraphKVGeometry,
	compatPrefillByDecode map[string][]string,
) (string, error) {
	if ln.PrefillGroupID != "" {
		pg, ok := groupsByID[ln.PrefillGroupID]
		if !ok || (pg.Role != inferencetopology.RuntimeRolePrefill && pg.Role != inferencetopology.RuntimeRoleBoth) {
			return "", fmt.Errorf("%w: lane %q prefill_group_id %q is not a prefill group", ErrRoleGraphLayoutUnsupported, ln.LaneID, ln.PrefillGroupID)
		}
		if err := checkPrefillDecodeGroupCompatibility(pg, dg, ranksByGroup[pg.ID], ranksByGroup[dg.ID], rankGeo); err != nil {
			return "", err
		}
		return pg.ID, nil
	}
	compat := compatPrefillByDecode[dg.ID]
	if len(compat) == 0 {
		return "", fmt.Errorf("%w: lane %q decode group %q has no compatible prefill group", ErrRoleGraphLayoutUnsupported, ln.LaneID, dg.ID)
	}
	if incumbentPrefill != "" && slices.Contains(compat, incumbentPrefill) {
		return incumbentPrefill, nil
	}
	return compat[0], nil
}

func resolveLaneKVGeometry(ln RoleGraphDecodeLane, r inferencetopology.RuntimeTopologyRank, dg inferencetopology.RuntimeTopologyGroup, base RoleGraphKVGeometry) (RoleGraphKVGeometry, error) {
	if ln.KVGeometry == nil {
		return base, nil
	}
	lg := *ln.KVGeometry
	if lg.PPStage != 0 && lg.PPStage != r.PPStage {
		return RoleGraphKVGeometry{}, fmt.Errorf("%w: lane %q kv_geometry pp_stage %d != rank %d", ErrRoleGraphLayoutUnsupported, ln.LaneID, lg.PPStage, r.PPStage)
	}
	if lg.TPRank != 0 && lg.TPRank != r.TPRank {
		return RoleGraphKVGeometry{}, fmt.Errorf("%w: lane %q kv_geometry tp_rank %d != rank %d", ErrRoleGraphLayoutUnsupported, ln.LaneID, lg.TPRank, r.TPRank)
	}
	if lg.EPRank != 0 && lg.EPRank != r.EPRank {
		return RoleGraphKVGeometry{}, fmt.Errorf("%w: lane %q kv_geometry ep_rank %d != rank %d", ErrRoleGraphLayoutUnsupported, ln.LaneID, lg.EPRank, r.EPRank)
	}
	if lg.KVFormat != "" && lg.KVFormat != dg.KVFormat {
		return RoleGraphKVGeometry{}, fmt.Errorf("%w: lane %q kv_format %q != decode group %q %q", ErrRoleGraphLayoutUnsupported, ln.LaneID, lg.KVFormat, dg.ID, dg.KVFormat)
	}
	if lg.BlockTokens < 0 || lg.NumKVHeads < 0 || lg.HeadDim < 0 || lg.VHeadDim < 0 || lg.KVLoraRank < 0 {
		return RoleGraphKVGeometry{}, fmt.Errorf("%w: lane %q negative kv_geometry dimension", ErrRoleGraphLayoutUnsupported, ln.LaneID)
	}
	if lg.LayerStart != 0 || lg.LayerEnd != 0 {
		if lg.LayerStart < r.LayerStart || lg.LayerEnd > r.LayerEnd || lg.LayerEnd <= lg.LayerStart {
			return RoleGraphKVGeometry{}, fmt.Errorf("%w: lane %q kv_geometry layers [%d,%d) outside rank band [%d,%d)",
				ErrRoleGraphLayoutUnsupported, ln.LaneID, lg.LayerStart, lg.LayerEnd, r.LayerStart, r.LayerEnd)
		}
	} else {
		lg.LayerStart = base.LayerStart
		lg.LayerEnd = base.LayerEnd
	}
	lg.PPStage = r.PPStage
	lg.TPRank = r.TPRank
	lg.EPRank = r.EPRank
	if lg.KVFormat == "" {
		lg.KVFormat = base.KVFormat
	}
	if lg.KVPrecision == "" {
		lg.KVPrecision = base.KVPrecision
	}
	if lg.BlockTokens == 0 {
		lg.BlockTokens = base.BlockTokens
	}
	if lg.NumKVHeads == 0 {
		lg.NumKVHeads = base.NumKVHeads
	}
	if lg.HeadDim == 0 {
		lg.HeadDim = base.HeadDim
	}
	if lg.VHeadDim == 0 {
		if lg.HeadDim != base.HeadDim {
			lg.VHeadDim = lg.HeadDim
		} else {
			lg.VHeadDim = base.VHeadDim
		}
	}
	if lg.KVLoraRank == 0 {
		lg.KVLoraRank = base.KVLoraRank
	}
	if lg.LayerGroup == "" {
		lg.LayerGroup = base.LayerGroup
	}
	return lg, nil
}

func laneMatchesCoords(ln RoleGraphDecodeLane, r inferencetopology.RuntimeTopologyRank) bool {
	if ln.PPStage != nil && *ln.PPStage != r.PPStage {
		return false
	}
	if ln.TPRank != nil && *ln.TPRank != r.TPRank {
		return false
	}
	if ln.EPRank != nil && *ln.EPRank != r.EPRank {
		return false
	}
	return true
}

func isDecodeCapableRole(role string) bool {
	return role == inferencetopology.RuntimeRoleDecode || role == inferencetopology.RuntimeRoleBoth
}

func sortRanksCanonical(ranks []inferencetopology.RuntimeTopologyRank) {
	sort.Slice(ranks, func(i, j int) bool {
		if ranks[i].PPStage != ranks[j].PPStage {
			return ranks[i].PPStage < ranks[j].PPStage
		}
		if ranks[i].TPRank != ranks[j].TPRank {
			return ranks[i].TPRank < ranks[j].TPRank
		}
		if ranks[i].EPRank != ranks[j].EPRank {
			return ranks[i].EPRank < ranks[j].EPRank
		}
		return ranks[i].ID < ranks[j].ID
	})
}

func hasKey[K comparable, V any](m map[K]V, k K) bool {
	_, ok := m[k]
	return ok
}

func wrapLayoutAndCause(err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%w: %w", ErrRoleGraphLayoutUnsupported, err)
}
