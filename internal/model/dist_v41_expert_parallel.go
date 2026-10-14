package model

import (
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"time"
)

// ErrV41ExpertParallel is fatal to the distributed request. It deliberately does
// not wrap ErrV41ForwardStage: a failed collective must never authorize local
// full-history replay against peers that already advanced their protocol.
var ErrV41ExpertParallel = errors.New("model: V4.1 expert-parallel request failed")

// V41ExpertParallelBinding declares the common checkpoint content and runtime
// source identities, each a lowercase SHA-256 hex digest. The caller must obtain
// these from its loader/build provenance. Agreement is not artifact attestation;
// this software transport does not qualify a sharded GGUF load or a device.
type V41ExpertParallelBinding struct {
	ArtifactSHA256 string
	SourceSHA256   string
}

type v41ExpertParallelCollective struct {
	g                                             *DistComm
	binding                                       [64]byte
	rounds, sentBytes, receivedBytes, gatherNanos atomic.Uint64
	panelExtraRows                                atomic.Uint64
	timeout                                       time.Duration
	failure                                       error // Serial, like DistComm; the first failed round is terminal.
}

// NewV41ExpertParallelCollective binds exact per-pick transport to an existing
// TCP group. It cannot be used as the generic reassociating GLM reduction. The
// native Session caller owns execution; production serve remains fenced until
// the loader verifies the binding and mixed Q2/Q3 owner-device residency.
func NewV41ExpertParallelCollective(g *DistComm, binding V41ExpertParallelBinding) (Collective, error) {
	if g == nil || g.Size() < 2 || g.Rank() < 0 || g.Rank() >= g.Size() {
		return nil, fmt.Errorf("%w: invalid process group", ErrV41ExpertParallel)
	}
	c := &v41ExpertParallelCollective{g: g, timeout: v41EPRoundTimeout}
	for i, value := range []string{binding.ArtifactSHA256, binding.SourceSHA256} {
		decoded, err := hex.DecodeString(value)
		if err != nil || len(decoded) != 32 || hex.EncodeToString(decoded) != value {
			return nil, fmt.Errorf("%w: binding %d requires lowercase SHA-256", ErrV41ExpertParallel, i)
		}
		copy(c.binding[i*32:], decoded)
	}
	return c, nil
}

func (*v41ExpertParallelCollective) AllReduceSum([][]float32) ([]float32, error) {
	return nil, fmt.Errorf("%w: rank pre-sums reassociate V4.1 arithmetic", ErrV41ExpertParallel)
}

func (*v41ExpertParallelCollective) AllGather([][]float32, TPPlan) ([]float32, error) {
	return nil, fmt.Errorf("%w: use the checked root-ordered reduction", ErrV41ExpertParallel)
}

// Abort closes socket handles without mutating DistComm's connection table, so
// it also safely interrupts a pending read. The normal group owner still Close's
// the group. A failed round is terminal; there is no reuse or local fallback.
func (g *DistComm) abortV41EP() {
	if g == nil {
		return
	}
	if g.coord != nil {
		_ = g.coord.Close()
	}
	for _, conn := range g.workers {
		if conn != nil {
			_ = conn.Close()
		}
	}
}

// Every owner contributes a fixed identity/status header, even with zero picks.
// Payload widths are header + owned picks*H, never ranks*topK*H padding. Using
// positive header widths preserves TPPlan's nonempty-band invariant.
const v41EPHeaderWords = 10
const v41EPWireVersion = uint32(0x56343102)
const v41EPRoundTimeout = 30 * time.Second

// Bound both aggregate host storage and each length-prefixed wire payload before
// allocation. This is an admission limit, not a throughput qualification.
const v41EPMaxFrameBytes = 64 << 20
const v41EPMaxWords = (v41EPMaxFrameBytes - 5) / 4

func v41EPGatherPlan(picks []routePick, owners TPPlan, width int) (TPPlan, error) {
	if err := owners.Validate(); err != nil || width <= 0 || width > v41EPMaxWords-v41EPHeaderWords || len(owners.Shards) > v41EPMaxWords/v41EPHeaderWords {
		return TPPlan{}, fmt.Errorf("%w: invalid ownership or row width", ErrV41ExpertParallel)
	}
	counts := make([]int, len(owners.Shards))
	for _, pick := range picks {
		if pick.expert < 0 || pick.expert >= owners.Dim || !finite32(pick.weight) {
			return TPPlan{}, fmt.Errorf("%w: invalid routed pick", ErrV41ExpertParallel)
		}
		for rank, band := range owners.Shards {
			if pick.expert >= band.Lo && pick.expert < band.Hi {
				counts[rank]++
				break
			}
		}
	}
	plan := TPPlan{Shards: make([]TPShard, len(counts))}
	for rank, count := range counts {
		cells, ok := checkedMulInt(count, width)
		if !ok || cells > v41EPMaxWords-v41EPHeaderWords-plan.Dim {
			return TPPlan{}, fmt.Errorf("%w: gather dimensions overflow", ErrV41ExpertParallel)
		}
		next := plan.Dim + v41EPHeaderWords + cells
		plan.Shards[rank] = TPShard{Rank: rank, Lo: plan.Dim, Hi: next}
		plan.Dim = next
	}
	return plan, plan.Validate()
}

func v41EPHeader(part []float32, digest [32]byte, failed bool) {
	part[0] = math.Float32frombits(v41EPWireVersion)
	part[1] = 0
	if failed {
		part[1] = math.Float32frombits(1)
	}
	for i := 0; i < 8; i++ {
		o := i * 4
		bits := uint32(digest[o]) | uint32(digest[o+1])<<8 | uint32(digest[o+2])<<16 | uint32(digest[o+3])<<24
		part[i+2] = math.Float32frombits(bits)
	}
}

var _ Collective = (*v41ExpertParallelCollective)(nil)

// V41ExpertParallelTransportStats counts completed gather/reduce/broadcast rounds
// on this rank only. CompletedGathers counts physical exchanges and retains
// its existing public field name. CompletedRows counts logical rows, credited
// only when the entire local exchange completes; scalar rounds credit one.
// As with the byte/time fields, concurrent snapshots are not transactional.
// It excludes decode broadcasts, failed partial I/O and TCP/IP overhead. It is
// mechanism telemetry, not throughput or device-execution qualification.
type V41ExpertParallelTransportStats struct {
	CompletedGathers   uint64
	CompletedRows      uint64
	SentFrameBytes     uint64
	ReceivedFrameBytes uint64
	GatherNanoseconds  uint64
}

func (m *Model) V41ExpertParallelTransportStats() V41ExpertParallelTransportStats {
	if m == nil {
		return V41ExpertParallelTransportStats{}
	}
	c, ok := m.epColl.(*v41ExpertParallelCollective)
	if !ok || c == nil {
		return V41ExpertParallelTransportStats{}
	}
	rounds := c.rounds.Load()
	return V41ExpertParallelTransportStats{CompletedGathers: rounds, CompletedRows: rounds + c.panelExtraRows.Load(), SentFrameBytes: c.sentBytes.Load(), ReceivedFrameBytes: c.receivedBytes.Load(), GatherNanoseconds: c.gatherNanos.Load()}
}

func (g *DistComm) v41EPDeadline(deadline time.Time) error {
	if g == nil {
		return fmt.Errorf("missing process group")
	}
	if g.rank != 0 {
		if g.coord == nil {
			return fmt.Errorf("closed coordinator connection")
		}
		return g.coord.SetDeadline(deadline)
	}
	if len(g.workers) != g.size {
		return fmt.Errorf("incomplete process group")
	}
	for rank := 1; rank < g.size; rank++ {
		if g.workers[rank] == nil {
			return fmt.Errorf("missing owner connection %d", rank)
		}
		if err := g.workers[rank].SetDeadline(deadline); err != nil {
			return err
		}
	}
	return nil
}

// Called under the coordinator mutex for both announcement and shutdown. Idle
// follower waits remain indefinite, but no selected root request can wait
// forever for a connected peer that stopped entering frame exchanges.
func (c *EPDecodeCoordinator) broadcast(payload []byte) (out []byte, err error) {
	if c.v41Timeout <= 0 {
		return c.g.BroadcastFromRoot(payload)
	}
	err = c.g.v41EPDeadline(time.Now().Add(c.v41Timeout))
	defer func() {
		if err == nil {
			err = c.g.v41EPDeadline(time.Time{})
		}
		if err != nil {
			c.g.abortV41EP()
		}
	}()
	if err != nil {
		return nil, err
	}
	return c.g.BroadcastFromRoot(payload)
}
