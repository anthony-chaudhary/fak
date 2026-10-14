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
}

// NewV41ExpertParallelCollective binds exact per-pick transport to an existing
// TCP group. It cannot be used as the generic reassociating GLM reduction. The
// native Session caller owns execution; production serve remains fenced until
// the loader verifies the binding and mixed Q2/Q3 owner-device residency.
func NewV41ExpertParallelCollective(g *DistComm, binding V41ExpertParallelBinding) (Collective, error) {
	if g == nil || g.Size() < 2 || g.Rank() < 0 || g.Rank() >= g.Size() {
		return nil, fmt.Errorf("%w: invalid process group", ErrV41ExpertParallel)
	}
	c := &v41ExpertParallelCollective{g: g}
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
	return nil, fmt.Errorf("%w: use the checked per-pick gather", ErrV41ExpertParallel)
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
const v41EPWireVersion = uint32(0x56343101)
const v41EPRoundTimeout = 30 * time.Second

func v41EPGatherPlan(picks []routePick, owners TPPlan, width int) (TPPlan, error) {
	if err := owners.Validate(); err != nil || width <= 0 {
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
		if !ok || cells > int(^uint(0)>>1)-v41EPHeaderWords-plan.Dim {
			return TPPlan{}, fmt.Errorf("%w: gather dimensions overflow", ErrV41ExpertParallel)
		}
		next := plan.Dim + v41EPHeaderWords + cells
		plan.Shards[rank] = TPShard{Rank: rank, Lo: plan.Dim, Hi: next}
		plan.Dim = next
	}
	return plan, plan.Validate()
}

func (c *v41ExpertParallelCollective) gather(part []float32, plan TPPlan) (out []float32, err error) {
	// Only the selected V4.1 operation sets deadlines. Idle follower frame
	// waits and every existing GLM collective keep their prior behavior.
	err = c.g.v41EPDeadline(time.Now().Add(v41EPRoundTimeout))
	defer func() {
		if err == nil {
			err = c.g.v41EPDeadline(time.Time{})
		}
		if err != nil {
			c.g.abortV41EP()
			err = fmt.Errorf("%w: gather: %v", ErrV41ExpertParallel, err)
		}
	}()
	if err != nil {
		return nil, err
	}
	opened := time.Now()
	out, err = c.g.AllGather(part, plan)
	if err == nil {
		// These are completed application frames (length/op/status/vector
		// headers included), not socket/IP overhead or failed partial I/O.
		if c.g.Rank() == 0 {
			c.sentBytes.Add(uint64((c.g.Size() - 1) * (9 + 4*len(out))))
			c.receivedBytes.Add(uint64(9*(c.g.Size()-1) + 4*(plan.Dim-len(part))))
		} else {
			c.sentBytes.Add(uint64(9 + 4*len(part)))
			c.receivedBytes.Add(uint64(9 + 4*len(out)))
		}
		c.gatherNanos.Add(uint64(time.Since(opened)))
		c.rounds.Add(1)
	}
	return out, err
}

func v41EPHeader(part []float32, digest [32]byte, failed bool) {
	part[0] = math.Float32frombits(v41EPWireVersion)
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

// V41ExpertParallelTransportStats counts completed gathers on this rank only.
// It excludes decode broadcasts, failed partial I/O and TCP/IP overhead. It is
// mechanism telemetry, not throughput or device-execution qualification.
type V41ExpertParallelTransportStats struct {
	CompletedGathers   uint64
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
	return V41ExpertParallelTransportStats{CompletedGathers: c.rounds.Load(), SentFrameBytes: c.sentBytes.Load(), ReceivedFrameBytes: c.receivedBytes.Load(), GatherNanoseconds: c.gatherNanos.Load()}
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
