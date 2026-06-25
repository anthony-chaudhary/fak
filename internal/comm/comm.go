package comm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Ready reports that the comm leaf is linked into the build.
func Ready() bool { return true }

const (
	// ToolBroadcast is the synthetic tool name a Broadcast submits per member so
	// the context-share shape is adjudicated at the floor like any other call.
	ToolBroadcast = "comm.broadcast"
	// ToolScatter is the synthetic tool name a Scatter submits per member so each
	// rank's goal slice crosses the adjudication floor independently.
	ToolScatter = "comm.scatter"
	// ToolGather is the synthetic tool name a Gather submits per member so the
	// fold is adjudicated at the floor like any other call.
	ToolGather = "comm.gather"
	// ToolBarrier is the synthetic tool name a Barrier submits per member so the
	// witness read-back fold is represented as N adjudicated arrivals.
	ToolBarrier = "comm.barrier"
)

var (
	// ErrEmptyGroup is returned when a Group is built with no members.
	ErrEmptyGroup = errors.New("comm: group has no members")
	// ErrDupMember is returned when two members share an identity (rank would not be
	// a function of the identity set).
	ErrDupMember = errors.New("comm: duplicate member identity")
	// ErrNoMember is returned when a rank or identity is not in the group.
	ErrNoMember = errors.New("comm: no such member")
	// ErrNoKernel is returned when a group op that admits a call has no kernel.
	ErrNoKernel = errors.New("comm: nil kernel")
	// ErrDenied is returned when the adjudication floor refuses a collective
	// submission.
	ErrDenied = errors.New("comm: collective denied")
	// ErrArity is returned when a collective's per-rank input count does not match
	// the group size.
	ErrArity = errors.New("comm: output count does not match group size")
	// ErrScopeWiden is returned when Broadcast would share a private Ref across
	// more than one member.
	ErrScopeWiden = errors.New("comm: broadcast would widen Ref scope")
)

// CollectiveRequest is the non-blocking I* handle for a collective expansion. It
// records the per-rank SubmissionHandle set produced by Kernel.Submit. Reap folds
// over those handles in rank order; no ABI edit is needed because the existing
// SubmissionHandle/StatusPending/Reap seam already carries the async identity.
type CollectiveRequest struct {
	Tool    string                 `json:"tool"`
	Handles []abi.SubmissionHandle `json:"handles"`
	State   abi.Status             `json:"state"`
}

// Reap blocks on every submitted rank in deterministic rank order. It leaves
// State as StatusPending if any result still reports pending; otherwise it marks
// the request OK once all handles have completed.
func (r *CollectiveRequest) Reap(ctx context.Context, k abi.Kernel) ([]*abi.Result, error) {
	if k == nil {
		return nil, ErrNoKernel
	}
	results := make([]*abi.Result, 0, len(r.Handles))
	pending := false
	for _, h := range r.Handles {
		res, err := k.Reap(ctx, h)
		if err != nil {
			return results, err
		}
		if res != nil && res.Status == abi.StatusPending {
			pending = true
		}
		results = append(results, res)
	}
	if pending {
		r.State = abi.StatusPending
	} else {
		r.State = abi.StatusOK
	}
	return results, nil
}

// GatherRequest is the non-blocking Gather handle. The rank-ordered votes are
// fixed at Submit time so Combine can run deterministically after the caller has
// reaped the underlying handles.
type GatherRequest struct {
	CollectiveRequest
	Votes  []modelroute.Vote
	Reduce modelroute.Reduction
}

// Combine folds the gathered votes in rank order.
func (r GatherRequest) Combine() (modelroute.Result, error) {
	return modelroute.Combine(r.Reduce, r.Votes)
}

// ReapCombine reaps the per-rank handles and then folds the gathered votes.
func (r *GatherRequest) ReapCombine(ctx context.Context, k abi.Kernel) (modelroute.Result, error) {
	if _, err := r.CollectiveRequest.Reap(ctx, k); err != nil {
		return modelroute.Result{}, err
	}
	return r.Combine()
}

// Member is one agent in a Group. ID is the stable identity rank is computed over
// (a lane id, a process tag, a trace id — any value unique within the group). Lane
// is the dos.toml lane the member holds; Weight rides into modelroute.Combine.
type Member struct {
	ID     string  `json:"id"`
	Lane   string  `json:"lane,omitempty"`
	Weight float64 `json:"weight,omitempty"`
}

// Membership is the rank-stamped value minted at spawn. It is the MPI_Comm_spawn
// membership: a member's place in a wave (WaveID), its parent trace, its rank/size
// within the group, and the lane it holds. It moves no bytes — it is an identity.
type Membership struct {
	WaveID        string `json:"wave_id"`
	ParentTraceID string `json:"parent_trace_id,omitempty"`
	Rank          int    `json:"rank"`
	Size          int    `json:"size"`
	Lane          string `json:"lane,omitempty"`
}

// Group is a deterministic, adjudicated agent communicator. Its members are stored
// in a canonical order (sorted by ID), so Rank is a pure function of the identity
// set: the same members always receive the same ranks regardless of the order they
// were passed to New. A Group moves no bytes and runs no collective (see the package
// doc's bright line to model.DistComm).
type Group struct {
	wave    string
	parent  string
	members []Member // canonical order: sorted by ID, deduplicated
}

// New builds a Group over members. Order does not matter: members are sorted by ID
// into the canonical rank order, so Rank/Size/Split are deterministic functions of
// the identity SET. A duplicate ID is refused (rank could not be a function of the
// set). waveID/parentTraceID stamp the Membership minted by Spawn.
func New(waveID, parentTraceID string, members []Member) (*Group, error) {
	if len(members) == 0 {
		return nil, ErrEmptyGroup
	}
	canon := make([]Member, len(members))
	copy(canon, members)
	sort.Slice(canon, func(i, j int) bool { return canon[i].ID < canon[j].ID })
	for i := 1; i < len(canon); i++ {
		if canon[i].ID == canon[i-1].ID {
			return nil, fmt.Errorf("%w: %q", ErrDupMember, canon[i].ID)
		}
	}
	return &Group{wave: waveID, parent: parentTraceID, members: canon}, nil
}

// Size is the number of members (MPI_Comm_size).
func (g *Group) Size() int { return len(g.members) }

// WaveID is the spawn wave this group belongs to.
func (g *Group) WaveID() string { return g.wave }

// Member returns the member at rank r in canonical order.
func (g *Group) Member(r int) (Member, error) {
	if r < 0 || r >= len(g.members) {
		return Member{}, fmt.Errorf("%w: rank %d of %d", ErrNoMember, r, len(g.members))
	}
	return g.members[r], nil
}

// Rank is id's position in the canonical (sorted) member set (MPI_Comm_rank). It is
// deterministic over the identity set: permuting the members passed to New does not
// change any member's rank.
func (g *Group) Rank(id string) (int, error) {
	// Canonical order is sorted by ID, so a binary search finds the rank.
	r := sort.Search(len(g.members), func(i int) bool { return g.members[i].ID >= id })
	if r < len(g.members) && g.members[r].ID == id {
		return r, nil
	}
	return -1, fmt.Errorf("%w: %q", ErrNoMember, id)
}

// Membership returns the spawn membership for the member at rank r — the value a
// spawned agent carries to know its place in the wave.
func (g *Group) Membership(r int) (Membership, error) {
	m, err := g.Member(r)
	if err != nil {
		return Membership{}, err
	}
	return Membership{
		WaveID:        g.wave,
		ParentTraceID: g.parent,
		Rank:          r,
		Size:          len(g.members),
		Lane:          m.Lane,
	}, nil
}

// Spawn mints the full rank-stamped membership roster in rank order. It is the
// MPI_Comm_spawn membership minting: a wave becomes a typed group of N members, each
// knowing its rank/size/lane, instead of N anonymous lanes.
func (g *Group) Spawn() []Membership {
	out := make([]Membership, len(g.members))
	for r, m := range g.members {
		out[r] = Membership{
			WaveID:        g.wave,
			ParentTraceID: g.parent,
			Rank:          r,
			Size:          len(g.members),
			Lane:          m.Lane,
		}
	}
	return out
}

// Split partitions the group: it returns the sub-group of members whose color(member)
// equals the given color, preserving canonical order so ranks within the split are
// stable (MPI_Comm_split). The split's lane is color itself — a split IS a
// dos-arbitrate lease keyed on the color, so two splits naming overlapping lanes
// serialize by refusal at the arbiter, not by any lock in this package. The returned
// Group carries the same wave/parent and re-ranks its members from 0.
//
// color maps a member to a lane name (its split key). A member mapped to "" is
// excluded from every split (it joins no color).
func (g *Group) Split(color func(Member) string) map[string]*Group {
	buckets := map[string][]Member{}
	for _, m := range g.members {
		c := color(m)
		if c == "" {
			continue
		}
		buckets[c] = append(buckets[c], m)
	}
	out := make(map[string]*Group, len(buckets))
	for c, ms := range buckets {
		// ms is already in canonical order (we iterated g.members in order); re-rank
		// from 0 by reusing New, which is order-independent and dedup-checked.
		sub, err := New(g.wave, g.parent, ms)
		if err != nil {
			// A split can only narrow an already-deduplicated set, so New cannot fail
			// here; guard defensively rather than panic.
			continue
		}
		out[c] = sub
	}
	return out
}

// SplitLane is Split with the color taken from each member's Lane field — the common
// case where the split key already IS the dos.toml lane the member holds. A member
// with an empty Lane joins no split.
func (g *Group) SplitLane() map[string]*Group {
	return g.Split(func(m Member) string { return m.Lane })
}

// Lanes returns the distinct lanes the group's members hold, in canonical (sorted)
// order — the lease partition a wave occupies. Members with an empty Lane are
// omitted.
func (g *Group) Lanes() []string {
	seen := map[string]struct{}{}
	var out []string
	for _, m := range g.members {
		if m.Lane == "" {
			continue
		}
		if _, ok := seen[m.Lane]; ok {
			continue
		}
		seen[m.Lane] = struct{}{}
		out = append(out, m.Lane)
	}
	sort.Strings(out)
	return out
}

// Broadcast adjudicates a scope-bounded context share once per member. The payload
// Ref is copied through unchanged: its Taint/Scope remain the bound on the share.
// A private ScopeAgent payload cannot be broadcast to a multi-member group because
// that would widen it across agents before the floor sees the call.
func (g *Group) Broadcast(ctx context.Context, k abi.Kernel, payload abi.Ref) (CollectiveRequest, error) {
	return g.IBroadcast(ctx, k, payload)
}

// IBroadcast is the non-blocking Broadcast variant. It returns the per-rank
// SubmissionHandles with State=StatusPending; callers can complete them via Reap.
func (g *Group) IBroadcast(ctx context.Context, k abi.Kernel, payload abi.Ref) (CollectiveRequest, error) {
	if err := g.checkBroadcastScope(payload); err != nil {
		return CollectiveRequest{}, err
	}
	return g.submitCollective(ctx, k, ToolBroadcast, func(rank int, m Member) abi.Ref {
		return payload
	}, nil)
}

// Scatter adjudicates one per-member goal slice per rank. goals[r] is the Ref
// handed to the member at rank r; the Ref's Taint/Scope ride unchanged.
func (g *Group) Scatter(ctx context.Context, k abi.Kernel, goals []abi.Ref) (CollectiveRequest, error) {
	return g.IScatter(ctx, k, goals)
}

// IScatter is the non-blocking Scatter variant.
func (g *Group) IScatter(ctx context.Context, k abi.Kernel, goals []abi.Ref) (CollectiveRequest, error) {
	if len(goals) != len(g.members) {
		return CollectiveRequest{}, fmt.Errorf("%w: %d goals for %d members", ErrArity, len(goals), len(g.members))
	}
	return g.submitCollective(ctx, k, ToolScatter, func(rank int, m Member) abi.Ref {
		return goals[rank]
	}, nil)
}

// Barrier adjudicates one witness-read-back arrival per member. It is not a
// hardware sync; it is a deterministic fold over N independent floor crossings.
func (g *Group) Barrier(ctx context.Context, k abi.Kernel) (CollectiveRequest, error) {
	return g.IBarrier(ctx, k)
}

// IBarrier is the non-blocking Barrier variant.
func (g *Group) IBarrier(ctx context.Context, k abi.Kernel) (CollectiveRequest, error) {
	return g.submitCollective(ctx, k, ToolBarrier, func(rank int, m Member) abi.Ref {
		return barrierArgs(g.wave, rank, len(g.members), m)
	}, map[string]string{"witness": "dos-witness-claim"})
}

// Gather adjudicates each member's output through the kernel (one Submit per member,
// the adjudication floor — no fan-out bypasses the chokepoint) and folds the results
// in RANK order through modelroute.Combine. outputs[r] is the output produced by the
// member at rank r; passing them indexed by rank is the caller's contract. The fold
// is deterministic on STRUCTURE: rank order is preserved into Combine even though the
// member outputs themselves come from non-bit-exact engines.
//
// reduce selects the modelroute fold (Concat for a gather, Vote for a quorum, …). A
// member whose Submit is refused fails the whole Gather closed — there is no
// collective that silently drops a refused call.
func (g *Group) Gather(ctx context.Context, k abi.Kernel, outputs []string, reduce modelroute.Reduction) (modelroute.Result, error) {
	req, err := g.IGather(ctx, k, outputs, reduce)
	if err != nil {
		return modelroute.Result{}, err
	}
	return req.Combine()
}

// IGather is the non-blocking Gather variant. It submits every member output
// through the floor in rank order, records the returned handles, and leaves the
// rank-ordered Combine fold for ReapCombine/Combine.
func (g *Group) IGather(ctx context.Context, k abi.Kernel, outputs []string, reduce modelroute.Reduction) (GatherRequest, error) {
	if k == nil {
		return GatherRequest{}, ErrNoKernel
	}
	if len(outputs) != len(g.members) {
		return GatherRequest{}, fmt.Errorf("%w: %d outputs for %d members", ErrArity, len(outputs), len(g.members))
	}
	votes := make([]modelroute.Vote, len(g.members))
	req, err := g.submitCollective(ctx, k, ToolGather, func(rank int, m Member) abi.Ref {
		return gatherArgs(g.wave, rank, len(g.members), m, outputs[rank])
	}, nil)
	if err != nil {
		return GatherRequest{}, err
	}
	for r, m := range g.members {
		votes[r] = modelroute.Vote{
			Member: modelroute.Member{Model: m.ID, Weight: m.Weight, Role: m.Lane},
			Output: outputs[r],
		}
	}
	return GatherRequest{CollectiveRequest: req, Votes: votes, Reduce: reduce}, nil
}

func (g *Group) checkBroadcastScope(payload abi.Ref) error {
	if len(g.members) > 1 && payload.Scope == abi.ScopeAgent {
		return ErrScopeWiden
	}
	return nil
}

func (g *Group) submitCollective(ctx context.Context, k abi.Kernel, tool string, args func(int, Member) abi.Ref, meta map[string]string) (CollectiveRequest, error) {
	if k == nil {
		return CollectiveRequest{}, ErrNoKernel
	}
	req := CollectiveRequest{Tool: tool, State: abi.StatusPending}
	for r, m := range g.members {
		call := &abi.ToolCall{
			Tool: tool,
			Args: args(r, m),
			Meta: collectiveMeta(g.wave, r, len(g.members), m, meta),
		}
		h, verdict := k.Submit(ctx, call)
		if verdict.Kind != abi.VerdictAllow {
			return req, fmt.Errorf("%w: %s rank %d (%s) by %s", ErrDenied, tool, r, abi.ReasonName(verdict.Reason), verdict.By)
		}
		req.Handles = append(req.Handles, h)
	}
	return req, nil
}

func collectiveMeta(wave string, rank, size int, m Member, extra map[string]string) map[string]string {
	meta := map[string]string{
		"comm":           "true",
		"readOnlyHint":   "true",
		"idempotentHint": "true",
		"wave":           wave,
		"rank":           fmt.Sprintf("%d", rank),
		"size":           fmt.Sprintf("%d", size),
		"member":         m.ID,
	}
	if m.Lane != "" {
		meta["lane"] = m.Lane
	}
	for k, v := range extra {
		meta[k] = v
	}
	return meta
}

// gatherArgs encodes the per-member gather descriptor as an inline, agent-scoped,
// tainted Ref — the default fail-closed (Tainted, ScopeAgent) shape; a Gather never
// widens sharing on its own.
func gatherArgs(wave string, rank, size int, m Member, output string) abi.Ref {
	body := map[string]any{
		"wave":   wave,
		"rank":   rank,
		"size":   size,
		"id":     m.ID,
		"lane":   m.Lane,
		"outlen": len(output),
	}
	encoded, _ := json.Marshal(body)
	return abi.Ref{
		Kind:   abi.RefInline,
		Inline: encoded,
		Len:    int64(len(encoded)),
		Scope:  abi.ScopeAgent,
		Taint:  abi.TaintTainted,
	}
}

func barrierArgs(wave string, rank, size int, m Member) abi.Ref {
	body := map[string]any{
		"wave":    wave,
		"rank":    rank,
		"size":    size,
		"id":      m.ID,
		"lane":    m.Lane,
		"barrier": "arrived",
	}
	encoded, _ := json.Marshal(body)
	return abi.Ref{
		Kind:   abi.RefInline,
		Inline: encoded,
		Len:    int64(len(encoded)),
		Scope:  abi.ScopeAgent,
		Taint:  abi.TaintTainted,
	}
}
