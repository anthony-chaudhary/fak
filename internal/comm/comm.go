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

// ToolGather is the synthetic tool name a Gather submits per member so the fold is
// adjudicated at the floor like any other call.
const ToolGather = "comm.gather"

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
	// ErrDenied is returned when the adjudication floor refuses a gather submission.
	ErrDenied = errors.New("comm: gather denied")
	// ErrArity is returned when a gather's output count does not match the group size.
	ErrArity = errors.New("comm: output count does not match group size")
)

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
	if k == nil {
		return modelroute.Result{}, ErrNoKernel
	}
	if len(outputs) != len(g.members) {
		return modelroute.Result{}, fmt.Errorf("%w: %d outputs for %d members", ErrArity, len(outputs), len(g.members))
	}
	votes := make([]modelroute.Vote, len(g.members))
	for r, m := range g.members {
		if err := g.adjudicateMember(ctx, k, r, m, outputs[r]); err != nil {
			return modelroute.Result{}, err
		}
		votes[r] = modelroute.Vote{
			Member: modelroute.Member{Model: m.ID, Weight: m.Weight, Role: m.Lane},
			Output: outputs[r],
		}
	}
	return modelroute.Combine(reduce, votes)
}

// adjudicateMember submits one comm.gather call for the member at rank r and refuses
// the gather if the floor does not Allow it.
func (g *Group) adjudicateMember(ctx context.Context, k abi.Kernel, rank int, m Member, output string) error {
	args := gatherArgs(g.wave, rank, len(g.members), m, output)
	call := &abi.ToolCall{
		Tool: ToolGather,
		Args: args,
		Meta: map[string]string{"comm": "true", "readOnlyHint": "true", "idempotentHint": "true"},
	}
	_, verdict := k.Submit(ctx, call)
	if verdict.Kind != abi.VerdictAllow {
		return fmt.Errorf("%w: rank %d (%s) by %s", ErrDenied, rank, abi.ReasonName(verdict.Reason), verdict.By)
	}
	return nil
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
