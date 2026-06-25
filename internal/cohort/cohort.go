package cohort

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/comm"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Ready reports that the cohort leaf is linked into the build.
func Ready() bool { return true }

var (
	// ErrNilGroup is returned when a Cohort is built without a comm.Group.
	ErrNilGroup = errors.New("cohort: nil group")
	// ErrNoMember is returned when a vote or failure report names an unknown member.
	ErrNoMember = errors.New("cohort: no such member")
	// ErrEmptyCohort is returned when Shrink would remove every member.
	ErrEmptyCohort = errors.New("cohort: shrink removed every member")
	// ErrInvalidQuorum is returned when Agree receives an unusable quorum floor.
	ErrInvalidQuorum = errors.New("cohort: invalid quorum floor")
	// ErrNoAgreement is returned when present votes do not satisfy the quorum floor.
	ErrNoAgreement = errors.New("cohort: no agreement")
)

// Cohort is a generation-stamped agent working group. The wrapped comm.Group
// supplies the deterministic ordered member set; generation advances only when a
// caller deliberately shrinks the group after reporting failed members.
type Cohort struct {
	group      *comm.Group
	generation uint64
}

// Agreement is the observable outcome of Agree.
type Agreement struct {
	Generation    uint64
	Agreed        bool
	Result        modelroute.Result
	Required      float64
	WinningWeight float64
	Present       int
	Missing       []string
}

// New wraps an existing comm.Group at generation.
func New(group *comm.Group, generation uint64) (*Cohort, error) {
	if group == nil {
		return nil, ErrNilGroup
	}
	if group.Size() == 0 {
		return nil, ErrEmptyCohort
	}
	return &Cohort{group: group, generation: generation}, nil
}

// FromMembers builds a generation-0 Cohort from a comm.Group member list.
func FromMembers(waveID, parentTraceID string, members []comm.Member) (*Cohort, error) {
	g, err := comm.New(waveID, parentTraceID, members)
	if err != nil {
		return nil, err
	}
	return New(g, 0)
}

// Group returns the deterministic member set this cohort wraps.
func (c *Cohort) Group() *comm.Group {
	if c == nil {
		return nil
	}
	return c.group
}

// Generation is the monotone shrink generation.
func (c *Cohort) Generation() uint64 {
	if c == nil {
		return 0
	}
	return c.generation
}

// Size reports the number of current members.
func (c *Cohort) Size() int {
	if c == nil || c.group == nil {
		return 0
	}
	return c.group.Size()
}

// Members returns the cohort members in rank order.
func (c *Cohort) Members() []comm.Member {
	if c == nil || c.group == nil {
		return nil
	}
	return membersOf(c.group)
}

// MemberIDs returns the cohort member IDs in rank order.
func (c *Cohort) MemberIDs() []string {
	ms := c.Members()
	ids := make([]string, len(ms))
	for i, m := range ms {
		ids[i] = m.ID
	}
	return ids
}

// MajorityFloor returns the unweighted majority quorum floor for size members.
func MajorityFloor(size int) float64 {
	if size <= 0 {
		return 0
	}
	return float64(size/2 + 1)
}

// MajorityFloor returns the unweighted majority quorum floor for this cohort.
func (c *Cohort) MajorityFloor() float64 {
	return MajorityFloor(c.Size())
}

// Shrink removes the reported failed members and returns a new generation. The
// survivors are collected in the current rank order so the next Combine input
// order remains deterministic.
func (c *Cohort) Shrink(failed ...string) (*Cohort, error) {
	if err := c.valid(); err != nil {
		return nil, err
	}
	dead := make(map[string]struct{}, len(failed))
	for _, id := range failed {
		if id == "" {
			return nil, fmt.Errorf("%w: empty member id", ErrNoMember)
		}
		if _, err := c.group.Rank(id); err != nil {
			return nil, fmt.Errorf("%w: %q", ErrNoMember, id)
		}
		dead[id] = struct{}{}
	}

	var survivors []comm.Member
	for _, m := range membersOf(c.group) {
		if _, failed := dead[m.ID]; failed {
			continue
		}
		survivors = append(survivors, m)
	}
	if len(survivors) == 0 {
		return nil, ErrEmptyCohort
	}

	parent := parentTraceID(c.group)
	g, err := comm.New(c.group.WaveID(), parent, survivors)
	if err != nil {
		return nil, err
	}
	return &Cohort{group: g, generation: c.generation + 1}, nil
}

// Agree folds present member outputs with modelroute.ReduceVote and requires the
// winning output's tally to meet quorumFloor. A missing member is not silently
// removed from the agreement: it is reported in Agreement.Missing and contributes
// no weight toward the floor, so absence can fail the agreement closed.
func (c *Cohort) Agree(outputs map[string]string, quorumFloor float64) (Agreement, error) {
	if err := c.valid(); err != nil {
		return Agreement{}, err
	}
	if quorumFloor <= 0 || math.IsNaN(quorumFloor) || math.IsInf(quorumFloor, 0) {
		return Agreement{}, fmt.Errorf("%w: %v", ErrInvalidQuorum, quorumFloor)
	}

	members := membersOf(c.group)
	known := make(map[string]struct{}, len(members))
	for _, m := range members {
		known[m.ID] = struct{}{}
	}
	for id := range outputs {
		if _, ok := known[id]; !ok {
			return Agreement{}, fmt.Errorf("%w: %q", ErrNoMember, id)
		}
	}

	var votes []modelroute.Vote
	var missing []string
	for _, m := range members {
		out, ok := outputs[m.ID]
		if !ok {
			missing = append(missing, m.ID)
			continue
		}
		votes = append(votes, modelroute.Vote{
			Member: modelroute.Member{Model: m.ID, Weight: m.Weight, Role: m.Lane},
			Output: out,
		})
	}

	agreement := Agreement{
		Generation: c.generation,
		Required:   quorumFloor,
		Present:    len(votes),
		Missing:    missing,
	}
	if len(votes) == 0 {
		return agreement, ErrNoAgreement
	}

	result, err := modelroute.Combine(modelroute.ReduceVote, votes)
	if err != nil {
		return agreement, err
	}
	agreement.Result = result
	agreement.WinningWeight = result.Tally[result.Output]
	agreement.Agreed = agreement.WinningWeight >= quorumFloor
	if !agreement.Agreed {
		return agreement, ErrNoAgreement
	}
	return agreement, nil
}

func (c *Cohort) valid() error {
	if c == nil || c.group == nil {
		return ErrNilGroup
	}
	if c.group.Size() == 0 {
		return ErrEmptyCohort
	}
	return nil
}

func membersOf(g *comm.Group) []comm.Member {
	out := make([]comm.Member, 0, g.Size())
	for r := 0; r < g.Size(); r++ {
		m, err := g.Member(r)
		if err == nil {
			out = append(out, m)
		}
	}
	return out
}

func parentTraceID(g *comm.Group) string {
	m, err := g.Membership(0)
	if err != nil {
		return ""
	}
	return m.ParentTraceID
}
