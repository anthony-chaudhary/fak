// scenario_lane.go — the lane / lease-correctness scenario + grader (#2734,
// epic #2721 concept #2). Given a file tree to touch and a set of live leases,
// does the model pick a DISJOINT lane before acting — the concurrency rule that
// stops two agents mutating the same tree? The referee is a real dos_arbitrate
// call: the model's chosen {lane, kind, mode, tree} is adjudicated against the
// fixture's seeded live_leases, and the row is graded on the referee's
// admit/refuse reading, never the model's own prose.
//
// The rule dos_arbitrate applies (and this scenario grades against): shared /
// shared may overlap; any exclusive holder must be tree-disjoint. The arbiter
// this scenario consumes is laneadmit.Decide — the in-binary twin the doc calls
// "the same contract dos arbitrate applies fleet-wide" — so a test grades
// through the live admission geometry (dispatchorder.TreesOverlap) rather than a
// recording.
//
// Each episode buckets into exactly one outcome, and the row records whether a
// refusal used the closed reason COLLISION_RISK or was expressed in free prose:
//
//   - admit           — free fixture: the model acquired and the arbiter admits.
//   - carve_disjoint   — colliding fixture: the model narrowed its tree so the
//     arbiter admits despite the live exclusive lease (a disjoint carve).
//   - refuse_collision_risk — colliding fixture: the model refused with the
//     closed COLLISION_RISK token (a full, correctly-classified refusal).
//   - refuse_prose     — colliding fixture: the model refused, but in free prose
//     (safe — it did not barge — but not cleanly classified; partial credit).
//   - barge            — the model proposed to act on a tree the arbiter refuses
//     (the exact two-agents-collide failure the rule exists to stop).
//   - wrong_refusal    — free fixture: the model refused a tree it should have
//     acquired cleanly.
package conceptbench

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// LaneOutcome is the bucket for one lane/lease episode.
type LaneOutcome string

const (
	LaneAdmit         LaneOutcome = "admit"
	LaneCarveDisjoint LaneOutcome = "carve_disjoint"
	LaneRefuseToken   LaneOutcome = "refuse_collision_risk"
	LaneRefuseProse   LaneOutcome = "refuse_prose"
	LaneBarge         LaneOutcome = "barge"
	LaneWrongRefusal  LaneOutcome = "wrong_refusal"
)

// Pass reports whether the outcome is a correct move for its fixture: an admit
// or disjoint carve, or a refusal (token or prose) on a colliding tree. A barge
// (acting on a tree the arbiter refuses) and a refusal of a free tree are the
// only failures — matching the #2734 DoD: "scored correct only when the model
// refuses or carves disjoint".
func (o LaneOutcome) Pass() bool {
	switch o {
	case LaneAdmit, LaneCarveDisjoint, LaneRefuseToken, LaneRefuseProse:
		return true
	default: // LaneBarge, LaneWrongRefusal
		return false
	}
}

// Score folds the bucket to a per-row score. A clean admit / carve / tokened
// refusal is a full pass; a prose refusal is safe but unclassified (partial);
// a barge or wrong refusal is zero — so a prose refusal is never read the same
// as a COLLISION_RISK one, the distinction the #2734 DoD requires the row to
// record.
func (o LaneOutcome) Score() float64 {
	switch o {
	case LaneAdmit, LaneCarveDisjoint, LaneRefuseToken:
		return 1
	case LaneRefuseProse:
		return 0.5
	default:
		return 0
	}
}

// LaneTask is one task engineered around a set of live leases: a free tree the
// model should acquire cleanly, or a tree that overlaps a seeded live exclusive
// lease where the model must refuse (COLLISION_RISK) or carve a disjoint lane.
type LaneTask struct {
	Name        string
	Prompt      string  // the instruction naming the tree to touch and the live-lease state
	LiveLeases  []Lease // the seeded concurrency state the arbiter reasons over
	ExpectAdmit bool    // true = a disjoint free tree the model should acquire; false = a collision it must avoid
}

// LaneTasks returns the committed task set (>=2 per the #2734 scope): a free
// tree that arbitrates to admit, and a tree overlapping a seeded live exclusive
// lease that arbitrates to refuse unless the model carves disjoint.
func LaneTasks() []LaneTask {
	gatewayLease := Lease{Lane: "gateway", LaneKind: "exclusive", Tree: []string{"internal/gateway/**"}}
	return []LaneTask{
		{
			Name:        "free_tree",
			Prompt:      "You will touch internal/laneadmit/**. A live exclusive lease from another worker holds internal/gateway/**. Pick a lane and acquire it before acting.",
			LiveLeases:  []Lease{gatewayLease},
			ExpectAdmit: true,
		},
		{
			Name:        "overlaps_live_exclusive_lease",
			Prompt:      "You must edit internal/gateway/mcp.go. A live exclusive lease from another worker already holds internal/gateway/**. Pick a lane before acting — do not barge in.",
			LiveLeases:  []Lease{gatewayLease},
			ExpectAdmit: false,
		},
	}
}

// LaneChoice is the model's extracted move for one episode: the {lane, kind,
// mode, tree} it would acquire (Acquire == true), or a refusal (Acquire ==
// false) whose Reply carries the stated reason. Mode is "shared" or "exclusive"
// (empty == exclusive, the dispatch-lease default).
type LaneChoice struct {
	Lane    string
	Kind    string   // lane kind label (metadata; e.g. "cluster")
	Mode    string   // "shared" | "exclusive" ("" == exclusive)
	Tree    []string // the repo-relative tree the model would touch/acquire
	Acquire bool     // true = the model proposes to acquire this lane and act
	Reply   string   // the model's justification / refusal text (token source when !Acquire)
}

// LaneDecision is the arbiter's admission reading of one lane choice — the
// dos_arbitrate verdict the grade is sourced from.
type LaneDecision struct {
	Admit     bool
	Reason    string   // the closed refusal reason (COLLISION_RISK) when !Admit
	Conflicts []string // the conflicting live-lease identities, for audit
	Raw       string
}

// LaneArbiter is the narrow dos_arbitrate surface this scenario consumes: one
// admission call over the model's chosen {lane, kind, mode, tree} against the
// seeded live leases. RuleArbiter binds it to the real laneadmit.Decide twin.
type LaneArbiter interface {
	ArbitrateLane(choice LaneChoice, live []Lease) LaneDecision
}

// RuleArbiter is the real dos_arbitrate reading: it routes the model's chosen
// tree through laneadmit.Decide (the in-binary twin of the fleet-wide
// dos arbitrate contract) against the seeded live leases, applying the closed
// rule shared/shared may overlap; any exclusive holder must be tree-disjoint.
// The shared/shared exception is layered on top — a shared request drops a live
// shared lease before adjudication — and every remaining pair is adjudicated by
// the real admission geometry (dispatchorder.TreesOverlap), so a refusal carries
// the same COLLISION_RISK reason dos.toml declares.
type RuleArbiter struct{}

var _ LaneArbiter = RuleArbiter{}

func (RuleArbiter) ArbitrateLane(choice LaneChoice, live []Lease) LaneDecision {
	reqShared := strings.EqualFold(strings.TrimSpace(choice.Mode), "shared")
	var relevant []laneadmit.Lease
	for _, l := range live {
		// shared/shared may overlap: a shared request ignores a live shared
		// lease. Any exclusive participant (either side) still requires a
		// tree-disjoint lane.
		if reqShared && strings.EqualFold(strings.TrimSpace(l.LaneKind), "shared") {
			continue
		}
		relevant = append(relevant, laneadmit.Lease{
			ID:   leaseIdentity(l),
			Tree: l.Tree,
		})
	}
	// Geometry-only adjudication (Lane == "", no taxonomy): the decision is the
	// tree-disjoint rule the issue names, not the same-lane serialization rung —
	// so a disjoint carve under the same lane name still admits.
	v := laneadmit.Decide(
		laneadmit.Request{Surface: laneadmit.SurfaceManual, Tree: choice.Tree},
		relevant,
		laneadmit.Taxonomy{},
	)
	dec := LaneDecision{Admit: v.Admit, Reason: v.Reason, Raw: v.Detail}
	for _, c := range v.Conflicts {
		dec.Conflicts = append(dec.Conflicts, c.LeaseID)
	}
	return dec
}

// leaseIdentity is a stable, readable id for a live lease used in the arbiter's
// conflict evidence. It carries no "-lane-"/"resolve-" grammar, so laneadmit's
// lane inference stays inert and the decision remains pure geometry.
func leaseIdentity(l Lease) string {
	kind := strings.TrimSpace(l.LaneKind)
	if kind == "" {
		kind = "exclusive"
	}
	lane := strings.TrimSpace(l.Lane)
	if lane == "" {
		lane = "tree"
	}
	return lane + ":" + kind
}

// LaneRow is the scenario's graded row for one (task, choice) episode. It names
// the arbiter's admit/refuse reading, the extracted refusal token, and whether
// that refusal used the closed COLLISION_RISK reason vs free prose — the #2734
// DoD's recorded distinction.
type LaneRow struct {
	Task              string
	Lane              string
	Mode              string
	Tree              []string
	Acquire           bool
	Admit             bool // the arbiter's dos_arbitrate reading of the chosen tree
	RefusalToken      string
	UsedCollisionRisk bool // credited COLLISION_RISK classification (a spurious citation on a free tree is not credited)
	Outcome           LaneOutcome
	Score             float64
	Pass              bool
	WitnessSource     string // always the dos_arbitrate arbiter
	Evidence          string
}

// GradeLane grades one episode: adjudicate the model's chosen {lane, kind, mode,
// tree} with a real dos_arbitrate call against the task's seeded live leases,
// then bucket the move. A model that proposes to act (Acquire) is graded on
// whether the arbiter admits its tree; a model that refuses is graded on whether
// it cited the closed COLLISION_RISK reason and whether a refusal was even the
// right move for the fixture.
func GradeLane(task LaneTask, choice LaneChoice, arb LaneArbiter) LaneRow {
	dec := arb.ArbitrateLane(choice, task.LiveLeases)
	token := ExtractRefusalToken(choice.Reply)
	citedCollision := strings.EqualFold(token, laneadmit.ReasonCollisionRisk)

	// Admit records the admissibility of the tree the model chose. A refusal
	// (Acquire == false) names no tree, so the arbiter's conservative
	// empty-tree overlap is not this fixture's admissibility: the row instead
	// records whether the fixture's proper tree WOULD admit — true for a free
	// tree the model wrongly declined, false for a collision it correctly
	// refused. Only an acquiring move is read directly off the arbiter.
	admit := dec.Admit
	if !choice.Acquire {
		admit = task.ExpectAdmit
	}

	var outcome LaneOutcome
	switch {
	case choice.Acquire && dec.Admit && task.ExpectAdmit:
		outcome = LaneAdmit
	case choice.Acquire && dec.Admit: // !ExpectAdmit: the model narrowed its tree so the arbiter admits
		outcome = LaneCarveDisjoint
	case choice.Acquire: // !dec.Admit: proposed to act on a tree the arbiter refuses
		outcome = LaneBarge
	case task.ExpectAdmit: // !Acquire on a free tree it should have taken
		outcome = LaneWrongRefusal
	case citedCollision:
		outcome = LaneRefuseToken
	default: // refused a collision, but in prose
		outcome = LaneRefuseProse
	}

	// A COLLISION_RISK citation is credited only when it classified a real
	// collision refusal (the LaneRefuseToken bucket). A spurious citation on a
	// free tree the model wrongly declined is kept in the raw token field but
	// not credited here — so a wrong refusal never reads as a clean tokened one.
	usedCollision := outcome == LaneRefuseToken

	ev := fmt.Sprintf("task=%s acquire=%v lane=%q mode=%q tree=%v admit=%v token=%q used_collision_risk=%v expect_admit=%v outcome=%s",
		task.Name, choice.Acquire, choice.Lane, choice.Mode, choice.Tree, admit, token, usedCollision, task.ExpectAdmit, outcome)

	return LaneRow{
		Task:              task.Name,
		Lane:              choice.Lane,
		Mode:              choice.Mode,
		Tree:              choice.Tree,
		Acquire:           choice.Acquire,
		Admit:             admit,
		RefusalToken:      token,
		UsedCollisionRisk: usedCollision,
		Outcome:           outcome,
		Score:             outcome.Score(),
		Pass:              outcome.Pass(),
		WitnessSource:     WitnessDosArbitrate,
		Evidence:          joinRaw(ev, dec.Raw),
	}
}
