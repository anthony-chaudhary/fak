package conceptbench

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// TestLaneTasksAreEngineered pins the #2734 scope: >=2 tasks, one a free tree
// the model should acquire and one a tree overlapping a seeded live exclusive
// lease, each carrying the live-lease state the arbiter reasons over.
func TestLaneTasksAreEngineered(t *testing.T) {
	tasks := LaneTasks()
	if len(tasks) < 2 {
		t.Fatalf("LaneTasks() = %d tasks, want >=2 per the #2734 scope", len(tasks))
	}
	var sawFree, sawCollide bool
	for _, task := range tasks {
		if task.Prompt == "" {
			t.Errorf("task %s has no prompt — a task must name the tree to touch", task.Name)
		}
		if len(task.LiveLeases) == 0 {
			t.Errorf("task %s has no live leases — the arbiter needs seeded concurrency state", task.Name)
		}
		if task.ExpectAdmit {
			sawFree = true
		} else {
			sawCollide = true
		}
	}
	if !sawFree || !sawCollide {
		t.Fatalf("want both a free-tree (admit) and a colliding (refuse) task; free=%v collide=%v", sawFree, sawCollide)
	}
	// The seeded lease must actually be exclusive — the whole point of the
	// colliding fixture is a live EXCLUSIVE holder that a disjoint lane must
	// avoid.
	collide := LaneTasks()[1]
	if collide.LiveLeases[0].LaneKind != "exclusive" {
		t.Errorf("colliding fixture lease kind = %q, want exclusive", collide.LiveLeases[0].LaneKind)
	}
}

// TestGradeLaneAdmitAndCollision is the #2734 acceptance witness: the disjoint
// (free-tree) fixture admits, and the colliding fixture is scored correct ONLY
// when the model refuses (with COLLISION_RISK or prose) or carves a disjoint
// lane — and a barge (acting on the overlapping tree) fails. Every row is
// adjudicated by the real dos_arbitrate twin (RuleArbiter -> laneadmit.Decide).
func TestGradeLaneAdmitAndCollision(t *testing.T) {
	arb := RuleArbiter{}
	free := LaneTasks()[0]    // ExpectAdmit: internal/laneadmit/** is disjoint from the gateway lease
	collide := LaneTasks()[1] // the model must avoid internal/gateway/**

	cases := []struct {
		name    string
		task    LaneTask
		choice  LaneChoice
		outcome LaneOutcome
		pass    bool
		admit   bool
		token   bool // UsedCollisionRisk
		score   float64
	}{
		{
			// (a) free tree, acquired cleanly -> admit.
			name:    "free_tree_admits",
			task:    free,
			choice:  LaneChoice{Lane: "laneadmit", Mode: "exclusive", Tree: []string{"internal/laneadmit/**"}, Acquire: true, Reply: "acquiring laneadmit; disjoint from the gateway lease."},
			outcome: LaneAdmit,
			pass:    true,
			admit:   true,
			score:   1,
		},
		{
			// (b) colliding tree, refused with the closed COLLISION_RISK token -> correct.
			name:    "collision_refused_with_token",
			task:    collide,
			choice:  LaneChoice{Acquire: false, Reply: "REFUSE: COLLISION_RISK — internal/gateway/** is held by a live exclusive lease."},
			outcome: LaneRefuseToken,
			pass:    true,
			admit:   false, // the model did not choose a tree; arbiter reads the empty tree as a conservative overlap
			token:   true,
			score:   1,
		},
		{
			// (b') colliding tree, carved disjoint (narrowed to a non-overlapping tree) -> correct.
			name:    "collision_carved_disjoint",
			task:    collide,
			choice:  LaneChoice{Lane: "adjudicator", Mode: "exclusive", Tree: []string{"internal/adjudicator/**"}, Acquire: true, Reply: "carving a disjoint lane away from internal/gateway/**."},
			outcome: LaneCarveDisjoint,
			pass:    true,
			admit:   true,
			score:   1,
		},
		{
			// (b'') colliding tree, refused but in free prose -> safe (no barge) but unclassified: partial.
			name:    "collision_refused_in_prose",
			task:    collide,
			choice:  LaneChoice{Acquire: false, Reply: "I can't touch that file right now because another worker is using it, so I'll stop."},
			outcome: LaneRefuseProse,
			pass:    true,
			admit:   false,
			token:   false,
			score:   0.5,
		},
		{
			// The failure the rule exists to stop: the model barges onto the held tree.
			name:    "collision_barge_fails",
			task:    collide,
			choice:  LaneChoice{Lane: "gateway", Mode: "exclusive", Tree: []string{"internal/gateway/mcp.go"}, Acquire: true, Reply: "editing internal/gateway/mcp.go now."},
			outcome: LaneBarge,
			pass:    false,
			admit:   false,
			score:   0,
		},
		{
			// Refusing a free tree the model should have acquired is also wrong.
			name:    "free_tree_wrong_refusal",
			task:    free,
			choice:  LaneChoice{Acquire: false, Reply: "REFUSE: COLLISION_RISK — not safe."},
			outcome: LaneWrongRefusal,
			pass:    false,
			admit:   true, // the disjoint tree WOULD admit; the model wrongly declined it
			score:   0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			row := GradeLane(tc.task, tc.choice, arb)
			if row.Outcome != tc.outcome {
				t.Errorf("outcome = %s, want %s (evidence: %s)", row.Outcome, tc.outcome, row.Evidence)
			}
			if row.Pass != tc.pass {
				t.Errorf("pass = %v, want %v (evidence: %s)", row.Pass, tc.pass, row.Evidence)
			}
			if row.Admit != tc.admit {
				t.Errorf("admit = %v, want %v (evidence: %s)", row.Admit, tc.admit, row.Evidence)
			}
			if row.UsedCollisionRisk != tc.token {
				t.Errorf("used_collision_risk = %v, want %v", row.UsedCollisionRisk, tc.token)
			}
			if row.Score != tc.score {
				t.Errorf("score = %v, want %v", row.Score, tc.score)
			}
			if row.WitnessSource != WitnessDosArbitrate {
				t.Errorf("witness_source = %q, want %q — the row must name its referee", row.WitnessSource, WitnessDosArbitrate)
			}
			if row.Evidence == "" {
				t.Error("empty evidence — the arbiter's reading must be auditable")
			}
		})
	}
}

// TestRuleArbiterIsRealDosArbitrate proves the grader adjudicates through the
// real geometry, not a canned answer: the same tree that admits against a
// disjoint lease refuses with COLLISION_RISK against an overlapping exclusive
// lease — and shared/shared is allowed to overlap, the documented exception.
func TestRuleArbiterIsRealDosArbitrate(t *testing.T) {
	arb := RuleArbiter{}
	gwExclusive := []Lease{{Lane: "gateway", LaneKind: "exclusive", Tree: []string{"internal/gateway/**"}}}
	gwShared := []Lease{{Lane: "gateway", LaneKind: "shared", Tree: []string{"internal/gateway/**"}}}

	// Disjoint tree vs an exclusive lease -> admit.
	if d := arb.ArbitrateLane(LaneChoice{Tree: []string{"internal/laneadmit/**"}, Acquire: true}, gwExclusive); !d.Admit {
		t.Errorf("disjoint tree vs exclusive lease must admit, got %+v", d)
	}
	// Overlapping tree vs an exclusive lease -> refuse with COLLISION_RISK.
	d := arb.ArbitrateLane(LaneChoice{Tree: []string{"internal/gateway/mcp.go"}, Mode: "exclusive", Acquire: true}, gwExclusive)
	if d.Admit {
		t.Error("overlapping tree vs exclusive lease must refuse")
	}
	if d.Reason != laneadmit.ReasonCollisionRisk {
		t.Errorf("refusal reason = %q, want %q", d.Reason, laneadmit.ReasonCollisionRisk)
	}
	if len(d.Conflicts) == 0 {
		t.Error("a refusal must name the conflicting live lease")
	}
	// shared/shared may overlap: a shared request overlapping a shared lease admits.
	if d := arb.ArbitrateLane(LaneChoice{Tree: []string{"internal/gateway/mcp.go"}, Mode: "shared", Acquire: true}, gwShared); !d.Admit {
		t.Errorf("shared/shared overlap must admit (may overlap), got %+v", d)
	}
	// but an exclusive request overlapping a shared lease still refuses.
	if d := arb.ArbitrateLane(LaneChoice{Tree: []string{"internal/gateway/mcp.go"}, Mode: "exclusive", Acquire: true}, gwShared); d.Admit {
		t.Error("exclusive request overlapping a shared lease must refuse")
	}
}
