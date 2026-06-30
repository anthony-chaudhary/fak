package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopindex"
)

func TestTaskHandoffNextStepIsLoopIndexWitness(t *testing.T) {
	rep := loopindex.Score(collectLoopIndex(repoRoot()))
	learn := rep.StageDetail[5]
	if learn.Name != loopindex.StageLearn {
		t.Fatalf("stage[5] = %q, want learn", learn.Name)
	}
	for _, p := range learn.Probes {
		if p.Name == "task_handoff_next_step" {
			if !p.Pass {
				t.Fatalf("task_handoff_next_step probe is false; `fak task handoff` is not witnessed by loop-index")
			}
			return
		}
	}
	t.Fatalf("learn stage missing task_handoff_next_step probe: %+v", learn.Probes)
}
