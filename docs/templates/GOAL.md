---
title: "GOAL.md loop template — objective, plan, and scratch"
description: "The starter GOAL.md a fak loop fills in: loop id, witness, and iteration budget in front-matter, then the objective, plan, and scratch sections."
# Note: Save goal specs with a unique name per goal (e.g., goals/GOAL-<slug>.md or GOAL-<slug>.md)
# to prevent clobbering, race conditions, and memory corruption across concurrent goals/subagents.
# Route fak loop drive via FAK_GOAL_SPEC="goals/GOAL-<slug>.md" or `fak loop drive --goal <path>`.
loop: goal
witness: commit-audit
budget: { max_iters: 20 }
---
<!-- Note: Save this template with a unique filename per goal (e.g., goals/GOAL-<slug>.md or GOAL-<slug>.md).
     A single static GOAL.md causes race conditions and memory clobbering in multi-agent workflows. -->
# Objective
State one unit of work the loop should complete.

# Plan
- [ ] Write the smallest concrete first step.

# Scratch / last-refusal
