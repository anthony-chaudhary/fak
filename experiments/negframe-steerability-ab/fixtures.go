package main

// fixturePair is one paired guard directive: the same instruction in a negative/prohibition
// frame (Arm A) and an affordance-first frame (Arm B). Arm A idioms are drawn from this repo's
// own steer-prose corpus (AGENTS.md/CLAUDE.md/skills style, per internal/negframe's own target
// list) and from the memory notes' recurring cautionary phrasing -- realistic guard-directive
// prose, not synthetic filler. ReframeIsMechanical records whether internal/negframe's own
// reframe RULE produced Arm B verbatim, modulo the sentence-initial capital letter (true), or
// whether Arm B is a hand-authored affordance-first rewrite instead -- either because the
// negation is judgement-tier and negframe deliberately does not auto-rewrite those, or because
// the mechanical template's bare-verb capture reads ungrammatically in context (false). See
// internal/negframe/negframe.go's rules table and the per-pair comments below for exceptions.
type fixturePair struct {
	ID                  string
	ArmA                string
	ArmB                string
	ReframeIsMechanical bool
}

// fixtures is the paired corpus this experiment scores. Kept in one file, in one literal, so the
// sample is auditable at a glance -- add a pair here to grow the corpus; TestFixtureCorpusIntegrity
// enforces that every Arm A carries at least one negframe finding and every Arm B carries none.
var fixtures = []fixturePair{
	{
		ID:                  "stamp-trailer",
		ArmA:                "Don't forget to stamp the commit with the (fak <leaf>) trailer.",
		ArmB:                "Remember to stamp the commit with the (fak <leaf>) trailer.",
		ReframeIsMechanical: true,
	},
	{
		ID:                  "ask-clarify",
		ArmA:                "Do not hesitate to ask for clarification before landing a risky change.",
		ArmB:                "Feel free to ask for clarification before landing a risky change.",
		ReframeIsMechanical: true,
	},
	{
		// negframe's own template for this rule is the bare captured verb ("you can skip
		// run the full suite..."), which is grammatically rough -- the rule intentionally
		// captures the base verb, not a gerund. Arm A still trips the library's mechanical
		// Absence rule (so its negativity score is real), but Arm B here is a hand-cleaned
		// affordance-first rewrite rather than the literal template output.
		ID:                  "skip-full-suite",
		ArmA:                "No need to run the full suite for a docs-only change.",
		ArmB:                "You can skip the full suite for a docs-only change.",
		ReframeIsMechanical: false,
	},
	{
		ID:                  "hedge-unlikely",
		ArmA:                "It is not unlikely that a stale lease blocks your lane, so check first.",
		ArmB:                "It is likely that a stale lease blocks your lane, so check first.",
		ReframeIsMechanical: true,
	},
	{
		ID:                  "trunk-only",
		ArmA:                "Never open a feature branch or new worktree.",
		ArmB:                "Work directly on the trunk (main).",
		ReframeIsMechanical: false,
	},
	{
		ID:                  "explicit-paths",
		ArmA:                "Do not use `git add -A` on the shared multi-session tree.",
		ArmB:                "Commit by explicit path with `git commit -- <paths>` on the shared multi-session tree.",
		ReframeIsMechanical: false,
	},
	{
		ID:                  "run-the-guard",
		ArmA:                "Don't skip the guard when landing a change.",
		ArmB:                "Run every change through the guard before landing it.",
		ReframeIsMechanical: false,
	},
	{
		ID:                  "verb-led-subject",
		ArmA:                "Avoid a noun-led commit subject.",
		ArmB:                "Lead every commit subject with a verb naming the concrete change.",
		ReframeIsMechanical: false,
	},
	{
		ID:                  "witnessed-claim",
		ArmA:                "Without a witnessed test, the claim stays unshipped.",
		ArmB:                "With a witnessed test attached, the claim ships.",
		ReframeIsMechanical: false,
	},
	{
		ID:                  "signoff-gate",
		ArmA:                "This action is not allowed without a signed-off commit.",
		ArmB:                "This action is permitted once the commit is signed off.",
		ReframeIsMechanical: false,
	},
	{
		ID:                  "closes-trailer",
		ArmA:                "The commit fails to close the issue if you leave out the Closes #N trailer.",
		ArmB:                "Include a Closes #N trailer so the commit closes the issue.",
		ReframeIsMechanical: false,
	},
	{
		ID:                  "force-push",
		ArmA:                "Never force-push to main.",
		ArmB:                "Push to main with a fast-forward only.",
		ReframeIsMechanical: false,
	},
	{
		ID:                  "amend-refusal",
		ArmA:                "Do not amend a commit that a peer may have already built on.",
		ArmB:                "Create a new commit on top when a peer may have already built on the prior one.",
		ReframeIsMechanical: false,
	},
	{
		ID:                  "no-verify",
		ArmA:                "Refuse to skip pre-commit hooks with `--no-verify`.",
		ArmB:                "Let pre-commit hooks run to completion on every commit.",
		ReframeIsMechanical: false,
	},
	{
		ID:                  "prestaged-path",
		ArmA:                "Don't forget to let `fak commit --path` stage the file itself.",
		ArmB:                "Remember to let `fak commit --path` stage the file itself.",
		ReframeIsMechanical: true,
	},
	{
		ID:                  "double-negative-bounded",
		ArmA:                "The retry budget is not unbounded, so respect the cap.",
		ArmB:                "The retry budget is bounded, so respect the cap.",
		ReframeIsMechanical: true,
	},
}
