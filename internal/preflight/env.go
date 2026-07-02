package preflight

// env.go — the runtime-environment preflight (#2079). An agent landing cold
// learns this host's constraints only after wasting tokens on them: that
// native `go test` is blocked here, that git/gh hang under a POSIX bash tool,
// that a peer already holds a lease on the very paths it is about to edit.
// PlanEnvPreflight folds those host facts into one machine-readable report so
// the agent can plan against real constraints before writing code.
//
// Like the workspace preflight above it, the fold is pure: the caller supplies
// every host observation as data, and no field causes I/O by itself.

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/testroute"
)

const (
	EnvPreflightSchema = "fak.preflight.env/1"

	// EnvVerdictClear means no live lease overlaps the declared paths.
	// EnvVerdictLeased means at least one does — coordinate before editing.
	EnvVerdictClear  = "CLEAR"
	EnvVerdictLeased = "LEASED"

	GitShellPowerShell = "powershell"
	GitShellBash       = "bash"
)

// EnvProbe is the caller-supplied host evidence. Data only: this package never
// shells out, checks PATH, reads the filesystem, or looks at a clock.
type EnvProbe struct {
	GOOS       string             `json:"goos,omitempty"`
	Test       testroute.Probe    `json:"test,omitempty"`
	Paths      []string           `json:"paths,omitempty"`
	LiveLeases []LeaseObservation `json:"live_leases,omitempty"`
}

// GitShellFact is the shell git/gh must run under on this host, with the why.
type GitShellFact struct {
	Shell  string `json:"shell"`
	Reason string `json:"reason"`
}

// EnvHazard is one known environment hazard plus its non-interactive fix, so
// the report pre-fills the route around the landmine instead of just naming it.
type EnvHazard struct {
	Kind string `json:"kind"`
	Why  string `json:"why"`
	Fix  string `json:"fix"`
}

// EnvReport is the machine-readable runtime constraint report.
type EnvReport struct {
	Schema             string             `json:"schema"`
	Verdict            string             `json:"verdict"`
	GOOS               string             `json:"goos,omitempty"`
	TestRoute          testroute.Route    `json:"test_route"`
	GitShell           GitShellFact       `json:"git_shell"`
	InteractiveHazards []EnvHazard        `json:"interactive_hazards,omitempty"`
	Paths              []string           `json:"paths,omitempty"`
	LiveLeases         []LeaseObservation `json:"live_leases,omitempty"`
}

// PlanEnvPreflight folds host facts into the runtime constraint report. With
// declared paths, LiveLeases carries only the leases whose tree overlaps them
// and the verdict flips to LEASED on any hit; with no declared paths every
// live lease is reported and the verdict stays CLEAR (a lease elsewhere does
// not block an undeclared write set). A lease with an empty tree is treated as
// global and overlaps everything.
func PlanEnvPreflight(p EnvProbe) EnvReport {
	paths := normalizeList(p.Paths)
	out := EnvReport{
		Schema:             EnvPreflightSchema,
		Verdict:            EnvVerdictClear,
		GOOS:               p.GOOS,
		TestRoute:          testroute.Decide(p.Test),
		GitShell:           gitShellFor(p.GOOS),
		InteractiveHazards: interactiveHazardsFor(p.GOOS),
		Paths:              paths,
		LiveLeases:         overlappingLeases(paths, p.LiveLeases),
	}
	if len(paths) > 0 && len(out.LiveLeases) > 0 {
		out.Verdict = EnvVerdictLeased
	}
	return out
}

func gitShellFor(goos string) GitShellFact {
	if goos == "windows" {
		return GitShellFact{
			Shell:  GitShellPowerShell,
			Reason: "git/gh under a POSIX bash tool hang until killed on this host class; run all git/gh through PowerShell",
		}
	}
	return GitShellFact{
		Shell:  GitShellBash,
		Reason: "no known git shell hazard on this host",
	}
}

// interactiveHazardsFor mirrors the repo-guard INTERACTIVE_HANG curation
// (internal/repoguard/interactive.go) at the advice level: only invocations
// that genuinely wedge or silently no-op without a TTY, each with the runnable
// non-interactive equivalent.
func interactiveHazardsFor(goos string) []EnvHazard {
	out := []EnvHazard{
		{
			Kind: "editor",
			Why:  "full-screen editors (vi/vim/nano/emacs) grab the terminal and wait for a human",
			Fix:  "use the harness Edit/Write tools, or a scripted edit: sed -i 's/OLD/NEW/' <file>",
		},
		{
			Kind: "git_interactive",
			Why:  "git rebase -i / add -i / -p open a sequence editor or hunk picker and wait",
			Fix:  "GIT_SEQUENCE_EDITOR=: git rebase ..., or whole-path forms: git add -- <paths>",
		},
		{
			Kind: "git_commit_editor",
			Why:  "git commit without -m/-F opens the commit-message editor and waits",
			Fix:  `git commit -s -m "<type>(<leaf>): <subject> (fak <leaf>)" -- <paths>`,
		},
		{
			Kind: "gh_auth_login",
			Why:  "gh auth login opens a browser/device-code prompt and waits",
			Fix:  "gh auth login --with-token < <token-file>  (or set GH_TOKEN)",
		},
	}
	if goos == "windows" {
		out = append(out, EnvHazard{
			Kind: "bash_git_hang",
			Why:  "git and gh invoked through a POSIX bash tool hang until the harness kills them (exit 143)",
			Fix:  "run git/gh through PowerShell",
		})
	}
	return out
}

func overlappingLeases(paths []string, live []LeaseObservation) []LeaseObservation {
	out := make([]LeaseObservation, 0, len(live))
	for _, lease := range live {
		l := lease
		l.Tree = normalizeList(lease.Tree)
		if len(paths) == 0 || dispatchorder.TreesOverlap(paths, l.Tree) {
			out = append(out, l)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].ID != out[j].ID {
			return out[i].ID < out[j].ID
		}
		return out[i].Holder < out[j].Holder
	})
	if len(out) == 0 {
		return nil
	}
	return out
}
