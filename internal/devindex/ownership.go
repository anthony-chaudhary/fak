package devindex

import (
	"fmt"
	"sort"
)

// CommandOwner names the independently buildable artifact that owns a command.
// Shared is reserved for a command whose executable contract genuinely belongs
// on both surfaces; it is not a synonym for a shared implementation package.
type CommandOwner string

const (
	OwnerRuntime CommandOwner = "runtime"
	OwnerDev     CommandOwner = "dev"
	OwnerShared  CommandOwner = "shared"
)

// CommandOwnership is the machine-readable product boundary for one top-level
// command spelling. CompatibilityName is the spelling retained at a legacy
// entry point while DispatchTarget names the artifact that owns execution.
type CommandOwnership struct {
	Name              string       `json:"name"`
	Owner             CommandOwner `json:"owner"`
	Rationale         string       `json:"rationale"`
	CompatibilityName string       `json:"compatibility_name"`
	DispatchTarget    string       `json:"dispatch_target"`
	DevReuse          DevReuse     `json:"dev_reuse"`
	DevReuseRationale string       `json:"dev_reuse_rationale"`
}

// SupplementalDevVerbs are pre-switch namespace commands that are intentionally
// absent from main.go's top-level switch. Keeping them beside the ownership
// contract prevents that implementation detail from making the inventory
// incomplete.
func SupplementalDevVerbs() []Verb {
	return []Verb{
		{Name: "index", Synopsis: "query the repository self-index and audit runtime/dev ownership", Aliases: []string{"devindex"}, Lane: "devindex", Tier: TierDev},
		{Name: "gh-spam-comments", Synopsis: "scan GitHub issue/PR comments for untrusted spam across reusable abuse families", Lane: "cmd", Tier: TierDev},
		{Name: "workspace", Synopsis: "map the local agentic-development workspace and guard decision stream", Lane: "cmd", Tier: TierDev},
	}
}

// OwnershipVerbs returns the complete command set, including pre-switch
// namespace-only commands, sorted by canonical spelling.
func OwnershipVerbs(verbs []Verb) []Verb {
	out := append([]Verb(nil), verbs...)
	out = append(out, SupplementalDevVerbs()...)
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// CommandOwnerships derives a total ownership inventory from the authoritative
// catalog. The existing tier decision is the migration input: front-door and
// internal product verbs stay in the runtime artifact; repository-development
// verbs move to fak-dev. Exceptions must be represented explicitly here rather
// than inferred from filenames.
func CommandOwnerships(verbs []Verb) []CommandOwnership {
	out := make([]CommandOwnership, 0, len(verbs))
	for _, verb := range verbs {
		owner := OwnerRuntime
		target := "fak"
		rationale := "product runtime/front-door command"
		if verb.Tier == TierDev {
			owner = OwnerDev
			target = "fak-dev"
			rationale = "develops, tests, releases, or operates the fak repository"
		}
		reuse, reuseRationale := ClassifyDevReuse(verb.Name, owner)
		out = append(out, CommandOwnership{
			Name:              verb.Name,
			Owner:             owner,
			Rationale:         rationale,
			CompatibilityName: verb.Name,
			DispatchTarget:    target,
			DevReuse:          reuse,
			DevReuseRationale: reuseRationale,
		})
	}
	return out
}

// ValidateCommandOwnership proves that inventory is a one-to-one, total map of
// catalog command names. It returns deterministic diagnostics for tests and CLI
// callers rather than failing on the first defect.
func ValidateCommandOwnership(verbs []Verb, inventory []CommandOwnership) []string {
	known := make(map[string]bool, len(verbs))
	for _, verb := range verbs {
		known[verb.Name] = true
	}
	seen := make(map[string]int, len(inventory))
	var problems []string
	for _, item := range inventory {
		seen[item.Name]++
		if !known[item.Name] {
			problems = append(problems, "unknown command: "+item.Name)
		}
		if item.Owner != OwnerRuntime && item.Owner != OwnerDev && item.Owner != OwnerShared {
			problems = append(problems, "invalid owner: "+item.Name)
		}
		if item.Rationale == "" {
			problems = append(problems, "missing rationale: "+item.Name)
		}
		if item.CompatibilityName == "" {
			problems = append(problems, "missing compatibility name: "+item.Name)
		}
		if item.DispatchTarget == "" {
			problems = append(problems, "missing dispatch target: "+item.Name)
		}
		if item.DevReuseRationale == "" {
			problems = append(problems, "empty dev reuse rationale: "+item.Name)
		}
		if !item.DevReuse.valid() {
			problems = append(problems, fmt.Sprintf("command %q has invalid dev reuse %q", item.Name, item.DevReuse))
		} else if item.Owner == OwnerDev && item.DevReuse == DevReuseNA {
			problems = append(problems, fmt.Sprintf("dev command %q has not-applicable dev reuse", item.Name))
		} else if item.Owner != OwnerDev && item.DevReuse != DevReuseNA {
			problems = append(problems, fmt.Sprintf("non-dev command %q has dev reuse %q", item.Name, item.DevReuse))
		}
	}
	for name := range known {
		switch seen[name] {
		case 0:
			problems = append(problems, "missing command: "+name)
		case 1:
		default:
			problems = append(problems, "duplicate command: "+name)
		}
	}
	sort.Strings(problems)
	return problems
}
