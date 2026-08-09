package main

// recover_bind.go — placeholder binding for `fak recover`.
//
// A config-class recovery is about a SPECIFIC path, env var, or address, so its
// catalog steps carry placeholders (<path>, <env>, <addr>) that a static plan
// cannot know. The bail site does know them, so it prints the recovery
// pre-bound:
//
//	next:   fak recover POLICY_LOAD_FAILED --set path=guard-policy.json
//
// bindRecoveryPlan substitutes those bindings everywhere they appear — argv,
// step summaries, notes, the plan summary — before anything is printed or run,
// so the operator reads a command they can paste and --execute runs a real one.
//
// Unbound placeholders are reported rather than shipped. Printing
// `fak policy --check <path>` as a dry run is fine (it is legible as a template),
// but EXECUTING a literal `<path>` would shell out an argument the operator
// never chose — on a POSIX shell it is a nonexistent file, and on PowerShell the
// angle brackets are redirection syntax. So --execute refuses while any
// placeholder is still unbound and names the ones it needs, which is itself the
// next step: re-run with --set.

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// recoveryPlaceholder matches a catalog placeholder: a lowercase token in angle
// brackets. Deliberately narrow so ordinary prose in a note ("a < b") is not
// mistaken for one.
var recoveryPlaceholder = regexp.MustCompile(`<[a-z][a-z0-9_-]*>`)

// parseRecoveryBindings turns repeated NAME=VALUE specs into the substitution
// map. An empty NAME or a spec with no '=' is a usage error, not a silent skip:
// a typo'd binding would otherwise leave the placeholder unbound and surface far
// away as a confusing "still unbound" refusal.
func parseRecoveryBindings(specs []string) (map[string]string, error) {
	if len(specs) == 0 {
		return nil, nil
	}
	out := make(map[string]string, len(specs))
	for _, spec := range specs {
		name, value, ok := strings.Cut(spec, "=")
		name = strings.TrimSpace(name)
		if !ok || name == "" {
			return nil, fmt.Errorf("--set %q is not NAME=VALUE", spec)
		}
		out[strings.ToLower(name)] = value
	}
	return out, nil
}

// bindRecoveryText substitutes every binding into s.
func bindRecoveryText(s string, bindings map[string]string) string {
	if len(bindings) == 0 || s == "" {
		return s
	}
	return recoveryPlaceholder.ReplaceAllStringFunc(s, func(match string) string {
		name := strings.ToLower(strings.TrimSuffix(strings.TrimPrefix(match, "<"), ">"))
		if v, ok := bindings[name]; ok {
			return v
		}
		return match
	})
}

// bindRecoveryPlan returns plan with every binding substituted throughout. The
// plan is copied, never mutated in place — recoveryPlans hands out fresh maps
// per call, but a caller that reuses one should not find it rewritten.
func bindRecoveryPlan(plan recoveryPlan, bindings map[string]string) recoveryPlan {
	if len(bindings) == 0 {
		return plan
	}
	bound := plan
	bound.Summary = bindRecoveryText(plan.Summary, bindings)
	if len(plan.Steps) > 0 {
		bound.Steps = make([]recoveryStep, len(plan.Steps))
		for i, step := range plan.Steps {
			next := step
			next.Summary = bindRecoveryText(step.Summary, bindings)
			if len(step.Argv) > 0 {
				next.Argv = make([]string, len(step.Argv))
				for j, arg := range step.Argv {
					next.Argv[j] = bindRecoveryText(arg, bindings)
				}
			}
			bound.Steps[i] = next
		}
	}
	if len(plan.Notes) > 0 {
		bound.Notes = make([]string, len(plan.Notes))
		for i, note := range plan.Notes {
			bound.Notes[i] = bindRecoveryText(note, bindings)
		}
	}
	return bound
}

// unboundRecoveryPlaceholders reports the placeholder names still present in the
// steps --execute would actually run (the Safe ones), sorted and de-duplicated.
// Only the executable path is scanned: a placeholder left in a note or in a
// manual step is documentation, and refusing to run over it would block a
// recovery that is otherwise complete.
func unboundRecoveryPlaceholders(plan recoveryPlan) []string {
	seen := map[string]bool{}
	for _, step := range plan.Steps {
		if !step.Safe {
			continue
		}
		for _, arg := range step.Argv {
			for _, match := range recoveryPlaceholder.FindAllString(arg, -1) {
				seen[strings.TrimSuffix(strings.TrimPrefix(match, "<"), ">")] = true
			}
		}
	}
	if len(seen) == 0 {
		return nil
	}
	out := make([]string, 0, len(seen))
	for name := range seen {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}
