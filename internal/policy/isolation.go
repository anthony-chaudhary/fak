// The isolation dial (issue #2013, epic #2000 M13): the declarative config
// surface mapping a task's declared TRUST LEVEL to a registered ToolExec
// backend, so isolation is a spend/threat-model spectrum the operator dials
// from the manifest — goroutine (trusted, in-process) → subprocess → container
// → gvisor → firecracker → remote (untrusted) — instead of a compiled-in
// choice.
//
// This is the FILE layer of the flags>env>file>defaults precedence ladder: the
// manifest declares which backends this deployment actually has registered and
// which backend each trust level dials to. Resolution is fail-closed by
// construction:
//
//   - an UNKNOWN trust level (or a declared-nowhere "untrusted") resolves to
//     the STRONGEST available backend, never a weaker one;
//   - "untrusted" may never dial to goroutine — refused at LOAD;
//   - if the strongest available backend IS goroutine (nothing stronger is
//     registered), unknown/untrusted work is refused rather than run
//     in-process;
//   - no dial configured at all refuses resolution outright.
package policy

import (
	"fmt"
	"sort"
	"strings"
)

// TrustUntrusted is the reserved trust level for work with no trust pedigree.
// It is the fail-closed pole of the dial: it may never map to goroutine, and
// when left undeclared it resolves to the strongest available backend.
const TrustUntrusted = "untrusted"

// backendGoroutine is the weakest (in-process, trust-only) backend — the one
// the fail-closed rules exist to keep untrusted work out of.
const backendGoroutine = "goroutine"

// isolationLadder is the closed backend vocabulary, ranked weakest → strongest.
// The ranks follow the issue's spectrum (goroutine → subprocess → container →
// gVisor → Firecracker/remote); firecracker and remote share the untrusted
// tier, with remote ranked above only so "strongest available" is a total,
// deterministic order.
var isolationLadder = map[string]int{
	backendGoroutine: 0,
	"subprocess":     1,
	"container":      2,
	"gvisor":         3,
	"firecracker":    4,
	"remote":         5,
}

// IsolationRule is the manifest's `isolation` block. Backends declares which
// ToolExec backends this deployment has registered (closed vocabulary,
// validated at load); Trust maps a declared trust level to one of them. Like
// RateLimit, this is manifest/runtime-only — NOT an adjudicator.Policy field
// (backend placement is separate from the name-level allow/deny floor).
type IsolationRule struct {
	Backends []string          `json:"backends"`
	Trust    map[string]string `json:"trust,omitempty"`
}

// compileIsolation validates a declared isolation block (absent => nil, no
// dial) at policy LOAD, so a typo'd backend, a dial to an unregistered
// backend, or an untrusted→goroutine mapping fails loud here, never at
// placement time. Names are normalized (trimmed, lowercased) so the resolved
// rule compares exactly.
func compileIsolation(r *IsolationRule) (*IsolationRule, error) {
	if r == nil {
		return nil, nil // absent => no dial configured
	}
	if len(r.Backends) == 0 {
		return nil, fmt.Errorf("isolation: declare at least one available backend in backends")
	}
	out := &IsolationRule{Backends: make([]string, 0, len(r.Backends))}
	declared := make(map[string]bool, len(r.Backends))
	for i, b := range r.Backends {
		name, err := normalizeIsolationBackend(b)
		if err != nil {
			return nil, fmt.Errorf("isolation.backends[%d]: %w", i, err)
		}
		if declared[name] {
			return nil, fmt.Errorf("isolation.backends[%d]: duplicate backend %q", i, name)
		}
		declared[name] = true
		out.Backends = append(out.Backends, name)
	}
	if len(r.Trust) > 0 {
		out.Trust = make(map[string]string, len(r.Trust))
		for level, b := range r.Trust {
			lv := strings.ToLower(strings.TrimSpace(level))
			if lv == "" {
				return nil, fmt.Errorf("isolation.trust: trust level is required")
			}
			if _, dup := out.Trust[lv]; dup {
				return nil, fmt.Errorf("isolation.trust: duplicate trust level %q", lv)
			}
			name, err := normalizeIsolationBackend(b)
			if err != nil {
				return nil, fmt.Errorf("isolation.trust[%s]: %w", lv, err)
			}
			if !declared[name] {
				return nil, fmt.Errorf("isolation.trust[%s]: backend %q is not declared in isolation.backends (available: %s)",
					lv, name, strings.Join(out.Backends, ", "))
			}
			if lv == TrustUntrusted && name == backendGoroutine {
				return nil, fmt.Errorf("isolation.trust[%s]: goroutine is never a legal backend for untrusted work (fail-closed floor)", TrustUntrusted)
			}
			out.Trust[lv] = name
		}
	}
	return out, nil
}

// BackendFor resolves a task's declared trust level to the ToolExec backend it
// must run on. Deterministic and fail-closed:
//
//   - a level the dial declares returns its mapped backend (load-time
//     validation already guaranteed untrusted ≠ goroutine);
//   - any other level — unknown, empty, or an undeclared "untrusted" —
//     resolves to the strongest available backend;
//   - if that strongest backend is goroutine, resolution REFUSES: unknown or
//     untrusted work never runs in-process;
//   - a nil rule (no isolation block in the manifest) refuses every
//     resolution — no dial means no placement, not a silent goroutine default.
func (r *IsolationRule) BackendFor(trustLevel string) (string, error) {
	if r == nil {
		return "", fmt.Errorf("isolation: no dial configured; refusing to place work (fail closed)")
	}
	lv := strings.ToLower(strings.TrimSpace(trustLevel))
	if b, ok := r.Trust[lv]; ok {
		return b, nil
	}
	strongest := r.strongest()
	if strongest == backendGoroutine {
		return "", fmt.Errorf("isolation: trust level %q is not declared and the strongest available backend is goroutine; refusing (unknown/untrusted work never runs in-process)", lv)
	}
	return strongest, nil
}

// strongest returns the highest-ranked backend the deployment declared
// available. Backends is non-empty for any compiled rule.
func (r *IsolationRule) strongest() string {
	best, bestRank := "", -1
	for _, b := range r.Backends {
		if rank := isolationLadder[b]; rank > bestRank {
			best, bestRank = b, rank
		}
	}
	return best
}

func normalizeIsolationBackend(s string) (string, error) {
	name := strings.ToLower(strings.TrimSpace(s))
	if name == "" {
		return "", fmt.Errorf("backend name is required")
	}
	if _, ok := isolationLadder[name]; !ok {
		return "", fmt.Errorf("unknown backend %q (want %s)", s, strings.Join(isolationBackendNames(), "|"))
	}
	return name, nil
}

// isolationBackendNames lists the closed backend vocabulary weakest-first, for
// error messages and the operator summary.
func isolationBackendNames() []string {
	names := make([]string, 0, len(isolationLadder))
	for n := range isolationLadder {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool { return isolationLadder[names[i]] < isolationLadder[names[j]] })
	return names
}
