// Model ALIAS registry: a runtime-reassignable name -> target rewrite that runs
// BEFORE the account roster at dispatch time (#11091).
//
// WHY IT EXISTS. Roster bindings are fixed at process start; an operator who wants
// "production-coder" to point at a new model today must restart the gateway. An
// AliasStore decouples the NAME a client sends from the model the roster binds, so
// an alias can be defined, reassigned, or removed at runtime and the very next
// completion redirects to the new target — no restart.
//
// WHERE IT SITS. AliasStore composes BEFORE Roster.Resolve: the requested model name
// is rewritten through the alias chain to a terminal name, and the roster then binds
// that terminal name to a concrete account/upstream target (internal/gateway
// bindChatRoute / resolveRoute). An alias therefore names another MODEL id (or
// another alias), never an account — account selection stays the roster's job.
//
// CYCLE / DEPTH SAFETY. Aliases form a directed graph name -> target. A cycle
// (A->B->A) or an over-deep chain would spin a resolver forever, so the invariant is
// enforced in three places: Validate rejects a cyclic/over-deep table at parse time,
// Set rejects a mutation that would create one against the live table, and Resolve
// defends itself with MaxAliasDepth so even a store mutated past its checks cannot
// livelock a request.
package modelroute

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"
)

// AliasVersion is the registry major version stamped into DefaultAliases and JSON.
const AliasVersion = "fak-alias-registry/1"

// MaxAliasDepth is the maximum number of EDGES (hops) a single alias chain may
// take. It is the ONE budget shared by Validate and Resolve: a chain of up to and
// including MaxAliasDepth edges is accepted and resolves to its terminal, while
// MaxAliasDepth+1 edges is refused. A longer chain is treated as a cycle (the
// resolver cannot distinguish a deep chain from a loop within the budget), so a
// request is refused rather than spinning. Kept well below any real chain length:
// an alias is a convenience rewrite, not a dependency graph.
const MaxAliasDepth = 10

// Alias is one name -> target edge. Both Name and Target are model ids (a target may
// itself be another alias, forming a chain).
type Alias struct {
	Name   string `json:"name"`
	Target string `json:"target"`
}

// AliasRegistry is the on-disk/JSON form of an alias table: a version tag plus the
// edge list, mirroring the Roster manifest's round-trip contract.
type AliasRegistry struct {
	Version string  `json:"version,omitempty"`
	Aliases []Alias `json:"aliases"`
}

// AliasStore is a concurrency-safe alias->target registry. Reads (Resolve, List,
// Registry) take the read lock; mutations (Set, Remove, Replace) take the write
// lock. A nil *AliasStore is inert and must be nil-guarded by the caller; the
// gateway call sites check for nil so an unconfigured server behaves exactly as
// before.
type AliasStore struct {
	mu      sync.RWMutex
	aliases map[string]string
}

// NewAliasStore validates and installs reg as the initial table. It returns an error
// on any invalid registry (empty/duplicate name, empty target, or a cycle/over-deep
// chain in the whole table) so a misconfigured alias file fails loud at startup.
func NewAliasStore(reg AliasRegistry) (*AliasStore, error) {
	if err := reg.Validate(); err != nil {
		return nil, err
	}
	s := &AliasStore{aliases: make(map[string]string, len(reg.Aliases))}
	for _, a := range reg.Aliases {
		s.aliases[a.Name] = a.Target
	}
	return s, nil
}

// Resolve follows the alias chain from name to its terminal target.
//
//   - (name, true, nil)  -> name IS an alias; target is the terminal resolved name
//     (the chain walked, up to MaxAliasDepth EDGES).
//   - (name, false, nil) -> name is NOT an alias; the caller keeps the original name.
//   - error              -> the chain cycles or exceeds MaxAliasDepth edges (the two
//     are indistinguishable within the budget, and both are refused rather than
//     spun).
//
// The whole walk runs under one read lock, so it sees a consistent snapshot even
// while a concurrent Set mutates the table.
func (s *AliasStore) Resolve(name string) (target string, isAlias bool, err error) {
	if s == nil {
		return name, false, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	cur := name
	hops := 0
	// Take at most MaxAliasDepth+1 hops: the terminal check runs on every iteration,
	// so a chain of exactly MaxAliasDepth edges terminates (the (MaxAliasDepth+1)-th
	// lookup misses the table) while a cycle or longer chain takes the extra hop and
	// is refused below — the exact bound validateAliasMap enforces at parse time.
	for i := 0; i <= MaxAliasDepth; i++ {
		next, ok := s.aliases[cur]
		if !ok {
			// The chain terminates at a non-alias name. It is an alias iff we took at
			// least one hop.
			return cur, cur != name, nil
		}
		cur = next
		hops++
		if hops > MaxAliasDepth {
			// Took the (MaxAliasDepth+1)-th edge: cycle or too deep within the budget.
			return "", false, fmt.Errorf("modelroute: alias %q: chain exceeds MaxAliasDepth (%d) — possible cycle", name, MaxAliasDepth)
		}
	}
	// Unreachable: the hops check above fires on the first over-budget edge. Kept
	// fail-closed so a future edit cannot silently fall through to a zero-value return.
	return "", false, fmt.Errorf("modelroute: alias %q: chain exceeds MaxAliasDepth (%d) — possible cycle", name, MaxAliasDepth)
}

// List returns the aliases sorted by Name for deterministic output.
func (s *AliasStore) List() []Alias {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Alias, 0, len(s.aliases))
	for n, t := range s.aliases {
		out = append(out, Alias{Name: n, Target: t})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Set inserts or hot-reassigns name -> target. It is cycle-checked against the LIVE
// table (not just the incoming edge): the prospective table is validated before the
// mutation is committed, so a Set that would close a cycle is refused and the
// installed table is left untouched. On success a subsequent Resolve returns the NEW
// target immediately.
func (s *AliasStore) Set(name, target string) error {
	if s == nil {
		return fmt.Errorf("modelroute: alias store is nil")
	}
	if name == "" {
		return fmt.Errorf("modelroute: alias set: empty name")
	}
	if target == "" {
		return fmt.Errorf("modelroute: alias set: %q has an empty target", name)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	// Build the prospective table and validate it wholesale — simpler and safer than
	// reasoning about the single new edge in isolation, and it catches a cycle formed
	// against any existing edge.
	next := make(map[string]string, len(s.aliases)+1)
	for n, t := range s.aliases {
		next[n] = t
	}
	next[name] = target
	if err := validateAliasMap(next); err != nil {
		return err
	}
	s.aliases[name] = target
	return nil
}

// Remove deletes name from the table, reporting whether it was present.
func (s *AliasStore) Remove(name string) bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.aliases[name]; !ok {
		return false
	}
	delete(s.aliases, name)
	return true
}

// Registry returns a snapshot of the table as an AliasRegistry. The returned slice is
// a copy, safe to retain after further mutation.
func (s *AliasStore) Registry() AliasRegistry {
	if s == nil {
		return AliasRegistry{Version: AliasVersion}
	}
	return AliasRegistry{Version: AliasVersion, Aliases: s.List()}
}

// Replace atomically swaps the whole table for reg after validating it. A rejected
// registry leaves the live table untouched (the hot-reload fail-loud contract: a bad
// edit never silently degrades the routing surface).
func (s *AliasStore) Replace(reg AliasRegistry) error {
	if s == nil {
		return fmt.Errorf("modelroute: alias store is nil")
	}
	if err := reg.Validate(); err != nil {
		return err
	}
	next := make(map[string]string, len(reg.Aliases))
	for _, a := range reg.Aliases {
		next[a.Name] = a.Target
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.aliases = next
	return nil
}

// Validate checks a registry is well-formed: every edge has a non-empty name and
// target, names are unique, and the whole table contains no cycle or over-deep
// chain. It is the parse-time and Replace-time boundary.
func (r AliasRegistry) Validate() error {
	if r.Version != "" && !hasPrefix(r.Version, "fak-alias-registry/") {
		return fmt.Errorf("modelroute: alias registry version %q is not fak-alias-registry/x", r.Version)
	}
	seen := make(map[string]bool, len(r.Aliases))
	table := make(map[string]string, len(r.Aliases))
	for i, a := range r.Aliases {
		if a.Name == "" {
			return fmt.Errorf("modelroute: alias %d has an empty name", i)
		}
		if a.Target == "" {
			return fmt.Errorf("modelroute: alias %q has an empty target", a.Name)
		}
		if seen[a.Name] {
			return fmt.Errorf("modelroute: duplicate alias name %q", a.Name)
		}
		seen[a.Name] = true
		table[a.Name] = a.Target
	}
	return validateAliasMap(table)
}

// JSON renders the registry as canonical indented JSON (stamping AliasVersion when
// absent), newline-terminated so `--aliases-dump > file` is clean.
func (r AliasRegistry) JSON() []byte {
	out := r
	if out.Version == "" {
		out.Version = AliasVersion
	}
	b, _ := json.MarshalIndent(out, "", "  ")
	return append(b, '\n')
}

// ParseAliases decodes and validates a registry. Unknown JSON fields are REJECTED
// (DisallowUnknownFields) so a typo fails loudly instead of silently changing a
// redirect.
func ParseAliases(b []byte) (AliasRegistry, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var r AliasRegistry
	if err := dec.Decode(&r); err != nil {
		return AliasRegistry{}, fmt.Errorf("modelroute: parse aliases: %w", err)
	}
	if err := r.Validate(); err != nil {
		return AliasRegistry{}, err
	}
	return r, nil
}

// LoadAliases reads and validates a registry from a file path.
func LoadAliases(path string) (AliasRegistry, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return AliasRegistry{}, fmt.Errorf("modelroute: read aliases %s: %w", path, err)
	}
	return ParseAliases(b)
}

// DefaultAliases is the illustrative starter `fak route --aliases-dump` emits: a
// couple of friendly names pointing at roster-resolvable model ids. Editing it shows
// how a client's stable alias can be repointed at runtime.
func DefaultAliases() AliasRegistry {
	return AliasRegistry{
		Version: AliasVersion,
		Aliases: []Alias{
			{Name: "production-coder", Target: "small"},
			{Name: "code-fast", Target: "qwen36-groq"},
		},
	}
}

// validateAliasMap reports whether the name->target map is acyclic and every chain
// takes at most MaxAliasDepth EDGES. An unknown intermediate is NOT an error here: a
// target naming a plain model id (not another alias) is the normal terminal case.
// The `steps > MaxAliasDepth` bound mirrors Resolve's walk exactly: a chain of
// exactly MaxAliasDepth edges is accepted by both; MaxAliasDepth+1 is refused.
func validateAliasMap(table map[string]string) error {
	for name := range table {
		cur := name
		steps := 0
		// Walk until we leave the alias table (a valid terminal) or exceed the depth
		// budget (a cycle / over-deep chain). We revisit `name` exactly when the chain
		// closes on itself, so hitting name again is also a cycle.
		for {
			next, ok := table[cur]
			if !ok {
				break // terminal: cur is a non-alias model id
			}
			cur = next
			steps++
			if steps > MaxAliasDepth || cur == name {
				return fmt.Errorf("modelroute: alias %q: chain forms a cycle or exceeds MaxAliasDepth (%d)", name, MaxAliasDepth)
			}
		}
	}
	return nil
}

// hasPrefix is strings.HasPrefix inlined to keep this file's import list to the
// stdlib minimum; it is not a hot path.
func hasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}
