package accounts

// Generated VIEWS of the canonical registry.
//
// The registry (registry.json) is the single source of truth for account identity AND
// policy. The two roster files other tools read — the dos roster (~/.claude/accounts.yaml)
// and the job switcher roster (job/config/claude_accounts.yaml) — are GENERATED projections
// of it, never hand-edited. This file owns that projection: a small deterministic YAML
// emitter plus one generator per view shape. No third-party YAML dependency (the module has
// zero external deps), so the emitter is hand-rolled and intentionally minimal — it covers
// exactly the value shapes these rosters use (maps, lists, strings, bools, ints).

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/maputil"
)

// ViewName enumerates the generated views.
type ViewName string

const (
	// ViewDos is the dos roster (~/.claude/accounts.yaml): name+config_dir rows plus a
	// `rotation:` and `defaults:` block. Read by `dos accounts`.
	ViewDos ViewName = "dos"
	// ViewJob is the job switcher roster (job/config/claude_accounts.yaml): richer rows
	// (chrome_profile/email/enabled/reserved), a `tombstoned_accounts:` block, and
	// defaults/rotation/launch. Read by job_search.account_switcher.
	ViewJob ViewName = "job"
)

// generatedHeader is the banner every generated view carries so a human (or a guard) can see
// at a glance the file is owned by `fak accounts sync`, with the rationale in the registry.
const generatedHeader = "# GENERATED from registry.json by `fak accounts sync` — do not hand-edit.\n" +
	"# Account identity + policy is canonical in ~/.claude-accounts/registry.json;\n" +
	"# per-account rationale lives in each home's `note` field there.\n"

// RenderView projects the registry into the YAML text of the named view. It is a PURE
// function of the registry (no disk reads), so `check` can compare on-disk bytes to a freshly
// rendered view and `sync` can write the same bytes.
func (r Registry) RenderView(view ViewName) (string, error) {
	switch view {
	case ViewDos:
		return r.renderDos(), nil
	case ViewJob:
		return r.renderJob(), nil
	default:
		return "", fmt.Errorf("accounts: unknown view %q", view)
	}
}

// activeHomes returns the live (non-tombstoned) homes in registry order — the rows that go in
// the `accounts:` list. tombstonedHomes returns the retired ones for `tombstoned_accounts:`.
func (r Registry) activeHomes() []Home {
	var out []Home
	for _, h := range r.Homes {
		if h.Active() {
			out = append(out, h)
		}
	}
	return out
}

func (r Registry) tombstonedHomes() []Home {
	var out []Home
	for _, h := range r.Homes {
		if !h.Active() {
			out = append(out, h)
		}
	}
	return out
}

// renderDos emits the dos roster: `accounts:` rows of name+config_dir for every active home,
// the active-default seat + the full role map (so a launcher/watchdog can pick the right seat
// without re-reading the registry), then the view's config blocks (rotation/defaults).
func (r Registry) renderDos() string {
	var b strings.Builder
	b.WriteString(generatedHeader)
	b.WriteString("accounts:\n")
	for _, h := range r.activeHomes() {
		b.WriteString("  - name: " + yamlScalar(h.Name) + "\n")
		b.WriteString("    config_dir: " + yamlScalar(h.Dir) + "\n")
	}
	// active_default: the name+config_dir of the ACTIVE-role seat — the account a bare launch /
	// watchdog should use. Emitted as a top-level scalar so the existing flat `accounts:`
	// parsers ignore it; a new consumer reads it directly. Omitted when no active role is set.
	if act, ok := r.Role(RoleActive); ok && act.Active() {
		b.WriteString("\nactive_default: " + yamlScalar(act.Name) + "\n")
		b.WriteString("active_default_dir: " + yamlScalar(act.Dir) + "\n")
	}
	// roles: the full role -> {name, config_dir} map, so a consumer can resolve any role (e.g.
	// the rehome anchor) from the view alone. Emitted in sorted role order for byte-stability.
	if len(r.Roles) > 0 {
		b.WriteString("\nroles:\n")
		for _, role := range maputil.SortedKeys(r.Roles) {
			h, ok := r.home(r.Roles[role])
			if !ok {
				continue // Validate rejects a dangling role, so a loaded registry never hits this.
			}
			b.WriteString("  " + role + ":\n")
			b.WriteString("    name: " + yamlScalar(h.Name) + "\n")
			b.WriteString("    config_dir: " + yamlScalar(h.Dir) + "\n")
		}
	}
	b.WriteString(r.renderBlocks(ViewDos))
	return b.String()
}

// renderJob emits the job roster: richer active rows, a tombstoned_accounts block built from
// the registry's tombstone audit fields, then defaults/rotation/launch.
func (r Registry) renderJob() string {
	var b strings.Builder
	b.WriteString(generatedHeader)
	b.WriteString("accounts:\n")
	for _, h := range r.activeHomes() {
		b.WriteString("  - name: " + yamlScalar(h.Name) + "\n")
		b.WriteString("    config_dir: " + yamlScalar(h.Dir) + "\n")
		b.WriteString("    chrome_profile: " + yamlScalar(h.ChromeProfile) + "\n")
		b.WriteString("    email: " + yamlScalar(h.Identity.Email) + "\n")
		b.WriteString("    enabled: " + strconv.FormatBool(h.EnabledOrDefault()) + "\n")
		if h.Reserved {
			b.WriteString("    reserved: true\n")
		}
	}
	if tomb := r.tombstonedHomes(); len(tomb) > 0 {
		b.WriteString("\ntombstoned_accounts:\n")
		for _, h := range tomb {
			b.WriteString("  - name: " + yamlScalar(h.Name) + "\n")
			b.WriteString("    config_dir: " + yamlScalar(h.Dir) + "\n")
			b.WriteString("    chrome_profile: " + yamlScalar(h.ChromeProfile) + "\n")
			b.WriteString("    email: " + yamlScalar(h.Identity.Email) + "\n")
			b.WriteString("    enabled: false\n")
			if h.TombstonedAt != "" {
				b.WriteString("    tombstoned_at: " + yamlScalar(h.TombstonedAt) + "\n")
			}
			if h.TombstoneReason != "" {
				b.WriteString("    tombstone_reason: " + yamlScalar(h.TombstoneReason) + "\n")
			}
			if h.RehomeTo != "" {
				b.WriteString("    rehome_to: " + yamlScalar(h.RehomeTo) + "\n")
			}
		}
	}
	b.WriteString(r.renderBlocks(ViewJob))
	return b.String()
}

// renderBlocks emits a view's config blocks (from registry.Views[view]) in BlockOrder, each
// as a top-level YAML mapping, separated from the rows above by a blank line. Returns "" when
// the view has no config.
func (r Registry) renderBlocks(view ViewName) string {
	vc, ok := r.Views[string(view)]
	if !ok || len(vc.Blocks) == 0 {
		return ""
	}
	var b strings.Builder
	for _, key := range orderedKeys(vc.Blocks, vc.BlockOrder) {
		b.WriteString("\n" + key + ":\n")
		b.WriteString(yamlValue(vc.Blocks[key], 1))
	}
	return b.String()
}

// orderedKeys returns m's keys in `order` first (those present), then any remaining keys
// sorted, so emission is deterministic regardless of JSON map iteration order.
func orderedKeys(m map[string]any, order []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, k := range order {
		if _, ok := m[k]; ok && !seen[k] {
			out = append(out, k)
			seen[k] = true
		}
	}
	var rest []string
	for k := range m {
		if !seen[k] {
			rest = append(rest, k)
		}
	}
	sort.Strings(rest)
	return append(out, rest...)
}

// yamlValue renders v (a value decoded from JSON: map, slice, string, bool, float64) as YAML
// indented to `depth` levels (2 spaces each). It handles exactly the shapes the roster config
// blocks use. A mapping's keys are emitted in `_order`-aware order when the map carries one,
// else sorted, for determinism.
func yamlValue(v any, depth int) string {
	indent := strings.Repeat("  ", depth)
	switch t := v.(type) {
	case map[string]any:
		var b strings.Builder
		for _, k := range orderedKeys(t, nil) {
			child := t[k]
			if isScalar(child) {
				b.WriteString(indent + k + ": " + yamlScalarAny(child) + "\n")
			} else {
				b.WriteString(indent + k + ":\n")
				b.WriteString(yamlValue(child, depth+1))
			}
		}
		return b.String()
	case []any:
		var b strings.Builder
		for _, item := range t {
			if isScalar(item) {
				b.WriteString(indent + "- " + yamlScalarAny(item) + "\n")
			} else {
				// A nested map/list under a list item: emit "- " then the value indented.
				b.WriteString(indent + "-\n")
				b.WriteString(yamlValue(item, depth+1))
			}
		}
		return b.String()
	default:
		return indent + yamlScalarAny(v) + "\n"
	}
}

// isScalar reports whether v renders as a single inline YAML scalar (not a block).
func isScalar(v any) bool {
	switch v.(type) {
	case map[string]any, []any:
		return false
	default:
		return true
	}
}

// yamlScalarAny renders a JSON-decoded scalar (string, bool, float64, nil) as a YAML scalar.
func yamlScalarAny(v any) string {
	switch t := v.(type) {
	case nil:
		return `""`
	case bool:
		return strconv.FormatBool(t)
	case float64:
		// JSON numbers decode to float64; emit integers without a trailing ".0".
		if t == float64(int64(t)) {
			return strconv.FormatInt(int64(t), 10)
		}
		return strconv.FormatFloat(t, 'g', -1, 64)
	case string:
		return yamlScalar(t)
	default:
		return yamlScalar(fmt.Sprint(v))
	}
}

// yamlScalar renders a string as a YAML scalar, quoting it when the value would otherwise be
// misparsed (empty, leading/trailing space, or a character that starts a non-plain scalar).
// It double-quotes and escapes, which is always safe; plain (unquoted) output is used only for
// values that are unambiguously plain, to keep the generated file readable.
func yamlScalar(s string) string {
	if s == "" {
		return `""`
	}
	if needsQuote(s) {
		// Double-quoted with the minimal escapes YAML needs.
		r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`, "\t", `\t`)
		return `"` + r.Replace(s) + `"`
	}
	return s
}

// needsQuote reports whether s must be quoted to round-trip as a plain YAML scalar.
func needsQuote(s string) bool {
	if s != strings.TrimSpace(s) {
		return true
	}
	// Reserved indicators at the start of a plain scalar, plus structural chars.
	switch s[0] {
	case '!', '&', '*', '-', '?', '{', '}', '[', ']', ',', '#', '|', '>', '@', '`', '"', '\'', '%', ' ':
		return true
	}
	if strings.ContainsAny(s, ":#\n\t") {
		return true
	}
	// Values that YAML would coerce to a non-string type if left plain.
	switch strings.ToLower(s) {
	case "true", "false", "null", "yes", "no", "on", "off", "~":
		return true
	}
	return false
}
