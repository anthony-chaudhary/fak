package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// guard_allow.go implements `fak guard allow` — the OPERATOR control surface for the
// capability floor's allow-list. The built-in guard floor is deliberately permissive
// (it admits the standard coding-agent toolset and refuses only the genuine-danger
// classes; see guard-default-policy.json), so a DEFAULT_DENY is usually a tool the
// floor simply never enumerated — a harness verb, an MCP tool, a new orchestration
// name. When the operator decides that tool is fine, this verb records it in a small,
// operator-authored OVERLAY the guard floor unions into its allow-list at launch.
//
// The overlay is the "add to always-allow policy" path, and it is out-of-band from
// the agent BY CONSTRUCTION: the operator runs `fak guard allow <tool>` in their own
// shell, so a wrapped agent can never grant itself a capability by editing its own
// context. It also only WIDENS the allow surface — the danger arg-rules (rm -rf, sudo,
// disk wipe, RCE pipe) and any explicit deny are untouched, so re-admitting a
// DEFAULT_DENY'd tool never loosens the danger floor.
//
// The overlay is a separate, tiny file rather than an edit to the base policy so the
// two stay reviewable apart: the base floor is the shipped default (or a `--policy`
// manifest), and the overlay is this host/repo's operator decisions, diffable on its
// own. `fak guard allow --from-journal` closes the loop from a block back to the fix:
// it reads the guarded session's audit journal, lists the tools that were blocked, and
// prints the exact command (or, with --add-all, applies it).
const guardAllowOverlayVersion = "fak-guard-allow/v1"

// guardAllowOverlayEnv points the overlay at a non-default path — e.g. a host-wide
// file shared across repos, or a location the operator keeps under their own control.
// When unset the overlay is repo-local (.fak/guard/allow.json), so it is scoped to the
// checkout and versionable beside it.
const guardAllowOverlayEnv = "FAK_GUARD_ALLOW_OVERLAY"

// guardAllowOverlay is the on-disk schema: two positive lists that union into the
// floor's Allow (exact tool names) and AllowPrefix (a tool-name prefix family). It is
// deliberately a strict SUBSET of the policy manifest — no deny, no arg-rules — because
// the overlay's whole job is to WIDEN allow, never to tighten anything.
type guardAllowOverlay struct {
	Version     string   `json:"version"`
	Allow       []string `json:"allow,omitempty"`
	AllowPrefix []string `json:"allow_prefix,omitempty"`
}

// guardAllowOverlayPath resolves the overlay file: the env override wins, else the
// repo-local default beside the guard audit journal's discovery root.
func guardAllowOverlayPath() string {
	if p := strings.TrimSpace(os.Getenv(guardAllowOverlayEnv)); p != "" {
		return p
	}
	return filepath.Join(findRepoRoot("."), ".fak", "guard", "allow.json")
}

// loadGuardAllowOverlay reads and validates the overlay. A MISSING file is not an
// error — it is the common "no operator overrides yet" case and yields an empty
// overlay (fail-open: the base floor stands unchanged). A present-but-malformed file
// fails loud, the same discipline the policy loader uses: an operator who believes
// they widened the floor must never be silently still enforcing the bare default.
func loadGuardAllowOverlay(path string) (guardAllowOverlay, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return guardAllowOverlay{Version: guardAllowOverlayVersion}, nil
		}
		return guardAllowOverlay{}, fmt.Errorf("guard allow overlay %s: %w", path, err)
	}
	var ov guardAllowOverlay
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&ov); err != nil {
		return guardAllowOverlay{}, fmt.Errorf("guard allow overlay %s: invalid: %w", path, err)
	}
	if v := strings.TrimSpace(ov.Version); v != "" && v != guardAllowOverlayVersion {
		return guardAllowOverlay{}, fmt.Errorf("guard allow overlay %s: unsupported version %q (want %s)", path, ov.Version, guardAllowOverlayVersion)
	}
	ov.Version = guardAllowOverlayVersion
	ov.Allow = guardAllowNormalize(ov.Allow)
	ov.AllowPrefix = guardAllowNormalize(ov.AllowPrefix)
	return ov, nil
}

// saveGuardAllowOverlay writes the overlay as pretty, newline-terminated JSON,
// creating the parent dir. It normalizes (trim/dedupe/sort) so the file stays a clean,
// reviewable diff regardless of the order tools were added.
func saveGuardAllowOverlay(path string, ov guardAllowOverlay) error {
	ov.Version = guardAllowOverlayVersion
	ov.Allow = guardAllowNormalize(ov.Allow)
	ov.AllowPrefix = guardAllowNormalize(ov.AllowPrefix)
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("guard allow overlay: mkdir %s: %w", dir, err)
		}
	}
	b, err := json.MarshalIndent(ov, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	return writeGuardAllowOverlayAtomic(path, b)
}

func writeGuardAllowOverlayAtomic(path string, body []byte) error {
	dir := filepath.Dir(path)
	if dir == "" {
		dir = "."
	}
	tmp, err := os.CreateTemp(dir, ".allow-*.json.tmp")
	if err != nil {
		return fmt.Errorf("guard allow overlay: create temp: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o644); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("guard allow overlay: chmod temp: %w", err)
	}
	if _, err := tmp.Write(body); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("guard allow overlay: write temp: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("guard allow overlay: sync temp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("guard allow overlay: close temp: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("guard allow overlay: replace %s: %w", path, err)
	}
	return nil
}

// guardApplyAllowOverlay unions the overlay's allow/allow_prefix into a runtime floor,
// returning the number of NEW entries added (0 when the overlay is empty or every
// entry was already on the floor). It ONLY widens the allow surface: an existing
// explicit deny or a danger arg-rule is left in place, so an operator can re-admit a
// DEFAULT_DENY'd tool without ever softening the genuine-danger floor.
func guardApplyAllowOverlay(rt *policy.Runtime, ov guardAllowOverlay) int {
	added := 0
	if len(ov.Allow) > 0 {
		if rt.Adjudicator.Allow == nil {
			rt.Adjudicator.Allow = map[string]bool{}
		}
		for _, t := range ov.Allow {
			if !rt.Adjudicator.Allow[t] {
				rt.Adjudicator.Allow[t] = true
				added++
			}
			policy.AttachShellDangerRules(rt, t)
		}
	}
	if len(ov.AllowPrefix) > 0 {
		existing := make(map[string]bool, len(rt.Adjudicator.AllowPrefix))
		for _, p := range rt.Adjudicator.AllowPrefix {
			existing[p] = true
		}
		for _, p := range ov.AllowPrefix {
			if !existing[p] {
				rt.Adjudicator.AllowPrefix = append(rt.Adjudicator.AllowPrefix, p)
				existing[p] = true
				added++
			}
		}
	}
	return added
}

// guardAllowNormalize trims, de-dupes, and sorts a name list so the overlay file and
// its rendered summary are deterministic no matter the input order.
func guardAllowShellAttachments(names []string) []string {
	var out []string
	for _, name := range guardAllowNormalize(names) {
		if sets := policy.ShellDangerRuleSetsFor(name); len(sets) > 0 {
			out = append(out, name+"="+strings.Join(sets, "+"))
		}
	}
	return out
}

func printGuardAllowShellAttachments(w io.Writer, names []string) {
	if attached := guardAllowShellAttachments(names); len(attached) > 0 {
		fmt.Fprintf(w, "  Attached inherited shell danger rules: %s.\n", strings.Join(attached, ", "))
	}
}

func guardAllowNormalize(in []string) []string {
	seen := make(map[string]bool, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

// guardAllowSubtract returns in with every element of remove dropped.
func guardAllowSubtract(in, remove []string) []string {
	rm := make(map[string]bool, len(remove))
	for _, r := range remove {
		rm[strings.TrimSpace(r)] = true
	}
	out := make([]string, 0, len(in))
	for _, s := range in {
		if rm[s] {
			continue
		}
		out = append(out, s)
	}
	return out
}

// cmdGuardAllow is the `fak guard allow` subcommand, peeled off in cmdGuard before the
// wrap-a-command flag parse. It is pure operator plumbing: it never touches a wrapped
// agent, only the overlay file this host's operator owns.
func cmdGuardAllow(argv []string) {
	fs := flag.NewFlagSet("guard allow", flag.ExitOnError)
	fs.Usage = func() { fmt.Fprintln(os.Stderr, guardAllowUsage()) }
	list := fs.Bool("list", false, "print the current operator allow overlay and its path, then exit")
	remove := fs.Bool("remove", false, "remove the named tool(s)/prefix(es) from the overlay instead of adding")
	prefix := fs.Bool("prefix", false, "treat the positional args as allow_prefix entries (a tool-name PREFIX) rather than exact names")
	fromJournal := fs.Bool("from-journal", false, "list the tools a guarded session BLOCKED (DEFAULT_DENY) from an audit journal, each with the exact command to allow it")
	journalPath := fs.String("journal", "", "the audit journal --from-journal reads (default: the newest repo-local guard journal)")
	fromClaudeSettings := fs.Bool("from-claude-settings", false, "import permissions.allow from Claude settings.json (+ settings.local.json, or a positional path) into the overlay; name-level entries only")
	addAll := fs.Bool("add-all", false, "with --from-journal / --from-claude-settings, add EVERY mappable entry found to the overlay in one step")
	_ = fs.Parse(argv)

	path := guardAllowOverlayPath()
	ov, err := loadGuardAllowOverlay(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak guard allow:", err)
		os.Exit(1)
	}

	switch {
	case *fromJournal:
		os.Exit(runGuardAllowFromJournal(os.Stdout, os.Stderr, path, &ov, *journalPath, *addAll))
	case *fromClaudeSettings:
		os.Exit(runGuardAllowFromClaudeSettings(os.Stdout, os.Stderr, path, &ov, fs.Args(), *addAll))
	case *list:
		printGuardAllowOverlay(os.Stdout, path, ov)
	default:
		names := guardAllowNormalize(fs.Args())
		if len(names) == 0 {
			fmt.Fprintln(os.Stderr, guardAllowUsage())
			os.Exit(2)
		}
		if *remove {
			before := len(ov.Allow) + len(ov.AllowPrefix)
			ov.Allow = guardAllowSubtract(ov.Allow, names)
			ov.AllowPrefix = guardAllowSubtract(ov.AllowPrefix, names)
			if err := saveGuardAllowOverlay(path, ov); err != nil {
				fmt.Fprintln(os.Stderr, "fak guard allow:", err)
				os.Exit(1)
			}
			fmt.Printf("fak guard allow: removed %d entr(ies) — overlay now:\n", before-(len(ov.Allow)+len(ov.AllowPrefix)))
			printGuardAllowOverlay(os.Stdout, path, ov)
			return
		}
		if *prefix {
			ov.AllowPrefix = append(ov.AllowPrefix, names...)
		} else {
			ov.Allow = append(ov.Allow, names...)
		}
		if err := saveGuardAllowOverlay(path, ov); err != nil {
			fmt.Fprintln(os.Stderr, "fak guard allow:", err)
			os.Exit(1)
		}
		fmt.Printf("fak guard allow: added %s to the operator allow overlay.\n", strings.Join(names, ", "))
		if !*prefix {
			printGuardAllowShellAttachments(os.Stdout, names)
		}
		fmt.Println("  Takes effect on the next `fak guard` launch (or POST /v1/fak/policy/reload on a running gateway).")
		printGuardAllowOverlay(os.Stdout, path, ov)
	}
}

// guardAllowBlockedTool is one DEFAULT_DENY'd tool name and how many times the guarded
// session refused it — the join key for "what did the floor block, and how do I allow it".
type guardAllowBlockedTool struct {
	name  string
	count int
}

// guardAllowBlockedTools folds the audit-journal rows into the DEFAULT_DENY'd tool
// names, most-blocked first. It reads ONLY the DEFAULT_DENY reason on purpose: that is
// the "never enumerated by the floor" class the overlay can fix. A POLICY_BLOCK (a
// danger arg-rule like rm -rf, or an explicit deny) is NOT surfaced here — allowing the
// tool name would not lift it (the tool is already allowed; its argument is the refusal),
// and it SHOULD stay blocked, so the overlay never invites loosening the danger floor.
func guardAllowBlockedTools(rows []journal.Row) []guardAllowBlockedTool {
	counts := map[string]int{}
	for _, r := range rows {
		if r.Verdict != "DENY" || r.Reason != "DEFAULT_DENY" {
			continue
		}
		if name := strings.TrimSpace(r.Tool); name != "" {
			counts[name]++
		}
	}
	out := make([]guardAllowBlockedTool, 0, len(counts))
	for n, c := range counts {
		out = append(out, guardAllowBlockedTool{name: n, count: c})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].count != out[j].count {
			return out[i].count > out[j].count
		}
		return out[i].name < out[j].name
	})
	return out
}

// runGuardAllowFromJournal reads a guard audit journal, lists the DEFAULT_DENY'd tools,
// and either prints the exact allow command (default) or, with addAll, records them all
// in the overlay. It fails soft on a missing/empty journal (no blocks to report), the
// same tolerance journal.ReadRows already gives a missing file.
func runGuardAllowFromJournal(stdout, stderr io.Writer, overlayPath string, ov *guardAllowOverlay, journalPath string, addAll bool) int {
	jp := strings.TrimSpace(journalPath)
	if jp == "" {
		jp = guardReadableAuditPath()
	}
	rows, err := journal.ReadRows(jp)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard allow: read journal %s: %v\n", jp, err)
		return 1
	}
	blocked := guardAllowBlockedTools(rows)
	if len(blocked) == 0 {
		fmt.Fprintf(stdout, "fak guard allow: no DEFAULT_DENY blocks found in %s — nothing to allow.\n", jp)
		return 0
	}
	fmt.Fprintf(stdout, "Blocked (DEFAULT_DENY) tool(s) in %s:\n", jp)
	names := make([]string, 0, len(blocked))
	for _, t := range blocked {
		fmt.Fprintf(stdout, "  %-28s x%d\n", t.name, t.count)
		names = append(names, t.name)
	}
	if addAll {
		ov.Allow = append(ov.Allow, names...)
		if err := saveGuardAllowOverlay(overlayPath, *ov); err != nil {
			fmt.Fprintln(stderr, "fak guard allow:", err)
			return 1
		}
		fmt.Fprintf(stdout, "\nAdded %d tool(s) to the operator allow overlay: %s\n", len(names), overlayPath)
		fmt.Fprintln(stdout, "  Takes effect on the next `fak guard` launch.")
		printGuardAllowOverlay(stdout, overlayPath, *ov)
		return 0
	}
	fmt.Fprintln(stdout, "\nTo always-allow these for future sessions (operator, out-of-band from the agent):")
	fmt.Fprintf(stdout, "  fak guard allow %s\n", strings.Join(names, " "))
	fmt.Fprintln(stdout, "  (or add them all at once: fak guard allow --from-journal --add-all)")
	return 0
}

// printGuardAllowOverlay renders the current overlay for --list and post-mutation echo.
func printGuardAllowOverlay(w io.Writer, path string, ov guardAllowOverlay) {
	fmt.Fprintf(w, "operator allow overlay: %s\n", path)
	if len(ov.Allow) == 0 && len(ov.AllowPrefix) == 0 {
		fmt.Fprintln(w, "  (empty — no extra tools allowed beyond the capability floor)")
		return
	}
	if len(ov.Allow) > 0 {
		fmt.Fprintf(w, "  allow (exact) : %s\n", strings.Join(ov.Allow, ", "))
	}
	if len(ov.AllowPrefix) > 0 {
		fmt.Fprintf(w, "  allow (prefix): %s\n", strings.Join(ov.AllowPrefix, ", "))
	}
}

// guardAllowUsage is the one-screen help for the subcommand.
func guardAllowUsage() string {
	return strings.Join([]string{
		"fak guard allow — the operator control for the always-allow overlay (out-of-band from the agent).",
		"",
		"usage:",
		"  fak guard allow <tool>...              add exact tool name(s) to the always-allow overlay",
		"  fak guard allow --prefix <prefix>...   add an allow_prefix (a tool-name PREFIX family) instead",
		"  fak guard allow --remove <name>...     remove entr(ies) from the overlay",
		"  fak guard allow --list                 print the current overlay and its path",
		"  fak guard allow --from-journal         list what a guarded session BLOCKED + the command to allow each",
		"  fak guard allow --from-journal --add-all   add every blocked tool in one step",
		"  fak guard allow --from-claude-settings [path]   import permissions.allow from .claude/settings.json (name-level only)",
		"  fak guard allow --from-claude-settings --add-all   apply that import to the overlay",
		"",
		"The overlay is an operator-authored file the guard floor UNIONS into its allow-list at launch",
		"(default .fak/guard/allow.json; override with $" + guardAllowOverlayEnv + "). It only WIDENS what is",
		"allowed — the genuine-danger arg-rules (rm -rf, sudo, disk wipe, RCE pipe) and explicit denies are",
		"untouched. Because you run it in your own shell, the wrapped agent can never grant itself a capability.",
		"Put flags before positional names (Go flag parsing stops at the first non-flag argument).",
	}, "\n")
}
