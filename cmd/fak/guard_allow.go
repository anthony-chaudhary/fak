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
	"time"

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
	// Expiry records an optional TTL per widening (#5179, epic #5170 Track D
	// "GATED-WIDEN safety rails"): it maps an Allow / AllowPrefix entry name to an
	// RFC3339 (UTC) instant after which the entry is auto-reverted. `fak guard allow
	// <tool> --ttl <duration>` writes now+duration here; a name ABSENT from the map is
	// permanent (the pre-#5179 semantics). Read paths drop an entry whose expiry has
	// passed (guardAllowDropExpired), so an operator widening "just for now" returns the
	// floor to its heaviest baseline on the next `fak guard` launch without a manual
	// removal — closing the drift the ticket names. Keyed by the entry string so one map
	// covers both the exact-name and the prefix list.
	Expiry map[string]string `json:"expiry,omitempty"`
}

type guardAllowOverlayLayer struct {
	Name string
	Path string
}

func guardAllowUserOverlayPath() string {
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return ""
	}
	return filepath.Join(home, ".fak", "guard", "allow.json")
}

// guardAllowOverlayPaths returns the effective ordered layers. An explicit env
// override remains the sole layer for backward compatibility; otherwise the
// per-user layer is unioned before the repo-local layer.
func guardAllowOverlayPaths() []guardAllowOverlayLayer {
	if p := strings.TrimSpace(os.Getenv(guardAllowOverlayEnv)); p != "" {
		return []guardAllowOverlayLayer{{Name: "env", Path: p}}
	}
	layers := make([]guardAllowOverlayLayer, 0, 2)
	if p := guardAllowUserOverlayPath(); p != "" {
		layers = append(layers, guardAllowOverlayLayer{Name: "user", Path: p})
	}
	return append(layers, guardAllowOverlayLayer{Name: "repo", Path: filepath.Join(findRepoRoot("."), ".fak", "guard", "allow.json")})
}

// guardAllowOverlayPath is the default WRITE target: env override when set,
// otherwise repo-local. Reads use guardAllowOverlayPaths.
func guardAllowOverlayPath() string {
	if p := strings.TrimSpace(os.Getenv(guardAllowOverlayEnv)); p != "" {
		return p
	}
	return filepath.Join(findRepoRoot("."), ".fak", "guard", "allow.json")
}

// guardAllowWritePathForScope resolves the WRITE target for a named scope from the
// guard_allow_scope.go precedence table: "session" is this session's overlay (the
// narrowest layer, and the one dropped at guard teardown — armed at boot, dropped in
// finishGuardChildAndReport, see guard_allow_scope.go), "user" the host-wide per-user
// file, anything else the default repo-local target (or the env override when one is set).
func guardAllowWritePathForScope(scope string) (string, error) {
	switch scope {
	case guardAllowScopeSession:
		return guardAllowSessionOverlayPath(), nil
	case "user":
		p := guardAllowUserOverlayPath()
		if p == "" {
			return "", errors.New("user home directory is unavailable")
		}
		return p, nil
	}
	return guardAllowOverlayPath(), nil
}

func guardAllowWritePath(user bool) (string, error) {
	if user {
		return guardAllowWritePathForScope("user")
	}
	return guardAllowWritePathForScope("repo")
}

func guardAllowOverlayLayerPaths() []string {
	layers := guardAllowLayersWithSessionScope(guardAllowOverlayPaths())
	out := make([]string, 0, len(layers))
	for _, layer := range layers {
		out = append(out, layer.Path)
	}
	return out
}

func loadGuardAllowOverlayLayers() (guardAllowOverlay, []guardAllowOverlayLayer, error) {
	merged := guardAllowOverlay{Version: guardAllowOverlayVersion}
	// The ENFORCEMENT read: guardAllowEffectiveReadLayers, not the raw stack, so a session
	// layer the boot reclaim could not clear is not merged into the floor (guard_allow_scope.go).
	layers := guardAllowEffectiveReadLayers()
	for _, layer := range layers {
		ov, err := loadGuardAllowOverlay(layer.Path)
		if err != nil {
			return guardAllowOverlay{}, layers, err
		}
		merged.Allow = append(merged.Allow, ov.Allow...)
		merged.AllowPrefix = append(merged.AllowPrefix, ov.AllowPrefix...)
	}
	merged.Allow = guardAllowNormalize(merged.Allow)
	merged.AllowPrefix = guardAllowNormalize(merged.AllowPrefix)
	return merged, layers, nil
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
	// Launch-boundary TTL auto-revert (#5179): every read path funnels through here, so an
	// entry past its expiry is dropped from the floor and from `--list` alike on the next
	// launch. A missing/permanent entry and an unparseable stamp are retained.
	ov, _ = guardAllowDropExpired(ov, guardAllowNow())
	guardAllowPruneOrphanExpiry(&ov)
	return ov, nil
}

// saveGuardAllowOverlay writes the overlay as pretty, newline-terminated JSON,
// creating the parent dir. It normalizes (trim/dedupe/sort) so the file stays a clean,
// reviewable diff regardless of the order tools were added.
func saveGuardAllowOverlay(path string, ov guardAllowOverlay) error {
	ov.Version = guardAllowOverlayVersion
	ov.Allow = guardAllowNormalize(ov.Allow)
	ov.AllowPrefix = guardAllowNormalize(ov.AllowPrefix)
	guardAllowPruneOrphanExpiry(&ov)
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

// guardAllowExpiryStamp is the on-disk expiry format: RFC3339 in UTC, so the file stays
// a stable, timezone-free, reviewable diff. `--ttl <d>` records now+d through this.
func guardAllowExpiryStamp(t time.Time) string { return t.UTC().Format(time.RFC3339) }

// guardAllowNow is the ONE clock every #5179 TTL decision reads: the read-time expiry
// check in loadGuardAllowOverlay, the `--list` remaining-TTL render, and the `--ttl`
// stamp `fak guard allow` writes. It is a package var rather than a direct time.Now()
// call so a test can pin the exact instant an entry crosses its expiry and witness both
// sides of that boundary against ONE on-disk overlay — the alternative is a test that
// either sleeps for the window (slow, and flaky under a loaded shared runner) or can only
// ever assert a stamp so far in the past that the boundary itself is never exercised.
// Production is unaffected: the default is time.Now, and nothing but a test reassigns it.
var guardAllowNow = time.Now

// guardAllowStampExpiry applies the `--ttl` semantics to the entries just added: a
// POSITIVE window records now+ttl on each name, and a ttl of ZERO (the default — "no
// expiry") CLEARS any stamp those names carried, so re-adding a "just for now" widening
// with no --ttl promotes it back to permanent (#5179). It returns the stamp written, or
// "" for the permanent case, so the operator echo quotes the instant that actually landed
// on disk instead of re-deriving it from a second clock read.
//
// A negative window never reaches here — cmdGuardAllow refuses it up front, because
// stamping an already-past expiry would add an entry the very next launch drops.
func guardAllowStampExpiry(ov *guardAllowOverlay, names []string, ttl time.Duration, now time.Time) string {
	if ttl <= 0 {
		for _, n := range names {
			delete(ov.Expiry, n)
		}
		return ""
	}
	if ov.Expiry == nil {
		ov.Expiry = map[string]string{}
	}
	stamp := guardAllowExpiryStamp(now.Add(ttl))
	for _, n := range names {
		ov.Expiry[n] = stamp
	}
	return stamp
}

// guardAllowDropExpired removes every Allow / AllowPrefix entry whose recorded expiry is
// at or before now, returning the pruned overlay and the sorted names dropped (#5179).
// It is the launch-boundary auto-revert: because every read path funnels through
// loadGuardAllowOverlay, an entry past its TTL is gone from the enforced floor AND from
// `fak guard allow --list` on the next `fak guard` launch, with no manual removal.
//
// Two entries are DELIBERATELY retained: one with no expiry (the permanent, pre-#5179
// case) and one whose stamp will not parse. A malformed timestamp must never silently
// revoke a widening an operator is relying on — the fail-safe direction here is to keep
// the grant and leave the bad stamp visible, not to drop on a parse error.
func guardAllowDropExpired(ov guardAllowOverlay, now time.Time) (guardAllowOverlay, []string) {
	if len(ov.Expiry) == 0 {
		return ov, nil
	}
	expired := make(map[string]bool, len(ov.Expiry))
	var dropped []string
	for name, stamp := range ov.Expiry {
		t, err := time.Parse(time.RFC3339, strings.TrimSpace(stamp))
		if err != nil {
			continue // unparseable stamp → treat as permanent, keep
		}
		if !now.Before(t) { // now >= expiry
			expired[name] = true
			dropped = append(dropped, name)
		}
	}
	if len(expired) == 0 {
		return ov, nil
	}
	out := guardAllowOverlay{Version: ov.Version}
	out.Allow = guardAllowSubtract(ov.Allow, dropped)
	out.AllowPrefix = guardAllowSubtract(ov.AllowPrefix, dropped)
	for name, stamp := range ov.Expiry {
		if expired[name] {
			continue
		}
		if out.Expiry == nil {
			out.Expiry = map[string]string{}
		}
		out.Expiry[name] = stamp
	}
	sort.Strings(dropped)
	return out, dropped
}

// guardAllowPruneOrphanExpiry drops Expiry keys that no longer name a live Allow /
// AllowPrefix entry, so removing a tool (or its natural drop) never leaves a dangling
// stamp behind. Time-independent by design: the time-based revert is guardAllowDropExpired
// on the read path; this only keeps the map a strict index of the two positive lists.
func guardAllowPruneOrphanExpiry(ov *guardAllowOverlay) {
	if len(ov.Expiry) == 0 {
		return
	}
	live := make(map[string]bool, len(ov.Allow)+len(ov.AllowPrefix))
	for _, n := range ov.Allow {
		live[n] = true
	}
	for _, n := range ov.AllowPrefix {
		live[n] = true
	}
	for name := range ov.Expiry {
		if !live[name] {
			delete(ov.Expiry, name)
		}
	}
	if len(ov.Expiry) == 0 {
		ov.Expiry = nil
	}
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
	list := fs.Bool("list", false, "print effective allow layers with per-layer provenance, then exit")
	user := fs.Bool("user", false, "write the per-user home overlay instead of the repo-local overlay")
	session := fs.Bool("session", false, "write the SESSION-scope overlay: the narrowest layer, applied last, so it is the last word over the repo/user/env layers. EPHEMERAL — a guard drops this layer both at its boot and at its session end, so the widening never survives into another session. Scoped PER LAUNCH: run this INSIDE a guarded session, where the guard's injected $FAK_GUARD_SESSION_ID names the file it is reading. Run outside one it lands in sessions/current.allow.json, which no live guard resolves — so nothing honors it")
	remove := fs.Bool("remove", false, "remove the named tool(s)/prefix(es) from the overlay instead of adding")
	prefix := fs.Bool("prefix", false, "treat the positional args as allow_prefix entries (a tool-name PREFIX) rather than exact names")
	ttl := fs.Duration("ttl", 0, "record an EXPIRY on the added entr(ies): e.g. --ttl 1h. On the first `fak guard` launch after the window they are auto-reverted (dropped from the floor and from --list); a re-add with no --ttl makes the entr(ies) permanent again. 0 = permanent (default).")
	fromJournal := fs.Bool("from-journal", false, "list the tools a guarded session BLOCKED (DEFAULT_DENY) from an audit journal, each with the exact command to allow it")
	journalPath := fs.String("journal", "", "the audit journal --from-journal reads (default: the newest repo-local guard journal)")
	fromClaudeSettings := fs.Bool("from-claude-settings", false, "import permissions.allow from Claude settings.json (+ settings.local.json, or a positional path) into the overlay; name-level entries only")
	fromMCPConfig := fs.Bool("from-mcp-config", false, "import one coarse allow_prefix per mcpServers key from .mcp.json (or a positional path)")
	addAll := fs.Bool("add-all", false, "with an import source, add EVERY mappable entry found to the overlay in one step")
	_ = fs.Parse(argv)

	// The write SCOPE (guard_allow_scope.go): --session is the narrowest layer (applied
	// last, so it is the last word), --user the host-wide one, and the default stays
	// repo-local. Naming both is a contradiction, not a precedence question, so refuse it
	// rather than silently picking one and writing the widening somewhere the operator did
	// not mean.
	writeScope := "repo"
	switch {
	case *session && *user:
		fmt.Fprintln(os.Stderr, "fak guard allow: --session and --user name different scopes; pick one")
		os.Exit(2)
	case *session:
		writeScope = guardAllowScopeSession
	case *user:
		writeScope = "user"
	}
	// --ttl only means anything when ADDING named entries: it stamps an expiry on each. It is
	// a contradiction with --remove and a no-op for the import/list modes, so refuse those
	// combinations rather than silently ignoring the flag. A negative window would add an
	// already-expired entry (dropped on the very next launch), which is never what an operator
	// means, so refuse it too.
	if *ttl != 0 {
		switch {
		case *ttl < 0:
			fmt.Fprintln(os.Stderr, "fak guard allow: --ttl must be a positive duration (e.g. --ttl 1h)")
			os.Exit(2)
		case *remove || *list || *fromJournal || *fromClaudeSettings || *fromMCPConfig:
			fmt.Fprintln(os.Stderr, "fak guard allow: --ttl applies only when adding named entr(ies); it cannot combine with --remove/--list/--from-*")
			os.Exit(2)
		}
	}
	path, err := guardAllowWritePathForScope(writeScope)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak guard allow:", err)
		os.Exit(1)
	}
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
	case *fromMCPConfig:
		os.Exit(runGuardAllowFromMCPConfig(os.Stdout, os.Stderr, path, &ov, fs.Args(), *addAll))
	case *list:
		// List every layer the READ path actually unions — session scope included, in
		// ascending precedence order — and name each layer's scope rank, so an operator
		// can tell which scope a widening lives in instead of inferring it from a path.
		for _, layer := range guardAllowLayersWithSessionScope(guardAllowOverlayPaths()) {
			layerOverlay, layerErr := loadGuardAllowOverlay(layer.Path)
			if layerErr != nil {
				fmt.Fprintln(os.Stderr, "fak guard allow:", layerErr)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stdout, "[%s layer] scope=%s precedence=%d%s\n",
				layer.Name, layer.Name, guardAllowScopeRank(layer.Name), guardAllowScopeDurabilityNote(layer.Name))
			printGuardAllowOverlay(os.Stdout, layer.Path, layerOverlay)
		}
	default:
		names := guardAllowNormalize(fs.Args())
		if len(names) == 0 {
			fmt.Fprintln(os.Stderr, guardAllowUsage())
			os.Exit(2)
		}
		if *remove {
			before := len(ov.Allow) + len(ov.AllowPrefix)
			guardAllowRemove(&ov, names)
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
		// Stamp (or clear) the per-entry expiry. --ttl records now+window; a re-add with no
		// --ttl clears any prior stamp, so an operator can promote a "just for now" widening
		// back to permanent by adding it again (#5179).
		stamp := guardAllowStampExpiry(&ov, names, *ttl, guardAllowNow())
		if err := saveGuardAllowOverlay(path, ov); err != nil {
			fmt.Fprintln(os.Stderr, "fak guard allow:", err)
			os.Exit(1)
		}
		fmt.Printf("fak guard allow: added %s to the operator allow overlay.\n", strings.Join(names, ", "))
		if stamp != "" {
			fmt.Printf("  Expires in %s (at %s) — auto-reverted on the first `fak guard` launch after that.\n", *ttl, stamp)
		}
		if !*prefix {
			printGuardAllowShellAttachments(os.Stdout, names)
		}
		fmt.Println("  Takes effect on the next `fak guard` launch (or POST /v1/fak/policy/reload on a running gateway).")
		printGuardAllowOverlay(os.Stdout, path, ov)
	}
}

func guardAllowRemove(ov *guardAllowOverlay, names []string) {
	ov.Allow = guardAllowSubtract(ov.Allow, names)
	ov.AllowPrefix = guardAllowSubtract(ov.AllowPrefix, names)
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

func guardAllowPrunedTools(rows []journal.Row) []guardAllowBlockedTool {
	counts := map[string]int{}
	for _, r := range rows {
		if r.Kind != journal.KindToolDefinitionPruned {
			continue
		}
		if name := strings.TrimSpace(r.Tool); name != "" {
			counts[name]++
		}
	}
	out := make([]guardAllowBlockedTool, 0, len(counts))
	for name, count := range counts {
		out = append(out, guardAllowBlockedTool{name: name, count: count})
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
// same tolerance journal.ReadRows already gives a missing file. The read is segment-aware
// (#6488): the list is a roll-up of every tool the guard ever blocked, so a tool blocked
// before a rotation cut must not drop off it.
func runGuardAllowFromJournal(stdout, stderr io.Writer, overlayPath string, ov *guardAllowOverlay, journalPath string, addAll bool) int {
	jp := strings.TrimSpace(journalPath)
	if jp == "" {
		jp = guardReadableAuditPath()
	}
	rows, err := journal.ReadAllSegments(jp)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard allow: read journal %s: %v\n", jp, err)
		return 1
	}
	rows = journal.WithoutCutAnchors(rows)
	blocked := guardAllowBlockedTools(rows)
	pruned := guardAllowPrunedTools(rows)
	if len(blocked) == 0 && len(pruned) == 0 {
		fmt.Fprintf(stdout, "fak guard allow: no DEFAULT_DENY blocks or pruned tool definitions found in %s — nothing to allow.\n", jp)
		return 0
	}
	names := make([]string, 0, len(blocked)+len(pruned))
	seen := map[string]bool{}
	if len(blocked) > 0 {
		fmt.Fprintf(stdout, "Blocked (DEFAULT_DENY) tool(s) in %s:\n", jp)
		for _, tool := range blocked {
			fmt.Fprintf(stdout, "  %-28s x%d\n", tool.name, tool.count)
			if !seen[tool.name] {
				seen[tool.name] = true
				names = append(names, tool.name)
			}
		}
	}
	if len(pruned) > 0 {
		fmt.Fprintln(stdout, "Never advertised (the floor pruned these definitions):")
		for _, tool := range pruned {
			fmt.Fprintf(stdout, "  %-28s x%d   fak guard allow %s\n", tool.name, tool.count, shellQuote(tool.name))
			if !seen[tool.name] {
				seen[tool.name] = true
				names = append(names, tool.name)
			}
		}
	}
	if addAll {
		ov.Allow = append(ov.Allow, names...)
		if err := saveGuardAllowOverlay(overlayPath, *ov); err != nil {
			fmt.Fprintln(stderr, "fak guard allow:", err)
			return 1
		}
		fmt.Fprintf(stdout, "\nAdded %d tool(s) to the operator allow overlay: %s\n", len(names), overlayPath)
		fmt.Fprintln(stdout, "  A live guard reloads this overlay automatically; otherwise it takes effect on the next launch.")
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
	printGuardAllowExpiries(w, ov, guardAllowNow())
}

// printGuardAllowExpiries renders the remaining TTL of each expiring entry (#5179), so an
// operator listing the overlay can see which widenings are temporary and how long they
// have left. Entries are sorted for a stable readout; an already-expired entry never
// reaches here because loadGuardAllowOverlay drops it before the render.
func printGuardAllowExpiries(w io.Writer, ov guardAllowOverlay, now time.Time) {
	if len(ov.Expiry) == 0 {
		return
	}
	names := make([]string, 0, len(ov.Expiry))
	for name := range ov.Expiry {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		t, err := time.Parse(time.RFC3339, strings.TrimSpace(ov.Expiry[name]))
		if err != nil {
			fmt.Fprintf(w, "  expires       : %s (unparseable stamp %q — treated as permanent)\n", name, ov.Expiry[name])
			continue
		}
		fmt.Fprintf(w, "  expires       : %s in %s (at %s)\n", name, t.Sub(now).Round(time.Second), guardAllowExpiryStamp(t))
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
		"  fak guard allow --ttl 1h <tool>        add with an EXPIRY: auto-reverted on the first launch past the window",
		"  fak guard allow --remove <name>...     remove entr(ies) from the overlay",
		"  fak guard allow --list                 print every effective layer with its scope, precedence and provenance",
		"  fak guard allow --user <tool>...       write the per-user home layer instead of repo-local",
		"  fak guard allow --session <tool>...    write the session-scope layer (narrowest, applied last; dropped at guard boot AND at session end, so it never outlives a session)",
		"  fak guard allow --from-journal         list what a guarded session BLOCKED + the command to allow each",
		"  fak guard allow --from-journal --add-all   add every blocked tool in one step",
		"  fak guard allow --from-claude-settings [path]   import permissions.allow from .claude/settings.json (name-level only)",
		"  fak guard allow --from-claude-settings --add-all   apply that import to the overlay",
		"  fak guard allow --from-mcp-config [path]   list one coarse allow prefix per .mcp.json server",
		"  fak guard allow --from-mcp-config --add-all   add every server prefix to the overlay",
		"",
		"The overlay is an operator-authored file the guard floor UNIONS into its allow-list at launch",
		"(default .fak/guard/allow.json; override with $" + guardAllowOverlayEnv + "). It only WIDENS what is",
		"allowed — the genuine-danger arg-rules (rm -rf, sudo, disk wipe, RCE pipe) and explicit denies are",
		"untouched. Because you run it in your own shell, the wrapped agent can never grant itself a capability.",
		"Put flags before positional names (Go flag parsing stops at the first non-flag argument).",
	}, "\n")
}
