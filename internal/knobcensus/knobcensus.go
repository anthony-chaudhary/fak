package knobcensus

import (
	"bufio"
	"github.com/anthony-chaudhary/fak/internal/sortkeys"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxknobs"
)

// Verdict is the closed two-token vocabulary the census files every knob under
// (issue #2210, epic #2208). Exactly one applies to each knob.
type Verdict string

const (
	// Intent encodes a choice the system cannot infer — which/what/how-much/
	// consent. Disposition: promote (epic #2208): it must have a full, consistent
	// route on every surface.
	Intent Verdict = "INTENT"
	// Housekeeping is derivable from telemetry — residency, warmth, TTLs,
	// breakpoints, eviction, hygiene. Disposition: automate (#2198 doctrine): the
	// default path must not require it; the knob survives as an operator override.
	Housekeeping Verdict = "HOUSEKEEPING"
)

// Surface is where a knob is exposed to the user.
type Surface string

const (
	SurfaceFlag  Surface = "flag"  // a cmd/fak flag registration
	SurfaceEnv   Surface = "env"   // a FAK_* environment lookup
	SurfaceSkill Surface = "skill" // a .claude/skills context skill (via #2199)
)

// Disposition and owner epic are a pure function of the verdict: INTENT is
// promoted by epic #2208, HOUSEKEEPING is retired by epic #2198.
func (v Verdict) Disposition() string {
	if v == Intent {
		return "promote"
	}
	return "automate"
}

func (v Verdict) OwnerEpic() string {
	if v == Intent {
		return "#2208"
	}
	return "#2198"
}

// Knob is one user-facing knob with its verdict and file:line provenance.
type Knob struct {
	Name          string    `json:"name"`
	Surface       Surface   `json:"surface"`
	Verdict       Verdict   `json:"verdict"`
	Disposition   string    `json:"disposition"`    // promote | automate (derived from verdict)
	OwnerEpic     string    `json:"owner_epic"`     // #2208 (INTENT) | #2198 (HOUSEKEEPING)
	File          string    `json:"file"`           // repo-relative, forward slashes
	Line          int       `json:"line"`           //
	RouteCoverage []Surface `json:"route_coverage"` // surfaces this knob's control is exposed on (its route)
	Evidence      string    `json:"evidence"`
}

// Key is the stable identity a knob is deduped on (surface:name).
func (k Knob) Key() string { return string(k.Surface) + ":" + k.Name }

// Route renders the knob's route-coverage as a compact "surface[+surface]" token
// for the human table (e.g. "flag", "flag+env"). For an INTENT knob this is the
// "each INTENT row names its route" witness the issue (#2210) requires; an INTENT
// knob routed on a single surface is the promotion gap epic #2208 tracks.
func (k Knob) Route() string {
	parts := make([]string, len(k.RouteCoverage))
	for i, s := range k.RouteCoverage {
		parts[i] = string(s)
	}
	return strings.Join(parts, "+")
}

// newKnob stamps the disposition + owner epic that follow from the verdict so a
// caller can never construct a knob whose disposition disagrees with its verdict.
func newKnob(surface Surface, name string, v Verdict, file string, line int, evidence string) Knob {
	return Knob{
		Name: name, Surface: surface, Verdict: v,
		Disposition: v.Disposition(), OwnerEpic: v.OwnerEpic(),
		File: file, Line: line, Evidence: evidence,
	}
}

// Census is a full, sorted scan of the tree's user-facing knobs with verdicts.
type Census struct {
	Knobs        []Knob `json:"knobs"`
	Intent       int    `json:"intent"`
	Housekeeping int    `json:"housekeeping"`
}

// Scan walks root and returns the sorted knob census. It is deterministic: the
// same tree yields byte-identical output (the "run twice → identical" witness).
//
// It CONSUMES #2199's context-knob inventory rather than re-deriving it (the
// "no second context count" contract): every context/cache-management knob is
// HOUSEKEEPING by doctrine — "the management of context and cache is super
// automatic by default" — so ctxknobs' inventory folds in verbatim, whatever
// operator-debug/user-required class #2199 assigned it (that axis is about the
// default path, orthogonal to intent-vs-housekeeping). The census then walks the
// BROADER user-facing surface (guard/session/account/model/fleet flags + FAK_*
// env) for the non-context behavior knobs and classifies each.
//
// A missing scan root (no cmd/fak) is not an error — that source contributes
// nothing.
func Scan(root string) (Census, error) {
	seen := map[string]bool{}
	var knobs []Knob
	add := func(k Knob) {
		if !seen[k.Key()] {
			seen[k.Key()] = true
			knobs = append(knobs, k)
		}
	}

	// (1) Consume #2199 — the context slice, all HOUSEKEEPING.
	inv, err := ctxknobs.Scan(root)
	if err != nil {
		return Census{}, err
	}
	for _, k := range inv.Knobs {
		add(newKnob(Surface(k.Kind), k.Name, Housekeeping, k.File, k.Line,
			"context/cache-management knob (via #2199) — automatic-by-default domain"))
	}

	// (2) Walk the broader behavior surface for the non-context knobs.
	walked, err := scanFlagsAndEnv(root)
	if err != nil {
		return Census{}, err
	}
	for _, k := range walked {
		add(k)
	}

	sort.Slice(knobs, func(i, j int) bool {
		a, b := knobs[i], knobs[j]
		return sortkeys.FileLine(a.File, a.Line, a.Key(), b.File, b.Line, b.Key())
	})

	// Route-coverage post-pass. Group by logical knob identity so the same control
	// exposed on two surfaces (flag --account and env FAK_ACCOUNT) is recognized as
	// one knob whose route covers both. Each row then names its route (the issue's
	// route-coverage column); an INTENT knob routed on a single surface is a
	// promotion gap #2208 must close. Deterministic: surfaces are sorted.
	surfacesByName := map[string]map[Surface]bool{}
	for _, k := range knobs {
		n := normalizeKnobName(k.Name)
		if surfacesByName[n] == nil {
			surfacesByName[n] = map[Surface]bool{}
		}
		surfacesByName[n][k.Surface] = true
	}
	for i := range knobs {
		set := surfacesByName[normalizeKnobName(knobs[i].Name)]
		surfaces := make([]Surface, 0, len(set))
		for s := range set {
			surfaces = append(surfaces, s)
		}
		sort.Slice(surfaces, func(a, b int) bool { return surfaces[a] < surfaces[b] })
		knobs[i].RouteCoverage = surfaces
	}

	census := Census{Knobs: knobs}
	for _, k := range knobs {
		switch k.Verdict {
		case Intent:
			census.Intent++
		case Housekeeping:
			census.Housekeeping++
		}
	}
	return census, nil
}

// --- flag / env scanning (cmd/fak) ---

var (
	reFlagReg  = regexp.MustCompile(`\b(?:flag|fs)\.(?:String|Bool|Int|Int64|Uint|Uint64|Float64|Duration|Var|Func|TextVar)(?:Var)?\(`)
	reFirstStr = regexp.MustCompile(`"([^"]*)"`)
	reEnv      = regexp.MustCompile(`\bos\.(?:Getenv|LookupEnv)\("(FAK_[A-Z0-9_]+)"\)`)
)

func scanFlagsAndEnv(root string) ([]Knob, error) {
	dir := filepath.Join(root, "cmd", "fak")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var knobs []Knob
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		rel := "cmd/fak/" + e.Name()
		fileKnobs, err := scanGoFile(filepath.Join(dir, e.Name()), rel)
		if err != nil {
			return nil, err
		}
		knobs = append(knobs, fileKnobs...)
	}
	return knobs, nil
}

func scanGoFile(path, rel string) ([]Knob, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var knobs []Knob
	seen := map[string]bool{} // dedup identical (surface,name) within a file
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
		if loc := reFlagReg.FindStringIndex(text); loc != nil {
			if m := reFirstStr.FindStringSubmatch(text[loc[1]-1:]); m != nil {
				knobs = admitKnob(knobs, seen, SurfaceFlag, m[1], rel, line)
			}
		}
		for _, m := range reEnv.FindAllStringSubmatch(text, -1) {
			knobs = admitKnob(knobs, seen, SurfaceEnv, m[1], rel, line)
		}
	}
	return knobs, sc.Err()
}

// admitKnob classifies one surface match and records it unless this (surface,name) pair
// was already seen in the file. The flag-registration and env lanes differ only in the
// surface they claim, so both admit through here.
func admitKnob(knobs []Knob, seen map[string]bool, surface Surface, name, rel string, line int) []Knob {
	k, ok := classify(surface, name, rel, line)
	if !ok || seen[k.Key()] {
		return knobs
	}
	seen[k.Key()] = true
	return append(knobs, k)
}

// intentTokens name a user choice the system cannot infer (which/what/consent).
// strongIntentTokens additionally OVERRIDE a housekeeping match: an account or a
// consent gate is INTENT even when its name also carries a timing/hygiene word.
var (
	strongIntentTokens = []string{"account", "model", "consent", "confirm", "approve", "persona", "objective", "goal"}
	intentTokens       = []string{
		"account", "model", "consent", "confirm", "approve", "persona",
		"objective", "goal", "force", "ship", "release", "publish",
		"target", "select", "profile", "owner",
	}
	// houseTokens name state derivable from telemetry — residency, warmth, TTLs,
	// breakpoints, eviction, hygiene, timing.
	houseTokens = []string{
		"ttl", "warm", "residen", "compact", "breakpoint", "evict", "hygiene",
		"prune", "trim", "sweep", "cooldown", "backoff", "retry", "refresh",
		"rotate", "headroom", "autoheal", "watchdog", "reclaim", "flush",
		"stale", "poll", "heartbeat", "skew", "ratelimit",
	}
)

// classify decides whether a flag/env name is a user-facing BEHAVIOR knob and,
// if so, its verdict. It is deliberately narrow (name-only, like #2199's
// isContextFlagName) so the census stays meaningful rather than sweeping in every
// plumbing/output/path flag. A name that matches neither vocabulary is not a
// behavior knob and is excluded — the census is the CONTROL surface, not the
// whole flag registry. When a name carries both an intent and a housekeeping
// word, a strong-intent token wins (a consent/account gate is never housekeeping).
func classify(surface Surface, name, file string, line int) (Knob, bool) {
	low := strings.ToLower(name)
	strongIntent := containsAny(low, strongIntentTokens)
	isIntent := strongIntent || containsAny(low, intentTokens)
	isHouse := containsAny(low, houseTokens)

	switch {
	case isHouse && !strongIntent:
		return newKnob(surface, name, Housekeeping, file, line,
			"name encodes telemetry-derivable state (residency/warmth/TTL/hygiene) — automate"), true
	case isIntent:
		return newKnob(surface, name, Intent, file, line,
			"name encodes a user choice the system cannot infer (which/what/consent) — promote"), true
	default:
		return Knob{}, false
	}
}

// normalizeKnobName folds a flag or env name to the logical knob identity used
// for route-coverage: the same control on two surfaces (flag "account", env
// "FAK_ACCOUNT") must normalize to one key. Lowercase, drop a FAK_ prefix, strip
// the "_"/"-" word separators the two surfaces spell differently.
func normalizeKnobName(name string) string {
	low := strings.ToLower(name)
	low = strings.TrimPrefix(low, "fak_")
	low = strings.TrimPrefix(low, "fak-")
	low = strings.ReplaceAll(low, "_", "")
	low = strings.ReplaceAll(low, "-", "")
	return low
}

func containsAny(s string, tokens []string) bool {
	for _, t := range tokens {
		if strings.Contains(s, t) {
			return true
		}
	}
	return false
}
