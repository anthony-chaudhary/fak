package newleaf

import (
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const (
	ModulePrefix = "github.com/anthony-chaudhary/fak/internal"
	TierMarker   = "// new-leaf:tier"
	LaneMarker   = "# new-leaf:lane"
	TreeMarker   = "# new-leaf:tree"
)

var (
	NameRE = regexp.MustCompile(`^[a-z][a-z0-9]*$`)
	Tiers  = map[string]int{"root": 0, "primitive": 1, "foundation-composite": 2, "mechanism": 3, "composer": 4, "integrator": 5}
)

type Options struct {
	Root     string
	Name     string
	Tier     string
	Register bool
	Summary  string
	DryRun   bool
}

type Report struct {
	Name         string   `json:"name"`
	Tier         string   `json:"tier"`
	Register     bool     `json:"register"`
	DryRun       bool     `json:"dry_run"`
	TierAdvisory string   `json:"tier_advisory,omitempty"`
	Edits        []string `json:"edits"`
	NextSteps    []string `json:"next_steps"`
}

func DocGo(name, tier string, n int, summary string) string {
	return fmt.Sprintf(
		"// Package %s is %s.\n//\n// Tier: %s (%d) - see internal/architest. This package may import only\n// packages whose tier is <= %d; an upward import fails the architest gate.\n// See AGENTS.md and internal/architest for the layering contract.\npackage %s\n",
		name, summary, tier, n, n, name,
	)
}

func ImplGo(name string, register bool) string {
	head := "package " + name + "\n"
	if register {
		head += "\nimport \"" + ModulePrefix + "/abi\"\n"
	}
	body := "\n// Ready reports that the leaf is wired. Replace this placeholder with the\n" +
		"// real surface this package exists to provide.\n" +
		"func Ready() bool { return true }\n"
	if register {
		body += "\n// init registers this leaf's driver against the frozen ABI before the\n" +
			"// kernel boots. Replace this placeholder with the real abi.Register* call.\n" +
			"func init() {\n\t_ = abi.ABIMinor\n}\n"
	}
	return head + body
}

func TestGo(name string) string {
	return "package " + name + "\n\n" +
		"import \"testing\"\n\n" +
		"// TestSpine drives the generated leaf's real surface end to end. Keep this\n" +
		"// representative path working while the proof envelope expands around it.\n" +
		"func TestSpine(t *testing.T) {\n" +
		"\tif !Ready() {\n" +
		"\t\tt.Fatal(\"generated leaf spine did not reach Ready\")\n" +
		"\t}\n" +
		"}\n"
}

// isMarkerLine reports whether ln IS the marker line rather than a line that
// merely MENTIONS the marker.
//
// The distinction is load-bearing, and a substring match got it wrong: dos.toml
// carries a prose comment documenting why some lanes landed by hand ("... so the
// `# new-leaf:lane` auto-insert never fired"), and that sentence sits INSIDE the
// [lanes].concurrent array. A Contains-based match treated it as a third marker,
// so every generated leaf was inserted into `concurrent` twice — once at the real
// marker, once above the prose — and the duplicate tripped the roster gate
// (CONCURRENT_LANES_OVERLAP, internal/architest TestLaneRosterHasNoDuplicates) on
// a tree the generator had just written. A real marker always OPENS its line, so
// anchoring the match to the trimmed prefix keeps documentation about the marker
// from acting as one.
func isMarkerLine(ln, marker string) bool {
	return strings.HasPrefix(strings.TrimSpace(ln), marker)
}

func InsertBeforeMarker(text, marker, line string) (string, error) {
	var out strings.Builder
	done := false
	for _, ln := range splitKeepLines(text) {
		if !done && isMarkerLine(ln, marker) {
			out.WriteString(line)
			done = true
		}
		out.WriteString(ln)
	}
	if !done {
		return "", fmt.Errorf("marker %q not found", marker)
	}
	return out.String(), nil
}

func InsertBeforeAllMarkers(text, marker, line string) (string, error) {
	var out strings.Builder
	hits := 0
	for _, ln := range splitKeepLines(text) {
		if isMarkerLine(ln, marker) {
			out.WriteString(line)
			hits++
		}
		out.WriteString(ln)
	}
	if hits == 0 {
		return "", fmt.Errorf("marker %q not found", marker)
	}
	return out.String(), nil
}

func AddRegistration(text, name string) (string, error) {
	imp := "\t_ \"" + ModulePrefix + "/" + name + "\"\n"
	if strings.Contains(text, imp) {
		return text, nil
	}
	idx := strings.LastIndex(text, "\n)")
	if idx < 0 {
		return "", fmt.Errorf("could not find import block close")
	}
	return text[:idx+1] + imp + text[idx+1:], nil
}

func AddLeafLane(text, name string) (string, error) {
	if strings.Contains(text, "[\"internal/"+name+"/**\"]") || strings.Contains(text, "fak/internal/"+name+"/**") {
		return text, nil
	}
	var err error
	text, err = InsertBeforeAllMarkers(text, LaneMarker, "  \""+name+"\",\n")
	if err != nil {
		return "", err
	}
	text, err = InsertBeforeMarker(text, TreeMarker, name+" = [\"internal/"+name+"/**\"]\n")
	if err != nil {
		return "", err
	}
	return text, nil
}

func Apply(opts Options) (Report, error) {
	name := opts.Name
	tier := opts.Tier
	n, ok := Tiers[tier]
	if !ok {
		return Report{}, fmt.Errorf("unknown tier %q", tier)
	}
	if !NameRE.MatchString(name) {
		return Report{}, fmt.Errorf("%q is not a valid lowercase Go package name", name)
	}
	if tier == "root" {
		return Report{}, fmt.Errorf("'root' is reserved for internal/abi; pick primitive or higher")
	}
	root := opts.Root
	if root == "" {
		var err error
		root, err = os.Getwd()
		if err != nil {
			return Report{}, err
		}
	}
	summary := opts.Summary
	if summary == "" {
		summary = "a tier-" + tier + " leaf (describe its single responsibility)"
	}

	leafDir := filepath.Join(root, "internal", name)
	if _, err := os.Stat(leafDir); err == nil {
		return Report{}, fmt.Errorf("%s already exists - refusing to overwrite", leafDir)
	}
	archPath := filepath.Join(root, "internal", "architest", "architest_test.go")
	regPath := filepath.Join(root, "internal", "registrations", "registrations.go")
	dosPath := filepath.Join(root, "dos.toml")

	archText, err := os.ReadFile(archPath)
	if err != nil {
		return Report{}, err
	}
	if strings.Contains(string(archText), "\""+name+"\":") {
		return Report{}, fmt.Errorf("tier table already declares %q", name)
	}
	newArch, err := InsertBeforeMarker(string(archText), TierMarker, fmt.Sprintf("\t%q: %d,\n", name, n))
	if err != nil {
		return Report{}, err
	}

	var newReg string
	if opts.Register {
		regText, err := os.ReadFile(regPath)
		if err != nil {
			return Report{}, err
		}
		newReg, err = AddRegistration(string(regText), name)
		if err != nil {
			return Report{}, err
		}
	}

	var newDos string
	if raw, err := os.ReadFile(dosPath); err == nil {
		newDos, err = AddLeafLane(string(raw), name)
		if err != nil {
			return Report{}, err
		}
	}

	files := map[string]string{
		filepath.Join(leafDir, "doc.go"):        DocGo(name, tier, n, summary),
		filepath.Join(leafDir, name+".go"):      ImplGo(name, opts.Register),
		filepath.Join(leafDir, name+"_test.go"): TestGo(name),
	}
	if !opts.DryRun {
		if err := os.MkdirAll(leafDir, 0o755); err != nil {
			return Report{}, err
		}
		for path, content := range files {
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				return Report{}, err
			}
		}
		if err := os.WriteFile(archPath, []byte(newArch), 0o644); err != nil {
			return Report{}, err
		}
		if opts.Register {
			if err := os.WriteFile(regPath, []byte(newReg), 0o644); err != nil {
				return Report{}, err
			}
		}
		if newDos != "" {
			if err := os.WriteFile(dosPath, []byte(newDos), 0o644); err != nil {
				return Report{}, err
			}
		}
	}

	report := Report{Name: name, Tier: tier, Register: opts.Register, DryRun: opts.DryRun}
	for _, p := range []string{
		filepath.Join("internal", name, "doc.go"),
		filepath.Join("internal", name, name+".go"),
		filepath.Join("internal", name, name+"_test.go"),
	} {
		report.Edits = append(report.Edits, filepath.ToSlash(p))
	}
	report.Edits = append(report.Edits, "internal/architest/architest_test.go (tier table)")
	if opts.Register {
		report.Edits = append(report.Edits, "internal/registrations/registrations.go (defconfig)")
	}
	if newDos != "" {
		report.Edits = append(report.Edits, "dos.toml (concurrency lane)")
	}
	report.NextSteps = []string{
		"1. name the end-to-end outcome and implement its smallest applied path through the real seam in internal/" + name + "/" + name + ".go",
		"2. go test ./internal/" + name + " ./internal/architest to capture the working spine test or command that drives the real " + name + " object end to end",
		"3. fak-dev issue fanout --title \"<feature>\" --leaf " + name + " --spine <commit|command|doc> --json",
		"4. expand the exhaustive proof envelope around that spine: failure paths, edge cases, platforms, concurrency, and soak",
		"5. measure the end-to-end baseline before optimizing, then keep only a net-true gain",
	}
	// Nudge toward the minimum-correct tier at creation (#4045): the skeleton
	// imports only abi (when registered) or nothing, so declaring above foundation
	// surfaces an advisory the moment the leaf is scaffolded — foundation stops
	// being the frictionless default without any creation being blocked.
	var scaffoldDeps []string
	if opts.Register {
		scaffoldDeps = []string{"abi"}
	}
	report.TierAdvisory = TierAdvisory(tier, scaffoldDeps, ParseTierTable(string(archText)))
	return report, nil
}

func (r Report) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// --- minimum-correct-tier suggestion (#4045) --------------------------------
//
// The imbalance the architest-foundation epic (#4041) chases is authored in one
// `new-leaf` call at a time: with no signal about what a leaf imports, the path of
// least resistance is always `--tier foundation`. These helpers compute the minimum
// tier consistent with the layered-DAG rule (a package may import only packages of
// tier <= its own, so its floor is max(tier(dep))) directly from the leaf's imports,
// so the author chooses from evidence. Advisory only — edit-time enforcement is #2082.

var tierRowRE = regexp.MustCompile(`"([a-z][a-z0-9]*)"\s*:\s*(\d+)`)

// TierNameForLevel reverses Tiers: a level (0..5) to its canonical tier name, or ""
// for an unknown level.
func TierNameForLevel(level int) string {
	for name, n := range Tiers {
		if n == level {
			return name
		}
	}
	return ""
}

// ParseTierTable extracts the package -> tier-level map from the architest tier
// table text (the `"name": N,` rows new-leaf itself maintains behind TierMarker).
// It is the read half of the table Apply writes into, so a suggestion is always
// computed against the same source of truth the gate enforces.
func ParseTierTable(archText string) map[string]int {
	out := map[string]int{}
	for _, m := range tierRowRE.FindAllStringSubmatch(archText, -1) {
		if n, err := strconv.Atoi(m[2]); err == nil {
			out[m[1]] = n
		}
	}
	return out
}

// ScanInternalDeps returns the sorted, unique internal leaf names imported by the
// non-test Go files in dir: an import of ".../internal/foo" or ".../internal/foo/bar"
// both yield "foo". It is the import scan the tier suggestion reasons over.
func ScanInternalDeps(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	seen := map[string]struct{}{}
	fset := token.NewFileSet()
	for _, e := range entries {
		nm := e.Name()
		if e.IsDir() || !strings.HasSuffix(nm, ".go") || strings.HasSuffix(nm, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(dir, nm), nil, parser.ImportsOnly)
		if err != nil {
			return nil, err
		}
		for _, imp := range f.Imports {
			p, err := strconv.Unquote(imp.Path.Value)
			if err != nil {
				continue
			}
			rest, ok := strings.CutPrefix(p, ModulePrefix+"/")
			if !ok {
				continue
			}
			leaf := rest
			if i := strings.IndexByte(leaf, '/'); i >= 0 {
				leaf = leaf[:i]
			}
			if leaf != "" {
				seen[leaf] = struct{}{}
			}
		}
	}
	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out, nil
}

// MinTier returns the minimum legal tier for a leaf with the given internal dependencies.
// No dependencies (or abi alone) is primitive. Any other internal dependency makes the leaf
// at least foundation-composite; a still-higher dependency raises that floor.
func MinTier(deps []string, tierOf map[string]int) (level int, governing string) {
	level = Tiers["primitive"]
	for _, d := range deps {
		if d != "abi" && level < Tiers["foundation-composite"] {
			level, governing = Tiers["foundation-composite"], d
		}
		if t, ok := tierOf[d]; ok && t > level {
			level, governing = t, d
		}
	}
	return level, governing
}

// TierAdvisory compares the author-declared tier against the minimum legal tier
// derived from the deps and returns a one-line advisory when they differ, or "" when
// the declared tier is already minimum-correct. Advisory only — never a block.
func TierAdvisory(declared string, deps []string, tierOf map[string]int) string {
	dl, ok := Tiers[declared]
	if !ok {
		return ""
	}
	level, gov := MinTier(deps, tierOf)
	name := TierNameForLevel(level)
	switch {
	case dl > level:
		if gov == "" {
			return fmt.Sprintf("tier advisory: imports only tier-≤%d packages → %q suffices (you declared %q)", level, name, declared)
		}
		return fmt.Sprintf("tier advisory: highest-tier import %q is tier-%d → %q suffices (you declared %q)", gov, level, name, declared)
	case dl < level:
		return fmt.Sprintf("tier advisory: imports %q (tier-%d) → tier must be ≥ %q (you declared %q); the architest gate rejects an upward import", gov, level, name, declared)
	}
	return ""
}

// Suggestion is the read-only minimum-correct-tier verdict for an existing leaf.
type Suggestion struct {
	Leaf           string   `json:"leaf"`
	DeclaredTier   string   `json:"declared_tier,omitempty"`
	SuggestedTier  string   `json:"suggested_tier"`
	SuggestedLevel int      `json:"suggested_level"`
	GoverningDep   string   `json:"governing_dep,omitempty"`
	InternalDeps   []string `json:"internal_deps"`
	Advisory       string   `json:"advisory,omitempty"`
}

// Suggest computes the minimum-correct tier for an existing internal leaf from its
// imports, read-only. A non-empty declared tier adds a comparison advisory. root
// defaults to the current working directory.
func Suggest(root, leaf, declared string) (Suggestion, error) {
	if !NameRE.MatchString(leaf) {
		return Suggestion{}, fmt.Errorf("%q is not a valid lowercase Go package name", leaf)
	}
	if declared != "" {
		if _, ok := Tiers[declared]; !ok {
			return Suggestion{}, fmt.Errorf("unknown tier %q", declared)
		}
	}
	if root == "" {
		var err error
		root, err = os.Getwd()
		if err != nil {
			return Suggestion{}, err
		}
	}
	archText, err := os.ReadFile(filepath.Join(root, "internal", "architest", "architest_test.go"))
	if err != nil {
		return Suggestion{}, err
	}
	tierOf := ParseTierTable(string(archText))
	deps, err := ScanInternalDeps(filepath.Join(root, "internal", leaf))
	if err != nil {
		return Suggestion{}, err
	}
	level, gov := MinTier(deps, tierOf)
	if deps == nil {
		deps = []string{}
	}
	s := Suggestion{
		Leaf:           leaf,
		DeclaredTier:   declared,
		SuggestedTier:  TierNameForLevel(level),
		SuggestedLevel: level,
		GoverningDep:   gov,
		InternalDeps:   deps,
	}
	if declared != "" {
		s.Advisory = TierAdvisory(declared, deps, tierOf)
	}
	return s, nil
}

// JSON renders the suggestion as indented JSON.
func (s Suggestion) JSON() ([]byte, error) {
	return json.MarshalIndent(s, "", "  ")
}

func splitKeepLines(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.SplitAfter(s, "\n")
	if parts[len(parts)-1] == "" {
		parts = parts[:len(parts)-1]
	}
	return parts
}
