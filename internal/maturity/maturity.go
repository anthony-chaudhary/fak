// Package maturity scores where each fak capability sits on its LIFECYCLE
// maturity ladder — and, crucially, what the next step to advance it is.
//
// The sibling scorecards ask "is the fleet's dev discipline healthy"
// (internal/conceptusage), "can an agent adopt fak" (agent-readiness), or "is a
// concept a durable product" (tools/product_scorecard.py). None of them asks the
// question an operator running a long-horizon program asks of every feature:
//
//	A v1 prototype can be legitimately COMPLETE — but is it tested? does fak
//	itself run it? is it benchmarked? is it the default, or still an opt-in?
//	Where is each capability in its lifecycle, and what is the NEXT thing that
//	would mature it?
//
// That is this scorecard. It places every declared capability (one per
// internal/<leaf> lane in dos.toml [lanes.trees]) on a closed lifecycle ladder,
// best last:
//
//	proposed -> prototyped -> tested -> dogfooded -> benchmarked -> default
//
// The load-bearing property — the same invariant the rest of the kernel carries
// — is that NO RUNG CAN BE REACHED BY EDITING THE CLAIM. Each rung is gated by
// evidence the capability's author did not write: code on disk, a *_test.go, an
// import from cmd/ (fak itself runs it), a Benchmark func / an authority row, a
// documented verb. To move a capability up the ladder you change the real tree,
// not a data file.
//
// Two structural ideas make this an "agentic culture" subsystem, not just a
// report:
//
//   - Immaturity is NOT a defect. A capability honestly sitting at `prototyped`
//     is a complete v1 that simply has not been matured yet — that is the normal,
//     expected state, and the operator should SEE it without it counting against
//     anyone. What IS a defect is a LADDER-SKIP: a capability that has high-rung
//     evidence (fak runs it; it is benchmarked) while a LOWER rung is unmet (it
//     has no tests). Appearing more mature than the evidence supports is the
//     overclaim this refuses — the maturity sibling of the product scorecard's
//     verdict-overclaim and the readiness ladder's READINESS_OVERCLAIM (#582/G1).
//
//   - Every gap is the next work item. For each capability the FIRST unmet rung
//     is rendered as a concrete, checkable next step ("wire it into a fak verb so
//     fak itself runs it"). `fak maturity next` is that backlog — the queue an
//     agent (or the issue-dispatch loop) pulls from to advance the fleet one rung
//     at a time. The desire to create the next work item is mechanized: the tree
//     itself says what is owed.
//
// Every number is re-derived from disk (dos.toml + the tree + a few top-level
// docs). The score cannot be moved by editing a JSON file — only by actually
// maturing a capability.
package maturity

import (
	"bufio"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	Schema = "fak-maturity-scorecard/1"
)

// Rung is a closed lifecycle level, total-ordered worst -> best. A value outside
// this set is a bug, not a lower score — the same discipline the closed refusal
// vocabulary applies to a reason token.
type Rung int

const (
	RungProposed   Rung = iota // 0 — a named capability with no code on disk yet
	RungPrototyped             // 1 — v1 code exists in the leaf (legitimately "complete")
	RungTested                 // 2 — the leaf carries unit tests (the QA rung)
	RungDogfooded              // 3 — fak ITSELF runs it: it is on the running binary's import path
	RungDefault                // 4 — it is a documented default surface: a fak verb in cli-reference
)

// MaxRung is the top of the monotonic ladder.
const MaxRung = RungDefault

// RungName is the closed display vocabulary, indexed by Rung. `benchmarked` is NOT
// a ladder rung — measurement is an orthogonal badge (a capability can be measured
// at any rung), so forcing it into the total order would manufacture false
// inversions. It is tracked per capability and is the natural step AFTER `default`.
var RungName = []string{"proposed", "prototyped", "tested", "dogfooded", "default"}

func (r Rung) String() string {
	if int(r) >= 0 && int(r) < len(RungName) {
		return RungName[r]
	}
	return "?"
}

// Capability is one declared leaf and the lifecycle facts re-derived for it. All
// the boolean facts are read from ground truth the author of the leaf did not write.
type Capability struct {
	Lane string `json:"lane"` // the dos.toml lane key (== leaf package name)
	Dir  string `json:"dir"`  // the leaf tree root, e.g. internal/adjudicator

	HasCode        bool `json:"has_code"`        // a non-test .go file exists
	HasTests       bool `json:"has_tests"`       // a *_test.go exists
	Dogfooded      bool `json:"dogfooded"`       // imported by a cmd/ package — fak runs it
	Benchmarked    bool `json:"benchmarked"`     // a Benchmark func or a BENCHMARK-AUTHORITY row
	DefaultSurface bool `json:"default_surface"` // a documented verb / named in llms.txt

	// Rung is the monotonic current lifecycle rung: the highest R such that EVERY
	// promotion predicate up to and including R holds. A gap caps it.
	Rung Rung `json:"rung"`
	// TopEvidence is the highest rung whose predicate holds, ignoring gaps below
	// it. TopEvidence > Rung means the capability skipped a lower rung.
	TopEvidence Rung `json:"top_evidence"`
	// Skip is true when high-rung evidence sits over an unmet lower rung — a
	// maturity inversion / overclaim (e.g. fak runs it but it has no tests).
	Skip bool `json:"skip"`

	// Next is the first unmet rung rendered as a concrete next work item, or nil
	// when the capability is already at the top of the ladder.
	Next *NextWork `json:"next,omitempty"`
}

// NextWork is one mechanically-derived "next thing that would mature this
// capability" — the unit the agentic-culture backlog is made of. It mirrors the
// CLAUDE.md "not yet" idiom: the gap, the missing witness, and the checkable step.
type NextWork struct {
	Lane     string `json:"lane"`
	FromRung Rung   `json:"from_rung"` // where the capability is now
	Gap      Rung   `json:"gap"`       // the first rung it is missing
	Title    string `json:"title"`     // imperative, ticket-shaped
	Witness  string `json:"witness"`   // the evidence that would close it
	Skip     bool   `json:"skip"`      // is filling this gap also resolving a ladder-skip?
}

// predicate is a named lifecycle rung and the fact that gates it.
type predicate struct {
	rung Rung
	want func(Capability) bool
}

// ladder is the closed promotion sequence above `prototyped`. `proposed` and
// `prototyped` are decided by HasCode alone (no predicate); everything from
// `tested` up is gated by evidence. These four rungs ARE naturally ordered: a
// documented default surface presupposes fak runs it, which presupposes it is
// tested, which presupposes it has code. (Measurement is a separate badge.)
var ladder = []predicate{
	{RungTested, func(c Capability) bool { return c.HasTests }},
	{RungDogfooded, func(c Capability) bool { return c.Dogfooded }},
	{RungDefault, func(c Capability) bool { return c.DefaultSurface }},
}

// adjudicate computes the monotonic current rung, the top-evidence rung, the
// skip flag, and the next work item — purely from the capability's facts. This
// is the testable core: facts in, lifecycle verdict out, no I/O.
func adjudicate(c Capability) Capability {
	if !c.HasCode {
		c.Rung, c.TopEvidence = RungProposed, RungProposed
		c.Next = &NextWork{
			Lane: c.Lane, FromRung: RungProposed, Gap: RungPrototyped,
			Title:   "prototype " + c.Lane + ": land a v1 in " + c.Dir,
			Witness: "a non-test .go file exists under " + c.Dir,
		}
		return c
	}
	// Code exists: at least prototyped. Walk the gated ladder.
	c.Rung = RungPrototyped
	c.TopEvidence = RungPrototyped
	monotonic := true
	gap := Rung(-1)
	for i := range ladder {
		p := ladder[i]
		held := p.want(c)
		if held && p.rung > c.TopEvidence {
			c.TopEvidence = p.rung
		}
		if monotonic && held {
			c.Rung = p.rung
		} else if monotonic && !held {
			monotonic = false
			gap = p.rung
		}
	}
	// A ladder-skip is the ONE honest inversion: fak relies on this capability (it
	// is dogfooded, a default surface, or measured) yet it has NO tests. Appearing
	// more depended-upon than it is QA'd is the maturity overclaim this refuses.
	// Mere immaturity (a complete v1 sitting at `prototyped`) is never a skip.
	c.Skip = !c.HasTests && (c.Dogfooded || c.DefaultSurface || c.Benchmarked)

	switch {
	case gap >= 0:
		c.Next = nextWorkFor(c, gap)
	case !c.Benchmarked:
		// At the top of the ladder (a documented default surface) but never measured
		// — the natural next step is to prove it with a number.
		c.Next = nextWorkFor(c, rungBenchmark)
	default:
		c.Next = nil // fully matured: default surface AND measured
	}
	return c
}

// rungBenchmark is a sentinel above the ladder used only to render the
// "benchmark it" next-work for a capability already at the top rung. It is NOT a
// monotonic rung (measurement is an orthogonal badge), so it never caps a rung.
const rungBenchmark Rung = MaxRung + 1

// nextWorkFor renders the first unmet rung (or the benchmark badge) as a checkable ticket.
func nextWorkFor(c Capability, gap Rung) *NextWork {
	nw := &NextWork{Lane: c.Lane, FromRung: c.Rung, Gap: gap, Skip: c.Skip}
	switch gap {
	case RungTested:
		nw.Title = "test " + c.Lane + ": add unit tests covering " + c.Dir
		nw.Witness = "a *_test.go in " + c.Dir + " (go test ./" + c.Dir + "/... passes)"
	case RungDogfooded:
		nw.Title = "dogfood " + c.Lane + ": wire it onto the running binary's path so fak itself runs it"
		nw.Witness = importPath(c.Lane) + " imported by cmd/, internal/registrations, or internal/kernel"
	case RungDefault:
		nw.Title = "default " + c.Lane + ": promote it to a documented default surface (a fak verb)"
		nw.Witness = c.Lane + " documented in docs/cli-reference.md"
	case rungBenchmark:
		nw.Title = "benchmark " + c.Lane + ": prove the default surface with a measured number"
		nw.Witness = "a func Benchmark* in " + c.Dir + " or a BENCHMARK-AUTHORITY.md row naming " + c.Lane
	}
	if c.Skip {
		nw.Title += " (LADDER-SKIP: fak relies on " + c.Lane + " but it has no tests)"
	}
	return nw
}

func importPath(lane string) string {
	return "github.com/anthony-chaudhary/fak/internal/" + lane
}

// ---- Options + payload (mirror internal/conceptusage) -----------------------

// Options pins the root and the tree facts so the score is deterministic for
// tests. The facts seam lets a test inject a synthetic tree without touching disk.
type Options struct {
	Root string
	// facts overrides the disk read for tests; nil means re-derive from Root.
	facts func(root string) []Capability
}

func (o Options) normalize() Options {
	if o.Root == "" {
		o.Root = "."
	}
	return o
}

// ScorecardPayload is the uniform control-pane envelope every fak scorecard emits.
type ScorecardPayload struct {
	Schema     string         `json:"schema"`
	OK         bool           `json:"ok"`
	Verdict    string         `json:"verdict"`
	Finding    string         `json:"finding"`
	Reason     string         `json:"reason"`
	NextAction string         `json:"next_action"`
	Workspace  string         `json:"workspace"`
	Corpus     map[string]any `json:"corpus"`
	Caps       []Capability   `json:"capabilities"`
	Backlog    []NextWork     `json:"backlog"`
}

// Build is the fold: re-derive every capability's facts, adjudicate each, and
// roll up the distribution + the ladder-skip debt + the next-work backlog.
func Build(opts Options) ScorecardPayload {
	opts = opts.normalize()
	root, _ := filepath.Abs(opts.Root)
	if root == "" {
		root = opts.Root
	}
	factsFn := opts.facts
	if factsFn == nil {
		factsFn = gatherFacts
	}
	caps := factsFn(root)
	for i := range caps {
		caps[i] = adjudicate(caps[i])
	}
	sort.SliceStable(caps, func(i, j int) bool { return caps[i].Lane < caps[j].Lane })

	dist := map[string]int{}
	for _, r := range RungName {
		dist[r] = 0
	}
	skips, benchmarked := 0, 0
	var rungSum int
	var backlog []NextWork
	for _, c := range caps {
		dist[c.Rung.String()]++
		rungSum += int(c.Rung)
		if c.Skip {
			skips++
		}
		if c.Benchmarked {
			benchmarked++
		}
		if c.Next != nil {
			backlog = append(backlog, *c.Next)
		}
	}
	n := len(caps)
	// The maturity index is the average current rung as a fraction of the top of
	// the ladder — a 0-100 fleet-maturity score that grows only as capabilities
	// genuinely advance. A ladder-skip docks the index (the inversion is real debt).
	score := 0
	if n > 0 {
		raw := 100.0 * float64(rungSum) / (float64(n) * float64(MaxRung))
		raw -= 100.0 * float64(skips) / float64(n) // each skip costs one capability's worth
		if raw < 0 {
			raw = 0
		}
		score = int(raw + 0.5)
	}
	grade := GradeLetter(score)

	// Rank the backlog: ladder-skips first (real overclaim debt), then lowest
	// current rung first (the least-mature capability is the most leverage), then
	// by lane for determinism.
	sort.SliceStable(backlog, func(i, j int) bool {
		if backlog[i].Skip != backlog[j].Skip {
			return backlog[i].Skip
		}
		if backlog[i].FromRung != backlog[j].FromRung {
			return backlog[i].FromRung < backlog[j].FromRung
		}
		return backlog[i].Lane < backlog[j].Lane
	})

	// maturity-debt = ladder-skips only. Immaturity itself is never debt — that is
	// the whole point. The CI-relevant signal is "no capability appears more mature
	// than its evidence supports."
	debt := skips
	ok := debt == 0

	verdict, finding, reason, next := "OK", "ladder_honest", "", ""
	atDefault := dist["default"]
	belowTested := dist["proposed"] + dist["prototyped"]
	distLine := distString(dist)
	if ok {
		reason = "maturity: " + itoa(n) + " capabilities, index " + itoa(score) + "/100 (" + grade +
			"); no ladder-skips (every capability is at most as mature as its evidence); " + distLine
		if len(backlog) > 0 {
			next = "advance the fleet one rung: `fak maturity next` lists " + itoa(len(backlog)) +
				" next work item(s); the least-mature capability is the most leverage"
		} else {
			next = "every capability is at the top of the ladder; hold the line"
		}
	} else {
		verdict, finding = "ACTION", "ladder_skip"
		reason = "maturity carries " + itoa(debt) + " ladder-skip(s) (capabilities that look more mature " +
			"than their evidence — e.g. fak runs them but they have no tests); index " + itoa(score) +
			"/100 (" + grade + "); " + distLine
		next = "retire the inversions worst-first: `fak maturity next` lists the skipped lower rung for each; " +
			"fill the missing lower rung (usually: add tests) before claiming the higher one"
	}

	return ScorecardPayload{
		Schema:     Schema,
		OK:         ok,
		Verdict:    verdict,
		Finding:    finding,
		Reason:     reason,
		NextAction: next,
		Workspace:  root,
		Corpus: map[string]any{
			"maturity_debt": debt,
			"score":         score,
			"grade":         grade,
			"capabilities":  n,
			"ladder_skips":  skips,
			"at_default":    atDefault,
			"below_tested":  belowTested,
			"benchmarked":   benchmarked,
			"backlog":       len(backlog),
			"distribution":  dist,
			"ladder":        RungName,
		},
		Caps:    caps,
		Backlog: backlog,
	}
}

func distString(dist map[string]int) string {
	parts := make([]string, 0, len(RungName))
	for _, r := range RungName {
		parts = append(parts, itoa(dist[r])+" "+r)
	}
	return strings.Join(parts, " · ")
}

// ---- evidence gathering (the impure shell, kept thin) -----------------------

var (
	laneTreeRe = regexp.MustCompile(`^([A-Za-z0-9_]+)\s*=\s*\[\s*"internal/([A-Za-z0-9_]+)/\*\*"`)
	importRe   = regexp.MustCompile(`github\.com/anthony-chaudhary/fak/internal/([A-Za-z0-9_]+)`)
	benchRe    = regexp.MustCompile(`(?m)^func Benchmark`)
)

// gatherFacts re-derives every declared capability's lifecycle facts from disk:
// the dos.toml lane roster, the leaf tree, the running-binary import set, and a
// few top-level docs. Read-only; a missing source degrades a fact to false (a
// conservative "not yet"), never a false pass.
func gatherFacts(root string) []Capability {
	lanes := parseLaneTrees(filepath.Join(root, "dos.toml"))
	// `dogfooded` = fak ITSELF runs it: the leaf is on the running binary's TRANSITIVE
	// import graph, seeded from the cmd/ packages and the registrations blank-imports
	// and closed over internal→internal edges. A leaf reachable from the binary is one
	// fak runs (even deep behind the kernel, like the canonicalizer behind a screen);
	// a leaf reachable from nothing but its own tests is genuinely not dogfooded yet.
	runImports := scanReachable(root)
	benchDoc := lowerFileWords(filepath.Join(root, "BENCHMARK-AUTHORITY.md"))
	// `default surface` = a documented fak verb. cli-reference.md is the verb
	// catalog; being named there (and being dogfooded, enforced by the ladder) is
	// the "promoted to the standard surface" signal. llms.txt is the doc MAP — it
	// names internal concepts that are not verbs, so it is deliberately NOT used here.
	surfaceDoc := lowerFileWords(filepath.Join(root, "docs", "cli-reference.md"))

	caps := make([]Capability, 0, len(lanes))
	for _, lane := range lanes {
		dir := "internal/" + lane
		abs := filepath.Join(root, "internal", lane)
		hasCode, hasTests, hasBench := scanLeaf(abs)
		_, dogfooded := runImports[lane]
		_, namedInBench := benchDoc[strings.ToLower(lane)]
		_, namedInSurface := surfaceDoc[strings.ToLower(lane)]
		caps = append(caps, Capability{
			Lane:           lane,
			Dir:            dir,
			HasCode:        hasCode,
			HasTests:       hasTests,
			Dogfooded:      dogfooded,
			Benchmarked:    hasBench || namedInBench,
			DefaultSurface: namedInSurface,
		})
	}
	return caps
}

// parseLaneTrees returns the leaf lanes declared in dos.toml [lanes.trees] whose
// tree is an internal/<leaf>/** glob — the first-class capability roster. Lanes
// pointing outside internal/ (cmd, docs, global, …) are area lanes, not leaf
// capabilities, and are skipped.
func parseLaneTrees(path string) []string {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	var lanes []string
	seen := map[string]struct{}{}
	inTrees := false
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1024*1024), 8*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if strings.HasPrefix(line, "[") {
			inTrees = line == "[lanes.trees]"
			continue
		}
		if !inTrees {
			continue
		}
		m := laneTreeRe.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		leaf := m[2]
		if _, ok := seen[leaf]; ok {
			continue
		}
		seen[leaf] = struct{}{}
		lanes = append(lanes, leaf)
	}
	return lanes
}

// scanLeaf reports whether the leaf dir has non-test code, test code, and a
// Benchmark func. One shallow read of the directory (the leaf's own files).
func scanLeaf(dir string) (hasCode, hasTests, hasBench bool) {
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		name := d.Name()
		if !strings.HasSuffix(name, ".go") {
			return nil
		}
		if strings.HasSuffix(name, "_test.go") {
			hasTests = true
			if !hasBench {
				if b, e := os.ReadFile(p); e == nil && benchRe.Match(b) {
					hasBench = true
				}
			}
			return nil
		}
		hasCode = true
		return nil
	})
	_ = err
	return hasCode, hasTests, hasBench
}

// scanReachable returns the set of internal leaves on the running binary's
// TRANSITIVE import graph. It seeds from the non-test imports of every cmd/
// package and internal/registrations (the ABI wiring the binary loads), then
// closes the seed set over internal→internal edges (non-test imports only — a
// test importing a leaf does not mean the binary runs it). A leaf in the result
// is one fak itself runs; one outside it self-tests but is not yet dogfooded.
func scanReachable(root string) map[string]struct{} {
	graph := internalImportGraph(filepath.Join(root, "internal"))
	seeds := importsUnder(filepath.Join(root, "cmd"))
	for leaf := range importsUnder(filepath.Join(root, "internal", "registrations")) {
		seeds[leaf] = struct{}{}
	}
	reach := map[string]struct{}{}
	var queue []string
	for s := range seeds {
		if _, ok := reach[s]; !ok {
			reach[s] = struct{}{}
			queue = append(queue, s)
		}
	}
	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		for dep := range graph[n] {
			if _, ok := reach[dep]; !ok {
				reach[dep] = struct{}{}
				queue = append(queue, dep)
			}
		}
	}
	return reach
}

// internalImportGraph maps each internal leaf to the set of internal leaves it
// imports (non-test files only).
func internalImportGraph(internalRoot string) map[string]map[string]struct{} {
	graph := map[string]map[string]struct{}{}
	entries, err := os.ReadDir(internalRoot)
	if err != nil {
		return graph
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		leaf := e.Name()
		deps := map[string]struct{}{}
		_ = filepath.WalkDir(filepath.Join(internalRoot, leaf), func(p string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			name := d.Name()
			if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
				return nil
			}
			if b, e := os.ReadFile(p); e == nil {
				for _, m := range importRe.FindAllStringSubmatch(string(b), -1) {
					if m[1] != leaf {
						deps[m[1]] = struct{}{}
					}
				}
			}
			return nil
		})
		graph[leaf] = deps
	}
	return graph
}

// importsUnder returns the set of internal leaves imported by any non-test .go
// file under root.
func importsUnder(root string) map[string]struct{} {
	out := map[string]struct{}{}
	_ = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		name := d.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			return nil
		}
		if b, e := os.ReadFile(p); e == nil {
			for _, m := range importRe.FindAllStringSubmatch(string(b), -1) {
				out[m[1]] = struct{}{}
			}
		}
		return nil
	})
	return out
}

// lowerFileWords returns the set of lowercase identifier-ish tokens in a file, so
// a capability named in a doc can be matched as a whole word. Missing file -> empty.
func lowerFileWords(path string) map[string]struct{} {
	out := map[string]struct{}{}
	b, err := os.ReadFile(path)
	if err != nil {
		return out
	}
	for _, tok := range wordRe.FindAllString(strings.ToLower(string(b)), -1) {
		out[tok] = struct{}{}
	}
	return out
}

var wordRe = regexp.MustCompile(`[a-z0-9_]+`)

// ---- grade + small helpers (mirror internal/conceptusage) -------------------

func GradeLetter(score int) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 80:
		return "B"
	case score >= 70:
		return "C"
	case score >= 60:
		return "D"
	default:
		return "F"
	}
}

func itoa(n int) string {
	// small, allocation-free for the common range; falls back for the rest.
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
