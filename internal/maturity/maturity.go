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
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/walkfiles"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
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
	RungDefault                // 4 — a passing runtime proof declares the capability active without opt-in
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

	HasCode        bool   `json:"has_code"`   // a non-test .go file exists
	HasTests       bool   `json:"has_tests"`  // a *_test.go exists
	Integrated     bool   `json:"integrated"` // reachable from a production command import graph
	Dogfooded      bool   `json:"dogfooded"`  // a declared runtime witness passes
	RuntimeProof   string `json:"runtime_proof,omitempty"`
	Benchmarked    bool   `json:"benchmarked"`     // a Benchmark func or a BENCHMARK-AUTHORITY row
	DefaultSurface bool   `json:"default_surface"` // a documented verb / named in llms.txt

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
// witnessed default-on behavior presupposes fak runs it, which presupposes it is
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
	// A ladder-skip is any higher-rung claim sitting above a missing prerequisite.
	// This catches both an exercised/default capability without tests and a documented
	// default with no runtime proof. Mere immaturity has no higher claim and is not a skip.
	c.Skip = c.TopEvidence > c.Rung

	switch {
	case gap >= 0:
		c.Next = nextWorkFor(c, gap)
	case !c.Benchmarked:
		// At the top of the ladder (witnessed default-on behavior) but never measured
		// — the natural next step is to prove it with a number.
		c.Next = nextWorkFor(c, rungBenchmark)
	default:
		c.Next = nil // fully matured: witnessed default-on behavior AND measured
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
		nw.Title = "dogfood " + c.Lane + ": exercise a real runtime path and record its passing command"
		nw.Witness = "a passing runtime command recorded for " + c.Lane + " in internal/maturity/runtime-proofs.json"
	case RungDefault:
		nw.Title = "default " + c.Lane + ": prove it runs without an opt-in action"
		nw.Witness = "a passing runtime proof for " + c.Lane + " with default_on=true and a concrete default_reason"
	case rungBenchmark:
		nw.Title = "benchmark " + c.Lane + ": prove the default surface with a measured number"
		nw.Witness = "a func Benchmark* in " + c.Dir + " or a BENCHMARK-AUTHORITY.md row naming " + c.Lane
	}
	if c.Skip {
		nw.Title += " (LADDER-SKIP: higher-rung evidence exists above this missing prerequisite)"
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
	// Witnesses overrides runtime witness loading for deterministic callers and tests.
	Witnesses func(root string) (map[string]RuntimeProof, error)
}

func (o Options) normalize() Options {
	if o.Root == "" {
		o.Root = "."
	}
	return o
}

// ScorecardPayload is the uniform control-pane envelope every fak scorecard emits.
type ScorecardPayload struct {
	Schema            string         `json:"schema"`
	OK                bool           `json:"ok"`
	Verdict           string         `json:"verdict"`
	Finding           string         `json:"finding"`
	Reason            string         `json:"reason"`
	NextAction        string         `json:"next_action"`
	Workspace         string         `json:"workspace"`
	Corpus            map[string]any `json:"corpus"`
	Caps              []Capability   `json:"capabilities"`
	Backlog           []NextWork     `json:"backlog"`
	RuntimeProofOK    bool           `json:"runtime_proof_ok"`
	RuntimeProofCount int            `json:"runtime_proof_count"`
	RuntimeProofError string         `json:"runtime_proof_error,omitempty"`
}

// Build is the fold: re-derive every capability's facts, adjudicate each, and
// roll up the distribution + the ladder-skip debt + the next-work backlog.
func Build(opts Options) ScorecardPayload {
	opts = opts.normalize()
	root := scorecard.WorkspaceRoot(opts.Root)
	factsFn := opts.facts
	if factsFn == nil {
		factsFn = gatherFacts
	}
	caps := factsFn(root)
	witnessFn := opts.Witnesses
	if witnessFn == nil {
		witnessFn = verifyRuntimeProofs
	}
	witnesses, proofLoadErr := witnessFn(root)
	for i := range caps {
		caps[i].Dogfooded = false
		if proofLoadErr == nil {
			if witness, ok := witnesses[caps[i].Lane]; ok {
				caps[i].Dogfooded = true
				caps[i].DefaultSurface = witness.DefaultOn
				caps[i].RuntimeProof = witness.Command
			}
		}
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
	grade := scorecard.GradeStd(float64(score))

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
	runtimeProofOK := proofLoadErr == nil
	ok := debt == 0 && runtimeProofOK

	verdict, finding, reason, next := "OK", "ladder_honest", "", ""
	atDefault := dist["default"]
	belowTested := dist["proposed"] + dist["prototyped"]
	switch {
	case proofLoadErr != nil:
		verdict = "FAIL"
		finding = "runtime_proof_unverified"
		reason = "maturity runtime proofs could not be verified: " + proofLoadErr.Error()
		next = "repair or replace the declared runtime proof artifact, then rerun `fak maturity`"
	case debt > 0:
		verdict = "RISKY"
		finding = "ladder_skip"
		reason = fmt.Sprintf(
			"maturity: %d capabilities, index %d/100 (%s); %d ladder-skip(s) where the tree relies on a capability above its completed rung; %s",
			len(caps), score, grade, debt, distString(dist),
		)
		next = fmt.Sprintf("retire the first ladder-skip: %s", backlog[0].Title)
	default:
		reason = fmt.Sprintf(
			"maturity: %d capabilities, index %d/100 (%s); no ladder-skips (every capability is at most as mature as its evidence); %s",
			len(caps), score, grade, distString(dist),
		)
		if len(backlog) > 0 {
			next = fmt.Sprintf("advance the fleet one rung: `fak maturity next` lists %d next work item(s); the least-mature capability is the most leverage", len(backlog))
		} else {
			next = "all declared capabilities are measured defaults; keep the scorecard green"
		}
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
		Caps:              caps,
		Backlog:           backlog,
		RuntimeProofOK:    runtimeProofOK,
		RuntimeProofCount: len(witnesses),
		RuntimeProofError: errorString(proofLoadErr),
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
func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func gatherFacts(root string) []Capability {
	lanes := parseLaneTrees(filepath.Join(root, "dos.toml"))
	// `integrated` = the leaf is on the running binary's TRANSITIVE
	// import graph, seeded from the cmd/ packages and the registrations blank-imports
	// and closed over internal→internal edges. A leaf reachable from the binary is one
	// fak runs (even deep behind the kernel, like the canonicalizer behind a screen);
	// a leaf reachable from nothing but its own tests is not integrated.
	runImports := scanReachable(root)
	benchDoc := lowerFileWords(filepath.Join(root, "BENCHMARK-AUTHORITY.md"))
	caps := make([]Capability, 0, len(lanes))
	for _, lane := range lanes {
		dir := "internal/" + lane
		abs := filepath.Join(root, "internal", lane)
		hasCode, hasTests, hasBench := scanLeaf(abs)
		_, integrated := runImports[lane]
		_, namedInBench := benchDoc[strings.ToLower(lane)]
		caps = append(caps, Capability{
			Lane:        lane,
			Dir:         dir,
			HasCode:     hasCode,
			HasTests:    hasTests,
			Integrated:  integrated,
			Benchmarked: hasBench || namedInBench,
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
	_ = walkfiles.Files(dir, func(p string, d os.DirEntry) error {
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
	return hasCode, hasTests, hasBench
}

// scanReachable returns the set of internal leaves on the running binary's
// TRANSITIVE import graph. It seeds from the non-test imports of every cmd/
// package and internal/registrations (the ABI wiring the binary loads), then
// closes the seed set over internal→internal edges (non-test imports only — a
// test importing a leaf does not mean the binary runs it). A leaf in the result
// is one fak itself runs; one outside it self-tests but is not yet dogfooded.
func scanReachable(root string) map[string]struct{} {
	return scanReachableWithGraph(root, internalImportGraph(filepath.Join(root, "internal")))
}

func scanReachableWithGraph(root string, graph map[string]map[string]struct{}) map[string]struct{} {
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
// imports (non-test files only). A leaf's dep set is exactly the imports under
// its own directory minus the self-edge, so it delegates to importsUnder rather
// than carrying a second copy of that walk.
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
		deps := importsUnder(filepath.Join(internalRoot, leaf))
		delete(deps, leaf)
		graph[leaf] = deps
	}
	return graph
}

// importsUnder returns the set of internal leaves imported by any non-test .go
// file under root.
func importsUnder(root string) map[string]struct{} {
	out := map[string]struct{}{}
	_ = walkfiles.Files(root, func(p string, d os.DirEntry) error {
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
