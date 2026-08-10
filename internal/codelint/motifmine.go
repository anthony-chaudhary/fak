package codelint

// motifmine is the structural workload-motif miner (#6211, epic #6208): it finds
// reusable coding-workload motifs that this repository already PRACTISES but has
// never NAMED — inspect→edit→verify, lease→isolate→land, reproduce→fix→witness,
// plan→fanout→reconcile — and maps each hit to a stable catalog ID.
//
// The one rule that makes the output worth reading: a motif is claimed ONLY from
// structural evidence — call edges to identified APIs, defer/go statements, and
// identifier data-flow inside a single function body. A declaration's NAME and its
// COMMENTS never contribute. That is enforced mechanically, not by convention: the
// parser runs WITHOUT parser.ParseComments, so comment text is not even in the AST
// the detectors see, and no detector reads FuncDecl.Name. A function called
// `inspectEditVerify` whose doc comment spells the motif out is therefore invisible
// to the miner unless its body really does read→mutate→re-read the same target.
// Names appear in a finding only as the Symbol label an operator navigates by.
//
// Every detector demands a LINKAGE that naming coincidence cannot fake: the same
// path argument across the read/write/read triple, the same receiver identifier
// across the acquire/defer-release pair, a data dependency from the seeded value
// into the subject call and on into the assertion, the same slice identifier
// planned and then ranged over. When a linkage is missing the miner ABSTAINS —
// low recall is the intended trade; a false motif poisons a vocabulary.
//
// Scope: this is a discovery aid over Go source. A finding is evidence that the
// motif is PRESENT at that range; the absence of a finding is never evidence that
// a motif is absent (an unmodelled spelling simply is not detected). Sibling
// issues own the authoritative catalog (#6210), trajectory mining (#6212), and the
// report CLI (#6213); this leaf owns the source-evidence detectors only.

import (
	"bytes"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// MotifMinerVersion versions the EVIDENCE RULES, not the code. Bump it whenever a
// detector's admission rule changes, so an archived finding stays interpretable
// against the rule that produced it.
const MotifMinerVersion = "codelint.motifmine/1"

// Catalog IDs are the stable join key between a mined finding and the versioned
// pattern catalog (#6210). They are deliberately spelled out here rather than
// derived from a detector ID: a detector may be rewritten, renamed, or split, and
// the catalog identity must survive that.
const (
	MotifCatalogInspectEditVerify   = "sp.inspect-edit-verify@1"
	MotifCatalogLeaseIsolateLand    = "sp.lease-isolate-land@1"
	MotifCatalogReproduceFixWitness = "sp.reproduce-fix-witness@1"
	MotifCatalogPlanFanoutReconcile = "sp.plan-fanout-reconcile@1"
)

// MotifRoleEvidence is one satisfied role of a motif, pinned to the source
// construct that satisfied it. Source is the printed expression — positive
// evidence an operator can read without opening the file, and the thing a reviewer
// checks the detector against.
type MotifRoleEvidence struct {
	Role   string `json:"role"`
	Line   int    `json:"line"`
	Source string `json:"source"`
}

// MotifFinding is one detected motif instance. It carries its own provenance
// (path + line range, detector, detector version, confidence, reason) so a finding
// copied out of the report is still checkable on its own.
type MotifFinding struct {
	CatalogID  string              `json:"catalog_id"`
	Motif      string              `json:"motif"`
	Detector   string              `json:"detector"`
	Version    string              `json:"detector_version"`
	Path       string              `json:"path"`
	Symbol     string              `json:"symbol"`
	StartLine  int                 `json:"start_line"`
	EndLine    int                 `json:"end_line"`
	Confidence float64             `json:"confidence"`
	Reason     string              `json:"reason"`
	Evidence   []MotifRoleEvidence `json:"evidence"`
}

// MotifSkip records one path the miner refused to read, with the policy reason —
// so "no findings here" is never confused with "never looked here".
type MotifSkip struct {
	Path   string `json:"path"`
	Reason string `json:"reason"`
}

// MotifDetector is a detector's self-description: what it claims, which catalog ID
// it maps to, and the exact structural rule it demands.
type MotifDetector struct {
	ID         string   `json:"id"`
	CatalogID  string   `json:"catalog_id"`
	Motif      string   `json:"motif"`
	Version    string   `json:"version"`
	Roles      []string `json:"roles"`
	Rule       string   `json:"rule"`
	Confidence float64  `json:"confidence"`
}

// MotifReport is a whole-tree mining run. Findings and Skipped are both sorted, so
// two runs over an unchanged tree marshal byte-for-byte identically.
type MotifReport struct {
	Version   string          `json:"version"`
	Scanned   int             `json:"files_scanned"`
	Detectors []MotifDetector `json:"detectors"`
	Findings  []MotifFinding  `json:"findings"`
	Skipped   []MotifSkip     `json:"skipped"`
}

// --- exclusion policy -------------------------------------------------------
//
// Explicit, path-shaped, and applied to directories (pruned whole) as well as
// files. Nothing here is inferred from content except the one generated-file
// marker Go itself standardizes, because a vendored or machine-emitted body is
// evidence about its generator, never about how this repo works.

var motifGeneratedMarker = regexp.MustCompile(`(?m)^// Code generated .* DO NOT EDIT\.$`)

// MotifExcludedPath applies the path policy to a slash-separated repo-relative
// path (a file or a directory) and returns the reason it is excluded, if it is.
func MotifExcludedPath(rel string) (string, bool) {
	rel = filepath.ToSlash(rel)
	segs := strings.Split(rel, "/")
	for _, s := range segs {
		if s == "" || s == "." {
			continue
		}
		low := strings.ToLower(s)
		switch {
		case low == "vendor" || low == "node_modules" || low == "third_party" || low == "thirdparty":
			return "vendor", true
		case strings.HasPrefix(s, "."):
			// .git, .scratch, .claude, .private — dot-prefixed trees are tooling
			// or per-session state, never repo source.
			return "hidden-or-private", true
		case low == "private" || low == "secret" || low == "secrets":
			return "private", true
		case low == "tmp" || low == "temp" || strings.Contains(low, "scratch"):
			return "scratch", true
		}
	}
	base := strings.ToLower(segs[len(segs)-1])
	if strings.HasSuffix(base, ".pb.go") || strings.HasSuffix(base, "_generated.go") ||
		strings.HasSuffix(base, ".gen.go") || strings.HasPrefix(base, "zz_generated") {
		return "generated", true
	}
	return "", false
}

// motifExcludedContent catches machine-emitted files whose path looks handwritten,
// via the marker line the Go toolchain standardizes. Only the header is scanned:
// the marker is a header convention, and a mention deeper in a file is prose.
func motifExcludedContent(src []byte) (string, bool) {
	head := src
	if len(head) > 4096 {
		head = head[:4096]
	}
	if motifGeneratedMarker.Match(head) {
		return "generated", true
	}
	return "", false
}

// --- structural facts -------------------------------------------------------

type motifCall struct {
	pos      token.Pos
	line     int
	recv     string
	sel      string
	qual     string
	args     []string
	src      string
	deferred bool
}

type motifFn struct {
	fset    *token.FileSet
	name    string
	start   int
	end     int
	tParam  string
	calls   []motifCall
	defers  []motifCall
	gos     []token.Pos
	ranges  []*ast.RangeStmt
	assigns []*ast.AssignStmt
	ifs     []*ast.IfStmt
	recvOps []token.Pos
}

func motifExprText(fset *token.FileSet, n ast.Node) string {
	var b bytes.Buffer
	if err := printer.Fprint(&b, fset, n); err != nil {
		return ""
	}
	return motifTrim(b.String())
}

// motifTrim collapses a printed expression onto one line and bounds its length, so
// a multi-line call renders as readable positive evidence in a JSON report.
func motifTrim(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > 120 {
		s = string(r[:117]) + "..."
	}
	return s
}

func motifMakeCall(fset *token.FileSet, c *ast.CallExpr) motifCall {
	mc := motifCall{pos: c.Pos(), line: fset.Position(c.Pos()).Line, src: motifExprText(fset, c)}
	switch f := c.Fun.(type) {
	case *ast.SelectorExpr:
		if id, ok := f.X.(*ast.Ident); ok {
			mc.recv = id.Name
		}
		mc.sel = f.Sel.Name
	case *ast.Ident:
		mc.sel = f.Name
	}
	mc.qual = mc.sel
	if mc.recv != "" {
		mc.qual = mc.recv + "." + mc.sel
	}
	for _, a := range c.Args {
		mc.args = append(mc.args, motifExprText(fset, a))
	}
	return mc
}

// motifTestingParam returns the name bound to a *testing.T parameter, by TYPE.
// A function is a test because of its signature, never because of its name — that
// asymmetry is what the reproduce→fix→witness negative fixture pins.
func motifTestingParam(fd *ast.FuncDecl) string {
	if fd.Type == nil || fd.Type.Params == nil {
		return ""
	}
	for _, f := range fd.Type.Params.List {
		star, ok := f.Type.(*ast.StarExpr)
		if !ok {
			continue
		}
		sel, ok := star.X.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "T" {
			continue
		}
		if pkg, ok := sel.X.(*ast.Ident); !ok || pkg.Name != "testing" {
			continue
		}
		for _, n := range f.Names {
			if n.Name != "_" {
				return n.Name
			}
		}
	}
	return ""
}

func motifCollect(fset *token.FileSet, fd *ast.FuncDecl) *motifFn {
	fn := &motifFn{
		fset:   fset,
		name:   fd.Name.Name,
		start:  fset.Position(fd.Pos()).Line,
		end:    fset.Position(fd.End()).Line,
		tParam: motifTestingParam(fd),
	}
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		switch v := n.(type) {
		case *ast.CallExpr:
			fn.calls = append(fn.calls, motifMakeCall(fset, v))
		case *ast.DeferStmt:
			fn.defers = append(fn.defers, motifMakeCall(fset, v.Call))
		case *ast.GoStmt:
			fn.gos = append(fn.gos, v.Pos())
		case *ast.RangeStmt:
			fn.ranges = append(fn.ranges, v)
		case *ast.AssignStmt:
			fn.assigns = append(fn.assigns, v)
		case *ast.IfStmt:
			fn.ifs = append(fn.ifs, v)
		case *ast.UnaryExpr:
			if v.Op == token.ARROW {
				fn.recvOps = append(fn.recvOps, v.Pos())
			}
		}
		return true
	})
	sort.Slice(fn.calls, func(i, j int) bool { return fn.calls[i].pos < fn.calls[j].pos })
	sort.Slice(fn.assigns, func(i, j int) bool { return fn.assigns[i].Pos() < fn.assigns[j].Pos() })
	sort.Slice(fn.ifs, func(i, j int) bool { return fn.ifs[i].Pos() < fn.ifs[j].Pos() })
	// A deferred call is also a CallExpr, so mark it: a deferred cleanup is the
	// release step of a lease, never the "edit" or the "land" of one.
	deferPos := make(map[token.Pos]bool, len(fn.defers))
	for _, d := range fn.defers {
		deferPos[d.pos] = true
	}
	for i := range fn.calls {
		fn.calls[i].deferred = deferPos[fn.calls[i].pos]
	}
	return fn
}

func motifMentions(text, name string) bool {
	if name == "" || name == "_" {
		return false
	}
	for i := 0; i+len(name) <= len(text); i++ {
		if text[i:i+len(name)] != name {
			continue
		}
		if i > 0 && motifIdentByte(text[i-1]) {
			continue
		}
		if j := i + len(name); j < len(text) && motifIdentByte(text[j]) {
			continue
		}
		return true
	}
	return false
}

func motifIdentByte(b byte) bool {
	return b == '_' || (b >= '0' && b <= '9') || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

// --- detector vocabularies --------------------------------------------------
//
// These name CALL EDGES (which package function is invoked), not declarations.
// A call edge is repository-native structure: it survives a rename of the caller
// and cannot be produced by a comment.

var motifReadCalls = map[string]bool{
	"os.ReadFile": true, "os.Open": true, "os.OpenFile": true,
	"os.Stat": true, "os.Lstat": true, "os.ReadDir": true,
}

var motifWriteCalls = map[string]bool{
	"os.WriteFile": true, "os.Create": true, "os.Rename": true,
	"os.MkdirAll": true, "os.Mkdir": true, "os.Truncate": true,
	"os.Chmod": true, "os.Remove": true, "os.RemoveAll": true,
}

// motifLeasePairs pins acquire→release to MATCHED pairs. An unmatched pair on the
// same receiver (Lock … Close) is not a lease, and the miner abstains.
var motifLeasePairs = map[string][]string{
	"Lock":    {"Unlock"},
	"RLock":   {"RUnlock"},
	"Acquire": {"Release", "Close"},
	"Begin":   {"Rollback", "Abort", "Close"},
	"Reserve": {"Release"},
	"Claim":   {"Release"},
	"Lease":   {"Release"},
}

var motifLandSel = map[string]bool{
	"Commit": true, "Land": true, "Flush": true, "Sync": true,
	"Publish": true, "Persist": true,
}

// --- detectors --------------------------------------------------------------

type motifRule struct {
	spec   MotifDetector
	detect func(*motifFn) ([]MotifRoleEvidence, string, bool)
}

var motifRules = []motifRule{
	{
		spec: MotifDetector{
			ID: "inspect-edit-verify/1", CatalogID: MotifCatalogInspectEditVerify,
			Motif: "inspect-edit-verify", Version: MotifMinerVersion,
			Roles:      []string{"inspect", "edit", "verify"},
			Rule:       "one function reads a target, mutates that SAME target expression, then reads it again; the shared first argument is the linkage",
			Confidence: 0.9,
		},
		detect: motifDetectInspectEditVerify,
	},
	{
		spec: MotifDetector{
			ID: "lease-isolate-land/1", CatalogID: MotifCatalogLeaseIsolateLand,
			Motif: "lease-isolate-land", Version: MotifMinerVersion,
			Roles:      []string{"lease", "isolate", "land"},
			Rule:       "an acquire call on receiver R is followed by a defer of its MATCHED release on the same R, and a non-deferred mutating/commit call lands work inside that window",
			Confidence: 0.85,
		},
		detect: motifDetectLeaseIsolateLand,
	},
	{
		spec: MotifDetector{
			ID: "reproduce-fix-witness/1", CatalogID: MotifCatalogReproduceFixWitness,
			Motif: "reproduce-fix-witness", Version: MotifMinerVersion,
			Roles:      []string{"reproduce", "subject", "witness"},
			Rule:       "a function taking *testing.T seeds state into a variable, feeds that variable to a subject call, and asserts on the subject's result via t.Fatal*/t.Error*; the chain is a data dependency, not a naming convention",
			Confidence: 0.8,
		},
		detect: motifDetectReproduceFixWitness,
	},
	{
		spec: MotifDetector{
			ID: "plan-fanout-reconcile/1", CatalogID: MotifCatalogPlanFanoutReconcile,
			Motif: "plan-fanout-reconcile", Version: MotifMinerVersion,
			Roles:      []string{"plan", "fanout", "reconcile"},
			Rule:       "a work list is built into an identifier, ranged over with a `go` statement in the loop body, and joined after the loop by a Wait call or a channel receive",
			Confidence: 0.85,
		},
		detect: motifDetectPlanFanoutReconcile,
	},
}

// MotifDetectors returns the detector set, ordered and copied — the report embeds
// it so an archived finding can be read against the rule that produced it.
func MotifDetectors() []MotifDetector {
	out := make([]MotifDetector, 0, len(motifRules))
	for _, r := range motifRules {
		spec := r.spec
		spec.Roles = append([]string(nil), r.spec.Roles...)
		out = append(out, spec)
	}
	return out
}

func motifDetectInspectEditVerify(fn *motifFn) ([]MotifRoleEvidence, string, bool) {
	for _, w := range fn.calls {
		if w.deferred || !motifWriteCalls[w.qual] || len(w.args) == 0 {
			continue
		}
		target := w.args[0]
		var before, after *motifCall
		for i := range fn.calls {
			c := &fn.calls[i]
			if !motifReadCalls[c.qual] || len(c.args) == 0 || c.args[0] != target {
				continue
			}
			if c.pos < w.pos && before == nil {
				before = c
			}
			if c.pos > w.pos && after == nil {
				after = c
			}
		}
		if before == nil || after == nil {
			continue
		}
		ev := []MotifRoleEvidence{
			{Role: "inspect", Line: before.line, Source: before.src},
			{Role: "edit", Line: w.line, Source: w.src},
			{Role: "verify", Line: after.line, Source: after.src},
		}
		return ev, "read, mutate, and re-read of the same target expression " + target, true
	}
	return nil, "", false
}

func motifDetectLeaseIsolateLand(fn *motifFn) ([]MotifRoleEvidence, string, bool) {
	for _, d := range fn.defers {
		if d.recv == "" {
			continue
		}
		var acquire *motifCall
		for i := range fn.calls {
			c := &fn.calls[i]
			if c.pos >= d.pos || c.recv != d.recv || c.deferred {
				continue
			}
			for _, rel := range motifLeasePairs[c.sel] {
				if rel == d.sel {
					acquire = c
					break
				}
			}
			if acquire != nil {
				break
			}
		}
		if acquire == nil {
			continue
		}
		var land *motifCall
		for i := range fn.calls {
			c := &fn.calls[i]
			if c.pos <= d.pos || c.deferred {
				continue
			}
			if motifWriteCalls[c.qual] || motifLandSel[c.sel] {
				land = c
				break
			}
		}
		if land == nil {
			continue
		}
		ev := []MotifRoleEvidence{
			{Role: "lease", Line: acquire.line, Source: acquire.src},
			{Role: "isolate", Line: fn.fset.Position(d.pos).Line, Source: "defer " + d.src},
			{Role: "land", Line: land.line, Source: land.src},
		}
		return ev, "acquire/release pair on the same receiver " + d.recv + " wraps a landing call", true
	}
	return nil, "", false
}

func motifDetectReproduceFixWitness(fn *motifFn) ([]MotifRoleEvidence, string, bool) {
	if fn.tParam == "" {
		return nil, "", false
	}
	type bound struct {
		name string
		pos  token.Pos
		src  string
	}
	var seeds []bound
	for _, a := range fn.assigns {
		if a.Tok != token.DEFINE || len(a.Rhs) != 1 {
			continue
		}
		switch a.Rhs[0].(type) {
		case *ast.CallExpr, *ast.CompositeLit:
		default:
			continue
		}
		for _, l := range a.Lhs {
			id, ok := l.(*ast.Ident)
			if !ok || id.Name == "_" {
				continue
			}
			seeds = append(seeds, bound{id.Name, a.Pos(), motifExprText(fn.fset, a)})
		}
	}
	for _, a := range fn.assigns {
		if len(a.Rhs) != 1 {
			continue
		}
		call, ok := a.Rhs[0].(*ast.CallExpr)
		if !ok {
			continue
		}
		mc := motifMakeCall(fn.fset, call)
		if mc.recv == fn.tParam {
			continue
		}
		var seed *bound
		for i := range seeds {
			if seeds[i].pos < a.Pos() && motifMentions(mc.src, seeds[i].name) {
				seed = &seeds[i]
				break
			}
		}
		if seed == nil {
			continue
		}
		for _, l := range a.Lhs {
			id, ok := l.(*ast.Ident)
			if !ok || id.Name == "_" {
				continue
			}
			for _, ifs := range fn.ifs {
				if ifs.Pos() <= a.Pos() || ifs.Cond == nil {
					continue
				}
				if !motifMentions(motifExprText(fn.fset, ifs.Cond), id.Name) {
					continue
				}
				assert := motifAssertIn(fn, ifs)
				if assert == nil {
					continue
				}
				ev := []MotifRoleEvidence{
					{Role: "reproduce", Line: fn.fset.Position(seed.pos).Line, Source: seed.src},
					{Role: "subject", Line: mc.line, Source: motifExprText(fn.fset, a)},
					{Role: "witness", Line: assert.line, Source: assert.src},
				}
				return ev, "seeded value " + seed.name + " flows into the subject call and its result " + id.Name + " is asserted on", true
			}
		}
	}
	return nil, "", false
}

// motifAssertIn returns the first t.Fatal*/t.Error* call inside an if body.
func motifAssertIn(fn *motifFn, ifs *ast.IfStmt) *motifCall {
	for i := range fn.calls {
		c := &fn.calls[i]
		if c.pos < ifs.Body.Pos() || c.pos > ifs.Body.End() || c.recv != fn.tParam {
			continue
		}
		if strings.HasPrefix(c.sel, "Fatal") || strings.HasPrefix(c.sel, "Error") {
			return c
		}
	}
	return nil
}

func motifDetectPlanFanoutReconcile(fn *motifFn) ([]MotifRoleEvidence, string, bool) {
	for _, r := range fn.ranges {
		id, ok := r.X.(*ast.Ident)
		if !ok || r.Body == nil {
			continue
		}
		var spawn token.Pos
		for _, g := range fn.gos {
			if g > r.Body.Pos() && g < r.Body.End() {
				spawn = g
				break
			}
		}
		if spawn == 0 {
			continue
		}
		var plan *ast.AssignStmt
		for _, a := range fn.assigns {
			if a.Pos() >= r.Pos() || len(a.Rhs) != 1 {
				continue
			}
			switch a.Rhs[0].(type) {
			case *ast.CallExpr, *ast.CompositeLit:
			default:
				continue
			}
			for _, l := range a.Lhs {
				if lid, ok := l.(*ast.Ident); ok && lid.Name == id.Name {
					plan = a
				}
			}
		}
		if plan == nil {
			continue
		}
		joinLine, joinSrc := 0, ""
		for i := range fn.calls {
			c := &fn.calls[i]
			if c.pos > r.End() && c.sel == "Wait" {
				joinLine, joinSrc = c.line, c.src
				break
			}
		}
		if joinSrc == "" {
			for _, p := range fn.recvOps {
				if p > r.End() {
					joinLine, joinSrc = fn.fset.Position(p).Line, "channel receive"
					break
				}
			}
		}
		if joinSrc == "" {
			continue
		}
		ev := []MotifRoleEvidence{
			{Role: "plan", Line: fn.fset.Position(plan.Pos()).Line, Source: motifExprText(fn.fset, plan)},
			{Role: "fanout", Line: fn.fset.Position(spawn).Line, Source: "go statement in range over " + id.Name},
			{Role: "reconcile", Line: joinLine, Source: joinSrc},
		}
		return ev, "work list " + id.Name + " is planned, fanned out with a go statement, and joined after the loop", true
	}
	return nil, "", false
}

// --- entry points -----------------------------------------------------------

// MineMotifsSource mines one Go source buffer. path is used only to label the
// findings. Comments are deliberately NOT parsed, so no detector can see them.
func MineMotifsSource(path string, src []byte) ([]MotifFinding, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, src, parser.SkipObjectResolution)
	if err != nil {
		return nil, err
	}
	out := []MotifFinding{}
	for _, d := range file.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}
		fn := motifCollect(fset, fd)
		for _, rule := range motifRules {
			ev, why, ok := rule.detect(fn)
			if !ok {
				continue
			}
			out = append(out, MotifFinding{
				CatalogID: rule.spec.CatalogID, Motif: rule.spec.Motif,
				Detector: rule.spec.ID, Version: rule.spec.Version,
				Path: filepath.ToSlash(path), Symbol: fn.name,
				StartLine: fn.start, EndLine: fn.end,
				Confidence: rule.spec.Confidence, Reason: why, Evidence: ev,
			})
		}
	}
	motifSortFindings(out)
	return out, nil
}

// MineMotifs mines a whole tree. It never fails on one bad file: an unreadable,
// excluded, or unparsable path becomes a Skip entry, so the report distinguishes
// "looked and found nothing" from "never looked".
func MineMotifs(root string) (MotifReport, error) {
	rep := MotifReport{
		Version:   MotifMinerVersion,
		Detectors: MotifDetectors(),
		Findings:  []MotifFinding{},
		Skipped:   []MotifSkip{},
	}
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		rel, relErr := filepath.Rel(root, p)
		if relErr != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		if rel == "." {
			return nil
		}
		if d.IsDir() {
			if reason, ex := MotifExcludedPath(rel); ex {
				rep.Skipped = append(rep.Skipped, MotifSkip{Path: rel + "/", Reason: reason})
				return fs.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(rel, ".go") {
			return nil
		}
		if reason, ex := MotifExcludedPath(rel); ex {
			rep.Skipped = append(rep.Skipped, MotifSkip{Path: rel, Reason: reason})
			return nil
		}
		src, readErr := os.ReadFile(p)
		if readErr != nil {
			rep.Skipped = append(rep.Skipped, MotifSkip{Path: rel, Reason: "unreadable"})
			return nil
		}
		if reason, ex := motifExcludedContent(src); ex {
			rep.Skipped = append(rep.Skipped, MotifSkip{Path: rel, Reason: reason})
			return nil
		}
		found, parseErr := MineMotifsSource(rel, src)
		if parseErr != nil {
			rep.Skipped = append(rep.Skipped, MotifSkip{Path: rel, Reason: "unparsable"})
			return nil
		}
		rep.Scanned++
		rep.Findings = append(rep.Findings, found...)
		return nil
	})
	motifSortFindings(rep.Findings)
	sort.SliceStable(rep.Skipped, func(i, j int) bool {
		if rep.Skipped[i].Path != rep.Skipped[j].Path {
			return rep.Skipped[i].Path < rep.Skipped[j].Path
		}
		return rep.Skipped[i].Reason < rep.Skipped[j].Reason
	})
	return rep, err
}

// motifSortFindings imposes the total order that makes two runs byte-identical.
func motifSortFindings(f []MotifFinding) {
	sort.SliceStable(f, func(i, j int) bool {
		a, b := f[i], f[j]
		switch {
		case a.Path != b.Path:
			return a.Path < b.Path
		case a.StartLine != b.StartLine:
			return a.StartLine < b.StartLine
		case a.CatalogID != b.CatalogID:
			return a.CatalogID < b.CatalogID
		default:
			return a.Symbol < b.Symbol
		}
	})
}

// MotifReportJSON renders a report as deterministic JSON (empty slices, never
// null, so a downstream consumer sees a stable shape).
func MotifReportJSON(r MotifReport) ([]byte, error) {
	if r.Findings == nil {
		r.Findings = []MotifFinding{}
	}
	if r.Skipped == nil {
		r.Skipped = []MotifSkip{}
	}
	if r.Detectors == nil {
		r.Detectors = []MotifDetector{}
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}
