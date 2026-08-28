package devindex

import (
	"fmt"
	"go/ast"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ExtractionReasonCode is the closed vocabulary for refusing source motion.
// Callers must be able to distinguish a safety decision without parsing prose.
type ExtractionReasonCode string

const (
	ReasonUnresolvedHandler  ExtractionReasonCode = "unresolved-handler"
	ReasonRuntimeOverlap     ExtractionReasonCode = "runtime-tier-overlap"
	ReasonUnknownTierOverlap ExtractionReasonCode = "unknown-tier-overlap"
	ReasonSharedDeclaration  ExtractionReasonCode = "shared-declaration"
	ReasonCgo                ExtractionReasonCode = "hazard-cgo"
	ReasonReflect            ExtractionReasonCode = "hazard-reflect"
	ReasonInit               ExtractionReasonCode = "hazard-init"
	ReasonLinkname           ExtractionReasonCode = "hazard-linkname"
	ReasonEmbed              ExtractionReasonCode = "hazard-embed"
	ReasonSelfExec           ExtractionReasonCode = "hazard-self-exec"
)

// ExtractionReason attributes one closed refusal reason to the source that
// caused it. Source is always repository-relative; Symbol is present when a
// declaration, rather than a file-wide mechanism, owns the edge.
type ExtractionReason struct {
	Code          ExtractionReasonCode `json:"code"`
	Source        string               `json:"source"`
	Symbol        string               `json:"symbol,omitempty"`
	EvidenceCount int                  `json:"evidence_count"`
}

// ExtractionDelta is a non-mutating projection of source and import-graph
// removal. Package counts are the packages that become unreachable from cmd/fak,
// not a build-time claim.
type ExtractionDelta struct {
	Commands         int   `json:"commands"`
	Files            int   `json:"files"`
	SourceBytes      int64 `json:"source_bytes"`
	DirectImports    int   `json:"direct_imports"`
	Packages         int   `json:"packages"`
	InternalPackages int   `json:"internal_packages"`
}

// ExtractionCandidate is one whole same-package component. Multiple commands
// share a row when their declaration closures overlap and therefore must move
// together. Reasons is empty exactly for rows in safe.
type ExtractionCandidate struct {
	Commands []string           `json:"commands"`
	Handlers []string           `json:"handlers"`
	Files    []string           `json:"files"`
	Imports  []string           `json:"imports,omitempty"`
	Delta    ExtractionDelta    `json:"projected_delta"`
	Reasons  []ExtractionReason `json:"reasons,omitempty"`
}

// ExtractionReport is the checked-in, fail-closed answer to "what TierDev source
// can leave cmd/fak next?" Every remaining runtime TierDev root appears exactly
// once in Safe or Excluded.
type ExtractionReport struct {
	Schema    string                 `json:"schema"`
	Safe      []ExtractionCandidate  `json:"safe"`
	Excluded  []ExtractionCandidate  `json:"excluded"`
	Counts    ExtractionReportCounts `json:"counts"`
	SafeDelta ExtractionDelta        `json:"safe_projected_delta"`
}

type ExtractionReportCounts struct {
	Commands   int `json:"commands"`
	Safe       int `json:"safe"`
	Excluded   int `json:"excluded"`
	Components int `json:"components"`
}

type extractionDecl struct {
	id     int
	symbol string
	file   *ast.File
	node   ast.Node
	deps   map[int]bool
}

type extractionRoot struct {
	command string
	handler string
	origin  string
	decls   map[int]bool
}

type extractionModel struct {
	pkg       *vsPkg
	decls     []*extractionDecl
	byName    map[string][]int
	fileDecls map[*ast.File][]int
	byNode    map[ast.Node][]int
}

// BuildRemainingExtractionReport derives the remaining runtime-owned TierDev
// source plan and projects the safe union against the current runtime graph.
func BuildRemainingExtractionReport(root string, nodes []ImportNode) (ExtractionReport, error) {
	pkg, err := vsLoadCmd(root)
	if err != nil {
		return ExtractionReport{}, err
	}
	report, safeFiles, err := buildRemainingExtractionReport(pkg)
	if err != nil {
		return ExtractionReport{}, err
	}
	projectExtractionDeltas(&report, pkg, nodes, safeFiles)
	return report, nil
}

func buildRemainingExtractionReport(pkg *vsPkg) (ExtractionReport, map[string]bool, error) {
	model := newExtractionModel(pkg)
	rows := vsTopRows(pkg)
	var roots []extractionRoot
	allHandlerRoots := map[int][]vsRow{}
	runtimeReach := map[int][]vsRow{}
	for _, row := range rows {
		if row.fn == "" {
			continue
		}
		ids := model.byName[row.fn]
		for _, id := range ids {
			allHandlerRoots[id] = append(allHandlerRoots[id], row)
		}
		tier, known := TierOf(row.name)
		if !known || tier != TierDev {
			for id := range model.closure(ids) {
				runtimeReach[id] = append(runtimeReach[id], row)
			}
			continue
		}
		root := extractionRoot{command: row.name, handler: row.fn, origin: row.origin}
		if len(ids) != 0 {
			root.decls = model.closure(ids)
		}
		roots = append(roots, root)
	}
	if len(roots) == 0 {
		return ExtractionReport{}, nil, fmt.Errorf("no remaining TierDev runtime handlers found")
	}

	// Roots with intersecting declaration closures form one atomic motion unit.
	parent := make([]int, len(roots))
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(i int) int {
		if parent[i] != i {
			parent[i] = find(parent[i])
		}
		return parent[i]
	}
	union := func(a, b int) {
		a, b = find(a), find(b)
		if a != b {
			parent[b] = a
		}
	}
	owners := map[int]int{}
	for i, root := range roots {
		for id := range root.decls {
			if prior, ok := owners[id]; ok {
				union(i, prior)
			} else {
				owners[id] = i
			}
		}
	}
	groups := map[int][]int{}
	for i := range roots {
		groups[find(i)] = append(groups[find(i)], i)
	}

	report := ExtractionReport{Schema: "fak-dev-extraction-report/1"}
	safeFiles := map[string]bool{}
	for _, indexes := range groups {
		component := map[int]bool{}
		commands, handlers := []string{}, []string{}
		for _, i := range indexes {
			commands = append(commands, roots[i].command)
			handlers = append(handlers, roots[i].handler)
			for id := range roots[i].decls {
				component[id] = true
			}
		}
		sort.Strings(commands)
		sort.Strings(handlers)
		candidate := ExtractionCandidate{Commands: compactStrings(commands), Handlers: compactStrings(handlers)}
		candidate.Files, candidate.Imports, candidate.Delta.SourceBytes = model.componentFiles(component)
		candidate.Delta.Commands = len(candidate.Commands)
		candidate.Delta.Files = len(candidate.Files)
		candidate.Reasons = model.componentReasons(component, indexes, roots, allHandlerRoots, runtimeReach)
		if len(candidate.Reasons) == 0 {
			report.Safe = append(report.Safe, candidate)
			for _, file := range candidate.Files {
				safeFiles[file] = true
			}
		} else {
			report.Excluded = append(report.Excluded, candidate)
		}
	}
	sortCandidates(report.Safe)
	sortCandidates(report.Excluded)
	for _, row := range report.Safe {
		report.Counts.Safe += len(row.Commands)
	}
	for _, row := range report.Excluded {
		report.Counts.Excluded += len(row.Commands)
	}
	report.Counts.Commands = report.Counts.Safe + report.Counts.Excluded
	report.Counts.Components = len(report.Safe) + len(report.Excluded)
	return report, safeFiles, nil
}

func newExtractionModel(pkg *vsPkg) *extractionModel {
	m := &extractionModel{pkg: pkg, byName: map[string][]int{}, fileDecls: map[*ast.File][]int{}, byNode: map[ast.Node][]int{}}
	add := func(symbol string, file *ast.File, node ast.Node) {
		id := len(m.decls)
		m.decls = append(m.decls, &extractionDecl{id: id, symbol: symbol, file: file, node: node, deps: map[int]bool{}})
		m.byName[symbol] = append(m.byName[symbol], id)
		m.fileDecls[file] = append(m.fileDecls[file], id)
		m.byNode[node] = append(m.byNode[node], id)
	}
	for _, file := range pkg.files {
		for _, decl := range file.Decls {
			switch decl := decl.(type) {
			case *ast.FuncDecl:
				symbol := decl.Name.Name
				if decl.Recv != nil {
					symbol = receiverName(decl.Recv) + "." + symbol
				}
				add(symbol, file, decl)
			case *ast.GenDecl:
				for _, spec := range decl.Specs {
					switch spec := spec.(type) {
					case *ast.TypeSpec:
						add(spec.Name.Name, file, spec)
					case *ast.ValueSpec:
						for _, name := range spec.Names {
							add(name.Name, file, spec)
						}
					}
				}
			}
		}
	}
	// Resolve same-package declaration references. Imported selectors remain a
	// package boundary; method selectors conservatively join every local method
	// with that name, which can only exclude motion rather than admit it falsely.
	for _, decl := range m.decls {
		imports := pkg.imports[decl.file]
		ast.Inspect(decl.node, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.Ident:
				ids := m.byName[n.Name]
				if n.Obj != nil {
					declNode, ok := n.Obj.Decl.(ast.Node)
					if !ok {
						return true
					}
					ids = m.byNode[declNode]
					if len(ids) == 0 { // parser-resolved local declaration
						return true
					}
				}
				for _, id := range ids {
					if id != decl.id {
						decl.deps[id] = true
					}
				}
			case *ast.SelectorExpr:
				if id, ok := n.X.(*ast.Ident); ok {
					if _, imported := imports[id.Name]; imported {
						return false
					}
				}
				for _, id := range m.localMethodTargets(decl, n) {
					decl.deps[id] = true
				}
			}
			return true
		})
	}
	return m
}

func (m *extractionModel) localMethodTargets(decl *extractionDecl, sel *ast.SelectorExpr) []int {
	// A selector whose receiver is a declared type is a method expression. A
	// selector on the current method receiver is resolved from its field type.
	var recv string
	if id, ok := sel.X.(*ast.Ident); ok {
		if fn, ok := decl.node.(*ast.FuncDecl); ok {
			if fn.Recv != nil && len(fn.Recv.List) != 0 {
				for _, name := range fn.Recv.List[0].Names {
					if name.Name == id.Name {
						recv = receiverName(fn.Recv)
					}
				}
			}
			if fn.Type.Params != nil {
				for _, field := range fn.Type.Params.List {
					for _, name := range field.Names {
						if name.Name == id.Name {
							recv = receiverExprName(field.Type)
						}
					}
				}
			}
		}
		if recv == "" && m.identNamesPackageType(id) {
			recv = id.Name
		}
	}
	if recv == "" {
		// A selector on an expression whose static receiver cannot be derived is
		// not permission to ignore package-local methods. Join every matching
		// method declaration; the conservative edge may exclude a component but
		// cannot admit source that would strand a method.
		var out []int
		for name, ids := range m.byName {
			if strings.HasSuffix(name, "."+sel.Sel.Name) {
				out = append(out, ids...)
			}
		}
		return out
	}
	return m.byName[recv+"."+sel.Sel.Name]
}

func (m *extractionModel) identNamesPackageType(id *ast.Ident) bool {
	if id.Obj != nil {
		node, ok := id.Obj.Decl.(ast.Node)
		if !ok {
			return false
		}
		for _, declID := range m.byNode[node] {
			if _, ok := m.decls[declID].node.(*ast.TypeSpec); ok {
				return true
			}
		}
		return false
	}
	for _, declID := range m.byName[id.Name] {
		if _, ok := m.decls[declID].node.(*ast.TypeSpec); ok {
			return true
		}
	}
	return false
}

func receiverExprName(expr ast.Expr) string {
	if star, ok := expr.(*ast.StarExpr); ok {
		expr = star.X
	}
	if id, ok := expr.(*ast.Ident); ok {
		return id.Name
	}
	return ""
}

func receiverName(fields *ast.FieldList) string {
	if fields == nil || len(fields.List) == 0 {
		return "?"
	}
	var expr ast.Expr = fields.List[0].Type
	if star, ok := expr.(*ast.StarExpr); ok {
		expr = star.X
	}
	if id, ok := expr.(*ast.Ident); ok {
		return id.Name
	}
	return "?"
}

func (m *extractionModel) closure(starts []int) map[int]bool {
	out := map[int]bool{}
	queue := append([]int(nil), starts...)
	for len(queue) != 0 {
		id := queue[0]
		queue = queue[1:]
		if out[id] {
			continue
		}
		out[id] = true
		// main.go is the runtime dispatch shell: its handler reference is replaced
		// by generated handoff metadata, while unrelated dispatch declarations stay.
		// Treating the shell as one movable file would collapse every command into a
		// single component and answer the source-ownership question vacuously.
		if filepath.Base(m.pkg.pathOf[m.decls[id].file]) != "main.go" {
			for _, sibling := range m.fileDecls[m.decls[id].file] {
				if !out[sibling] {
					queue = append(queue, sibling)
				}
			}
		}
		for dep := range m.decls[id].deps {
			if !out[dep] {
				queue = append(queue, dep)
			}
		}
	}
	return out
}

func (m *extractionModel) componentFiles(component map[int]bool) ([]string, []string, int64) {
	fileSet, imports := map[string]bool{}, map[string]bool{}
	var bytes int64
	for id := range component {
		file := m.decls[id].file
		path := m.pkg.pathOf[file]
		fileSet[path] = true
		for _, imp := range m.pkg.imports[file] {
			imports[imp] = true
		}
	}
	files := sortedKeys(fileSet)
	for _, path := range files {
		if info, err := os.Stat(filepath.Join(m.pkgRoot(), filepath.FromSlash(path))); err == nil {
			bytes += info.Size()
		}
	}
	return files, sortedKeys(imports), bytes
}

func (m *extractionModel) pkgRoot() string {
	for file, path := range m.pkg.pathOf {
		abs := m.pkg.fset.Position(file.Package).Filename
		return strings.TrimSuffix(filepath.ToSlash(abs), path)
	}
	return ""
}

func (m *extractionModel) componentReasons(component map[int]bool, indexes []int, roots []extractionRoot, handlers, runtimeReach map[int][]vsRow) []ExtractionReason {
	var reasons []ExtractionReason
	for _, i := range indexes {
		if len(roots[i].decls) == 0 {
			reasons = append(reasons, ExtractionReason{Code: ReasonUnresolvedHandler, Source: roots[i].origin, Symbol: roots[i].handler})
		}
	}
	componentHandlers := map[string]bool{}
	for _, i := range indexes {
		componentHandlers[roots[i].handler] = true
	}
	for id := range component {
		decl := m.decls[id]
		for _, hazard := range sourceHazards(decl.file) {
			code := hazardReason(hazard)
			if code != "" {
				reasons = append(reasons, ExtractionReason{Code: code, Source: m.pkg.pathOf[decl.file]})
			}
		}
		for _, row := range runtimeReach[id] {
			tier, known := TierOf(row.name)
			if !known {
				reasons = append(reasons, ExtractionReason{Code: ReasonUnknownTierOverlap, Source: m.pkg.pathOf[decl.file], Symbol: row.name})
			} else if tier != TierDev {
				reasons = append(reasons, ExtractionReason{Code: ReasonRuntimeOverlap, Source: m.pkg.pathOf[decl.file], Symbol: row.name})
			}
		}
	}
	// Any incoming same-package reference would be stranded after moving the
	// component. Dispatch-table references to its handler are the one intended
	// cut edge; generated handoff rewires those at extraction time.
	for _, outside := range m.decls {
		if component[outside.id] {
			continue
		}
		for dep := range outside.deps {
			if !component[dep] || m.isDispatchCutEdge(outside, m.decls[dep], componentHandlers) {
				continue
			}
			reasons = append(reasons, ExtractionReason{Code: ReasonSharedDeclaration, Source: m.pkg.pathOf[outside.file], Symbol: outside.symbol})
		}
	}
	sort.Slice(reasons, func(i, j int) bool {
		a, b := reasons[i], reasons[j]
		if a.Code != b.Code {
			return a.Code < b.Code
		}
		if a.Source != b.Source {
			return a.Source < b.Source
		}
		return a.Symbol < b.Symbol
	})
	return summarizeReasons(reasons)
}

func (m *extractionModel) isDispatchCutEdge(from, to *extractionDecl, handlers map[string]bool) bool {
	if !handlers[to.symbol] {
		return false
	}
	base := filepath.Base(m.pkg.pathOf[from.file])
	return base == "main.go" && (from.symbol == "main" || strings.HasPrefix(from.symbol, "dispatch"))
}

func hazardReason(hazard string) ExtractionReasonCode {
	switch hazard {
	case "cgo":
		return ReasonCgo
	case "reflection":
		return ReasonReflect
	case "init":
		return ReasonInit
	case "linkname":
		return ReasonLinkname
	case "embed":
		return ReasonEmbed
	case "self-exec":
		return ReasonSelfExec
	}
	return ""
}

func summarizeReasons(in []ExtractionReason) []ExtractionReason {
	type reasonFact struct {
		code   ExtractionReasonCode
		source string
		symbol string
	}
	distinct := make(map[reasonFact]bool, len(in))
	for _, fact := range in {
		distinct[reasonFact{code: fact.Code, source: fact.Source, symbol: fact.Symbol}] = true
	}
	facts := make([]reasonFact, 0, len(distinct))
	for fact := range distinct {
		facts = append(facts, fact)
	}
	sort.Slice(facts, func(i, j int) bool {
		if facts[i].code != facts[j].code {
			return facts[i].code < facts[j].code
		}
		if facts[i].source != facts[j].source {
			return facts[i].source < facts[j].source
		}
		return facts[i].symbol < facts[j].symbol
	})
	var out []ExtractionReason
	for _, fact := range facts {
		if len(out) == 0 || out[len(out)-1].Code != fact.code {
			out = append(out, ExtractionReason{
				Code: fact.code, Source: fact.source, Symbol: fact.symbol, EvidenceCount: 1,
			})
			continue
		}
		out[len(out)-1].EvidenceCount++
	}
	return out
}

func sortCandidates(rows []ExtractionCandidate) {
	sort.Slice(rows, func(i, j int) bool {
		return strings.Join(rows[i].Commands, "\x00") < strings.Join(rows[j].Commands, "\x00")
	})
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for key := range m {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}

func projectExtractionDeltas(report *ExtractionReport, pkg *vsPkg, nodes []ImportNode, safeFiles map[string]bool) {
	for i := range report.Safe {
		projectCandidateDelta(&report.Safe[i], importsOutsideFiles(pkg, extractionStringSet(report.Safe[i].Files)), nodes)
	}
	for i := range report.Excluded {
		projectCandidateDelta(&report.Excluded[i], importsOutsideFiles(pkg, extractionStringSet(report.Excluded[i].Files)), nodes)
	}
	for _, row := range report.Safe {
		report.SafeDelta.Commands += row.Delta.Commands
		report.SafeDelta.SourceBytes += row.Delta.SourceBytes
	}
	report.SafeDelta.Files = len(safeFiles)
	outsideImports := importsOutsideFiles(pkg, safeFiles)
	unionImports := map[string]bool{}
	for _, row := range report.Safe {
		for _, imp := range row.Imports {
			if !outsideImports[imp] {
				unionImports[imp] = true
			}
		}
	}
	report.SafeDelta.DirectImports, report.SafeDelta.Packages, report.SafeDelta.InternalPackages = projectedGraphLoss(nodes, unionImports)
}

func importsOutsideFiles(pkg *vsPkg, moved map[string]bool) map[string]bool {
	out := map[string]bool{}
	for _, file := range pkg.files {
		if moved[pkg.pathOf[file]] {
			continue
		}
		for _, path := range pkg.imports[file] {
			out[path] = true
		}
	}
	return out
}

func extractionStringSet(values []string) map[string]bool {
	out := make(map[string]bool, len(values))
	for _, value := range values {
		out[value] = true
	}
	return out
}

func projectCandidateDelta(row *ExtractionCandidate, outside map[string]bool, nodes []ImportNode) {
	removed := map[string]bool{}
	for _, imp := range row.Imports {
		if !outside[imp] {
			removed[imp] = true
		}
	}
	row.Delta.DirectImports, row.Delta.Packages, row.Delta.InternalPackages = projectedGraphLoss(nodes, removed)
}

func projectedGraphLoss(nodes []ImportNode, removed map[string]bool) (int, int, int) {
	adj := map[string][]string{}
	for _, node := range nodes {
		adj[node.ImportPath] = node.Imports
	}
	root := "github.com/anthony-chaudhary/fak/cmd/fak"
	activeRemoved := map[string]bool{}
	for _, path := range adj[root] {
		if removed[path] {
			activeRemoved[path] = true
		}
	}
	reach := func(cut bool) map[string]bool {
		seen := map[string]bool{}
		queue := []string{root}
		for len(queue) != 0 {
			cur := queue[0]
			queue = queue[1:]
			if seen[cur] {
				continue
			}
			seen[cur] = true
			for _, next := range adj[cur] {
				if cut && cur == root && activeRemoved[next] {
					continue
				}
				queue = append(queue, next)
			}
		}
		return seen
	}
	before, after := reach(false), reach(true)
	lost, internal := 0, 0
	for path := range before {
		if !after[path] {
			lost++
			if strings.HasPrefix(path, moduleInternalPrefix) {
				internal++
			}
		}
	}
	return len(activeRemoved), lost, internal
}
