package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"strconv"
	"strings"
)

type tokenAssertionSpec struct{ Terms []string }

type tokenExecutableProof struct {
	Path       string
	Function   string
	Assertions []tokenAssertionSpec
}

type tokenLiteralProof struct {
	Path, Function, Callee string
	Literals               []string
}

type tokenProofBinding struct {
	Effect tokenExecutableProof
	Lock   tokenExecutableProof
	Note   tokenLiteralProof
	Gate   *tokenGateDeclaration
}

type tokenGateState string

const tokenGateOpen tokenGateState = "OPEN"

type tokenGateDeclaration struct {
	Schema       string
	ID           string
	State        tokenGateState
	PassArtifact string
}

type tokenGatePass struct {
	Schema string `json:"schema"`
	ID     string `json:"id"`
	Status string `json:"status"`
}

var deferColdToolsGate = tokenGateDeclaration{
	Schema: "fak-token-required-gate/1", ID: "#3536", State: tokenGateOpen,
	PassArtifact: "docs/notes/defer-cold-tools-live-dogfood-pass.json",
}

func (g tokenGateDeclaration) valid() bool {
	return g.Schema == "fak-token-required-gate/1" && g.ID == "#3536" && g.State == tokenGateOpen && g.PassArtifact == "docs/notes/defer-cold-tools-live-dogfood-pass.json"
}

func tokenGateBlocker(s tokenDefaultSources, gate *tokenGateDeclaration) string {
	if gate == nil {
		return ""
	}
	if !gate.valid() {
		return "invalid required-gate declaration"
	}
	if !parseTokenGatePass(s.read(gate.PassArtifact), *gate) {
		return gate.ID + " OPEN: strict committed PASS receipt missing"
	}
	return ""
}

func parseTokenGatePass(raw string, gate tokenGateDeclaration) bool {
	if raw == "" || !gate.valid() || hasDuplicateJSONKeys([]byte(raw)) {
		return false
	}
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.DisallowUnknownFields()
	var pass tokenGatePass
	if err := dec.Decode(&pass); err != nil {
		return false
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return false
	}
	return pass.Schema == "fak-token-gate-pass/1" && pass.ID == gate.ID && pass.Status == "PASS"
}

func hasDuplicateJSONKeys(raw []byte) bool {
	dec := json.NewDecoder(bytes.NewReader(raw))
	var walk func() bool
	walk = func() bool {
		tok, err := dec.Token()
		if err != nil {
			return true
		}
		delim, ok := tok.(json.Delim)
		if !ok {
			return false
		}
		switch delim {
		case '{':
			seen := map[string]bool{}
			for dec.More() {
				key, err := dec.Token()
				if err != nil {
					return true
				}
				name, ok := key.(string)
				if !ok || seen[name] {
					return true
				}
				seen[name] = true
				if walk() {
					return true
				}
			}
			_, err = dec.Token()
			return err != nil
		case '[':
			for dec.More() {
				if walk() {
					return true
				}
			}
			_, err = dec.Token()
			return err != nil
		default:
			return true
		}
	}
	return walk()
}

func astTerms(node ast.Node) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(node, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.Ident:
			out[x.Name] = true
		case *ast.BasicLit:
			if x.Kind == token.STRING {
				if s, err := strconv.Unquote(x.Value); err == nil {
					out[s] = true
					out["literal:"+s] = true
				}
			}
		}
		return true
	})
	return out
}

func staticallyFalse(expr ast.Expr, constants map[string]bool) bool {
	switch x := expr.(type) {
	case *ast.Ident:
		if x.Name == "false" {
			return true
		}
		value, known := constants[x.Name]
		return known && !value
	case *ast.ParenExpr:
		return staticallyFalse(x.X, constants)
	case *ast.UnaryExpr:
		return x.Op == token.NOT && staticallyTrue(x.X, constants)
	case *ast.BinaryExpr:
		if x.Op == token.LAND {
			return staticallyFalse(x.X, constants) || staticallyFalse(x.Y, constants)
		}
		if x.Op == token.LOR {
			return staticallyFalse(x.X, constants) && staticallyFalse(x.Y, constants)
		}
	}
	return false
}

func staticallyTrue(expr ast.Expr, constants map[string]bool) bool {
	switch x := expr.(type) {
	case *ast.Ident:
		if x.Name == "true" {
			return true
		}
		value, known := constants[x.Name]
		return known && value
	case *ast.ParenExpr:
		return staticallyTrue(x.X, constants)
	case *ast.UnaryExpr:
		return x.Op == token.NOT && staticallyFalse(x.X, constants)
	case *ast.BinaryExpr:
		if x.Op == token.LAND {
			return staticallyTrue(x.X, constants) && staticallyTrue(x.Y, constants)
		}
		if x.Op == token.LOR {
			return staticallyTrue(x.X, constants) || staticallyTrue(x.Y, constants)
		}
	}
	return false
}

func testingFailureCall(expr ast.Expr, receiver string) bool {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	id, receiverOK := sel.X.(*ast.Ident)
	if receiverOK && id.Name == receiver {
		switch sel.Sel.Name {
		case "Fatal", "Fatalf", "Error", "Errorf", "Fail", "FailNow":
			return true
		}
	}
	return false
}

func testFailureIn(block *ast.BlockStmt, receiver string, constants map[string]bool) bool {
	var statementsFail func([]ast.Stmt) bool
	statementsFail = func(stmts []ast.Stmt) bool {
		for _, raw := range stmts {
			switch stmt := raw.(type) {
			case *ast.ExprStmt:
				if testingFailureCall(stmt.X, receiver) {
					return true
				}
			case *ast.BlockStmt:
				if statementsFail(stmt.List) {
					return true
				}
			case *ast.IfStmt:
				if !staticallyFalse(stmt.Cond, constants) && statementsFail(stmt.Body.List) {
					return true
				}
				if stmt.Else != nil && !staticallyTrue(stmt.Cond, constants) {
					switch branch := stmt.Else.(type) {
					case *ast.BlockStmt:
						if statementsFail(branch.List) {
							return true
						}
					case *ast.IfStmt:
						if testFailureIn(&ast.BlockStmt{List: []ast.Stmt{branch}}, receiver, constants) {
							return true
						}
					}
				}
			case *ast.ForStmt:
				if (stmt.Cond == nil || !staticallyFalse(stmt.Cond, constants)) && statementsFail(stmt.Body.List) {
					return true
				}
			case *ast.RangeStmt:
				if statementsFail(stmt.Body.List) {
					return true
				}
			case *ast.LabeledStmt:
				if statementsFail([]ast.Stmt{stmt.Stmt}) {
					return true
				}
			}
		}
		return false
	}
	return statementsFail(block.List)
}

func localBoolConstants(fn *ast.FuncDecl) map[string]bool {
	constants := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		decl, ok := n.(*ast.DeclStmt)
		if !ok {
			return true
		}
		gen, ok := decl.Decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.CONST {
			return true
		}
		for _, raw := range gen.Specs {
			spec, ok := raw.(*ast.ValueSpec)
			if !ok {
				continue
			}
			for i, name := range spec.Names {
				if i >= len(spec.Values) {
					continue
				}
				if value, ok := spec.Values[i].(*ast.Ident); ok && (value.Name == "true" || value.Name == "false") {
					constants[name.Name] = value.Name == "true"
				}
			}
		}
		return false
	})
	return constants
}

func findTestFunction(file *ast.File, name string) (*ast.FuncDecl, string) {
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != name || fn.Body == nil || fn.Type.Params == nil || len(fn.Type.Params.List) == 0 {
			continue
		}
		param := fn.Type.Params.List[0]
		star, ok := param.Type.(*ast.StarExpr)
		if !ok {
			continue
		}
		sel, typed := star.X.(*ast.SelectorExpr)
		if !typed {
			continue
		}
		pkg, pkgOK := sel.X.(*ast.Ident)
		if ok && typed && pkgOK && pkg.Name == "testing" && sel.Sel.Name == "T" && len(param.Names) == 1 {
			return fn, param.Names[0].Name
		}
	}
	return nil, ""
}

func (s tokenDefaultSources) executableProof(p tokenExecutableProof) bool {
	if p.Path == "" || p.Function == "" || len(p.Assertions) == 0 {
		return false
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, p.Path, s.read(p.Path), 0)
	if err != nil {
		return false
	}
	fn, receiver := findTestFunction(file, p.Function)
	if fn == nil {
		return false
	}
	constants := localBoolConstants(fn)
	for _, want := range p.Assertions {
		matched := false
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			stmt, ok := n.(*ast.IfStmt)
			if !ok || staticallyFalse(stmt.Cond, constants) || !testFailureIn(stmt.Body, receiver, constants) {
				return true
			}
			terms := astTerms(stmt.Cond)
			for _, term := range want.Terms {
				present := terms[term]
				for got := range terms {
					if strings.HasPrefix(got, "literal:") && strings.Contains(strings.TrimPrefix(got, "literal:"), term) {
						present = true
					}
				}
				if !present {
					return true
				}
			}
			matched = true
			return false
		})
		if !matched {
			return false
		}
	}
	return true
}

func calleeName(expr ast.Expr) string {
	switch x := expr.(type) {
	case *ast.Ident:
		return x.Name
	case *ast.SelectorExpr:
		return calleeName(x.X) + "." + x.Sel.Name
	}
	return ""
}

func (s tokenDefaultSources) literalProof(p tokenLiteralProof) bool {
	if p.Path == "" || len(p.Literals) == 0 {
		return false
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, p.Path, s.read(p.Path), 0)
	if err != nil {
		return false
	}
	var root ast.Node = file
	if p.Function != "" {
		fn, _ := findTestFunction(file, p.Function)
		if fn == nil {
			for _, decl := range file.Decls {
				if candidate, ok := decl.(*ast.FuncDecl); ok && candidate.Name.Name == p.Function {
					fn = candidate
				}
			}
		}
		if fn == nil {
			return false
		}
		root = fn.Body
	}
	matched := false
	ast.Inspect(root, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || (p.Callee != "" && calleeName(call.Fun) != p.Callee) {
			return true
		}
		var values strings.Builder
		for _, arg := range call.Args {
			lit, ok := arg.(*ast.BasicLit)
			if !ok || lit.Kind != token.STRING {
				continue
			}
			value, err := strconv.Unquote(lit.Value)
			if err != nil {
				continue
			}
			values.WriteString(value)
			values.WriteByte('\n')
		}
		all := true
		for _, required := range p.Literals {
			all = all && strings.Contains(values.String(), required)
		}
		matched = matched || all
		return !matched
	})
	return matched
}

func (s tokenDefaultSources) registeredCompressorProof(path, name string) bool {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, s.read(path), 0)
	if err != nil {
		return false
	}
	hasName, hasRegister := false, false
	ast.Inspect(file, func(n ast.Node) bool {
		if lit, ok := n.(*ast.BasicLit); ok && lit.Kind == token.STRING {
			value, err := strconv.Unquote(lit.Value)
			hasName = hasName || (err == nil && value == name)
		}
		if call, ok := n.(*ast.CallExpr); ok && calleeName(call.Fun) == "Register" && len(call.Args) == 1 {
			hasRegister = true
		}
		return true
	})
	return hasName && hasRegister
}

type tokenEffectivenessRow struct {
	Key          string   `json:"key"`
	Default      string   `json:"default"`
	Configured   string   `json:"configured"`
	Owner        string   `json:"owner"`
	Mechanism    string   `json:"runtime_mechanism"`
	EffectMetric string   `json:"effect_metric"`
	WitnessKind  string   `json:"witness_kind"`
	Witness      string   `json:"witness"`
	Paths        []string `json:"paths"`
	Control      string   `json:"control"`
	Observed     string   `json:"observed"`
	Scope        string   `json:"scope"`
	Provenance   []string `json:"provenance"`
	Blocker      string   `json:"blocker,omitempty"`
}

type tokenEffectivenessReport struct {
	Schema string                  `json:"schema"`
	OK     bool                    `json:"ok"`
	Debt   int                     `json:"debt"`
	Rows   []tokenEffectivenessRow `json:"rows"`
	Note   string                  `json:"note"`
}

// tokenEffectivenessEvidence maps the source-derived defaults roster to the
// narrowest real effect witness and its control. It intentionally does not
// duplicate default state: collectTokenDefaultsScorecard remains authoritative.
var tokenEffectivenessEvidence = map[string]tokenEffectivenessRow{
	"provider_cache": {
		Owner: "internal/gateway", Mechanism: "provider prompt-cache prefix",
		EffectMetric: "cache-read tokens unlocked by stable fak breakpoint placement", WitnessKind: "exact-token A/B",
		Witness: "go test ./internal/gateway -run TestFakPlacementUnlocksProviderCacheSavings", Paths: []string{"internal/gateway/provider_cache_fak_placement_savings_test.go"},
		Control: "same turns with naive last-block placement", Scope: "synthetic provider-cache accounting; broader workload arms remain #6684 and #6690",
	},
	"toolfloor": {
		Owner: "internal/gateway", Mechanism: "unreachable tool-definition pruning",
		EffectMetric: "removed tool definitions and exact request-byte reduction while preserving the stable prefix", WitnessKind: "exact-byte A/B",
		Witness: "go test ./internal/gateway -run TestInboundToolsPrunesDeniedKeepsPrefix", Paths: []string{"internal/gateway/messages_inbound_tools_test.go"},
		Control: "nil/all-allowed tool-floor predicate", Scope: "synthetic Anthropic request fixture",
	},
	"mcptoolfilter": {
		Owner: "internal/gateway", Mechanism: "native MCP tools/list filtering",
		EffectMetric: "exact descriptor bytes removed with held-query recall and capability parity", WitnessKind: "exact-byte A/B",
		Witness: "go test ./internal/gateway -run TestNativeMCPFilterABProof", Paths: []string{"internal/gateway/mcp_filter_ab_test.go"},
		Control: "FAK_ABLATE_MCP_TOOL_FILTER=1", Scope: "native MCP descriptor registry and held intent queries",
	},
	"defercoldtools": {
		Owner: "internal/gateway", Mechanism: "outbound cold-tool deferral",
		EffectMetric: "resident tool-definition count with deterministic recovery; wire bytes may increase", WitnessKind: "resident-definition A/B",
		Witness: "go test ./internal/gateway -run 'TestDeferColdToolsABFiresOverRegistry|TestResidentToolDefsPartition'", Paths: []string{"internal/gateway/tooldefer_export_test.go"},
		Control: "--defer-cold-tools=false", Scope: "canonical registry fixture; this is not an exact wire-byte saving claim",
	},
	"vdso": {
		Owner: "internal/vdso", Mechanism: "repeated tool-call fast path",
		EffectMetric: "prompt tokens and engine calls avoided on one frozen trace", WitnessKind: "same-trace ablation",
		Witness: "fak ablate --sweep vdso --baseline all-off", Paths: []string{"internal/maturity/runtime-proofs.json", "internal/ablate"},
		Control: "all-off arm", Scope: "tau2 airline smoke fixture; run reports the current exact delta",
	},
	"compacthistory": {
		Owner: "internal/gateway", Mechanism: "history compaction",
		EffectMetric: "fak-authored token-equivalent shed net of metadata with provider-cache accounting separated", WitnessKind: "exact fixture proof",
		Witness: "go test ./internal/gateway -run TestFakCompactionShedNetSavingOnClaudeCodePath", Paths: []string{"internal/gateway/messages_compact_test.go"},
		Control: "--compact-history-budget 0", Scope: "Claude Code-shaped request fixture; live firing is in GET /debug/vars",
	},
	"elideresult": {
		Owner: "internal/gateway", Mechanism: "oversized result elision",
		EffectMetric: "exact request-byte reduction while retaining prefix and bounded head/tail evidence", WitnessKind: "exact-byte A/B",
		Witness: "go test ./internal/gateway -run TestMaybeElideOnShrinksKeepsPrefix", Paths: []string{"internal/gateway/messages_elide_test.go"},
		Control: "--elide-result-bytes 0", Scope: "oversized old tool_result fixture",
	},
	"elidestale": {
		Owner: "internal/gateway", Mechanism: "superseded-read elision",
		EffectMetric: "exact request-byte reduction plus verbatim restoration of elided reads", WitnessKind: "round-trip A/B",
		Witness: "go test ./internal/gateway -run TestMaybeElideStaleReadsRoundTrip", Paths: []string{"internal/gateway/messages_elide_stale_test.go"},
		Control: "--elide-stale-reads=false", Scope: "superseded Read/Edit request fixture; live saved tokens are in GET /debug/vars",
	},
	"ctxview": {
		Owner: "internal/agent", Mechanism: "budgeted ctxplan materialized view",
		EffectMetric: "rendered tokens under a fixed budget with verbatim demand-page recovery", WitnessKind: "bounded-view proof",
		Witness: "go test ./internal/agent -run TestCtxSeamRenderTurnPlansViewAndKeepsExactRecall", Paths: []string{"internal/agent/ctxplan_seam_test.go"},
		Control: "disabled seam identity path", Scope: "recorded session fixture; the default ablate smoke trace currently reports no-op, not savings",
	},
	"headroomcompressor": {
		Owner: "internal/headroom", Mechanism: "registered result-compressor family",
		EffectMetric: "matched-quality result bytes removed at the live result-admission seam", WitnessKind: "required matched-quality live-wire A/B",
		Witness: "missing: matched-quality live-wire compressor receipt", Paths: []string{"internal/headroom"},
		Control: "selected noop identity compressor", Scope: "registered family is default-off; native selection remains opt-in and lacks a matched-quality live-wire default proof",
	},
}

func assertion(terms ...string) tokenAssertionSpec { return tokenAssertionSpec{Terms: terms} }

var tokenProofCatalog = map[string]tokenProofBinding{
	"provider_cache": {
		Effect: tokenExecutableProof{Path: "internal/gateway/provider_cache_fak_placement_savings_test.go", Function: "TestFakPlacementUnlocksProviderCacheSavings", Assertions: []tokenAssertionSpec{assertion("offCacheRead"), assertion("savingsUSD")}},
		Lock:   tokenExecutableProof{Path: "internal/gateway/provider_cache_fak_placement_savings_test.go", Function: "TestFakPlacementUnlocksProviderCacheSavings", Assertions: []tokenAssertionSpec{assertion("bytes", "Equal", "reqOff", "orig"), assertion("bytes", "Equal", "reqOn", "orig")}},
	},
	"toolfloor": {
		Effect: tokenExecutableProof{Path: "internal/gateway/messages_inbound_tools_test.go", Function: "TestInboundToolsPrunesDeniedKeepsPrefix", Assertions: []tokenAssertionSpec{assertion("len", "req", "Raw", "orig"), assertion("bytes", "Equal", "orig", "req")}},
		Lock:   tokenExecutableProof{Path: "cmd/fak/token_defaults_test.go", Function: "TestTokenDefault_ToolFloorDefaultsOn", Assertions: []tokenAssertionSpec{assertion("strings", "Contains", "ToolFloorDenies:")}},
	},
	"mcptoolfilter": {
		Effect: tokenExecutableProof{Path: "internal/gateway/mcp_filter_ab_test.go", Function: "TestNativeMCPFilterABProof", Assertions: []tokenAssertionSpec{assertion("receipt", "SavedBytes"), assertion("controlReceipt", "SavedBytes")}},
		Lock:   tokenExecutableProof{Path: "internal/gateway/mcp_filter_ab_test.go", Function: "TestNativeMCPFilterABProof", Assertions: []tokenAssertionSpec{assertion("receipt", "Mode", "active"), assertion("controlReceipt", "Mode", "bypass")}},
		Note:   tokenLiteralProof{Path: "cmd/fak/mcp_filter_proof.go", Function: "runMCPFilterProof", Callee: "fmt.Fprintf", Literals: []string{"tasks", "recall", "first-call routes", "saved %d descriptor bytes"}},
	},
	"defercoldtools": {
		Effect: tokenExecutableProof{Path: "internal/gateway/tooldefer_export_test.go", Function: "TestDeferColdToolsABFiresOverRegistry", Assertions: []tokenAssertionSpec{assertion("Changed"), assertion("len", "Armed", "Ablated")}},
		Lock:   tokenExecutableProof{Path: "cmd/fak/defer_cold_tools_wiring_test.go", Function: "TestDeferColdToolsFlagDeclaredOnBothDoors", Assertions: []tokenAssertionSpec{assertion("DefaultDeferColdTools"), assertion("strings", "Contains", "defer-cold-tools")}},
		Note:   tokenLiteralProof{Path: "cmd/fak/guard.go", Function: "cmdManageCommand", Callee: "fs.Bool", Literals: []string{"defer-cold-tools", "Pass =false to opt out", "nothing goes silently missing"}}, Gate: &deferColdToolsGate,
	},
	"vdso": {
		Effect: tokenExecutableProof{Path: "internal/bench/bench_test.go", Function: "TestRunArm_VDSOAblationChangesPath", Assertions: []tokenAssertionSpec{assertion("on", "VDSOHits"), assertion("off", "VDSOHits")}},
		Lock:   tokenExecutableProof{Path: "cmd/fak/token_defaults_test.go", Function: "TestTokenDefault_VdsoDefaultsOn", Assertions: []tokenAssertionSpec{assertion("strings", "Contains", "vdso"), assertion("regexp", "MatchString")}},
	},
	"compacthistory": {
		Effect: tokenExecutableProof{Path: "internal/gateway/messages_compact_test.go", Function: "TestFakCompactionShedNetSavingOnClaudeCodePath", Assertions: []tokenAssertionSpec{assertion("CompactionShedTokens"), assertion("baseSavingUSD")}},
		Lock:   tokenExecutableProof{Path: "cmd/fak/token_defaults_test.go", Function: "TestTokenDefault_CompactHistoryDefaultsOn", Assertions: []tokenAssertionSpec{assertion("DefaultCompactHistoryBudget"), assertion("strings", "Contains", "compact-history-budget")}},
		Note:   tokenLiteralProof{Path: "cmd/fak/guard.go", Function: "cmdManageCommand", Callee: "fs.Int", Literals: []string{"compact-history-budget", "Pass 0 to disable", "BYTE-IDENTICAL"}},
	},
	"elideresult": {
		Effect: tokenExecutableProof{Path: "internal/gateway/messages_elide_test.go", Function: "TestMaybeElideOnShrinksKeepsPrefix", Assertions: []tokenAssertionSpec{assertion("len", "req", "Raw", "orig"), assertion("bytes", "Equal", "orig", "req")}},
		Lock:   tokenExecutableProof{Path: "cmd/fak/token_defaults_test.go", Function: "TestTokenDefault_ElideDefaultsOn", Assertions: []tokenAssertionSpec{assertion("DefaultElideResultBytes"), assertion("strings", "Contains", "elide-result-bytes")}},
		Note:   tokenLiteralProof{Path: "cmd/fak/guard.go", Function: "cmdManageCommand", Callee: "fs.Int", Literals: []string{"elide-result-bytes", "bounded head+tail", "0 disables"}},
	},
	"elidestale": {
		Effect: tokenExecutableProof{Path: "internal/gateway/messages_elide_stale_test.go", Function: "TestMaybeElideStaleReadsRoundTrip", Assertions: []tokenAssertionSpec{assertion("len", "req", "Raw", "orig"), assertion("got", "Bytes", "big")}},
		Lock:   tokenExecutableProof{Path: "cmd/fak/elide_stale_default_test.go", Function: "TestElideStaleReadsDefaultsOn", Assertions: []tokenAssertionSpec{assertion("DefaultElideStaleReads"), assertion("strings", "Contains", "elide-stale-reads")}},
		Note:   tokenLiteralProof{Path: "cmd/fak/guard.go", Function: "cmdManageCommand", Callee: "fs.Bool", Literals: []string{"elide-stale-reads", "lossy but restorable", "Pass =false to opt out"}},
	},
	"ctxview": {
		Effect: tokenExecutableProof{Path: "internal/agent/ctxplan_seam_test.go", Function: "TestCtxSeamRenderTurnPlansViewAndKeepsExactRecall", Assertions: []tokenAssertionSpec{assertion("tokens", "budget"), assertion("recovered", "orig")}},
		Lock:   tokenExecutableProof{Path: "cmd/fak/token_defaults_test.go", Function: "TestTokenDefault_CtxViewDefaultsOn", Assertions: []tokenAssertionSpec{assertion("DefaultCtxViewBudget"), assertion("strings", "Contains", "ctx-view-budget")}},
		Note:   tokenLiteralProof{Path: "cmd/fak/guard.go", Function: "cmdManageCommand", Callee: "fs.Int", Literals: []string{"ctx-view-budget", "pass 0 to disable", "falls open"}},
	},
	"headroomcompressor": {
		Note: tokenLiteralProof{Path: "internal/headroom/status.go", Function: "Status", Literals: []string{"disabled"}},
	},
}

func buildTokenEffectivenessReport(scorecard map[string]any) tokenEffectivenessReport {
	return buildTokenEffectivenessReportWithSources(scorecard, loadTokenDefaultSources(repoRoot()))
}

func buildTokenEffectivenessReportWithSources(scorecard map[string]any, sources tokenDefaultSources) tokenEffectivenessReport {
	corpus, _ := scorecard["corpus"].(map[string]any)
	status, _ := corpus["lever_status"].([]map[string]any)
	if status == nil {
		if raw, ok := corpus["lever_status"].([]any); ok {
			for _, item := range raw {
				if row, ok := item.(map[string]any); ok {
					status = append(status, row)
				}
			}
		}
	}
	report := tokenEffectivenessReport{Schema: "fak-token-effectiveness/2", OK: true, Note: "configured/default-on is posture, not effectiveness; captured requires executable treatment/control assertions, and committed gates resolve without network state"}
	for _, lever := range status {
		key, _ := lever["key"].(string)
		evidence, mapped := tokenEffectivenessEvidence[key]
		evidence.Key = key
		if on, _ := lever["on"].(bool); on {
			evidence.Default = "on"
			evidence.Configured = "default_on"
		} else {
			evidence.Default = "off"
			evidence.Configured = "default_off"
		}
		binding, proofMapped := tokenProofCatalog[key]
		if binding.Effect.Path != "" {
			evidence.Provenance = append(evidence.Provenance, binding.Effect.Path+"#"+binding.Effect.Function)
		}
		if binding.Lock.Path != "" {
			evidence.Provenance = append(evidence.Provenance, binding.Lock.Path+"#"+binding.Lock.Function)
		}
		if binding.Note.Path != "" {
			evidence.Provenance = append(evidence.Provenance, binding.Note.Path+"#"+binding.Note.Function)
		}
		evidence.Observed = "captured"
		if !mapped || !proofMapped || evidence.EffectMetric == "" || evidence.Witness == "" || evidence.Control == "" || !sources.executableProof(binding.Effect) {
			evidence.Observed = "missing"
		}
		if gateBlocker := tokenGateBlocker(sources, binding.Gate); gateBlocker != "" {
			evidence.Observed = "blocked"
			evidence.Blocker = gateBlocker
			evidence.Provenance = append(evidence.Provenance, binding.Gate.Schema+":"+binding.Gate.ID, binding.Gate.PassArtifact)
		}
		// Off, explicitly gated promotion candidates remain visible without
		// becoming unsafe-default debt. Non-captured default-on methods are HARD.
		on, _ := lever["on"].(bool)
		if evidence.Observed != "captured" && on {
			report.Debt++
			report.OK = false
		}
		report.Rows = append(report.Rows, evidence)
	}
	return report
}

func writeTokenEffectivenessReport(stdout, stderr io.Writer, scorecard map[string]any, asJSON bool) int {
	report := buildTokenEffectivenessReport(scorecard)
	if asJSON {
		if err := writeIndentedJSON(stdout, report); err != nil {
			fmt.Fprintf(stderr, "fak token-defaults-scorecard --effectiveness: encode json: %v\n", err)
			return 1
		}
		return okExit(report.OK)
	}
	fmt.Fprintf(stdout, "token-saving effectiveness: %d methods, witness debt %d\n", len(report.Rows), report.Debt)
	fmt.Fprintln(stdout, "method          configured  observed  witness                 effect/control")
	for _, row := range report.Rows {
		fmt.Fprintf(stdout, "%-15s %-11s %-9s %-23s %s; control: %s\n", row.Key, row.Configured, row.Observed, row.WitnessKind, row.EffectMetric, row.Control)
	}
	fmt.Fprintln(stdout, "\n"+strings.TrimSpace(report.Note))
	return okExit(report.OK)
}
