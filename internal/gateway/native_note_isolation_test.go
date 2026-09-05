package gateway

// native_note_isolation_test.go — #2414's remaining witness: the prose adjudicationNote
// machinery is proxy-ONLY by STRUCTURE, not merely unobserved in one scenario.
//
// TestDeniedCallReturnsStructuredError (native_honest_receipt_test.go) proves a denied
// native call comes back as a typed tool_result receipt and that no "[fak] ..." splice
// appears in that run. But a dynamic absence check only covers the path that one script
// happens to walk — the issue's own confusion risk ("verifying the prose adjudicationNote
// is genuinely absent on the native path, not merely unused in the test's specific
// scenario"). A later edit could splice a note into a native branch the script never
// reaches and every dynamic witness would stay green.
//
// So this witness is STATIC: it builds the package-local call graph from the source and
// asserts the prose-note helpers are UNREACHABLE from the owned loop's two wire entry
// points. The graph is deliberately over-approximated (a call is an edge whenever its
// callee name matches any package-local func or method name, regardless of receiver
// type), so it can only ever report a path that does not exist — never miss one that
// does. That is the safe direction for a "provably absent" claim.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"sort"
	"strings"
	"testing"
)

// nativeEntryPoints are the owned loop's wire entry points — EVERY served path on which fak
// owns the transcript, which is what makes the absence claim structural rather than
// per-scenario. The first two take a native /v1/messages turn (internal/gateway/messages.go's
// `if s.native` branch returns immediately after calling them). The third is the NDJSON
// agent-sessions wire (POST /v1/fak/agent/sessions), which drives agent.RunGovernedArm — the
// same owned loop under the kernel-governed arm. It is listed here because "the native path"
// in #2414 means the owned loop, not one URL: a prose note spliced into the agent-sessions
// handler would splice it into a transcript fak authors just as much as a /v1/messages one,
// and guarding only the /v1/messages pair would let exactly that regression land green.
var nativeEntryPoints = []string{"serveNativeMessages", "serveNativeMessagesStream", "handleFakAgentSessions"}

// proseNoteHelpers is the prose-splice machinery that the wire forbids fak from authoring
// as a real tool_result, so the proxy path folds it into the assistant's own voice. In the
// owned loop fak controls the transcript and emits a typed agent.ToolReceipt instead, so
// none of these may be reachable from a native entry point.
var proseNoteHelpers = []string{
	"adjudicationNote",
	"prependAdjudicationContentNote",
	"denySummary",
	"livelockInBandNote",
	"renderRefusalNotes",
	"prependTextBlock",
}

// callGraph maps a package-local func/method name to the set of names it calls. Methods are
// keyed by method name alone: without type checking a receiver cannot be resolved, and
// merging same-named methods over-approximates reachability, which is the conservative
// direction for an unreachability assertion.
type callGraph map[string]map[string]bool

// buildGatewayCallGraph parses the non-test sources of this package and records one edge per
// call expression. Selector calls qualified by an imported package name (agent.Foo) are
// skipped so an import's method name cannot forge a package-local edge.
func buildGatewayCallGraph(t *testing.T) callGraph {
	t.Helper()
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", func(fi fs.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, 0)
	if err != nil {
		t.Fatalf("parse package sources: %v", err)
	}
	pkg, ok := pkgs["gateway"]
	if !ok {
		t.Fatalf("package gateway not found in parsed sources")
	}

	// Pass 1: every package-local func/method name, so pass 2 can tell a local call from a
	// call into an imported package or a struct field.
	local := map[string]bool{}
	for _, file := range pkg.Files {
		for _, decl := range file.Decls {
			if fn, ok := decl.(*ast.FuncDecl); ok {
				local[fn.Name.Name] = true
			}
		}
	}

	// Pass 2: edges. Function literals nest inside a decl's body, so ast.Inspect attributes
	// a closure's calls to the enclosing func — correct here, since the closure is reachable
	// exactly when its enclosing func runs.
	g := callGraph{}
	for _, file := range pkg.Files {
		imports := importNames(file)
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			if g[fn.Name.Name] == nil {
				g[fn.Name.Name] = map[string]bool{}
			}
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				switch callee := call.Fun.(type) {
				case *ast.Ident:
					if local[callee.Name] {
						g[fn.Name.Name][callee.Name] = true
					}
				case *ast.SelectorExpr:
					// pkg.Foo() is a call into an import, never a package-local func.
					if x, ok := callee.X.(*ast.Ident); ok && imports[x.Name] {
						return true
					}
					if local[callee.Sel.Name] {
						g[fn.Name.Name][callee.Sel.Name] = true
					}
				}
				return true
			})
		}
	}
	return g
}

func importNames(file *ast.File) map[string]bool {
	names := map[string]bool{}
	for _, imp := range file.Imports {
		name := ""
		if imp.Name != nil {
			name = imp.Name.Name
		} else {
			path := strings.Trim(imp.Path.Value, `"`)
			if i := strings.LastIndex(path, "/"); i >= 0 {
				name = path[i+1:]
			} else {
				name = path
			}
		}
		names[name] = true
	}
	return names
}

// sortedCallees returns a func's callees in a stable order, so a red path is deterministic
// across runs rather than reshuffling with Go's map iteration.
func sortedCallees(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// reachPath returns the call chain from start to target, or nil when target is unreachable.
// Breadth-first, so the reported chain is a shortest one — the most readable witness when
// this test reds.
func reachPath(g callGraph, start, target string) []string {
	if start == target {
		return []string{start}
	}
	prev := map[string]string{start: ""}
	queue := []string{start}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, next := range sortedCallees(g[cur]) {
			if _, seen := prev[next]; seen {
				continue
			}
			prev[next] = cur
			if next == target {
				chain := []string{next}
				for at := cur; at != ""; at = prev[at] {
					chain = append([]string{at}, chain...)
				}
				return chain
			}
			queue = append(queue, next)
		}
	}
	return nil
}

// TestNativePathHasNoProseAdjudicationNote is the structural half of #2414's "the prose
// adjudicationNote is provably absent on the native path": no prose-note helper is
// reachable from either owned-loop entry point, so the absence holds for EVERY native run,
// not just the scripted one.
func TestNativePathHasNoProseAdjudicationNote(t *testing.T) {
	g := buildGatewayCallGraph(t)

	for _, entry := range nativeEntryPoints {
		callees, ok := g[entry]
		if !ok {
			t.Fatalf("native entry point %q not found in the package call graph — "+
				"the owned loop was renamed and this witness no longer guards it", entry)
		}
		// Non-vacuity: a node with NO outgoing edges is unreachable-from by construction, so
		// every assertion below would pass without proving anything. Each owned-loop entry
		// point demonstrably calls package-local helpers, so an empty callee set means the
		// graph builder failed to read this body rather than that the path is clean.
		if len(callees) == 0 {
			t.Fatalf("native entry point %q has no package-local callees — buildGatewayCallGraph "+
				"did not read its body, so the unreachability assertions below are vacuous", entry)
		}
		for _, helper := range proseNoteHelpers {
			if path := reachPath(g, entry, helper); path != nil {
				t.Errorf("owned loop reaches the proxy-only prose note helper %q:\n\t%s\n"+
					"the native path must author a typed agent.ToolReceipt on the originating "+
					"call ID (#2414), never splice a \"[fak] ...\" note into the model's own voice",
					helper, strings.Join(path, " -> "))
			}
		}
	}
}

// TestProxyPathStillReachesAdjudicationNote keeps the assertion above HONEST. An
// unreachability claim passes vacuously if the graph builder is broken (a typo in an entry
// name, a parse that silently dropped files, an edge rule that records nothing). The proxy
// buffered turn provably DOES splice the note — messages.go's completeAnthropicTurn calls
// adjudicationNote directly — so a graph that cannot see that edge cannot be trusted to
// prove absence anywhere. This also pins the demotion the issue asks for: the notes
// machinery stays alive as the PROXY-only compatibility shim rather than being deleted.
func TestProxyPathStillReachesAdjudicationNote(t *testing.T) {
	g := buildGatewayCallGraph(t)
	if path := reachPath(g, "completeAnthropicTurn", "adjudicationNote"); path == nil {
		t.Fatalf("the proxy buffered turn no longer reaches adjudicationNote — either the " +
			"compatibility shim was dropped or buildGatewayCallGraph is broken, which would " +
			"make TestNativePathHasNoProseAdjudicationNote pass vacuously")
	}
}

// TestNativeNoteIsolationRefusalNotesNoRunnableGuardCommands asserts that renderRefusalNotes
// and deniedToolResult for DEFAULT_DENY or POLICY_BLOCK contain zero runnable "fak guard allow"
// shell strings in agent-visible text, while operator remediation is preserved in the header or metadata (#11504).
func TestNativeNoteIsolationRefusalNotesNoRunnableGuardCommands(t *testing.T) {
	cases := []struct {
		reason string
		tool   string
	}{
		{"DEFAULT_DENY", "exec_command"},
		{"POLICY_BLOCK", "rm_rf"},
	}
	for _, tc := range cases {
		adj := ToolAdjudication{
			Tool:     tc.tool,
			Admitted: false,
			Verdict: WireVerdict{
				Kind:        "DENY",
				Reason:      tc.reason,
				Disposition: "TERMINAL",
			},
		}

		notes, _ := renderRefusalNotes(adj)
		if strings.Contains(notes, "fak guard allow") {
			t.Errorf("renderRefusalNotes(%s) contains runnable 'fak guard allow': %q", tc.reason, notes)
		}
		if strings.Contains(notes, "`fak guard") {
			t.Errorf("renderRefusalNotes(%s) contains backtick 'fak guard': %q", tc.reason, notes)
		}

		result := deniedToolResult(adj)
		if strings.Contains(result, "fak guard allow") {
			t.Errorf("deniedToolResult(%s) contains runnable 'fak guard allow': %q", tc.reason, result)
		}
		if strings.Contains(result, "`fak guard") {
			t.Errorf("deniedToolResult(%s) contains backtick 'fak guard': %q", tc.reason, result)
		}
	}
}
