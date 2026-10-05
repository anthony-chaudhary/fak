package stepobs

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// kernel_help_domain_drift_test.go — the regression lock that keeps the
// fak_engine_kernel_seconds HELP text honest about the timer_domain vocabulary.
//
// A HELP string that enumerates a closed label set and omits two members is worse
// than no enumeration: an operator reads it as the complete set and concludes a
// series they are looking at is malformed. #TICKET-15 added two Vulkan domains
// (vulkan_performance_query, vulkan_performance_query_unavailable) and the HELP
// text did not learn about them, so this drift went live.
//
// The guard derives the emitible domain set FROM THE PRODUCER SOURCE and asserts
// the rendered HELP names every one of them. A new backend domain that is not
// documented reds here instead of shipping undocumented.
//
// fak-test:runtime fast est=1s — parses a fixed handful of files and renders text.

// hdComputeDir is internal/compute relative to this package.
const hdComputeDir = "../compute"

// hdProducerFiles are the internal/compute files that can mint a TimerDomain.
// Every literal assignment (Event literal field or plain assignment) AND every
// string constant whose name mentions TimerDomain is collected from these.
var hdProducerFiles = []string{"cpuref.go", "cuda.go", "metal.go", "vulkan_kernel_obs.go"}

// hdKnownDomains is the vocabulary the renderer documents. It is asserted to be a
// SUBSET of what the producer source actually mints: the source is the truth about
// what can appear, this list is the claim about what is documented, and the HELP
// assertion below joins them.
var hdKnownDomains = []string{
	"host_monotonic",
	"cuda_event",
	"metal_command_buffer",
	"vulkan_performance_query",
	"vulkan_performance_query_unavailable",
}

// hdEmitibleTimerDomains returns every timer_domain string the producer source can
// put on a computetrace.Event: literal struct-field assignments, literal plain
// assignments, and string constants whose name mentions TimerDomain. It FAILS CLOSED
// when a named producer file is unreadable or unparsable, and when the derived set
// is empty — an empty scan would make the drift assertion vacuous.
func hdEmitibleTimerDomains(t *testing.T) map[string]string {
	t.Helper()
	out := map[string]string{}
	fset := token.NewFileSet()
	for _, name := range hdProducerFiles {
		path := filepath.Join(hdComputeDir, name)
		src, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read producer %s: %v (a missing producer must fail closed)", path, err)
		}
		file, err := parser.ParseFile(fset, path, src, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("parse producer %s: %v", path, err)
		}
		ast.Inspect(file, func(n ast.Node) bool {
			switch node := n.(type) {
			case *ast.KeyValueExpr:
				if id, ok := node.Key.(*ast.Ident); ok && id.Name == "TimerDomain" {
					if lit, ok := node.Value.(*ast.BasicLit); ok && lit.Kind == token.STRING {
						if v, err := strconv.Unquote(lit.Value); err == nil {
							out[v] = name
						}
					}
				}
			case *ast.AssignStmt:
				for i, lhs := range node.Lhs {
					sel, ok := lhs.(*ast.SelectorExpr)
					if !ok || sel.Sel.Name != "TimerDomain" || i >= len(node.Rhs) {
						continue
					}
					if lit, ok := node.Rhs[i].(*ast.BasicLit); ok && lit.Kind == token.STRING {
						if v, err := strconv.Unquote(lit.Value); err == nil {
							out[v] = name
						}
					}
				}
			case *ast.GenDecl:
				if node.Tok != token.CONST {
					return true
				}
				for _, spec := range node.Specs {
					vs, ok := spec.(*ast.ValueSpec)
					if !ok {
						continue
					}
					for i, id := range vs.Names {
						if !strings.Contains(id.Name, "TimerDomain") || i >= len(vs.Values) {
							continue
						}
						if lit, ok := vs.Values[i].(*ast.BasicLit); ok && lit.Kind == token.STRING {
							if v, err := strconv.Unquote(lit.Value); err == nil {
								out[v] = name
							}
						}
					}
				}
			}
			return true
		})
	}
	if len(out) == 0 {
		t.Fatalf("no timer_domain could be derived from %v; the drift assertion would be vacuous",
			hdProducerFiles)
	}
	return out
}

// TestKernelSecondsHelpDocumentsEveryEmitibleTimerDomain is the drift lock: every
// timer_domain the compute layer can put on a kernel event must be named in the
// rendered fak_engine_kernel_seconds HELP text, and every domain the renderer claims
// must actually be mintable. Either direction of drift reds here.
func TestKernelSecondsHelpDocumentsEveryEmitibleTimerDomain(t *testing.T) {
	emitible := hdEmitibleTimerDomains(t)

	var buf bytes.Buffer
	New().WritePrometheus(&buf)
	render := buf.String()

	help := ""
	for _, line := range strings.Split(render, "\n") {
		if strings.HasPrefix(line, "# HELP "+MetricKernelSeconds+" ") {
			help = strings.TrimPrefix(line, "# HELP "+MetricKernelSeconds+" ")
		}
	}
	if help == "" {
		t.Fatalf("rendered exposition carries no # HELP line for %s", MetricKernelSeconds)
	}

	for _, domain := range sortedKeys(emitible) {
		if !strings.Contains(help, domain) {
			t.Fatalf("%s HELP does not document the timer_domain %q that %s can emit; a reader "+
				"consulting the HELP would conclude the series is malformed\nhelp: %s",
				MetricKernelSeconds, domain, emitible[domain], help)
		}
	}
	for _, domain := range hdKnownDomains {
		if _, ok := emitible[domain]; !ok {
			t.Fatalf("the renderer documents timer_domain %q but no producer in %v mints it; "+
				"the documented vocabulary has drifted ahead of the producer", domain, hdProducerFiles)
		}
	}

	// The unavailable marker is only meaningful if it is distinguishable from a
	// measurement, so both Vulkan domains must be present and distinct in the HELP.
	if !strings.Contains(help, "vulkan_performance_query_unavailable") {
		t.Fatal("the Vulkan unavailable timer_domain is undocumented: a consumer cannot tell a " +
			"real kernel that ran without a readable device timer from one that was measured")
	}
}

// sortedKeys returns the map keys in a stable order so a failure lists domains
// deterministically.
func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
