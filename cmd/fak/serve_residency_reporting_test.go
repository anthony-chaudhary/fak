package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// fak-test:runtime fast est=5ms lane=default
// Estimate only; source-authored for #13775, not measured.
func TestServeActivatedExpertRingPlanTextReportsPolicyNotResidency(t *testing.T) {
	ring := serveActivatedExpertRing{Fit: ggufload.ActivatedExpertFit{DeviceBaseBytes: 63 << 30}, RingBytes: 3 << 30}
	for _, bound := range []int64{0, 12 << 30} {
		got := serveActivatedExpertRingPlanText(ring, ggufload.Q4KLoadOptionEffects{StreamedDenseQ4K: true, StreamedDenseBounded: true, StreamedDenseBytes: bound})
		for _, want := range []string{"planned dense device charge=", "planned device expert-ring capacity=", "actual residency unmeasured", "bounded dense HOST working-set ceiling=" + bytesText(uint64(bound))} {
			if !strings.Contains(got, want) {
				t.Errorf("missing %q in %q", want, got)
			}
		}
		if strings.Contains(got, "dense base device-resident") {
			t.Fatal("plan claims measured residency")
		}
	}
	unbounded := serveActivatedExpertRingPlanText(ring, ggufload.Q4KLoadOptionEffects{StreamedDenseQ4K: true})
	if !strings.Contains(unbounded, "without an explicit host working-set ceiling") || strings.Contains(unbounded, "ceiling=0") {
		t.Fatal(unbounded)
	}
}

// fak-test:runtime fast est=10ms lane=default
// Static reachability control, not a successful load or device execution witness.
func TestServePostLoadReservationReaderIsOnSuccessfulResidentLoadTail(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "serve_load_helpers.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var owner *ast.FuncDecl
	for _, d := range f.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok && fn.Name.Name == "loadResidentQ4KDevice" {
			owner = fn
		}
	}
	if owner == nil {
		t.Fatal("resident load tail missing")
	}
	var load, report token.Pos
	calls := 0
	ast.Inspect(owner.Body, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if name, ok := call.Fun.(*ast.Ident); ok {
				switch name.Name {
				case "loadResidentQ4KProfiledFor":
					load = call.Pos()
				case "serveVulkanPostLoadReservations":
					report = call.Pos()
					calls++
				}
			}
		}
		return true
	})
	if load == 0 || calls != 1 || report <= load {
		t.Fatalf("load=%d report=%d calls=%d", load, report, calls)
	}
}
