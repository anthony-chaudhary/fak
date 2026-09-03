package agentopt

import (
	"strings"
	"testing"
)

const sampleGoFile = `package mathutil

import (
	"fmt"
)

// Add calculates the sum of two integers.
func Add(a, b int) int {
	result := a + b // sum_impl
	return result
}

// Multiply calculates the product of two integers.
func Multiply(a, b int) int {
	product := a * b // mult_impl
	return product
}

// Power raises base to exponent power.
func Power(base, exp int) int {
	acc := 1 // pow_impl
	for i := 0; i < exp; i++ {
		acc *= base
	}
	return acc
}
`

// TestDemandDemandPageReader proves lazy loading and correct page fault handling.
func TestDemandPagedContextLoader(t *testing.T) {
	// Bounded capacity of 2 pages
	loader := NewPagedFileLoader(2)

	if loader.Capacity() != 2 {
		t.Fatalf("expected capacity 2, got %d", loader.Capacity())
	}

	// 1. Load outline: parses file structure without loading full bodies
	slices, err := loader.LoadOutline("mathutil.go", sampleGoFile)
	if err != nil {
		t.Fatalf("unexpected LoadOutline error: %v", err)
	}

	if len(slices) != 4 {
		t.Fatalf("expected 4 slices (preamble + 3 funcs), got %d", len(slices))
	}

	// Verify all slices are initially un-loaded (lazy loading invariant)
	for i, s := range slices {
		if s.IsLoaded {
			t.Fatalf("slice %d (%s) should not be loaded initially", i, s.Signature)
		}
		if s.Content != "" {
			t.Fatalf("slice %d should have empty content before being paged in", i)
		}
		if s.LineCount() <= 0 {
			t.Fatalf("slice %d invalid line count: %d", i, s.LineCount())
		}
	}

	// Initial page fault count must be 0
	if loader.PageFaultCount() != 0 {
		t.Fatalf("expected 0 page faults initially, got %d", loader.PageFaultCount())
	}

	// Context should show structural outline without function implementation bodies
	initialCtx := loader.GetLoadedOutline()
	if strings.Contains(initialCtx, "sum_impl") {
		t.Fatalf("initial context should not contain Add body (sum_impl)")
	}
	if strings.Contains(initialCtx, "mult_impl") {
		t.Fatalf("initial context should not contain Multiply body (mult_impl)")
	}
	if strings.Contains(initialCtx, "pow_impl") {
		t.Fatalf("initial context should not contain Power body (pow_impl)")
	}
	if !strings.Contains(initialCtx, "[PAGED OUT]") {
		t.Fatalf("initial context should mark pages as [PAGED OUT]")
	}
	if !strings.Contains(initialCtx, "func Add") {
		t.Fatalf("initial context should include Add signature in structural outline")
	}

	// 2. Request Page 1 (func Add): triggers first Page Fault
	content1, err := loader.RequestPage(1)
	if err != nil {
		t.Fatalf("RequestPage(1) failed: %v", err)
	}
	if !strings.Contains(content1, "sum_impl") {
		t.Fatalf("expected page 1 to contain sum_impl, got %s", content1)
	}
	if loader.PageFaultCount() != 1 {
		t.Fatalf("expected PageFaultCount=1 after first request, got %d", loader.PageFaultCount())
	}

	loaded := loader.LoadedPages()
	if len(loaded) != 1 || loaded[0] != 1 {
		t.Fatalf("expected LoadedPages=[1], got %v", loaded)
	}

	// Context should now display Page 1 body while Pages 0, 2, 3 remain outlines
	ctxAfterPage1 := loader.GetLoadedOutline()
	if !strings.Contains(ctxAfterPage1, "sum_impl") {
		t.Fatalf("context should now include sum_impl")
	}
	if strings.Contains(ctxAfterPage1, "mult_impl") {
		t.Fatalf("context should still omit mult_impl")
	}

	// 3. Buffer Hit: Request Page 1 again (should NOT increment page faults)
	content1Repeat, err := loader.RequestPage(1)
	if err != nil {
		t.Fatalf("repeat RequestPage(1) failed: %v", err)
	}
	if content1Repeat != content1 {
		t.Fatalf("repeat content did not match original content")
	}
	if loader.PageFaultCount() != 1 {
		t.Fatalf("expected PageFaultCount to remain 1 on buffer hit, got %d", loader.PageFaultCount())
	}

	// 4. Request Page 2 (func Multiply): triggers second Page Fault
	content2, err := loader.RequestPage(2)
	if err != nil {
		t.Fatalf("RequestPage(2) failed: %v", err)
	}
	if !strings.Contains(content2, "mult_impl") {
		t.Fatalf("expected page 2 to contain mult_impl, got %s", content2)
	}
	if loader.PageFaultCount() != 2 {
		t.Fatalf("expected PageFaultCount=2, got %d", loader.PageFaultCount())
	}

	// Both pages 1 and 2 should be resident (capacity is 2)
	loaded = loader.LoadedPages()
	if len(loaded) != 2 || loaded[0] != 1 || loaded[1] != 2 {
		t.Fatalf("expected LoadedPages=[1, 2], got %v", loaded)
	}

	// 5. Pruneion on Capacity Overflow:
	// Request Page 3 (func Power).
	// Page 1 was accessed twice (steps 2 and 3), Page 2 was accessed once (step 4).
	// Page 2 has lower access frequency, so Page 2 must be pruneed.
	content3, err := loader.RequestPage(3)
	if err != nil {
		t.Fatalf("RequestPage(3) failed: %v", err)
	}
	if !strings.Contains(content3, "pow_impl") {
		t.Fatalf("expected page 3 to contain pow_impl, got %s", content3)
	}
	if loader.PageFaultCount() != 3 {
		t.Fatalf("expected PageFaultCount=3 after pruneion, got %d", loader.PageFaultCount())
	}

	// Resident pages should now be [1, 3] (frequently referenced Page 1 retained)
	loaded = loader.LoadedPages()
	if len(loaded) != 2 || loaded[0] != 1 || loaded[1] != 3 {
		t.Fatalf("expected LoadedPages=[1, 3] after pruneing Page 2, got %v", loaded)
	}

	ctxAfterPage3 := loader.GetLoadedOutline()
	if !strings.Contains(ctxAfterPage3, "sum_impl") {
		t.Fatalf("frequently referenced Page 1 (sum_impl) should remain resident")
	}
	if !strings.Contains(ctxAfterPage3, "pow_impl") {
		t.Fatalf("newly paged Page 3 (pow_impl) should be resident")
	}
	if strings.Contains(ctxAfterPage3, "mult_impl") {
		t.Fatalf("pruneed Page 2 (mult_impl) should no longer be resident in context")
	}

	// 6. Refaulting Pruneed Page:
	// Requesting Page 2 again triggers Page Fault 4
	content2Refault, err := loader.RequestPage(2)
	if err != nil {
		t.Fatalf("refault RequestPage(2) failed: %v", err)
	}
	if !strings.Contains(content2Refault, "mult_impl") {
		t.Fatalf("expected refaulted content to contain mult_impl")
	}
	if loader.PageFaultCount() != 4 {
		t.Fatalf("expected PageFaultCount=4 on refault, got %d", loader.PageFaultCount())
	}

	// Pruneion between Page 1 (access count 2) and Page 3 (access count 1):
	// Page 3 is pruneed, so resident set is [1, 2]
	loaded = loader.LoadedPages()
	if len(loaded) != 2 || loaded[0] != 1 || loaded[1] != 2 {
		t.Fatalf("expected LoadedPages=[1, 2], got %v", loaded)
	}

	// 7. Fault history verification
	history := loader.FaultHistory()
	if len(history) != 4 {
		t.Fatalf("expected 4 recorded faults, got %d", len(history))
	}
	if history[0].PageIndex != 1 || history[0].PrunedPage != -1 {
		t.Fatalf("fault 0 mismatch: %+v", history[0])
	}
	if history[1].PageIndex != 2 || history[1].PrunedPage != -1 {
		t.Fatalf("fault 1 mismatch: %+v", history[1])
	}
	if history[2].PageIndex != 3 || history[2].PrunedPage != 2 {
		t.Fatalf("fault 2 expected pruneion of page 2, got: %+v", history[2])
	}
	if history[3].PageIndex != 2 || history[3].PrunedPage != 3 {
		t.Fatalf("fault 3 expected pruneion of page 3, got: %+v", history[3])
	}

	// 8. Bounds checking and error paths
	if _, err := loader.RequestPage(-1); err == nil {
		t.Fatalf("expected error for page index -1")
	}
	if _, err := loader.RequestPage(999); err == nil {
		t.Fatalf("expected error for page index 999 out of bounds")
	}
}

func TestDemandDemandPageReader_InterfaceCompliance(t *testing.T) {
	var loader DemandPageReader = NewDemandPageReader(3)

	slices, err := loader.LoadOutline("test.go", sampleGoFile)
	if err != nil {
		t.Fatalf("LoadOutline via interface failed: %v", err)
	}
	if len(slices) == 0 {
		t.Fatalf("expected non-empty slices via interface")
	}

	body, err := loader.RequestPage(0)
	if err != nil {
		t.Fatalf("RequestPage via interface failed: %v", err)
	}
	if !strings.Contains(body, "package mathutil") {
		t.Fatalf("expected package preamble, got: %s", body)
	}

	if loader.PageFaultCount() != 1 {
		t.Fatalf("expected 1 page fault, got %d", loader.PageFaultCount())
	}

	ctx := loader.GetLoadedOutline()
	if !strings.Contains(ctx, "package mathutil") {
		t.Fatalf("loaded context missing package declaration")
	}
}

func TestDemandDemandPageReader_PythonAndMarkdown(t *testing.T) {
	pyCode := `import math

def calculate(x):
    # calculate square root
    return math.sqrt(x)

class Transformer:
    def __init__(self, dim):
        self.dim = dim
`
	pyLoader := NewPagedFileLoader(2)
	pySlices, err := pyLoader.LoadOutline("math_ops.py", pyCode)
	if err != nil {
		t.Fatalf("python LoadOutline failed: %v", err)
	}
	if len(pySlices) != 3 {
		t.Fatalf("expected 3 python slices (imports, def, class), got %d", len(pySlices))
	}

	pyContent, err := pyLoader.RequestPage(1)
	if err != nil {
		t.Fatalf("python RequestPage(1) failed: %v", err)
	}
	if !strings.Contains(pyContent, "calculate square root") {
		t.Fatalf("expected python function body, got: %s", pyContent)
	}

	mdContent := `# Introduction
This is the intro.

## Section 1
First section details.

## Section 2
Second section details.
`
	mdLoader := NewPagedFileLoader(2)
	mdSlices, err := mdLoader.LoadOutline("doc.md", mdContent)
	if err != nil {
		t.Fatalf("markdown LoadOutline failed: %v", err)
	}
	if len(mdSlices) != 3 {
		t.Fatalf("expected 3 markdown sections, got %d", len(mdSlices))
	}

	sec1, err := mdLoader.RequestPage(1)
	if err != nil {
		t.Fatalf("markdown RequestPage(1) failed: %v", err)
	}
	if !strings.Contains(sec1, "First section details") {
		t.Fatalf("expected markdown section 1 body, got: %s", sec1)
	}
}

func TestDemandDemandPageReader_ChunkedFallback(t *testing.T) {
	var builder strings.Builder
	for i := 1; i <= 100; i++ {
		builder.WriteString("line item data\n")
	}

	loader := NewPagedFileLoader(2)
	slices, err := loader.LoadOutline("data.txt", builder.String())
	if err != nil {
		t.Fatalf("LoadOutline on text file failed: %v", err)
	}

	// 100 lines chunked by 40 -> 3 slices (40, 40, 20 lines)
	if len(slices) != 3 {
		t.Fatalf("expected 3 chunked slices, got %d", len(slices))
	}

	if slices[0].StartLine != 1 || slices[0].EndLine != 40 {
		t.Fatalf("slice 0 lines mismatch: %d-%d", slices[0].StartLine, slices[0].EndLine)
	}
	if slices[1].StartLine != 41 || slices[1].EndLine != 80 {
		t.Fatalf("slice 1 lines mismatch: %d-%d", slices[1].StartLine, slices[1].EndLine)
	}
	if slices[2].StartLine != 81 || slices[2].EndLine != 100 {
		t.Fatalf("slice 2 lines mismatch: %d-%d", slices[2].StartLine, slices[2].EndLine)
	}
}

func TestDemandDemandPageReader_ValidationErrors(t *testing.T) {
	loader := NewPagedFileLoader(2)

	// Request page before outline loaded
	if _, err := loader.RequestPage(0); err == nil {
		t.Fatalf("expected error requesting page before outline is loaded")
	}

	// Empty content
	if _, err := loader.LoadOutline("", ""); err == nil {
		t.Fatalf("expected error loading empty content and path")
	}
}
