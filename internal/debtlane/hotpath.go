package debtlane

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

// Detector dimensions for hot-path performance debt on critical paths (#12363).
const (
	DimAllocationCopyDebt DetectorDimension = "allocation_copy_debt" // Loop-local heap growth and byte/string copies on critical paths
	DimBlockingCallDebt   DetectorDimension = "blocking_call_debt"   // Filesystem/subprocess I/O, sleep, and mutex acquisition on critical paths
	DimHotpathDebt        DetectorDimension = "hotpath_debt"         // General hot-path performance debt token
)

// Aliases for dimensional lookup compatibility.
const (
	DimHotpathAllocation = DimAllocationCopyDebt
	DimHotpathBlocking   = DimBlockingCallDebt
)

// IsCriticalPath reports whether a lane lies on a production performance critical path.
// Peripheral, stewardship, non-runtime operational surfaces (tools, skills, docs, examples),
// and disconnected leaves unreachable from production roots are excluded to keep noise bounded.
func IsCriticalPath(lane *DebtLane) bool {
	if lane == nil {
		return false
	}
	// Only core and enabling criticalities qualify for performance critical paths.
	if lane.Criticality != CriticalityCore && lane.Criticality != CriticalityEnabling {
		return false
	}
	// Non-Go or off-spine operational surfaces are excluded.
	surface := classifySurface(lane.UnitOfWork)
	if surface != SurfaceInternal && surface != SurfacePkg && surface != SurfacePlatform {
		return false
	}
	// If evidence recorded reachability from production roots, disconnected leaves are off-spine.
	if lane.Evidence.HasCode && !lane.Evidence.Integrated {
		return false
	}
	return true
}

// InspectHotPathDebt scans non-test Go files in unitDir for allocation, copy,
// and blocking-call performance debt when lane is on a performance critical path.
// It reports static suspicion with exact source spans, not measured latency.
func InspectHotPathDebt(lane *DebtLane, unitDir string, surface SurfaceClass) []FindingProvenance {
	if !IsCriticalPath(lane) || unitDir == "" {
		return nil
	}

	entries, err := os.ReadDir(unitDir)
	if err != nil {
		return nil
	}

	var findings []FindingProvenance
	fset := token.NewFileSet()

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		path := filepath.Join(unitDir, e.Name())
		relPath := filepathToSlash(filepath.Join(lane.UnitOfWork, e.Name()))

		fileNode, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			continue
		}

		fileFindings := inspectFileHotPathAST(lane, relPath, fileNode, fset, surface)
		findings = append(findings, fileFindings...)
	}

	return findings
}

func inspectFileHotPathAST(lane *DebtLane, relPath string, fileNode *ast.File, fset *token.FileSet, surface SurfaceClass) []FindingProvenance {
	var findings []FindingProvenance
	var stack []ast.Node

	ast.Inspect(fileNode, func(n ast.Node) bool {
		if n == nil {
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
			return true
		}
		stack = append(stack, n)

		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}

		pos := fset.Position(call.Pos())
		span := fmt.Sprintf("%s:%d:%d", relPath, pos.Line, pos.Column)

		// 1. Loop-local heap growth: make / append inside loop body
		if isInLoopBody(stack) {
			if ident, ok := call.Fun.(*ast.Ident); ok {
				switch ident.Name {
				case "make":
					findings = append(findings, FindingProvenance{
						Dimension: string(DimAllocationCopyDebt),
						Surface:   string(surface),
						Lane:      lane.Lane,
						Path:      span,
						Severity:  "warning",
						Message:   fmt.Sprintf("hot-path allocation (static suspicion): loop-local heap growth via make at %s", span),
					})
				case "append":
					findings = append(findings, FindingProvenance{
						Dimension: string(DimAllocationCopyDebt),
						Surface:   string(surface),
						Lane:      lane.Lane,
						Path:      span,
						Severity:  "warning",
						Message:   fmt.Sprintf("hot-path allocation (static suspicion): loop-local heap growth via append at %s", span),
					})
				}
			}
		}

		// 2. Byte / string copies: string([]byte) or []byte(string) conversions
		if isByteStringConversion(call) {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimAllocationCopyDebt),
				Surface:   string(surface),
				Lane:      lane.Lane,
				Path:      span,
				Severity:  "warning",
				Message:   fmt.Sprintf("hot-path copy (static suspicion): byte/string conversion creates heap copy at %s", span),
			})
		}

		// 3. Filesystem or subprocess I/O
		if isIO, desc := isFilesystemOrSubprocessCall(call); isIO {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimBlockingCallDebt),
				Surface:   string(surface),
				Lane:      lane.Lane,
				Path:      span,
				Severity:  "warning",
				Message:   fmt.Sprintf("hot-path blocking call (static suspicion): filesystem or subprocess I/O (%s) on critical path at %s", desc, span),
			})
		}

		// 4. Sleep
		if isTimeSleep(call) {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimBlockingCallDebt),
				Surface:   string(surface),
				Lane:      lane.Lane,
				Path:      span,
				Severity:  "warning",
				Message:   fmt.Sprintf("hot-path blocking call (static suspicion): time.Sleep on critical path at %s", span),
			})
		}

		// 5. Mutex acquisition (coarse locks)
		if isLock, op := isMutexAcquisition(call); isLock {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimBlockingCallDebt),
				Surface:   string(surface),
				Lane:      lane.Lane,
				Path:      span,
				Severity:  "warning",
				Message:   fmt.Sprintf("hot-path blocking call (static suspicion): mutex acquisition (%s) on critical path at %s", op, span),
			})
		}

		return true
	})

	return findings
}

func isInLoopBody(stack []ast.Node) bool {
	for i := len(stack) - 1; i >= 1; i-- {
		switch parent := stack[i-1].(type) {
		case *ast.ForStmt:
			if parent.Body == stack[i] {
				return true
			}
		case *ast.RangeStmt:
			if parent.Body == stack[i] {
				return true
			}
		}
	}
	return false
}

func isByteStringConversion(call *ast.CallExpr) bool {
	if len(call.Args) != 1 {
		return false
	}
	// Case A: string(b)
	if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "string" {
		// Ignore string("literal")
		if lit, ok := call.Args[0].(*ast.BasicLit); ok && lit.Kind == token.STRING {
			return false
		}
		return true
	}
	// Case B: []byte(s)
	if arrType, ok := call.Fun.(*ast.ArrayType); ok && arrType.Len == nil {
		if eltIdent, ok := arrType.Elt.(*ast.Ident); ok {
			if eltIdent.Name == "byte" || eltIdent.Name == "uint8" {
				return true
			}
		}
	}
	return false
}

func isFilesystemOrSubprocessCall(call *ast.CallExpr) (bool, string) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false, ""
	}

	if xIdent, ok := sel.X.(*ast.Ident); ok {
		switch xIdent.Name {
		case "os":
			switch sel.Sel.Name {
			case "Open", "OpenFile", "ReadFile", "WriteFile", "Create", "Remove",
				"RemoveAll", "Mkdir", "MkdirAll", "Stat", "Lstat", "ReadDir", "Truncate":
				return true, fmt.Sprintf("os.%s", sel.Sel.Name)
			}
		case "ioutil":
			switch sel.Sel.Name {
			case "ReadFile", "WriteFile", "ReadAll", "ReadDir":
				return true, fmt.Sprintf("ioutil.%s", sel.Sel.Name)
			}
		case "io":
			if sel.Sel.Name == "ReadAll" {
				return true, "io.ReadAll"
			}
		case "exec":
			switch sel.Sel.Name {
			case "Command", "CommandContext", "LookPath":
				return true, fmt.Sprintf("exec.%s", sel.Sel.Name)
			}
		}
	}

	if sel.Sel.Name == "CombinedOutput" {
		return true, "exec.CombinedOutput"
	}

	return false, ""
}

func isTimeSleep(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	xIdent, ok := sel.X.(*ast.Ident)
	if !ok {
		return false
	}
	return xIdent.Name == "time" && sel.Sel.Name == "Sleep"
}

func isMutexAcquisition(call *ast.CallExpr) (bool, string) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false, ""
	}
	if sel.Sel.Name == "Lock" || sel.Sel.Name == "RLock" {
		return true, sel.Sel.Name
	}
	return false, ""
}
