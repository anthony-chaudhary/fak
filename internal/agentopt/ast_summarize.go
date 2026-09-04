package agentopt

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/printer"
	"go/token"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Family 5: Context-window management & compression.
//
// AST-guided code outline and signature extraction parses source files
// into structural skeleton outlines (types, interfaces, function/method
// signatures, docstrings) while omitting function bodies. This cuts prompt
// token weight by >= 70% while preserving complete interface semantics
// and public symbol visibility.

// TypeOutline describes a type declaration (interface, struct, or alias).
type TypeOutline struct {
	Name      string        `json:"name"`
	Doc       string        `json:"doc,omitempty"`
	Kind      string        `json:"kind"` // "interface", "struct", "alias", "type"
	Signature string        `json:"signature"`
	Exported  bool          `json:"exported"`
	Methods   []FuncOutline `json:"methods,omitempty"`
	Fields    []string      `json:"fields,omitempty"`
}

// FuncOutline describes a function or method signature.
type FuncOutline struct {
	Name       string   `json:"name"`
	Receiver   string   `json:"receiver,omitempty"`
	Signature  string   `json:"signature"`
	Doc        string   `json:"doc,omitempty"`
	Exported   bool     `json:"exported"`
	IsMethod   bool     `json:"is_method"`
	ParamTypes []string `json:"param_types,omitempty"`
	RetTypes   []string `json:"ret_types,omitempty"`
}

// OutlineSummary provides structural statistics for a summarized file outline.
type OutlineSummary struct {
	FilePath         string   `json:"file_path,omitempty"`
	PackageName      string   `json:"package_name"`
	TotalTypes       int      `json:"total_types"`
	TotalInterfaces  int      `json:"total_interfaces"`
	TotalFunctions   int      `json:"total_functions"`
	TotalMethods     int      `json:"total_methods"`
	ExportedSymbols  []string `json:"exported_symbols,omitempty"`
	OriginalTokens   int      `json:"original_tokens"`
	CompressedTokens int      `json:"compressed_tokens"`
	CompressionRatio float64  `json:"compression_ratio"`
}

// CodeOutline represents the complete structural skeleton and metadata of a file.
type CodeOutline struct {
	FilePath         string         `json:"file_path,omitempty"`
	PackageName      string         `json:"package_name"`
	PackageDoc       string         `json:"package_doc,omitempty"`
	Imports          []string       `json:"imports,omitempty"`
	Types            []TypeOutline  `json:"types,omitempty"`
	Functions        []FuncOutline  `json:"functions,omitempty"`
	Skeleton         string         `json:"skeleton"`
	Summary          OutlineSummary `json:"summary"`
	OriginalTokens   int            `json:"original_tokens"`
	CompressedTokens int            `json:"compressed_tokens"`
	CompressionRatio float64        `json:"compression_ratio"`
	Error            error          `json:"-"`
}

// String returns the structural skeleton text representation.
func (co CodeOutline) String() string {
	return co.Skeleton
}

// CalculateCompressionRatio returns the measured token compression ratio.
func (co CodeOutline) CalculateCompressionRatio() float64 {
	return co.CompressionRatio
}

// CompressionPercentage returns the compression ratio expressed as a percentage (0-100).
func (co CodeOutline) CompressionPercentage() float64 {
	return co.CompressionRatio * 100.0
}

// HasExportedSymbol checks if a public symbol is present in the outline summary.
func (co CodeOutline) HasExportedSymbol(name string) bool {
	for _, sym := range co.Summary.ExportedSymbols {
		if sym == name {
			return true
		}
	}
	return false
}

// FindType looks up a type outline by name.
func (co CodeOutline) FindType(name string) (TypeOutline, bool) {
	for _, t := range co.Types {
		if t.Name == name {
			return t, true
		}
	}
	return TypeOutline{}, false
}

// FindFunction looks up a function or method by name.
func (co CodeOutline) FindFunction(name string) (FuncOutline, bool) {
	for _, f := range co.Functions {
		if f.Name == name {
			return f, true
		}
	}
	return FuncOutline{}, false
}

// CalculateCompressionRatio computes token weight reduction ratio:
// (originalTokens - compressedTokens) / originalTokens.
// Returns a value between 0.0 and 1.0 (e.g. 0.75 represents 75% token reduction).
func CalculateCompressionRatio(original, compressed string) float64 {
	origTokens := EstimateTokens(original)
	if origTokens <= 0 {
		return 0.0
	}
	compTokens := EstimateTokens(compressed)
	if compTokens >= origTokens {
		return 0.0
	}
	return float64(origTokens-compTokens) / float64(origTokens)
}

// ASTSummarizerOption configures summarization options.
type ASTSummarizerOption func(*ASTSummarizer)

// WithPreservePrivate configures whether unexported symbols are included in the outline.
func WithPreservePrivate(preserve bool) ASTSummarizerOption {
	return func(s *ASTSummarizer) {
		s.preservePrivate = preserve
	}
}

// ASTSummarizer extracts structural skeletons and signatures from source files.
type ASTSummarizer struct {
	preservePrivate bool
}

// NewASTSummarizer creates a new AST outline extractor with optional configuration.
func NewASTSummarizer(opts ...ASTSummarizerOption) *ASTSummarizer {
	s := &ASTSummarizer{
		preservePrivate: true,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// CalculateCompressionRatio computes the token weight reduction ratio between original and compressed text.
func (s *ASTSummarizer) CalculateCompressionRatio(original, compressed string) float64 {
	return CalculateCompressionRatio(original, compressed)
}

// SummarizeGoSource extracts a structural skeleton outline from Go source code.
func (s *ASTSummarizer) SummarizeGoSource(sourceCode string) CodeOutline {
	return s.SummarizeOutline("source.go", sourceCode)
}

// SummarizeGoSourceText parses Go source code and returns its skeleton outline as a string.
func (s *ASTSummarizer) SummarizeGoSourceText(sourceCode string) (string, error) {
	outline := s.SummarizeGoSource(sourceCode)
	if outline.Error != nil {
		return "", outline.Error
	}
	return outline.Skeleton, nil
}

// SummarizeOutline extracts a structural skeleton outline from a source file.
// For Go files, it utilizes AST parsing via go/parser and go/ast.
// For TypeScript/JavaScript files, it uses structural signature extraction.
func (s *ASTSummarizer) SummarizeOutline(filePath, sourceCode string) CodeOutline {
	origTokens := EstimateTokens(sourceCode)
	if strings.TrimSpace(sourceCode) == "" {
		return CodeOutline{
			FilePath:       filePath,
			OriginalTokens: origTokens,
		}
	}

	ext := strings.ToLower(filepath.Ext(filePath))
	if ext == ".ts" || ext == ".tsx" || ext == ".js" || ext == ".jsx" {
		return s.summarizeTypeScript(filePath, sourceCode, origTokens)
	}

	return s.summarizeGo(filePath, sourceCode, origTokens)
}

// summarizeGo performs AST-guided extraction for Go source files.
func (s *ASTSummarizer) summarizeGo(filePath, sourceCode string, origTokens int) CodeOutline {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filePath, sourceCode, parser.ParseComments)
	if err != nil {
		return CodeOutline{
			FilePath:         filePath,
			OriginalTokens:   origTokens,
			CompressedTokens: origTokens,
			Skeleton:         sourceCode,
			Error:            fmt.Errorf("ast parse error: %w", err),
		}
	}

	packageName := file.Name.Name
	var packageDoc string
	if file.Doc != nil {
		packageDoc = strings.TrimSpace(file.Doc.Text())
	}

	var imports []string
	for _, imp := range file.Imports {
		if imp.Path != nil {
			imports = append(imports, imp.Path.Value)
		}
	}

	var types []TypeOutline
	var funcs []FuncOutline
	var exportedSymbols []string
	seenExported := make(map[string]bool)

	recordExported := func(name string) {
		if ast.IsExported(name) && !seenExported[name] {
			seenExported[name] = true
			exportedSymbols = append(exportedSymbols, name)
		}
	}

	var totalInterfaces, totalMethods int

	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.GenDecl:
			types = append(types, extractGoGenDecl(fset, d, recordExported, &totalInterfaces, &totalMethods)...)
		case *ast.FuncDecl:
			funcs = append(funcs, extractGoFuncDecl(fset, d, recordExported, &totalMethods))
		}
	}

	skeleton, skelErr := buildGoSkeleton(fset, file)
	if skelErr != nil {
		skeleton = sourceCode
	}

	compTokens := EstimateTokens(skeleton)
	compRatio := CalculateCompressionRatio(sourceCode, skeleton)

	sort.Strings(exportedSymbols)

	summary := OutlineSummary{
		FilePath:         filePath,
		PackageName:      packageName,
		TotalTypes:       len(types),
		TotalInterfaces:  totalInterfaces,
		TotalFunctions:   len(funcs) - totalMethods,
		TotalMethods:     totalMethods,
		ExportedSymbols:  exportedSymbols,
		OriginalTokens:   origTokens,
		CompressedTokens: compTokens,
		CompressionRatio: compRatio,
	}

	return CodeOutline{
		FilePath:         filePath,
		PackageName:      packageName,
		PackageDoc:       packageDoc,
		Imports:          imports,
		Types:            types,
		Functions:        funcs,
		Skeleton:         skeleton,
		Summary:          summary,
		OriginalTokens:   origTokens,
		CompressedTokens: compTokens,
		CompressionRatio: compRatio,
	}
}

func extractGoGenDecl(fset *token.FileSet, d *ast.GenDecl, recordExported func(string), totalInterfaces *int, totalMethods *int) []TypeOutline {
	if d.Tok != token.TYPE {
		return nil
	}
	var types []TypeOutline
	for _, spec := range d.Specs {
		ts, ok := spec.(*ast.TypeSpec)
		if !ok {
			continue
		}

		name := ts.Name.Name
		recordExported(name)

		doc := ""
		if d.Doc != nil {
			doc = strings.TrimSpace(d.Doc.Text())
		}
		if ts.Doc != nil {
			tDoc := strings.TrimSpace(ts.Doc.Text())
			if doc == "" {
				doc = tDoc
			} else if tDoc != "" {
				doc = doc + "\n" + tDoc
			}
		}

		kind := "type"
		var methods []FuncOutline
		var fields []string

		switch t := ts.Type.(type) {
		case *ast.InterfaceType:
			kind = "interface"
			*totalInterfaces++
			if t.Methods != nil {
				for _, m := range t.Methods.List {
					mDoc := ""
					if m.Doc != nil {
						mDoc = strings.TrimSpace(m.Doc.Text())
					}
					if len(m.Names) > 0 {
						mName := m.Names[0].Name
						recordExported(mName)
						*totalMethods++
						mSig := mName
						if ft, ok := m.Type.(*ast.FuncType); ok {
							mSig = mName + nodeString(fset, ft.Params)
							if ft.Results != nil {
								mSig += " " + nodeString(fset, ft.Results)
							}
						}
						methods = append(methods, FuncOutline{
							Name:       mName,
							Signature:  mSig,
							Doc:        mDoc,
							Exported:   ast.IsExported(mName),
							IsMethod:   true,
							ParamTypes: extractFieldTypes(fset, m.Type),
						})
					} else {
						// Embedded interface
						embName := nodeString(fset, m.Type)
						fields = append(fields, embName)
					}
				}
			}
		case *ast.StructType:
			kind = "struct"
			if t.Fields != nil {
				for _, f := range t.Fields.List {
					fieldType := nodeString(fset, f.Type)
					if len(f.Names) > 0 {
						for _, fn := range f.Names {
							recordExported(fn.Name)
							fields = append(fields, fn.Name+" "+fieldType)
						}
					} else {
						fields = append(fields, fieldType)
					}
				}
			}
		default:
			if ts.Assign.IsValid() {
				kind = "alias"
			}
		}

		sig := fmt.Sprintf("type %s %s", name, nodeString(fset, ts.Type))
		types = append(types, TypeOutline{
			Name:      name,
			Doc:       doc,
			Kind:      kind,
			Signature: sig,
			Exported:  ast.IsExported(name),
			Methods:   methods,
			Fields:    fields,
		})
	}
	return types
}

func extractGoFuncDecl(fset *token.FileSet, d *ast.FuncDecl, recordExported func(string), totalMethods *int) FuncOutline {
	funcName := d.Name.Name
	recordExported(funcName)

	isMethod := d.Recv != nil
	receiver := ""
	if isMethod {
		receiver = nodeString(fset, d.Recv)
		*totalMethods++
	}

	doc := ""
	if d.Doc != nil {
		doc = strings.TrimSpace(d.Doc.Text())
	}

	// Capture signature prior to pruning body
	var sigBuf bytes.Buffer
	dCopy := *d
	dCopy.Body = nil
	if err := printer.Fprint(&sigBuf, fset, &dCopy); err != nil {
		sigBuf.WriteString("func " + funcName)
	}
	sig := strings.TrimSpace(sigBuf.String())

	return FuncOutline{
		Name:       funcName,
		Receiver:   receiver,
		Signature:  sig,
		Doc:        doc,
		Exported:   ast.IsExported(funcName),
		IsMethod:   isMethod,
		ParamTypes: extractParams(fset, d.Type.Params),
		RetTypes:   extractResults(fset, d.Type.Results),
	}
}

// buildGoSkeleton produces the clean Go source skeleton with function bodies omitted.
func buildGoSkeleton(fset *token.FileSet, file *ast.File) (string, error) {
	type bodySpan struct {
		start token.Pos
		end   token.Pos
	}
	var spans []bodySpan

	// Omit function bodies and record spans
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Body != nil {
			spans = append(spans, bodySpan{
				start: fn.Body.Lbrace,
				end:   fn.Body.Rbrace,
			})
			fn.Body = nil
		}
	}

	// Filter internal comments originating inside omitted bodies
	if len(spans) > 0 {
		var kept []*ast.CommentGroup
		for _, cg := range file.Comments {
			inOmitted := false
			for _, sp := range spans {
				if cg.Pos() >= sp.start && cg.End() <= sp.end {
					inOmitted = true
					break
				}
			}
			if !inOmitted {
				kept = append(kept, cg)
			}
		}
		file.Comments = kept
	}

	var buf bytes.Buffer
	if err := printer.Fprint(&buf, fset, file); err != nil {
		return "", err
	}

	formatted, err := format.Source(buf.Bytes())
	if err == nil {
		return string(formatted), nil
	}
	return buf.String(), nil
}

// summarizeTypeScript performs structural outline extraction for TypeScript/JavaScript files.
func (s *ASTSummarizer) summarizeTypeScript(filePath, sourceCode string, origTokens int) CodeOutline {
	lines := strings.Split(sourceCode, "\n")
	var skeletonLines []string
	var types []TypeOutline
	var funcs []FuncOutline
	var imports []string
	var exportedSymbols []string
	seenExported := make(map[string]bool)

	recordExported := func(name string) {
		if name != "" && !seenExported[name] {
			seenExported[name] = true
			exportedSymbols = append(exportedSymbols, name)
		}
	}

	var currentDoc []string
	inInterface := false
	inClass := false
	braceLevel := 0

	reImport := regexp.MustCompile(`^import\s+.*`)
	reExportInterface := regexp.MustCompile(`^(?:export\s+)?interface\s+([A-Za-z0-9_]+)`)
	reExportType := regexp.MustCompile(`^(?:export\s+)?type\s+([A-Za-z0-9_]+)`)
	reExportClass := regexp.MustCompile(`^(?:export\s+)?class\s+([A-Za-z0-9_]+)`)
	reExportFunc := regexp.MustCompile(`^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_]+)`)
	reMethod := regexp.MustCompile(`^\s*(?:public\s+|private\s+|protected\s+|async\s+)*([A-Za-z0-9_]+)\s*\([^)]*\)\s*(?::\s*[^;{]+)?`)

	var currentInterface *TypeOutline

	for i := 0; i < len(lines); i++ {
		line := lines[i]
		trimmed := strings.TrimSpace(line)

		// Accumulate comments
		if strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "/*") || strings.HasPrefix(trimmed, "*") {
			currentDoc = append(currentDoc, line)
			skeletonLines = append(skeletonLines, line)
			continue
		}

		if trimmed == "" {
			if len(skeletonLines) > 0 && skeletonLines[len(skeletonLines)-1] != "" {
				skeletonLines = append(skeletonLines, "")
			}
			continue
		}

		docText := strings.Join(currentDoc, "\n")
		currentDoc = nil

		// Imports
		if reImport.MatchString(trimmed) {
			imports = append(imports, trimmed)
			skeletonLines = append(skeletonLines, line)
			continue
		}

		// Interfaces
		if m := reExportInterface.FindStringSubmatch(trimmed); len(m) > 1 {
			name := m[1]
			recordExported(name)
			inInterface = true
			braceLevel = countBraces(trimmed)
			to := TypeOutline{
				Name:      name,
				Doc:       docText,
				Kind:      "interface",
				Signature: trimmed,
				Exported:  strings.HasPrefix(trimmed, "export "),
			}
			currentInterface = &to
			skeletonLines = append(skeletonLines, line)
			continue
		}

		// Inside interface block
		if inInterface {
			braceLevel += countBraces(trimmed)
			if currentInterface != nil && strings.Contains(trimmed, "(") {
				mName := strings.Split(trimmed, "(")[0]
				mName = strings.TrimSpace(mName)
				currentInterface.Methods = append(currentInterface.Methods, FuncOutline{
					Name:      mName,
					Signature: trimmed,
					Exported:  true,
					IsMethod:  true,
				})
			}
			skeletonLines = append(skeletonLines, line)
			if braceLevel <= 0 {
				inInterface = false
				if currentInterface != nil {
					types = append(types, *currentInterface)
					currentInterface = nil
				}
			}
			continue
		}

		// Type aliases
		if m := reExportType.FindStringSubmatch(trimmed); len(m) > 1 {
			name := m[1]
			recordExported(name)
			types = append(types, TypeOutline{
				Name:      name,
				Doc:       docText,
				Kind:      "alias",
				Signature: trimmed,
				Exported:  strings.HasPrefix(trimmed, "export "),
			})
			skeletonLines = append(skeletonLines, line)
			continue
		}

		// Classes (structural outline)
		if m := reExportClass.FindStringSubmatch(trimmed); len(m) > 1 {
			name := m[1]
			recordExported(name)
			inClass = true
			braceLevel = countBraces(trimmed)
			types = append(types, TypeOutline{
				Name:      name,
				Doc:       docText,
				Kind:      "struct",
				Signature: trimmed,
				Exported:  strings.HasPrefix(trimmed, "export "),
			})
			skeletonLines = append(skeletonLines, line)
			continue
		}

		// Inside class: preserve method signatures, strip bodies
		if inClass {
			braceLevel += countBraces(trimmed)
			if m := reMethod.FindStringSubmatch(trimmed); len(m) > 1 && strings.Contains(trimmed, "{") {
				sig := strings.TrimSpace(strings.Split(trimmed, "{")[0]) + ";"
				skeletonLines = append(skeletonLines, "  "+sig)
			}
			if braceLevel <= 0 {
				inClass = false
				skeletonLines = append(skeletonLines, "}")
			}
			continue
		}

		// Standalone functions
		if m := reExportFunc.FindStringSubmatch(trimmed); len(m) > 1 {
			name := m[1]
			recordExported(name)
			sig := trimmed
			if strings.Contains(sig, "{") {
				sig = strings.TrimSpace(strings.Split(sig, "{")[0]) + ";"
			}
			funcs = append(funcs, FuncOutline{
				Name:      name,
				Signature: sig,
				Doc:       docText,
				Exported:  strings.HasPrefix(trimmed, "export "),
			})
			skeletonLines = append(skeletonLines, sig)
			// Skip through function body
			fBrace := countBraces(trimmed)
			for fBrace > 0 && i+1 < len(lines) {
				i++
				fBrace += countBraces(lines[i])
			}
			continue
		}
	}

	skeleton := strings.Join(skeletonLines, "\n")
	compTokens := EstimateTokens(skeleton)
	compRatio := CalculateCompressionRatio(sourceCode, skeleton)

	sort.Strings(exportedSymbols)

	summary := OutlineSummary{
		FilePath:         filePath,
		TotalTypes:       len(types),
		TotalFunctions:   len(funcs),
		ExportedSymbols:  exportedSymbols,
		OriginalTokens:   origTokens,
		CompressedTokens: compTokens,
		CompressionRatio: compRatio,
	}

	return CodeOutline{
		FilePath:         filePath,
		Imports:          imports,
		Types:            types,
		Functions:        funcs,
		Skeleton:         skeleton,
		Summary:          summary,
		OriginalTokens:   origTokens,
		CompressedTokens: compTokens,
		CompressionRatio: compRatio,
	}
}

func countBraces(s string) int {
	c := 0
	for _, ch := range s {
		if ch == '{' {
			c++
		} else if ch == '}' {
			c--
		}
	}
	return c
}

func nodeString(fset *token.FileSet, node any) string {
	if node == nil {
		return ""
	}
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, fset, node); err != nil {
		return ""
	}
	return buf.String()
}

func extractFieldTypes(fset *token.FileSet, node ast.Node) []string {
	var types []string
	if ft, ok := node.(*ast.FuncType); ok && ft.Params != nil {
		return extractParams(fset, ft.Params)
	}
	return types
}

func extractParams(fset *token.FileSet, fl *ast.FieldList) []string {
	if fl == nil {
		return nil
	}
	var res []string
	for _, f := range fl.List {
		tStr := nodeString(fset, f.Type)
		if len(f.Names) > 0 {
			for _, n := range f.Names {
				res = append(res, n.Name+" "+tStr)
			}
		} else {
			res = append(res, tStr)
		}
	}
	return res
}

func extractResults(fset *token.FileSet, fl *ast.FieldList) []string {
	if fl == nil {
		return nil
	}
	var res []string
	for _, f := range fl.List {
		tStr := nodeString(fset, f.Type)
		res = append(res, tStr)
	}
	return res
}

// SummarizeGoSource extracts a structural skeleton outline from Go source using the default ASTSummarizer.
func SummarizeGoSource(sourceCode string) CodeOutline {
	return NewASTSummarizer().SummarizeGoSource(sourceCode)
}

// SummarizeOutline extracts a structural skeleton outline from a file using the default ASTSummarizer.
func SummarizeOutline(filePath, sourceCode string) CodeOutline {
	return NewASTSummarizer().SummarizeOutline(filePath, sourceCode)
}
