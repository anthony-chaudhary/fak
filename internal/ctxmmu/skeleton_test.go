package ctxmmu

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

func TestSkeletonizeGo_Comprehensive(t *testing.T) {
	src := []byte(`// Package sample demonstrates various Go language constructs
// for testing AST skeletonization.
package sample

import (
	"context"
	"fmt"
	"io"
)

// DefaultTimeout is the default duration.
const DefaultTimeout = 30

// GlobalCounter tracks total invocations.
var GlobalCounter int

// Worker defines the contract for background job runners.
type Worker interface {
	// Work executes the assigned task.
	Work(ctx context.Context, id string) (int, error)
	// Stop gracefully halts the worker.
	Stop() error
}

// Config specifies worker parameters.
type Config struct {
	// ID is the unique identifier.
	ID string ` + "`" + `json:"id"` + "`" + `
	// Concurrency is the maximum concurrent workers.
	Concurrency int ` + "`" + `json:"concurrency"` + "`" + `
}

// Service manages active worker instances.
type Service struct {
	cfg Config
	out io.Writer
}

// NewService constructs a new Service instance.
func NewService(cfg Config, out io.Writer) (*Service, error) {
	// Validate configuration
	if cfg.Concurrency <= 0 {
		return nil, fmt.Errorf("invalid concurrency: %d", cfg.Concurrency)
	}
	s := &Service{
		cfg: cfg,
		out: out,
	}
	return s, nil
}

// Run executes the service loop until context cancellation.
func (s *Service) Run(ctx context.Context) error {
	// Internal loop implementation
	for {
		select {
		case <-ctx.Done():
			// Clean shutdown
			return ctx.Err()
		default:
			fmt.Fprintln(s.out, "tick")
		}
	}
}

// Status returns a status string from value receiver.
func (s Service) Status() string {
	// Format the status
	return fmt.Sprintf("id=%s", s.cfg.ID)
}

// ExternalAssemblyStub has no body.
func ExternalAssemblyStub()
`)

	skeleton, err := SkeletonizeGo(src)
	if err != nil {
		t.Fatalf("SkeletonizeGo failed: %v", err)
	}

	skelStr := string(skeleton)

	// 1. Output must be valid parseable Go code.
	fset := token.NewFileSet()
	parsed, parseErr := parser.ParseFile(fset, "skeleton.go", skeleton, parser.ParseComments)
	if parseErr != nil {
		t.Fatalf("Skeletonized output failed to parse as valid Go: %v\nOutput:\n%s", parseErr, skelStr)
	}

	// 2. Package declaration and doc must be preserved.
	if parsed.Name.Name != "sample" {
		t.Errorf("expected package name 'sample', got '%s'", parsed.Name.Name)
	}
	if !strings.Contains(skelStr, "Package sample demonstrates") {
		t.Errorf("expected package docstring to be preserved")
	}

	// 3. Imports must be preserved.
	for _, imp := range []string{`"context"`, `"fmt"`, `"io"`} {
		if !strings.Contains(skelStr, imp) {
			t.Errorf("expected import %s to be preserved in skeleton", imp)
		}
	}

	// 4. Constants and variables must be preserved.
	if !strings.Contains(skelStr, "DefaultTimeout = 30") {
		t.Errorf("expected DefaultTimeout const preserved")
	}
	if !strings.Contains(skelStr, "GlobalCounter int") {
		t.Errorf("expected GlobalCounter var preserved")
	}

	// 5. Types and interfaces must be preserved.
	if !strings.Contains(skelStr, "type Worker interface") {
		t.Errorf("expected Worker interface preserved")
	}
	if !strings.Contains(skelStr, "Work(ctx context.Context, id string) (int, error)") {
		t.Errorf("expected Worker.Work signature preserved")
	}
	if !strings.Contains(skelStr, "type Config struct") {
		t.Errorf("expected Config struct preserved")
	}
	if !strings.Contains(skelStr, "type Service struct") {
		t.Errorf("expected Service struct preserved")
	}

	// 6. Docstrings on functions, types, and methods must be preserved.
	if !strings.Contains(skelStr, "NewService constructs a new Service instance.") {
		t.Errorf("expected NewService docstring preserved")
	}
	if !strings.Contains(skelStr, "Run executes the service loop") {
		t.Errorf("expected Run docstring preserved")
	}
	if !strings.Contains(skelStr, "Status returns a status string") {
		t.Errorf("expected Status docstring preserved")
	}
	if !strings.Contains(skelStr, "Worker defines the contract") {
		t.Errorf("expected Worker docstring preserved")
	}

	// 7. Function bodies must be elided (no implementation logic, no internal comments).
	forbiddenSubstrings := []string{
		"invalid concurrency",
		"Internal loop implementation",
		"Clean shutdown",
		"tick",
		"Format the status",
		"select {",
		"for {",
	}
	for _, sub := range forbiddenSubstrings {
		if strings.Contains(skelStr, sub) {
			t.Errorf("skeleton should NOT contain implementation details: %q found", sub)
		}
	}

	// 8. Verify all functions with bodies in parsed AST now have empty bodies.
	for _, decl := range parsed.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			if fn.Name.Name == "ExternalAssemblyStub" {
				if fn.Body != nil {
					t.Errorf("external stub should still have nil body")
				}
			} else {
				if fn.Body == nil {
					t.Errorf("expected non-nil empty body for %s", fn.Name.Name)
				} else if len(fn.Body.List) != 0 {
					t.Errorf("expected 0 statements in body of %s, got %d", fn.Name.Name, len(fn.Body.List))
				}
			}
		}
	}

	// 9. Size reduction check: skeleton should be significantly smaller than source.
	if len(skeleton) >= len(src) {
		t.Errorf("expected skeleton size (%d) to be strictly smaller than source (%d)", len(skeleton), len(src))
	}
}

func TestSkeletonizeGo_InvalidSyntax(t *testing.T) {
	tests := []struct {
		name string
		src  []byte
	}{
		{
			name: "completely invalid text",
			src:  []byte("not a go source file at all !@#$%^&*()"),
		},
		{
			name: "unclosed brace in function",
			src:  []byte("package foo\nfunc bar() {\n"),
		},
		{
			name: "missing package clause",
			src:  []byte("var x = 10\n"),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out, err := SkeletonizeGo(tc.src)
			if err == nil {
				t.Fatalf("expected error for invalid syntax, got nil error and output:\n%s", string(out))
			}
			if out != nil {
				t.Errorf("expected nil output on error, got %d bytes", len(out))
			}
		})
	}
}

func TestSkeletonizeGo_EmptyBodiesAndClosures(t *testing.T) {
	src := []byte(`package minimal

// AlreadyEmpty has no statements in body.
func AlreadyEmpty() {}

// HasClosure has nested function literals.
func HasClosure() {
	fn := func(a, b int) int {
		return a + b
	}
	_ = fn(1, 2)
}
`)

	skeleton, err := SkeletonizeGo(src)
	if err != nil {
		t.Fatalf("SkeletonizeGo failed: %v", err)
	}

	fset := token.NewFileSet()
	parsed, err := parser.ParseFile(fset, "test.go", skeleton, parser.ParseComments)
	if err != nil {
		t.Fatalf("failed to parse skeleton: %v", err)
	}

	for _, decl := range parsed.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			if fn.Body == nil || len(fn.Body.List) != 0 {
				t.Errorf("function %s body should be empty, got: %v", fn.Name.Name, fn.Body)
			}
		}
	}

	skelStr := string(skeleton)
	if strings.Contains(skelStr, "return a + b") {
		t.Errorf("closure body leaked into skeleton")
	}
}

func TestSkeletonizeGo_StructsAndInterfaces(t *testing.T) {
	src := []byte(`package types

// Reader wraps the basic Read method.
type Reader interface {
	Read(p []byte) (n int, err error)
}

// Closer wraps the basic Close method.
type Closer interface {
	Close() error
}

// ReadCloser is the interface that groups the basic Read and Close methods.
type ReadCloser interface {
	Reader
	Closer
}

// Buffer is a variable-sized buffer of bytes.
type Buffer struct {
	buf      []byte
	off      int
	bootstrap [64]byte
}

// String returns the contents of the unread portion of the buffer.
func (b *Buffer) String() string {
	if b == nil {
		return "<nil>"
	}
	return string(b.buf[b.off:])
}
`)

	skeleton, err := SkeletonizeGo(src)
	if err != nil {
		t.Fatalf("SkeletonizeGo failed: %v", err)
	}

	fset := token.NewFileSet()
	_, err = parser.ParseFile(fset, "types.go", skeleton, parser.ParseComments)
	if err != nil {
		t.Fatalf("failed to parse skeleton: %v", err)
	}

	skelStr := string(skeleton)
	if !strings.Contains(skelStr, "type ReadCloser interface") {
		t.Errorf("ReadCloser interface missing")
	}
	if !strings.Contains(skelStr, "Reader\n\tCloser") && !strings.Contains(skelStr, "Reader") {
		t.Errorf("embedded interfaces missing")
	}
	if strings.Contains(skelStr, "<nil>") {
		t.Errorf("body implementation leaked into skeleton")
	}
}
