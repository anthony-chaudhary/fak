package agentopt

import (
	"testing"
)

// expectedFunctionSpan defines known line spans for functions and methods in test sources.
type expectedFunctionSpan struct {
	name      string
	startLine int
	endLine   int
	isMethod  bool
	scope     string
}

func TestSyntaxBoundaryChunking(t *testing.T) {
	chunker := NewSyntaxChunker()

	t.Run("GoSourceZeroSplitFunctions", func(t *testing.T) {
		goSource := `package samplepkg

import (
	"fmt"
	"strings"
)

// Config represents service settings.
type Config struct {
	Host string
	Port int
}

// Validator validates inputs.
type Validator interface {
	Validate(input string) error
}

// FormatStatus aliases status strings.
type FormatStatus string

// NewConfig constructs a default config.
func NewConfig(host string, port int) *Config {
	return &Config{
		Host: host,
		Port: port,
	}
}

// FormatAddress formats the server address.
func (c *Config) FormatAddress() string {
	if c.Port <= 0 {
		return c.Host
	}
	return fmt.Sprintf("%s:%d", c.Host, c.Port)
}

// IsLocal reports whether host is local.
func (c Config) IsLocal() bool {
	return c.Host == "localhost" || c.Host == "127.0.0.1"
}

// ProcessItems processes a list of tokens.
func ProcessItems(items []string) []string {
	var result []string
	for _, item := range items {
		trimmed := strings.TrimSpace(item)
		if len(trimmed) > 0 {
			result = append(result, trimmed)
		}
	}
	return result
}
`
		filePath := "internal/agentopt/sample.go"
		chunks := chunker.ChunkSource(filePath, goSource)

		if len(chunks) == 0 {
			t.Fatalf("expected chunks from Go source, got 0")
		}

		// Functions and methods defined in goSource:
		// 1. NewConfig: lines 24-30 (Doc starts on 24, func on 25, ends on 30)
		// 2. FormatAddress: lines 32-38 (Doc starts on 32, func on 33, ends on 38)
		// 3. IsLocal: lines 40-43 (Doc starts on 40, func on 41, ends on 43)
		// 4. ProcessItems: lines 45-56 (Doc starts on 45, func on 46, ends on 56)
		functions := []expectedFunctionSpan{
			{name: "NewConfig", startLine: 22, endLine: 28, isMethod: false, scope: ""},
			{name: "FormatAddress", startLine: 30, endLine: 36, isMethod: true, scope: "Config"},
			{name: "IsLocal", startLine: 38, endLine: 41, isMethod: true, scope: "Config"},
			{name: "ProcessItems", startLine: 43, endLine: 53, isMethod: false, scope: ""},
		}

		splitFunctionCount := 0

		for _, fn := range functions {
			var containingChunk *CodeChunk
			for _, chunk := range chunks {
				if chunk.StartLine <= fn.startLine && chunk.EndLine >= fn.endLine {
					containingChunk = &chunk
					break
				}
			}

			if containingChunk == nil {
				t.Errorf("function %s (lines %d-%d) was not fully contained in any single chunk",
					fn.name, fn.startLine, fn.endLine)
				splitFunctionCount++
				continue
			}

			// Verify zero chunk boundaries split the function interior
			for _, chunk := range chunks {
				if chunk.StartLine > fn.startLine && chunk.StartLine <= fn.endLine {
					t.Errorf("chunk boundary starts inside function %s: chunk.StartLine=%d inside [%d, %d]",
						fn.name, chunk.StartLine, fn.startLine, fn.endLine)
					splitFunctionCount++
				}
				if chunk.EndLine >= fn.startLine && chunk.EndLine < fn.endLine {
					t.Errorf("chunk boundary ends inside function %s: chunk.EndLine=%d inside [%d, %d]",
						fn.name, chunk.EndLine, fn.startLine, fn.endLine)
					splitFunctionCount++
				}
			}

			// Verify breadcrumb attributes on the containing chunk
			if containingChunk.PackageName != "samplepkg" {
				t.Errorf("function %s: expected PackageName=samplepkg, got %q", fn.name, containingChunk.PackageName)
			}
			if containingChunk.FilePath != filePath {
				t.Errorf("function %s: expected FilePath=%s, got %q", fn.name, filePath, containingChunk.FilePath)
			}
			if fn.isMethod {
				if containingChunk.Boundary != BoundaryMethod {
					t.Errorf("function %s: expected Boundary=%s, got %s", fn.name, BoundaryMethod, containingChunk.Boundary)
				}
				if containingChunk.EnclosingScope != fn.scope {
					t.Errorf("function %s: expected EnclosingScope=%q, got %q", fn.name, fn.scope, containingChunk.EnclosingScope)
				}
			} else {
				if containingChunk.Boundary != BoundaryFunction {
					t.Errorf("function %s: expected Boundary=%s, got %s", fn.name, BoundaryFunction, containingChunk.Boundary)
				}
			}
		}

		if splitFunctionCount != 0 {
			t.Fatalf("Go source: confirmed %d split functions across chunks (expected 0)", splitFunctionCount)
		}
		t.Logf("Go source: confirmed 0 split functions across %d functions/methods", len(functions))

		// Check struct and interface chunks
		foundStruct := false
		foundInterface := false
		for _, c := range chunks {
			if c.Boundary == BoundaryStruct && c.EnclosingScope == "Config" {
				foundStruct = true
			}
			if c.Boundary == BoundaryInterface && c.EnclosingScope == "Validator" {
				foundInterface = true
			}
		}
		if !foundStruct {
			t.Errorf("expected BoundaryStruct chunk for Config")
		}
		if !foundInterface {
			t.Errorf("expected BoundaryInterface chunk for Validator")
		}
	})

	t.Run("TypeScriptSourceZeroSplitFunctions", func(t *testing.T) {
		tsSource := `import { BaseLogger } from './logger';

// Account represents a user profile.
export interface Account {
    id: string;
    username: string;
}

export type AccountRole = "admin" | "viewer";

// AccountService manages account actions.
export class AccountService {
    private logger: BaseLogger;

    constructor(logger: BaseLogger) {
        this.logger = logger;
    }

    // findAccount retrieves account by identifier.
    async findAccount(id: string): Promise<Account | null> {
        if (!id) {
            return null;
        }
        return { id, username: "admin" };
    }

    // verifyAccess checks permission level.
    verifyAccess(role: AccountRole): boolean {
        return role === "admin";
    }
}

// createDefaultAccount initializes guest credentials.
export function createDefaultAccount(): Account {
    return {
        id: "guest-0",
        username: "guest",
    };
}

export const formatAccountName = (acc: Account): string => {
    return acc.username.trim();
};
`
		filePath := "src/services/account.ts"
		chunks := chunker.ChunkSource(filePath, tsSource)

		if len(chunks) == 0 {
			t.Fatalf("expected chunks from TypeScript source, got 0")
		}

		// Functions and methods in tsSource:
		// 1. constructor: lines 15-17
		// 2. findAccount: lines 19-25 (Doc on 19, method on 20, ends on 25)
		// 3. verifyAccess: lines 27-30 (Doc on 27, method on 28, ends on 30)
		// 4. createDefaultAccount: lines 33-39 (Doc on 33, func on 34, ends on 39)
		// 5. formatAccountName: lines 41-43
		functions := []expectedFunctionSpan{
			{name: "constructor", startLine: 15, endLine: 17, isMethod: true, scope: "AccountService"},
			{name: "findAccount", startLine: 19, endLine: 25, isMethod: true, scope: "AccountService"},
			{name: "verifyAccess", startLine: 27, endLine: 30, isMethod: true, scope: "AccountService"},
			{name: "createDefaultAccount", startLine: 33, endLine: 39, isMethod: false, scope: ""},
			{name: "formatAccountName", startLine: 41, endLine: 43, isMethod: false, scope: ""},
		}

		splitFunctionCount := 0

		for _, fn := range functions {
			var containingChunk *CodeChunk
			for _, chunk := range chunks {
				if chunk.StartLine <= fn.startLine && chunk.EndLine >= fn.endLine {
					containingChunk = &chunk
					break
				}
			}

			if containingChunk == nil {
				t.Errorf("TypeScript function %s (lines %d-%d) was not fully contained in any single chunk",
					fn.name, fn.startLine, fn.endLine)
				splitFunctionCount++
				continue
			}

			// Verify zero chunk boundaries split the function interior
			for _, chunk := range chunks {
				if chunk.StartLine > fn.startLine && chunk.StartLine <= fn.endLine {
					t.Errorf("chunk boundary starts inside TS function %s: chunk.StartLine=%d inside [%d, %d]",
						fn.name, chunk.StartLine, fn.startLine, fn.endLine)
					splitFunctionCount++
				}
				if chunk.EndLine >= fn.startLine && chunk.EndLine < fn.endLine {
					t.Errorf("chunk boundary ends inside TS function %s: chunk.EndLine=%d inside [%d, %d]",
						fn.name, chunk.EndLine, fn.startLine, fn.endLine)
					splitFunctionCount++
				}
			}

			// Verify breadcrumb attributes
			if containingChunk.FilePath != filePath {
				t.Errorf("TS function %s: expected FilePath=%s, got %q", fn.name, filePath, containingChunk.FilePath)
			}
			if fn.isMethod {
				if containingChunk.Boundary != BoundaryMethod {
					t.Errorf("TS function %s: expected Boundary=%s, got %s", fn.name, BoundaryMethod, containingChunk.Boundary)
				}
				if containingChunk.EnclosingScope != fn.scope {
					t.Errorf("TS function %s: expected EnclosingScope=%q, got %q", fn.name, fn.scope, containingChunk.EnclosingScope)
				}
			} else {
				if containingChunk.Boundary != BoundaryFunction {
					t.Errorf("TS function %s: expected Boundary=%s, got %s", fn.name, BoundaryFunction, containingChunk.Boundary)
				}
			}
		}

		if splitFunctionCount != 0 {
			t.Fatalf("TypeScript source: confirmed %d split functions across chunks (expected 0)", splitFunctionCount)
		}
		t.Logf("TypeScript source: confirmed 0 split functions across %d functions/methods", len(functions))

		// Check interface and type chunks
		foundInterface := false
		foundType := false
		for _, c := range chunks {
			if c.Boundary == BoundaryInterface && c.EnclosingScope == "Account" {
				foundInterface = true
			}
			if c.Boundary == BoundaryType && c.EnclosingScope == "AccountRole" {
				foundType = true
			}
		}
		if !foundInterface {
			t.Errorf("expected BoundaryInterface chunk for Account")
		}
		if !foundType {
			t.Errorf("expected BoundaryType chunk for AccountRole")
		}
	})
}

func TestSyntaxBoundaryBreadcrumbs(t *testing.T) {
	chunk := CodeChunk{
		FilePath:       "internal/agentopt/syntax_chunk.go",
		PackageName:    "agentopt",
		EnclosingScope: "SyntaxChunker",
		StartLine:      10,
		EndLine:        25,
		Boundary:       BoundaryMethod,
		Content:        "func (s *SyntaxChunker) ChunkSource(...) []CodeChunk",
	}

	expectedBreadcrumb := "internal/agentopt/syntax_chunk.go > agentopt > SyntaxChunker"
	if bc := chunk.Breadcrumb(); bc != expectedBreadcrumb {
		t.Fatalf("expected breadcrumb %q, got %q", expectedBreadcrumb, bc)
	}
}

func TestSyntaxBoundaryEmptySource(t *testing.T) {
	chunker := NewSyntaxChunker()
	if chunks := chunker.ChunkSource("test.go", ""); len(chunks) != 0 {
		t.Fatalf("expected 0 chunks for empty source, got %d", len(chunks))
	}
	if chunks := chunker.ChunkSource("test.go", "   \n\t  "); len(chunks) != 0 {
		t.Fatalf("expected 0 chunks for whitespace source, got %d", len(chunks))
	}
}

func TestSyntaxBoundaryGenericFallback(t *testing.T) {
	chunker := NewSyntaxChunker()
	raw := `first paragraph line 1
first paragraph line 2

second paragraph line 1
second paragraph line 2`

	chunks := chunker.ChunkSource("docs/notes.txt", raw)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 block chunks for generic fallback, got %d", len(chunks))
	}
	if chunks[0].Boundary != BoundaryBlock || chunks[1].Boundary != BoundaryBlock {
		t.Fatalf("expected BoundaryBlock, got %s and %s", chunks[0].Boundary, chunks[1].Boundary)
	}
}
