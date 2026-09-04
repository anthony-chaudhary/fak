package citeverify

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestClaimSymbolsQuoted(t *testing.T) {
	tests := []struct {
		name  string
		claim string
		want  []string
	}{
		{
			name:  "single quoted symbol",
			claim: "checks `Target` existence",
			want:  []string{"Target"},
		},
		{
			name:  "multiple quoted symbols",
			claim: "implements `First` and `Second` handlers",
			want:  []string{"First", "Second"},
		},
		{
			name:  "qualified identifier within quotes",
			claim: "invokes `pkg.MyStruct.Method()`",
			want:  []string{"pkg.MyStruct.Method"},
		},
		{
			name:  "duplicate quoted symbols are deduplicated",
			claim: "`Foo` and `Foo` again",
			want:  []string{"Foo"},
		},
		{
			name:  "symbol with underscores and hyphens",
			claim: "handles `k_v-cache_v2` logic",
			want:  []string{"k_v-cache_v2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := claimSymbols(tt.claim)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("claimSymbols(%q) = %v, want %v", tt.claim, got, tt.want)
			}
		})
	}
}

func TestClaimSymbolsUnquotedFallback(t *testing.T) {
	tests := []struct {
		name  string
		claim string
		want  []string
	}{
		{
			name:  "unquoted words longer than 2 characters",
			claim: "Verify claims accurately",
			want:  []string{"Verify", "claims", "accurately"},
		},
		{
			name:  "unquoted words of length <= 2 are filtered out",
			claim: "is an id to do",
			want:  nil,
		},
		{
			name:  "empty claim produces no symbols",
			claim: "",
			want:  nil,
		},
		{
			name:  "claim with numbers and special chars",
			claim: "run 123 !@# $%",
			want:  []string{"run"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := claimSymbols(tt.claim)
			if len(got) == 0 && len(tt.want) == 0 {
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("claimSymbols(%q) = %v, want %v", tt.claim, got, tt.want)
			}
		})
	}
}

func TestContainsSymbol(t *testing.T) {
	tests := []struct {
		name    string
		line    string
		symbols []string
		want    bool
	}{
		{
			name:    "exact match",
			line:    "func Target() error {",
			symbols: []string{"Target"},
			want:    true,
		},
		{
			name:    "substring match",
			line:    "type CustomTargetHandler struct",
			symbols: []string{"Target"},
			want:    true,
		},
		{
			name:    "no matching symbol",
			line:    "func OtherFunc() int {",
			symbols: []string{"Target"},
			want:    false,
		},
		{
			name:    "empty symbols list",
			line:    "func Target() error {",
			symbols: []string{},
			want:    false,
		},
		{
			name:    "multiple symbols with second matching",
			line:    "func Second() {",
			symbols: []string{"First", "Second"},
			want:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := containsSymbol(tt.line, tt.symbols); got != tt.want {
				t.Errorf("containsSymbol(%q, %v) = %v, want %v", tt.line, tt.symbols, got, tt.want)
			}
		})
	}
}

func TestCodeExtension(t *testing.T) {
	valid := []string{
		"file.go", "mod.rs", "lib.c", "api.h", "impl.cc", "core.cpp",
		"Main.java", "Script.kt", "index.js", "app.jsx", "server.ts",
		"page.tsx", "test.py", "runner.rb", "build.sh", "deploy.ps1",
		"program.cs", "util.swift", "UPPER.GO", "Mixed.Py",
	}
	for _, path := range valid {
		if !codeExtension(path) {
			t.Errorf("codeExtension(%q) = false, want true", path)
		}
	}

	invalid := []string{
		"README.md", "data.json", "config.yaml", "doc.txt",
		"binary.exe", "blob.bin", "style.css", "markup.html",
		"noextension", "",
	}
	for _, path := range invalid {
		if codeExtension(path) {
			t.Errorf("codeExtension(%q) = true, want false", path)
		}
	}
}

func TestUnsafePath(t *testing.T) {
	unsafe := []string{
		".git/config.go",
		".env.go",
		"dir/.hidden/source.go",
		"id_rsa",
		"keys/id_ed25519",
		"config/secret_keys.go",
		"credential_cache.go",
		"server.pem",
		"tls.key",
	}
	for _, path := range unsafe {
		if !unsafePath(path) {
			t.Errorf("unsafePath(%q) = false, want true", path)
		}
	}

	safe := []string{
		"pkg/source.go",
		"internal/citeverify/citeverify.go",
		"main.go",
	}
	for _, path := range safe {
		if unsafePath(path) {
			t.Errorf("unsafePath(%q) = true, want false", path)
		}
	}
}

func TestCitationRegexParsing(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantHits int
	}{
		{
			name:     "standard relative path",
			input:    "pkg/source.go:42",
			wantHits: 1,
		},
		{
			name:     "leading dot slash",
			input:    "./pkg/source.go:10",
			wantHits: 1,
		},
		{
			name:     "parent directory relative",
			input:    "../shared/source.go:99",
			wantHits: 1,
		},
		{
			name:     "windows absolute path",
			input:    `C:\repo\pkg\source.go:15`,
			wantHits: 1,
		},
		{
			name:     "multiple citations in one string",
			input:    "see pkg/a.go:1 and pkg/b.go:2 for details",
			wantHits: 2,
		},
		{
			name:     "no line number does not match",
			input:    "pkg/source.go",
			wantHits: 0,
		},
		{
			name:     "non numeric suffix does not match",
			input:    "pkg/source.go:abc",
			wantHits: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			matches := citationRE.FindAllStringSubmatch(tt.input, -1)
			if len(matches) != tt.wantHits {
				t.Errorf("citationRE matches for %q = %d, want %d", tt.input, len(matches), tt.wantHits)
			}
		})
	}
}

func TestVerifyEdgeCases(t *testing.T) {
	root := t.TempDir()
	sourcePath := filepath.Join(root, "source.go")
	body := "package main\n\n// line 3\nfunc Target() {}\n"
	if err := os.WriteFile(sourcePath, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Run("non-positive line number contradicts", func(t *testing.T) {
		if got := Verify("`Target`", []string{"source.go:0"}, root); got != Contradicts {
			t.Errorf("Verify with line 0: got %v, want %v", got, Contradicts)
		}
	})

	t.Run("oversized file returns unknown", func(t *testing.T) {
		largePath := filepath.Join(root, "large.go")
		f, err := os.Create(largePath)
		if err != nil {
			t.Fatal(err)
		}
		// Write more than maxSourceBytes (1MB)
		chunk := make([]byte, 64*1024)
		for i := 0; i < 17; i++ { // 17 * 64KB = 1088KB > 1MB
			if _, err := f.Write(chunk); err != nil {
				f.Close()
				t.Fatal(err)
			}
		}
		f.Close()

		if got := Verify("`Target`", []string{"large.go:1"}, root); got != Unknown {
			t.Errorf("Verify on oversized file: got %v, want %v", got, Unknown)
		}
	})

	t.Run("unresolvable claim with no symbols returns unknown", func(t *testing.T) {
		// Claim with only short words (< 3 chars) and no backticks -> no symbols
		if got := Verify("is an to", []string{"source.go:4"}, root); got != Unknown {
			t.Errorf("Verify with no symbols: got %v, want %v", got, Unknown)
		}
	})
}
