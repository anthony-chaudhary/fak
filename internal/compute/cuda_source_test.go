package compute

import (
	"bufio"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

var cudaQuotedIncludeRE = regexp.MustCompile(`^\s*#\s*include\s*"([^"\r\n]+)"`)

// cudaSourceClosure reads each CUDA source root and its repository-local quoted
// includes exactly once. It makes source-contract tests follow focused fragments
// such as cuda/*.cuh without requiring every assertion to know their filenames.
func cudaSourceClosure(t *testing.T, roots ...string) string {
	t.Helper()
	base, err := os.Getwd()
	if err != nil {
		t.Fatalf("resolve CUDA source base: %v", err)
	}
	base, err = filepath.Abs(base)
	if err != nil {
		t.Fatalf("resolve CUDA source base: %v", err)
	}

	const (
		visiting = 1
		visited  = 2
	)
	state := make(map[string]int)
	var out strings.Builder
	var visit func(string)
	visit = func(name string) {
		t.Helper()
		path := name
		if !filepath.IsAbs(path) {
			path = filepath.Join(base, path)
		}
		path, err = filepath.Abs(filepath.Clean(path))
		if err != nil {
			t.Fatalf("resolve CUDA source %q: %v", name, err)
		}
		rel, err := filepath.Rel(base, path)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			t.Fatalf("CUDA source %q escapes package directory", name)
		}
		switch state[path] {
		case visiting:
			t.Fatalf("CUDA quoted-include cycle at %s", filepath.ToSlash(rel))
		case visited:
			return
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read CUDA source %s: %v", filepath.ToSlash(rel), err)
		}
		state[path] = visiting
		out.Write(data)
		if len(data) == 0 || data[len(data)-1] != '\n' {
			out.WriteByte('\n')
		}
		scanner := bufio.NewScanner(strings.NewReader(string(data)))
		for scanner.Scan() {
			match := cudaQuotedIncludeRE.FindStringSubmatch(scanner.Text())
			if match != nil {
				visit(filepath.Join(filepath.Dir(path), match[1]))
			}
		}
		if err := scanner.Err(); err != nil {
			t.Fatalf("scan CUDA source %s: %v", filepath.ToSlash(rel), err)
		}
		state[path] = visited
	}
	for _, root := range roots {
		visit(root)
	}
	return out.String()
}
