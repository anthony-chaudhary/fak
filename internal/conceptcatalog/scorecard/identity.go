package scorecard

import (
	"crypto/sha256"
	"embed"
	"fmt"
	"io/fs"
	"sort"
	"strings"
	"sync"
)

// The analyzer runs from the compiled binary, not from the tree it folds, so a
// witness cached by tree or archive identity is only reusable while the code
// that produced it is unchanged. Embedding the package's own sources gives a
// build-time identity of exactly that code.
//
//go:embed *.go
var sourceFS embed.FS

var sourceDigest = sync.OnceValue(func() string {
	d, err := SourceDigestFS(sourceFS)
	if err != nil {
		panic("scorecard: embedded sources unreadable: " + err.Error())
	}
	return d
})

// SourceDigest identifies the scorecard analyzer compiled into this binary.
func SourceDigest() string { return sourceDigest() }

// SourceDigestFS hashes every non-test .go file at the root of fsys by name and
// content, in sorted order.
func SourceDigestFS(fsys fs.FS) (string, error) {
	entries, err := fs.ReadDir(fsys, ".")
	if err != nil {
		return "", err
	}
	var names []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		names = append(names, name)
	}
	if len(names) == 0 {
		return "", fmt.Errorf("no analyzer sources")
	}
	sort.Strings(names)
	h := sha256.New()
	for _, name := range names {
		body, readErr := fs.ReadFile(fsys, name)
		if readErr != nil {
			return "", readErr
		}
		fmt.Fprintf(h, "%s\x00%d\x00", name, len(body))
		h.Write(body)
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}
