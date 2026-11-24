package architest

import (
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// Public fak must build with no private companion.
const privateModulePrefix = "github.com/anthony-chaudhary/fak-private"

var standaloneSkipDirs = map[string]bool{
	"vendor":            true,
	"third_party":       true,
	"testdata":          true,
	"_scratch":          true,
	"_wip-preservation": true,
}

func TestStandaloneNoPrivateImport(t *testing.T) {
	root := findRepoRoot(t)

	gomod, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		t.Fatalf("read go.mod: %v", err)
	}
	if strings.Contains(string(gomod), "fak-private") {
		t.Errorf("go.mod mentions fak-private (require or replace)")
	}

	var files []string
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if path != root && (strings.HasPrefix(name, ".") || standaloneSkipDirs[name]) {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(path, ".go") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", root, err)
	}

	var (
		mu       sync.Mutex
		problems []string
		wg       sync.WaitGroup
	)
	paths := make(chan string)
	for range runtime.GOMAXPROCS(0) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			fset := token.NewFileSet()
			for path := range paths {
				rel, _ := filepath.Rel(root, path)
				rel = filepath.ToSlash(rel)
				f, perr := parser.ParseFile(fset, path, nil, parser.ImportsOnly)
				if perr != nil {
					mu.Lock()
					problems = append(problems, "parse "+rel+": "+perr.Error())
					mu.Unlock()
					continue
				}
				for _, imp := range f.Imports {
					p, _ := strconv.Unquote(imp.Path.Value)
					if strings.HasPrefix(p, privateModulePrefix) {
						mu.Lock()
						problems = append(problems, rel+" imports private module path "+strconv.Quote(p))
						mu.Unlock()
					}
				}
			}
		}()
	}
	for _, f := range files {
		paths <- f
	}
	close(paths)
	wg.Wait()
	for _, p := range problems {
		t.Error(p)
	}
}
