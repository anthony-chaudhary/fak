package unwiredscore

import (
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Predicate reports whether the state described as absent by a prose pin has
// become true. A true result intentionally fails the pin: these assertions
// fire on success so stale "not wired" prose cannot survive the cutover.
type Predicate func(root, packagePath string) (bool, error)

// ClaimPin binds a negative prose claim to the predicate that makes it stale.
// ProsePath is repository-relative and ClaimText must occur verbatim.
type ClaimPin struct {
	Package   string
	ProsePath string
	ClaimText string
	Predicate Predicate
}

// CheckClaimPin enforces both sides of a negative-claim pin. Removing the prose
// without removing the pin fails, as does making the predicate true while the
// prose remains. The errors name the text that must be removed at cutover.
func CheckClaimPin(root string, pin ClaimPin) error {
	if strings.TrimSpace(pin.ClaimText) == "" {
		return fmt.Errorf("unwired claim pin: claim text is empty")
	}
	if pin.Predicate == nil {
		return fmt.Errorf("unwired claim pin for %q: predicate is nil", pin.Package)
	}
	body, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(pin.ProsePath)))
	if err != nil {
		return fmt.Errorf("unwired claim pin %q: read prose %s: %w", pin.Package, pin.ProsePath, err)
	}
	if !strings.Contains(string(body), pin.ClaimText) {
		return fmt.Errorf("unwired claim pin %q remains but prose %q is gone from %s; delete the pin with the prose", pin.Package, pin.ClaimText, pin.ProsePath)
	}
	becameTrue, err := pin.Predicate(root, pin.Package)
	if err != nil {
		return fmt.Errorf("unwired claim pin %q: evaluate predicate: %w", pin.Package, err)
	}
	if becameTrue {
		return fmt.Errorf("unwired claim pin fired on success: %q is now wired; delete prose %q from %s and remove this pin", pin.Package, pin.ClaimText, pin.ProsePath)
	}
	return nil
}

// ExternalImporter is the common "no call site outside this package"
// predicate. It parses Go imports, so mentions in comments, strings, and test
// fixtures do not count as callers. The package path is repository-relative,
// for example "internal/unwiredscore".
func ExternalImporter(root, packagePath string) (bool, error) {
	modulePath, err := moduleName(root)
	if err != nil {
		return false, err
	}
	want := strings.TrimSuffix(modulePath, "/") + "/" + strings.TrimPrefix(filepath.ToSlash(packagePath), "/")
	packageDir := filepath.Clean(filepath.Join(root, filepath.FromSlash(packagePath)))
	found := false
	err = filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if path != root && (entry.Name() == ".git" || entry.Name() == "vendor" || strings.HasPrefix(entry.Name(), ".")) {
				return filepath.SkipDir
			}
			return nil
		}
		if found || filepath.Ext(path) != ".go" || within(path, packageDir) {
			return nil
		}
		file, parseErr := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
		if parseErr != nil {
			return fmt.Errorf("parse imports %s: %w", path, parseErr)
		}
		for _, spec := range file.Imports {
			value, unquoteErr := strconv.Unquote(spec.Path.Value)
			if unquoteErr == nil && value == want {
				found = true
				break
			}
		}
		return nil
	})
	return found, err
}

func within(path, dir string) bool {
	rel, err := filepath.Rel(dir, path)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func moduleName(root string) (string, error) {
	body, readErr := os.ReadFile(filepath.Join(root, "go.mod"))
	if readErr != nil {
		return "", fmt.Errorf("read go.mod: %w", readErr)
	}
	for _, line := range strings.Split(string(body), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == "module" {
			return fields[1], nil
		}
	}
	return "", fmt.Errorf("go.mod has no module directive")
}

// MeasurementPin binds a number quoted in prose to a live measurement.
type MeasurementPin struct {
	ProsePath string
	ClaimText string
	Want      int
	Measure   func(root string) (int, error)
}

// CheckMeasurementPin refuses a missing prose claim and a stale measured
// number. ClaimText should include the rendered number, making the tie
// bidirectional rather than merely comparing two hidden integers.
func CheckMeasurementPin(root string, pin MeasurementPin) error {
	if pin.Measure == nil {
		return fmt.Errorf("measurement pin for %s: measure function is nil", pin.ProsePath)
	}
	body, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(pin.ProsePath)))
	if err != nil {
		return fmt.Errorf("measurement pin: read prose %s: %w", pin.ProsePath, err)
	}
	if !strings.Contains(string(body), pin.ClaimText) {
		return fmt.Errorf("measurement pin remains but prose %q is gone from %s; delete the pin with the prose", pin.ClaimText, pin.ProsePath)
	}
	got, err := pin.Measure(root)
	if err != nil {
		return fmt.Errorf("measurement pin for %s: %w", pin.ProsePath, err)
	}
	if got != pin.Want {
		return fmt.Errorf("stale measured prose %q in %s: says %d, measurement is %d; update the prose and pin together", pin.ClaimText, pin.ProsePath, pin.Want, got)
	}
	return nil
}
