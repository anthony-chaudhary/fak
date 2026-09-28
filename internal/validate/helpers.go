package validate

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/affectedtests"
)

func validateSkipWalkDir(name string) bool {
	return strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_")
}

func packagePatternsForRoot(root string, packages []string, fileToPkg map[string]string) []string {
	pkgDirs := make(map[string]string)
	for file, pkg := range fileToPkg {
		dir := filepath.ToSlash(filepath.Dir(file))
		if dir == "." {
			dir = ""
		}
		if old, ok := pkgDirs[pkg]; !ok || len(dir) < len(old) {
			pkgDirs[pkg] = dir
		}
	}
	out := make([]string, 0, len(packages))
	for _, pkg := range packages {
		if dir, ok := pkgDirs[pkg]; ok {
			if dir == "" {
				out = append(out, ".")
			} else {
				out = append(out, "./"+dir)
			}
			continue
		}
		out = append(out, pkg)
	}
	return out
}

func appendUniqueStrings(dst []string, values ...string) []string {
	seen := make(map[string]bool, len(dst)+len(values))
	for _, value := range dst {
		seen[value] = true
	}
	for _, value := range values {
		if !seen[value] {
			seen[value] = true
			dst = append(dst, value)
		}
	}
	return dst
}

func posixQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "'\"'\"'") + "'"
}

func validateCommandDetail(out []byte, err error) string {
	if detail := strings.TrimSpace(string(out)); detail != "" {
		return detail
	}
	return err.Error()
}

func defaultValidateWSLTests(goos string) bool { return goos == "windows" }

func validateTestRunner(goos string, wsl bool) string {
	if goos == "windows" && wsl {
		return "wsl.exe bash -lc go test"
	}
	return "go test"
}

// validateGoTestTimeoutMaxMargin caps the head start go test's own -timeout alarm
// gets over the validate deadline.
const validateGoTestTimeoutMaxMargin = 5 * time.Minute

// validateGoTestTimeout derives go test's -timeout from the validate budget left at
// now. Without it Go's 10m default governed even under `fak commit
// --build-check-timeout 90m`, so a green-but-slow cmd/fak suite on a loaded host died
// as "test timed out after 10m0s" (#13558). The alarm is set a margin (a tenth of the
// remaining budget, at most validateGoTestTimeoutMaxMargin) short of the deadline so
// that for a truly hung test Go's panic and goroutine dump name it before the context
// kill discards them. A context with no deadline returns 0: nothing external bounds
// the run, so Go's own default stays in charge.
func validateGoTestTimeout(ctx context.Context, now time.Time) time.Duration {
	deadline, ok := ctx.Deadline()
	if !ok {
		return 0
	}
	remaining := deadline.Sub(now)
	margin := min(remaining/10, validateGoTestTimeoutMaxMargin)
	// -timeout 0 disables the alarm, so an exhausted budget still gets a positive one.
	return max((remaining - margin).Truncate(time.Second), time.Second)
}

// validateTestArgs builds the isolated go test argv; timeout 0 omits -timeout.
func validateTestArgs(testRun string, timeout time.Duration, targets []string) []string {
	// Tests require filesystem-valid source paths from runtime.Caller(0) (e.g. for
	// package introspection and dos.toml validation); do not pass -trimpath here (#9788).
	args := []string{"test", "-count=1"}
	if timeout > 0 {
		args = append(args, "-timeout", timeout.String())
	}
	if testRun != "" {
		args = append(args, "-run", testRun)
	}
	return append(args, targets...)
}

func validateGoCheckArgs(mode string, targets []string) []string {
	// validate materializes the same committed bytes under a fresh root on every run.
	// Normalize that disposable root in compile identities so the shared Go cache can
	// reuse artifacts without changing the developer/release build's debug-path contract.
	if mode == "build" {
		return append([]string{mode, "-trimpath", "-buildvcs=false"}, targets...)
	}
	return append([]string{mode, "-trimpath"}, targets...)
}

type validateGoTestEvent struct {
	Action  string `json:"Action"`
	Package string `json:"Package"`
}

func parseValidateTestObservation(output string, packages []string, commandComplete bool) affectedtests.TestObservation {
	observed := make(map[string]affectedtests.PackageObservation, len(packages))
	terminal := make(map[string]bool, len(packages))
	for _, pkg := range packages {
		observed[pkg] = affectedtests.PackageObservation{Package: pkg}
	}
	dec := json.NewDecoder(strings.NewReader(output))
	for {
		var event validateGoTestEvent
		if err := dec.Decode(&event); err != nil {
			break
		}
		observation, wanted := observed[event.Package]
		if !wanted {
			continue
		}
		switch event.Action {
		case "fail":
			observation.Failed = true
			observed[event.Package] = observation
			terminal[event.Package] = true
		case "pass", "skip":
			terminal[event.Package] = true
		}
	}
	result := affectedtests.TestObservation{Complete: commandComplete, Packages: make([]affectedtests.PackageObservation, 0, len(packages))}
	for _, pkg := range packages {
		result.Packages = append(result.Packages, observed[pkg])
		if !terminal[pkg] {
			result.Complete = false
		}
	}
	return result
}

func validateAuditHead(root, tip string, paths []string) string {
	hash := sha256.New()
	_, _ = io.WriteString(hash, tip+"\x00")
	for _, rel := range paths {
		_, _ = io.WriteString(hash, rel+"\x00")
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			_, _ = io.WriteString(hash, "<deleted>\x00")
			continue
		}
		_, _ = hash.Write(data)
		_, _ = io.WriteString(hash, "\x00")
	}
	return fmt.Sprintf("%s+mine:%x", tip, hash.Sum(nil)[:6])
}

func validateJSONTestArgs(args []string) []string {
	out := append([]string(nil), args...)
	if len(out) == 0 {
		return out
	}
	return append(out[:1], append([]string{"-json"}, out[1:]...)...)
}

func validateAllPackages(fileToPkg map[string]string) []string {
	seen := make(map[string]struct{}, len(fileToPkg))
	for _, pkg := range fileToPkg {
		seen[pkg] = struct{}{}
	}
	packages := make([]string, 0, len(seen))
	for pkg := range seen {
		packages = append(packages, pkg)
	}
	sort.Strings(packages)
	return packages
}

func hasDeletedMinePath(root string, paths []string) bool {
	for _, rel := range paths {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel))); os.IsNotExist(err) {
			return true
		}
	}
	return false
}

func ownedTestRunExpression(root string, paths []string) (string, error) {
	for _, rel := range paths {
		if rel != "" && !strings.HasSuffix(strings.ToLower(rel), "_test.go") {
			return "", nil
		}
	}
	seen := map[string]bool{}
	var names []string
	for _, rel := range paths {
		if !strings.HasSuffix(strings.ToLower(rel), "_test.go") {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), filepath.Join(root, filepath.FromSlash(rel)), nil, 0)
		if err != nil {
			return "", err
		}
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv != nil || fn.Name == nil {
				continue
			}
			name := fn.Name.Name
			if !(strings.HasPrefix(name, "Test") || strings.HasPrefix(name, "Example") || strings.HasPrefix(name, "Fuzz")) || seen[name] {
				continue
			}
			seen[name] = true
			names = append(names, regexp.QuoteMeta(name))
		}
	}
	if len(names) == 0 {
		return "", nil
	}
	sort.Strings(names)
	return "^(" + strings.Join(names, "|") + ")$", nil
}

func requestedMinePaths(raw []string) []string {
	out := make([]string, 0, len(raw))
	for _, path := range raw {
		out = append(out, filepath.ToSlash(filepath.Clean(path)))
	}
	return out
}

func subtractValidatePaths(all, checked []string) []string {
	done := make(map[string]bool, len(checked))
	for _, path := range checked {
		done[path] = true
	}
	left := make([]string, 0, len(all)-len(done))
	for _, path := range all {
		if !done[path] {
			left = append(left, path)
		}
	}
	return left
}

func validateWriterIsTerminal(w io.Writer) bool {
	type statter interface {
		Stat() (os.FileInfo, error)
	}
	f, ok := w.(statter)
	if !ok {
		return false
	}
	info, err := f.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

func validatePhaseOrder(testOnly, auditSelection, smoke bool) []string {
	phases := []string{"resolve_root", "resolve_ref", "wsl_preflight", "normalize_mine", "extract_tip", "base_graph", "overlay"}
	if !testOnly {
		phases = append(phases, "gofmt")
	}
	phases = append(phases, "list_graph", "test_select")
	if !testOnly {
		phases = append(phases, "build", "vet")
	}
	phases = append(phases, "test")
	if auditSelection {
		phases = append(phases, "test_audit_full")
	}
	if smoke {
		phases = append(phases, "smoke")
	}
	return phases
}
