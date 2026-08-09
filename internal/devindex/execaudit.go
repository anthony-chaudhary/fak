package devindex

// C7 of epic #1287 (#5648): the EXECUTABLE audit. The verb spine (freshness.go's
// UndeclaredVerbs) measures the ~200 subcommands INSIDE the single cmd/fak binary.
// It says nothing about the other ~100 `main` packages in this module: whether they
// carry a test, and whether anything outside their own directory ever builds, runs,
// installs or documents them. A package can compile forever while being absent from
// every build target, script, dispatcher and doc — "buildable but unwired" — and fak
// has fixed that defect one incident at a time. This file turns the class into a
// measured invariant.
//
// The three rules that make the measurement mean something:
//
//   - DERIVED, NOT LISTED. The domain comes from the Go toolchain (`go list`) asking
//     for `main` packages across the whole module, so it is recursive and finds nested
//     executables a `cmd/*` glob cannot. It is the same "COVERAGE comes from source,
//     the curated overlay is only a fallback" discipline verbs.go already states.
//   - TWO AXES, NOT ONE. "Has an adjacent test" and "is reached from outside itself"
//     need opposite fixes, so they stay separate fields. Test existence is a SOURCE
//     fact read out of the toolchain's package metadata (TestGoFiles), never an exit
//     code — `go test ./cmd/<pkg>` exits 0 on a package with no tests at all, which is
//     precisely the vacuous evidence this audit exists to replace.
//   - SELF-REFERENCE IS NOT REACHABILITY. A package does not become reachable because
//     its own source names it, nor because an inventory row (a catalog table, a survey
//     doc, this audit's own JSON) lists it. Evidence must be an INVOCATION FORM — a
//     `go build/run/install/test` naming the package, a `./binary` command, or the
//     package path inside a string literal in Go code outside it — carried by a file
//     outside the package directory. Prose that merely names the package is excluded,
//     because an audit satisfiable by prose about itself measures nothing.
//
// Fail-closed: when the toolchain cannot be run, or resolves an EMPTY executable set,
// the result is could-not-establish-domain, never a green audit over zero packages. A
// naive fold over an empty domain finds zero problems and reports success, which is
// the one bug that would quietly retire the instrument.
//
// This file is the DETECTION half, like freshness.go: it reports rows and a verdict.
// Turning the verdict into a blocking CI check is an operator policy decision layered
// on top (cmd/fak's `fak index execaudit` exits non-zero on a failing audit).

import (
	"fmt"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ExecEvidenceClass names HOW an executable is reached from outside itself. The
// classes stay DISTINGUISHED rather than collapsed to a bool: a build target, a
// dispatcher registration, an installer line, a script and a documented runnable
// example are not interchangeable, and which one holds is exactly what a fixer needs.
type ExecEvidenceClass string

const (
	// ExecEvidenceBuildTarget: a Makefile / *.mk / CI-workflow line that builds it.
	ExecEvidenceBuildTarget ExecEvidenceClass = "build-target"
	// ExecEvidenceDispatch: Go code outside the package naming its path in a string
	// literal — the spawn/registration edge (a main package cannot be imported).
	ExecEvidenceDispatch ExecEvidenceClass = "dispatcher-registration"
	// ExecEvidenceInstaller: a `go install` line — the "how a user gets the binary" edge.
	ExecEvidenceInstaller ExecEvidenceClass = "installer"
	// ExecEvidenceScript: a shell/PowerShell/Python script that builds or runs it.
	ExecEvidenceScript ExecEvidenceClass = "script"
	// ExecEvidenceDocExample: a documented RUNNABLE example (a command line in prose),
	// as distinct from an inventory row that merely names the package.
	ExecEvidenceDocExample ExecEvidenceClass = "documented-example"
)

// ExecEvidence is one reachability edge: the class, the repo-relative file and line
// that carry it, and the matched text. The locator is kept so a reader can check the
// claim instead of trusting the audit's own summary.
type ExecEvidence struct {
	Class ExecEvidenceClass `json:"class"`
	File  string            `json:"file"`
	Line  int               `json:"line"`
	Text  string            `json:"text"`
}

// ExecStatus is the closed per-package verdict vocabulary, derived from the two
// independent axes (HasTest, Evidence) plus pin state. It is a LABEL over the axes,
// never a replacement for them.
type ExecStatus string

const (
	// ExecStatusOK: an adjacent test AND at least one outside invocation edge.
	ExecStatusOK ExecStatus = "ok"
	// ExecStatusUntested: wired up, but no adjacent test (the "wired-but-untested" state).
	ExecStatusUntested ExecStatus = "untested"
	// ExecStatusUnreachable: tested, but nothing outside the package invokes it.
	ExecStatusUnreachable ExecStatus = "unreachable"
	// ExecStatusOrphan: neither tested nor reached — the "buildable-but-unwired" state.
	ExecStatusOrphan ExecStatus = "orphan"
	// ExecStatusPinned: failing an axis, but covered by a live reasoned exception.
	ExecStatusPinned ExecStatus = "pinned"
)

// ExecPackage is one row of the audit: an executable package, its two independent
// axes, the evidence that established reachability, and whether a pin admits it.
type ExecPackage struct {
	ImportPath string         `json:"import_path"`
	Dir        string         `json:"dir"` // repo-relative, slash-separated (host-independent)
	Binary     string         `json:"binary"`
	HasTest    bool           `json:"has_test"`
	TestFiles  int            `json:"test_files"`
	Evidence   []ExecEvidence `json:"evidence,omitempty"`
	Status     ExecStatus     `json:"status"`
	// PinReason is set only when a live pin admits an otherwise-failing package.
	PinReason string `json:"pin_reason,omitempty"`
}

// Reachable reports whether anything outside the package invokes it.
func (p ExecPackage) Reachable() bool { return len(p.Evidence) > 0 }

// ExecPin is a temporary, REASONED exception: a package allowed to fail an axis
// while a named condition holds. Following internal/architest's tier map, the reason
// is mandatory — a bare ignore list converts a measured exception into invisible
// permanent debt, and the whole value of a pin is that it fails once stale.
type ExecPin struct {
	Package string `json:"package"` // full import path
	Reason  string `json:"reason"`
	// Until is an optional YYYY-MM-DD expiry. Empty means the pin is bounded only by
	// its own justification (it goes stale when the package leaves the domain or the
	// condition it excuses no longer holds).
	Until string `json:"until,omitempty"`
}

// ExecPinState is a pin folded against the live tree: whether it is still doing work,
// or has gone stale and must be removed.
type ExecPinState struct {
	ExecPin
	Stale bool   `json:"stale"`
	Why   string `json:"why"`
}

// ExecAuditResult is the whole answer, JSON-shaped so the DENOMINATOR is auditable
// rather than asserted: the total domain size, every row, every admitted exception,
// and the named failures.
type ExecAuditResult struct {
	Schema string `json:"schema"`
	// Established is false when the executable domain could not be resolved at all.
	Established bool   `json:"established"`
	Status      string `json:"status"` // "ok" | "fail" | ExecDomainNotEstablished
	Reason      string `json:"reason,omitempty"`
	// Domain is the TOTAL number of executable packages discovered — the denominator.
	Domain     int            `json:"domain"`
	Tested     int            `json:"tested"`
	Reached    int            `json:"reached"`
	Packages   []ExecPackage  `json:"packages"`
	Exceptions []ExecPinState `json:"exceptions"`
	Failures   []string       `json:"failures"`
	StalePins  []string       `json:"stale_pins"`
}

const (
	// ExecAuditSchema versions the emitted witness.
	ExecAuditSchema = "fak.devindex.exec-audit.v1"
	// ExecDomainNotEstablished is the fail-closed status: the audit could not resolve
	// the executable domain, so it reports THAT rather than a green run over nothing.
	ExecDomainNotEstablished = "could-not-establish-domain"
)

// ExecAuditOptions parameterizes one audit run.
type ExecAuditOptions struct {
	// Root is the module root to audit.
	Root string
	// Pins are the admitted exceptions. Nil means "no exception is admitted".
	Pins []ExecPin
	// Now dates pin expiry. Zero means time.Now().
	Now time.Time
}

// AuditExecutables derives every `main` package in the module at Root, joins each
// against an adjacent-test probe and a reachability sweep over the tree outside the
// package, folds the admitted pins, and returns the typed result.
//
// It FAILS CLOSED: if `go list` cannot run, or resolves no executable at all, the
// returned result carries Established=false and Status=ExecDomainNotEstablished
// alongside a non-nil error. Callers must never read that as a clean audit.
func AuditExecutables(opts ExecAuditOptions) (*ExecAuditResult, error) {
	res := &ExecAuditResult{
		Schema:     ExecAuditSchema,
		Status:     ExecDomainNotEstablished,
		Packages:   []ExecPackage{},
		Exceptions: []ExecPinState{},
		Failures:   []string{},
		StalePins:  []string{},
	}
	root := opts.Root
	if root == "" {
		root = "."
	}
	pkgs, moduleDir, err := discoverExecPackages(root)
	if err != nil {
		res.Reason = err.Error()
		return res, err
	}
	if len(pkgs) == 0 {
		err := fmt.Errorf("devindex: no executable (main) package resolved under %s — an empty domain is not a clean audit", root)
		res.Reason = err.Error()
		return res, err
	}
	// Resolve the corpus from the MODULE root the toolchain itself reported, so package
	// dirs and evidence file paths share one frame (a caller-supplied root and go list's
	// view can differ by symlink or short-name on some hosts, which would silently break
	// the "is this file inside the package" self-reference check).
	corpus, tracked := execAuditCorpus(moduleDir)
	if tracked {
		// The domain obeys the same repository rule as the evidence, or the audit grades
		// a tree nobody committed. See keepTrackedExecPackages.
		pkgs = keepTrackedExecPackages(corpus, pkgs)
		if len(pkgs) == 0 {
			err := fmt.Errorf("devindex: no TRACKED executable (main) package resolved under %s — an empty domain is not a clean audit", root)
			res.Reason = err.Error()
			return res, err
		}
	}
	res.Established = true
	res.Domain = len(pkgs)

	scanExecReachability(moduleDir, corpus, pkgs)

	now := opts.Now
	if now.IsZero() {
		now = time.Now()
	}
	byPath := map[string]*ExecPackage{}
	for i := range pkgs {
		byPath[pkgs[i].ImportPath] = &pkgs[i]
	}
	livePins := map[string]string{}
	for _, st := range foldExecPins(opts.Pins, byPath, now) {
		res.Exceptions = append(res.Exceptions, st)
		if st.Stale {
			res.StalePins = append(res.StalePins, st.Package)
			continue
		}
		livePins[st.Package] = st.Reason
	}

	for i := range pkgs {
		p := &pkgs[i]
		if p.HasTest {
			res.Tested++
		}
		if p.Reachable() {
			res.Reached++
		}
		p.Status = execStatusFor(*p)
		if p.Status == ExecStatusOK {
			continue
		}
		if reason, ok := livePins[p.ImportPath]; ok {
			p.Status = ExecStatusPinned
			p.PinReason = reason
			continue
		}
		res.Failures = append(res.Failures, p.ImportPath)
	}
	res.Packages = pkgs

	switch {
	case len(res.Failures) > 0 || len(res.StalePins) > 0:
		res.Status = "fail"
		res.Reason = fmt.Sprintf("%d/%d executable package(s) untested or unreachable, %d stale pin(s)",
			len(res.Failures), res.Domain, len(res.StalePins))
	default:
		res.Status = "ok"
	}
	return res, nil
}

// execStatusFor labels a row from its two independent axes.
func execStatusFor(p ExecPackage) ExecStatus {
	switch {
	case p.HasTest && p.Reachable():
		return ExecStatusOK
	case p.Reachable():
		return ExecStatusUntested
	case p.HasTest:
		return ExecStatusUnreachable
	default:
		return ExecStatusOrphan
	}
}

// foldExecPins checks every pin against the live tree and returns its state, sorted
// by package. A pin is STALE — and reds the audit — when it carries no reason, when
// its package has left the executable domain, when the package now passes on its own
// (the exception has done its job and is now just noise), or when its declared expiry
// has passed. That staleness is the entire difference between a reasoned pin and an
// ignore list.
func foldExecPins(pins []ExecPin, byPath map[string]*ExecPackage, now time.Time) []ExecPinState {
	out := make([]ExecPinState, 0, len(pins))
	for _, pin := range pins {
		st := ExecPinState{ExecPin: pin}
		p, known := byPath[pin.Package]
		switch {
		case strings.TrimSpace(pin.Reason) == "":
			st.Stale, st.Why = true, "pin carries no reason — an unreasoned pin is an ignore list"
		case !known:
			st.Stale, st.Why = true, "package is no longer in the executable domain"
		case p.HasTest && p.Reachable():
			st.Stale, st.Why = true, "package now has an adjacent test AND outside reachability — the exception is no longer doing work"
		default:
			if pin.Until != "" {
				exp, err := time.Parse("2006-01-02", pin.Until)
				switch {
				case err != nil:
					st.Stale, st.Why = true, "pin expiry "+pin.Until+" is not a YYYY-MM-DD date"
				case now.After(exp):
					st.Stale, st.Why = true, "pin expired on "+pin.Until
				}
			}
		}
		out = append(out, st)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Package < out[j].Package })
	return out
}

// discoverExecPackages asks the Go toolchain for every `main` package in the module,
// recursively — never a directory glob, which would both miss nested executables (this
// module has several below experiments/) and miscount directories that are not
// packages. Test-file counts come from the same metadata, so "has a test" is a source
// fact and not an exit code.
//
// `go list -e` is deliberate: on a shared, peer-dirty checkout one unrelated broken
// package must not erase the whole domain. A toolchain that cannot run at all still
// errors, which is what fails the audit closed.
// It returns the rows plus the MODULE root the toolchain resolved, which is the frame
// every emitted path is relative to.
func discoverExecPackages(root string) ([]ExecPackage, string, error) {
	const sep = "\x1f"
	const tmpl = `{{if eq .Name "main"}}{{.ImportPath}}` + sep + `{{.Dir}}` + sep +
		`{{len .TestGoFiles}}` + sep + `{{len .XTestGoFiles}}` + sep +
		`{{if .Module}}{{.Module.Dir}}{{end}}{{end}}`
	cmd := exec.Command("go", "list", "-e", "-f", tmpl, "./...")
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return nil, "", fmt.Errorf("devindex: go list could not establish the executable domain under %s: %s", root, firstLine(msg))
	}
	moduleDir := ""
	var pkgs []ExecPackage
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		f := strings.Split(line, sep)
		if len(f) != 5 || f[4] == "" {
			continue
		}
		moduleDir = f[4]
		rel := relSlash(f[4], f[1])
		if rel == "" {
			continue
		}
		n := atoiSafe(f[2]) + atoiSafe(f[3])
		pkgs = append(pkgs, ExecPackage{
			ImportPath: f[0],
			Dir:        rel,
			Binary:     path.Base(rel),
			HasTest:    n > 0,
			TestFiles:  n,
		})
	}
	if moduleDir == "" {
		absRoot, aerr := filepath.Abs(root)
		if aerr != nil {
			absRoot = root
		}
		moduleDir = absRoot
	}
	sort.Slice(pkgs, func(i, j int) bool { return pkgs[i].ImportPath < pkgs[j].ImportPath })
	return pkgs, moduleDir, nil
}

func firstLine(s string) string {
	if i := strings.IndexAny(s, "\r\n"); i >= 0 {
		return s[:i]
	}
	return s
}

func atoiSafe(s string) int {
	n := 0
	for _, r := range s {
		if r < '0' || r > '9' {
			return 0
		}
		n = n*10 + int(r-'0')
	}
	return n
}

// relSlash returns dir relative to root in slash form, so the emitted witness is
// identical on every host (an absolute C:\ path is not a portable denominator).
func relSlash(root, dir string) string {
	rel, err := filepath.Rel(root, dir)
	if err != nil {
		return ""
	}
	rel = filepath.ToSlash(rel)
	if rel == "." || strings.HasPrefix(rel, "..") {
		return ""
	}
	return rel
}
