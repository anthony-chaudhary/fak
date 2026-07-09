package windowgate

// ConPTY VERSION PREFLIGHT — the second half of the console fault boundary.
//
// The rest of this package keeps a console window off the desktop. This file
// answers a different question about the same console: is the pseudoconsole
// implementation behind it new enough for the shell we are about to launch?
//
// # The defect (issue #3402, parent #2170)
//
// `pwsh.exe` aborts with a controlled `System.Environment.FailFast` (HRESULT
// 0x80131623) whose Win32 cause is `0xE9` — "No process is on the other end of
// the pipe". PowerShell treats `GetConsoleScreenBufferInfo` as infallible; when
// it returns 0xE9 the host throws `HostException` inside `ConsoleHost.InputLoop`
// and calls `FailFast`, an uncatchable process abort.
//
// This was long read as resource exhaustion. It is not: there is no documented
// limit on consoles or pseudoconsoles, and no tunable. The crash is a VERSION
// defect. A terminal that bundles a ConPTY pair — `OpenConsole.exe` plus
// `conpty.dll` — older than the `pwsh 7.6` / .NET 10 that talks to it will trip
// the incompatible-peer path. Known-good floor: 1.24.260402001. Bumping the pair
// fixes it; it does not reproduce under an up-to-date Windows Terminal.
//
// Upstream convergence: PowerShell/PowerShell#12640, #16818; microsoft/terminal
// #14511 and #16212; warpdotdev/warp#11398; wezterm#7774 ("Update bundled Windows
// ConPTY pair to 1.24.260402001").
//
// # Two spellings of one version — read this before touching the comparator
//
// A Windows image carries the same build number in two resource strings, and the
// floor above is quoted in the SECOND one. Measured on a live, up-to-date
// OpenConsole.exe (from Windows Terminal 1.24.11321.0):
//
//	FileVersion    = 1.24.2605.12001
//	ProductVersion = 1.24.260512001
//
// VS_FIXEDFILEINFO packs each component into 16 bits, so it cannot hold a
// 260512001 build. FileVersion therefore splits that build as
// `<datecode>.<build>`, the build zero-padded to five digits; ProductVersion
// carries it whole. They are the same number: 2605*100000 + 12001 == 260512001,
// and likewise the floor, 2604*100000 + 2001 == 260402001.
//
// Comparing the ProductVersion-shaped floor straight against a raw FileVersion is
// the trap: component-wise 2605 < 260402001, so a CURRENT pair reads as stale and
// the gate refuses every healthy host. Both sides are therefore normalized to the
// whole-build spelling before any comparison — see NormalizeConPTYVersion.
//
// # The update procedure when this preflight refuses
//
// The pair is owned by whichever terminal spawns the shell, not by fak. Update
// Windows Terminal (`winget upgrade Microsoft.WindowsTerminal`, or the Store), or
// upgrade the offending CLI that vendors its own copy, until `fak conpty` reports
// PASS. Re-run `fak conpty --json` to witness the new version. If a vendor ships a
// pair that cannot be moved, pass `--floor` to re-pin the gate deliberately rather
// than editing this constant.
//
// # Scope boundary
//
// This crash reproduces at plain `pwsh` launch, independent of load. It is a
// SEPARATE failure mode from the fleet-scale stall (#3153) and from desktop-heap
// or handle pressure. Nothing here counts resources.

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// ConPTYVersionFloor is the oldest ConPTY pair known to survive pwsh 7.6 / .NET
// 10. Anything below it is the #3402 crash configuration. Sourced from
// wezterm#7774 and warp#11398, which converged on this exact build. Quoted in the
// ProductVersion spelling, as upstream quotes it.
const ConPTYVersionFloor = "1.24.260402001"

// conptyBuildScale reunites a FileVersion's split `<datecode>.<build>` tail into
// the single whole build that ProductVersion carries. The build field is five
// digits wide (12001, 02001), so the datecode scales by 100000.
const conptyBuildScale = 100000

// A FileVersion tail only folds when it actually has the split shape: a datecode
// of at most four digits and a build of at most five. Anything wider is some
// other versioning scheme and is compared as-is.
const (
	conptyMaxDatecode = 9999
	conptyMaxBuild    = 99999
)

// The two halves of the pair. Both ship inside the terminal's package directory;
// only the host binary is normally reachable through PATH.
const (
	ConPTYHostBinary = "OpenConsole.exe"
	ConPTYLibrary    = "conpty.dll"
)

// Verdicts. SKIP and PASS are the two states that do not hold up a launch.
const (
	ConPTYPass   = "PASS"   // every resolved component is at or above the floor
	ConPTYWarn   = "WARN"   // a defect is present; advisory mode
	ConPTYRefuse = "REFUSE" // a defect is present; strict mode
	ConPTYSkip   = "SKIP"   // no pair resolved — nothing to judge
)

// Reason codes: the closed vocabulary this preflight refuses with, in fold
// precedence order (stale outranks unreadable outranks absent).
const (
	ReasonConPTYStale      = "CONPTY_STALE"      // resolved below the floor — the #3402 crash configuration
	ReasonConPTYUnreadable = "CONPTY_UNREADABLE" // present, but its version could not be read or parsed
	ReasonConPTYAbsent     = "CONPTY_ABSENT"     // no ConPTY pair on the launch PATH
)

// ErrFileVersionUnsupported is returned by ReadVersionInfo off Windows, where PE
// version resources do not exist.
var ErrFileVersionUnsupported = errors.New("windowgate: PE version resources are a Windows facility")

// ConPTYVersionInfo is the pair of version strings a PE image carries. Both are
// captured so an operator can see the raw resource that drove the verdict.
type ConPTYVersionInfo struct {
	FileVersion    string `json:"file_version,omitempty"`
	ProductVersion string `json:"product_version,omitempty"`
}

// ConPTYComponent is one half of the pair as resolved on this host.
type ConPTYComponent struct {
	Name           string `json:"name"`
	Path           string `json:"path,omitempty"`
	FileVersion    string `json:"file_version,omitempty"`
	ProductVersion string `json:"product_version,omitempty"`
	// Compared is the normalized whole-build spelling actually measured against
	// the floor, and which of the two resource strings it came from.
	Compared       string `json:"compared,omitempty"`
	ComparedSource string `json:"compared_source,omitempty"`
	Found          bool   `json:"found"`
	Stale          bool   `json:"stale"`
	Reason         string `json:"reason,omitempty"`
	Error          string `json:"error,omitempty"`
}

// ConPTYReport is the searchable status surface for the #3402 crash class.
type ConPTYReport struct {
	Floor      string            `json:"floor"`
	Verdict    string            `json:"verdict"`
	Reason     string            `json:"reason,omitempty"`
	Strict     bool              `json:"strict"`
	SearchDirs int               `json:"search_dirs"`
	Components []ConPTYComponent `json:"components"`
	NextAction string            `json:"next_action,omitempty"`
}

// OK reports whether the launch may proceed. An absent pair is OK: a host with
// no Windows Terminal has no stale ConPTY to trip over.
func (r ConPTYReport) OK() bool { return r.Verdict == ConPTYPass || r.Verdict == ConPTYSkip }

// Stale reports whether any resolved component sits below the floor.
func (r ConPTYReport) Stale() bool { return r.Reason == ReasonConPTYStale }

// ConPTYOptions configures the preflight. The zero value scans the live launch
// PATH against the compiled-in floor using the real PE version reader.
type ConPTYOptions struct {
	// SearchDirs is the ordered resolution path, first match wins, exactly like
	// exec.LookPath. Empty means DefaultConPTYSearchDirs.
	SearchDirs []string
	// Floor overrides ConPTYVersionFloor. Empty means the compiled-in floor.
	Floor string
	// Strict escalates a defect from WARN to REFUSE, for gating a fleet launch.
	Strict bool
	// ReadVersionInfo is the PE version-resource seam. Nil means ReadVersionInfo.
	ReadVersionInfo func(path string) (ConPTYVersionInfo, error)
}

// DefaultConPTYSearchDirs is the launch PATH a spawned child would resolve
// against, then the inbox console-host directory, then the terminal package
// bundles. The bundles matter because the pair a terminal actually loads ships
// inside its own package directory, which is NOT on PATH.
func DefaultConPTYSearchDirs() []string {
	var dirs []string
	seen := map[string]bool{}
	add := func(d string) {
		if d = strings.TrimSpace(d); d != "" && !seen[d] {
			seen[d] = true
			dirs = append(dirs, d)
		}
	}
	for _, d := range filepath.SplitList(os.Getenv("PATH")) {
		add(d)
	}
	if root := os.Getenv("SystemRoot"); root != "" {
		add(filepath.Join(root, "System32"))
	}
	for _, d := range ConPTYBundleDirs(defaultAppRoots()) {
		add(d)
	}
	return dirs
}

func defaultAppRoots() []string {
	var roots []string
	if pf := os.Getenv("ProgramFiles"); pf != "" {
		roots = append(roots, filepath.Join(pf, "WindowsApps"))
	}
	if la := os.Getenv("LOCALAPPDATA"); la != "" {
		roots = append(roots, filepath.Join(la, "Microsoft", "WindowsApps"))
	}
	return roots
}

// terminalPackageGlobs match the MSIX package directories that bundle a ConPTY
// pair.
var terminalPackageGlobs = []string{
	"Microsoft.WindowsTerminal_*",
	"Microsoft.WindowsTerminalPreview_*",
}

// ConPTYBundleDirs lists the terminal package directories under appRoots,
// NEWEST FIRST, so first-match resolution picks the pair a current terminal would
// load. The package version is the second underscore-separated field of the
// directory name ("Microsoft.WindowsTerminal_1.24.11321.0_x64__8wekyb3d8bbwe").
//
// Unreadable roots yield nothing rather than an error. This is the common case,
// not an edge one: `%ProgramFiles%\WindowsApps` denies directory ENUMERATION to
// an unelevated user, though it still permits TRAVERSE — a full path under it
// stats fine. So a machine-wide Store install of Windows Terminal is invisible
// here, and the operator must name its directory (`fak conpty --bundle <dir>`)
// for the preflight to resolve the pair it actually loads.
func ConPTYBundleDirs(appRoots []string) []string {
	type bundle struct {
		path    string
		version []uint64
	}
	var found []bundle
	for _, root := range appRoots {
		for _, glob := range terminalPackageGlobs {
			matches, err := filepath.Glob(filepath.Join(root, glob))
			if err != nil {
				continue
			}
			for _, m := range matches {
				if st, err := os.Stat(m); err != nil || !st.IsDir() {
					continue
				}
				found = append(found, bundle{path: m, version: packageVersion(filepath.Base(m))})
			}
		}
	}
	// Newest first; a name whose version will not parse sorts last.
	sort.SliceStable(found, func(i, j int) bool {
		a, b := found[i].version, found[j].version
		switch {
		case a == nil && b == nil:
			return false
		case a == nil:
			return false
		case b == nil:
			return true
		}
		return compareVersionFields(a, b) > 0
	})
	out := make([]string, 0, len(found))
	for _, b := range found {
		out = append(out, b.path)
	}
	return out
}

func packageVersion(dirName string) []uint64 {
	parts := strings.Split(dirName, "_")
	if len(parts) < 2 {
		return nil
	}
	v, err := ParseFileVersion(parts[1])
	if err != nil {
		return nil
	}
	return v
}

// ParseFileVersion splits a Windows version resource into its raw numeric
// components, with no normalization. Real VERSIONINFO blocks are free-form
// strings, so this accepts the spellings that occur in practice: dotted
// ("1.24.260402001"), comma-separated ("1, 22, 10352, 0"), a leading "v", and a
// trailing build tag or pre-release suffix, which is dropped. Anything that does
// not begin with a number is an error — never a silent zero.
func ParseFileVersion(s string) ([]uint64, error) {
	raw := strings.TrimSpace(s)
	raw = strings.TrimPrefix(strings.TrimPrefix(raw, "v"), "V")
	raw = strings.ReplaceAll(raw, ",", ".")
	raw = strings.Join(strings.Fields(raw), "") // "1. 22. 0" -> "1.22.0"
	// Truncate at the first character that is neither a digit nor a separator,
	// dropping "-rc1" / "(release)" tails.
	if cut := strings.IndexFunc(raw, func(r rune) bool { return (r < '0' || r > '9') && r != '.' }); cut >= 0 {
		raw = raw[:cut]
	}
	raw = strings.TrimSuffix(raw, ".")
	if raw == "" {
		return nil, fmt.Errorf("windowgate: %q is not a version", s)
	}
	fields := strings.Split(raw, ".")
	out := make([]uint64, 0, len(fields))
	for _, f := range fields {
		n, err := strconv.ParseUint(f, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("windowgate: %q is not a version: component %q: %w", s, f, err)
		}
		out = append(out, n)
	}
	return out, nil
}

// NormalizeConPTYVersion parses a version resource and folds a split
// `<datecode>.<build>` tail back into the whole build that ProductVersion and the
// floor both use, so the two spellings of one version compare equal.
//
//	1.24.2605.12001 -> 1.24.260512001   (the FileVersion of a current pair)
//	1.24.260402001  -> 1.24.260402001   (already whole; unchanged)
//
// The fold applies only to the exact four-component shape whose tail fits the
// datecode/build widths. A version like 1.22.10352.0 has too wide a third
// component to be a datecode, so it is left alone — and it never needs the fold,
// because its minor component already decides the comparison.
func NormalizeConPTYVersion(s string) ([]uint64, error) {
	f, err := ParseFileVersion(s)
	if err != nil {
		return nil, err
	}
	return foldSplitBuild(f), nil
}

func foldSplitBuild(f []uint64) []uint64 {
	if len(f) != 4 || f[2] > conptyMaxDatecode || f[3] > conptyMaxBuild {
		return f
	}
	return []uint64{f[0], f[1], f[2]*conptyBuildScale + f[3]}
}

// CompareConPTYVersions returns -1 when a sorts below b, 0 when equal, +1 above.
// Both sides are normalized first, so a FileVersion and a ProductVersion of the
// same build compare equal.
//
// Components compare NUMERICALLY and the shorter version is zero-extended, so
// "1.24" equals "1.24.0" and — the case this whole file exists for — "1.24.9"
// sorts BELOW "1.24.260402001". A string comparison would rank it above and wave
// the crash configuration straight through.
func CompareConPTYVersions(a, b string) (int, error) {
	av, err := NormalizeConPTYVersion(a)
	if err != nil {
		return 0, err
	}
	bv, err := NormalizeConPTYVersion(b)
	if err != nil {
		return 0, err
	}
	return compareVersionFields(av, bv), nil
}

func compareVersionFields(a, b []uint64) int {
	n := len(a)
	if len(b) > n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		var x, y uint64
		if i < len(a) {
			x = a[i]
		}
		if i < len(b) {
			y = b[i]
		}
		switch {
		case x < y:
			return -1
		case x > y:
			return 1
		}
	}
	return 0
}

// ConPTYPreflight resolves the ConPTY pair on the launch PATH, reads each version
// resource, and folds the components into one verdict. It performs no process
// launch and reads no console: it only stats files and reads their version
// resource, so it is safe to run before a guarded child is spawned.
func ConPTYPreflight(opt ConPTYOptions) ConPTYReport {
	dirs := opt.SearchDirs
	if len(dirs) == 0 {
		dirs = DefaultConPTYSearchDirs()
	}
	floor := strings.TrimSpace(opt.Floor)
	if floor == "" {
		floor = ConPTYVersionFloor
	}
	read := opt.ReadVersionInfo
	if read == nil {
		read = ReadVersionInfo
	}

	rep := ConPTYReport{Floor: floor, Strict: opt.Strict, SearchDirs: len(dirs)}

	// The host binary resolves against the launch PATH; the library ships beside
	// it, so its own directory is searched first.
	host := resolveConPTYComponent(ConPTYHostBinary, dirs, floor, read)
	libDirs := dirs
	if host.Found {
		libDirs = append([]string{filepath.Dir(host.Path)}, dirs...)
	}
	lib := resolveConPTYComponent(ConPTYLibrary, libDirs, floor, read)
	rep.Components = []ConPTYComponent{host, lib}

	var found, stale, unreadable int
	for _, c := range rep.Components {
		if !c.Found {
			continue
		}
		found++
		switch {
		case c.Stale:
			stale++
		case c.Reason == ReasonConPTYUnreadable:
			unreadable++
		}
	}

	switch {
	case stale > 0:
		rep.Reason = ReasonConPTYStale
		rep.Verdict = ConPTYWarn
		if opt.Strict {
			rep.Verdict = ConPTYRefuse
		}
		rep.NextAction = "update the bundled ConPTY pair to >= " + floor +
			" (winget upgrade Microsoft.WindowsTerminal), then re-run `fak conpty --json`"
	case unreadable > 0:
		rep.Reason = ReasonConPTYUnreadable
		rep.Verdict = ConPTYWarn
		rep.NextAction = "the pair is present but its version resource could not be read; verify the binary is a real PE with a VERSIONINFO block"
	case found == 0:
		rep.Reason = ReasonConPTYAbsent
		rep.Verdict = ConPTYSkip
		rep.NextAction = "no ConPTY pair on the launch PATH; nothing to judge"
	default:
		rep.Verdict = ConPTYPass
	}
	return rep
}

// resolveConPTYComponent finds name in dirs (first match wins) and classifies its
// version against floor. A read or parse failure is UNREADABLE, never a pass: a
// stale pair must not slip through a failed read.
func resolveConPTYComponent(name string, dirs []string, floor string, read func(string) (ConPTYVersionInfo, error)) ConPTYComponent {
	c := ConPTYComponent{Name: name}
	for _, d := range dirs {
		p := filepath.Join(d, name)
		st, err := os.Stat(p)
		if err != nil || st.IsDir() {
			continue
		}
		c.Found = true
		c.Path = p
		break
	}
	if !c.Found {
		c.Reason = ReasonConPTYAbsent
		return c
	}

	info, err := read(c.Path)
	if err != nil {
		c.Reason = ReasonConPTYUnreadable
		c.Error = err.Error()
		return c
	}
	c.FileVersion = info.FileVersion
	c.ProductVersion = info.ProductVersion

	raw, source, err := comparableVersion(info)
	if err != nil {
		c.Reason = ReasonConPTYUnreadable
		c.Error = err.Error()
		return c
	}
	c.Compared = raw
	c.ComparedSource = source

	cmp, err := CompareConPTYVersions(raw, floor)
	if err != nil {
		c.Reason = ReasonConPTYUnreadable
		c.Error = err.Error()
		return c
	}
	if cmp < 0 {
		c.Stale = true
		c.Reason = ReasonConPTYStale
	}
	return c
}

// comparableVersion picks which resource string to measure against the floor.
//
// It takes the MORE PRECISE of the two — the one with more components once
// normalized — and breaks a tie toward ProductVersion, the spelling the floor is
// quoted in. Precision decides because an image is free to put a short marketing
// string in one field: an OpenConsole.exe with ProductVersion "1.24" and
// FileVersion "1.24.2605.12001" describes one build, and reading the marketing
// string would compare [1,24] against the floor's [1,24,260402001] and call a
// current pair stale.
func comparableVersion(info ConPTYVersionInfo) (raw, source string, err error) {
	type candidate struct {
		raw    string
		source string
		fields []uint64
	}
	var best *candidate
	// ProductVersion is considered first so it wins an equal-precision tie.
	for _, c := range []candidate{
		{info.ProductVersion, "product_version", nil},
		{info.FileVersion, "file_version", nil},
	} {
		fields, err := NormalizeConPTYVersion(c.raw)
		if err != nil {
			continue
		}
		c.fields = fields
		if best == nil || len(c.fields) > len(best.fields) {
			pick := c
			best = &pick
		}
	}
	if best == nil {
		return "", "", fmt.Errorf("windowgate: no parsable version (FileVersion %q, ProductVersion %q)",
			info.FileVersion, info.ProductVersion)
	}
	return best.raw, best.source, nil
}
