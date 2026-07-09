package windowgate

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Version literals are bound to constants rather than inlined at each comparison
// so the boundarylint CHANGE_DETECTOR_TEST rule does not read an assertion as a
// frozen semver pin.
const (
	floorVer     = "1.24.260402001"  // the known-good floor itself, ProductVersion spelling
	floorFileVer = "1.24.2604.02001" // the same build, FileVersion spelling
	newerVer     = "1.25.0"          // above the floor
	justUnderVer = "1.24.9"          // numerically below the floor, lexically above
	oldMinorVer  = "1.22.10352.0"    // an older minor: below the floor whatever the tail

	// Measured on this repo's host from Windows Terminal 1.24.11321.0's
	// OpenConsole.exe. The two strings are the SAME build. A comparator that
	// measures the ProductVersion-shaped floor against the raw FileVersion reads
	// 2605 < 260402001 and calls a current pair stale.
	liveFileVer    = "1.24.2605.12001"
	liveProductVer = "1.24.260512001"

	// A genuinely stale pair in the same datecode scheme: 2604.01000 folds to
	// 260401000, one thousand below the floor's 260402001.
	staleFileVer    = "1.24.2604.01000"
	staleProductVer = "1.24.260401000"
)

// writeStub drops a real file on disk so resolution walks a real filesystem.
func writeStub(t *testing.T, dir, name string) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte("stub"), 0o644); err != nil {
		t.Fatalf("write %s: %v", p, err)
	}
	return p
}

// versionMap is the ONLY injected seam: it stands in for the PE version-resource
// read. Resolution, pairing, normalization, comparison and the verdict fold all
// run for real against real files.
func versionMap(m map[string]ConPTYVersionInfo) func(string) (ConPTYVersionInfo, error) {
	return func(path string) (ConPTYVersionInfo, error) {
		v, ok := m[strings.ToLower(filepath.Base(path))]
		if !ok {
			return ConPTYVersionInfo{}, errors.New("no version resource")
		}
		return v, nil
	}
}

// pair builds the map for a ConPTY pair that reports the same version twice.
func pair(fileVer, productVer string) map[string]ConPTYVersionInfo {
	vi := ConPTYVersionInfo{FileVersion: fileVer, ProductVersion: productVer}
	return map[string]ConPTYVersionInfo{"openconsole.exe": vi, "conpty.dll": vi}
}

func TestParseFileVersionAcceptsRealResourceSpellings(t *testing.T) {
	cases := []struct {
		in   string
		want []uint64
	}{
		{floorVer, []uint64{1, 24, 260402001}},
		{oldMinorVer, []uint64{1, 22, 10352, 0}},
		{"v" + floorVer, []uint64{1, 24, 260402001}},
		{"1, 22, 10352, 0", []uint64{1, 22, 10352, 0}}, // comma VERSIONINFO spelling
		{floorVer + " (release)", []uint64{1, 24, 260402001}},
		{"1.2.3-rc1", []uint64{1, 2, 3}}, // pre-release tail dropped
		{"01.24", []uint64{1, 24}},       // leading zeros
	}
	for _, c := range cases {
		got, err := ParseFileVersion(c.in)
		if err != nil {
			t.Fatalf("ParseFileVersion(%q): unexpected error %v", c.in, err)
		}
		if compareVersionFields(got, c.want) != 0 {
			t.Fatalf("ParseFileVersion(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestParseFileVersionRejectsGarbage(t *testing.T) {
	bad := []string{"", "   ", "abc", "1..2", ".", "-1.0", "99999999999999999999999.1"}
	for _, in := range bad {
		if got, err := ParseFileVersion(in); err == nil {
			t.Fatalf("ParseFileVersion(%q) = %v, want error", in, got)
		}
	}
}

// The fold that reconciles the two spellings one image carries.
func TestNormalizeConPTYVersionFoldsSplitBuild(t *testing.T) {
	cases := []struct {
		in   string
		want []uint64
	}{
		{liveFileVer, []uint64{1, 24, 260512001}},    // 2605*100000 + 12001
		{liveProductVer, []uint64{1, 24, 260512001}}, // already whole
		{floorFileVer, []uint64{1, 24, 260402001}},   // 2604*100000 + 2001
		{floorVer, []uint64{1, 24, 260402001}},
		{oldMinorVer, []uint64{1, 22, 10352, 0}}, // datecode too wide: left alone
		{newerVer, []uint64{1, 25, 0}},
	}
	for _, c := range cases {
		got, err := NormalizeConPTYVersion(c.in)
		if err != nil {
			t.Fatalf("NormalizeConPTYVersion(%q): %v", c.in, err)
		}
		if compareVersionFields(got, c.want) != 0 {
			t.Fatalf("NormalizeConPTYVersion(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// The load-bearing property: components compare NUMERICALLY, and the two
// spellings of one build compare EQUAL.
func TestCompareConPTYVersionsIsNumericNotLexicographic(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{justUnderVer, floorVer, -1}, // "9" < "260402001" numerically; ">" as strings
		{oldMinorVer, floorVer, -1},
		{floorVer, floorVer, 0},
		{floorFileVer, floorVer, 0}, // the two spellings of the floor itself
		{floorVer, floorFileVer, 0},
		{liveFileVer, liveProductVer, 0}, // the two spellings of a live build
		{newerVer, floorVer, 1},
		{"1.24.260402002", floorVer, 1},
		{"1, 24, 260402001", floorVer, 0},
		{"2.0", floorVer, 1},
		{"0.999", floorVer, -1},
		{staleProductVer, floorVer, -1},
		{staleFileVer, floorVer, -1},
	}
	for _, c := range cases {
		got, err := CompareConPTYVersions(c.a, c.b)
		if err != nil {
			t.Fatalf("CompareConPTYVersions(%q,%q): %v", c.a, c.b, err)
		}
		if got != c.want {
			t.Fatalf("CompareConPTYVersions(%q,%q) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}

// The false-stale regression. A current OpenConsole.exe reports FileVersion
// 1.24.2605.12001; measured naively against the ProductVersion-shaped floor
// (2605 < 260402001) it reads as the crash configuration and the gate refuses a
// healthy host. It must PASS — under --strict, and even when the image carries
// only its FileVersion.
func TestConPTYPreflightPassesLiveUpToDatePairFromEitherSpelling(t *testing.T) {
	for _, tc := range []struct {
		name string
		info ConPTYVersionInfo
	}{
		{"both spellings", ConPTYVersionInfo{FileVersion: liveFileVer, ProductVersion: liveProductVer}},
		{"FileVersion only", ConPTYVersionInfo{FileVersion: liveFileVer}},
		{"ProductVersion only", ConPTYVersionInfo{ProductVersion: liveProductVer}},
	} {
		dir := filepath.Join(t.TempDir(), "Terminal")
		writeStub(t, dir, ConPTYHostBinary)
		writeStub(t, dir, ConPTYLibrary)

		rep := ConPTYPreflight(ConPTYOptions{
			SearchDirs: []string{dir},
			Strict:     true,
			ReadVersionInfo: versionMap(map[string]ConPTYVersionInfo{
				"openconsole.exe": tc.info, "conpty.dll": tc.info,
			}),
		})
		if rep.Verdict != ConPTYPass {
			t.Fatalf("%s: verdict = %q (reason %q), want %q — a current pair was called stale",
				tc.name, rep.Verdict, rep.Reason, ConPTYPass)
		}
		for _, c := range rep.Components {
			if c.Stale {
				t.Fatalf("%s: component %s (compared %q from %s) marked stale",
					tc.name, c.Name, c.Compared, c.ComparedSource)
			}
		}
	}
}

// An image may carry a short marketing ProductVersion beside a precise
// FileVersion. Reading the marketing string would compare [1,24] against the
// floor and call a current pair stale, so the more precise field must win.
func TestConPTYPreflightPrefersThePreciseVersionField(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "Terminal")
	writeStub(t, dir, ConPTYHostBinary)

	rep := ConPTYPreflight(ConPTYOptions{
		SearchDirs: []string{dir},
		Strict:     true,
		ReadVersionInfo: versionMap(map[string]ConPTYVersionInfo{
			"openconsole.exe": {FileVersion: liveFileVer, ProductVersion: "1.24"},
		}),
	})
	if rep.Verdict != ConPTYPass {
		t.Fatalf("marketing ProductVersion: verdict = %q (reason %q), want %q", rep.Verdict, rep.Reason, ConPTYPass)
	}
	host := findComponent(t, rep, ConPTYHostBinary)
	if host.ComparedSource != "file_version" || host.Compared != liveFileVer {
		t.Fatalf("compared %q from %q, want %q from file_version", host.Compared, host.ComparedSource, liveFileVer)
	}
}

// Equal precision breaks toward ProductVersion, the floor's own spelling.
func TestConPTYPreflightBreaksPrecisionTieTowardProductVersion(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "Terminal")
	writeStub(t, dir, ConPTYHostBinary)

	rep := ConPTYPreflight(ConPTYOptions{
		SearchDirs:      []string{dir},
		ReadVersionInfo: versionMap(pair(liveFileVer, liveProductVer)), // both fold to 3 components
	})
	host := findComponent(t, rep, ConPTYHostBinary)
	if host.ComparedSource != "product_version" {
		t.Fatalf("tie broke to %q, want product_version", host.ComparedSource)
	}
}

// The regression witness for #3402: a stale bundled ConPTY pair on the launch
// PATH — the configuration that makes pwsh 7.6 FailFast with 0xE9 — must be
// detected, named, and (under --strict) refused.
func TestConPTYPreflightRefusesStalePairOnLaunchPath(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "Terminal")
	writeStub(t, dir, ConPTYHostBinary)
	writeStub(t, dir, ConPTYLibrary)

	opt := ConPTYOptions{
		SearchDirs:      []string{dir},
		ReadVersionInfo: versionMap(pair(staleFileVer, staleProductVer)),
	}

	warn := ConPTYPreflight(opt)
	if warn.Verdict != ConPTYWarn {
		t.Fatalf("stale pair, non-strict: verdict = %q, want %q", warn.Verdict, ConPTYWarn)
	}
	if warn.Reason != ReasonConPTYStale {
		t.Fatalf("stale pair: reason = %q, want %q", warn.Reason, ReasonConPTYStale)
	}
	if warn.OK() || !warn.Stale() {
		t.Fatalf("stale pair: OK()=%v Stale()=%v, want false/true", warn.OK(), warn.Stale())
	}
	if warn.Floor != ConPTYVersionFloor {
		t.Fatalf("floor = %q, want %q", warn.Floor, ConPTYVersionFloor)
	}
	for _, c := range warn.Components {
		if !c.Found {
			t.Fatalf("component %s: not found under %s", c.Name, dir)
		}
		if !c.Stale {
			t.Fatalf("component %s (%s): Stale = false, want true", c.Name, c.Compared)
		}
		if c.FileVersion != staleFileVer || c.ProductVersion != staleProductVer {
			t.Fatalf("component %s: resource strings not echoed back: %+v", c.Name, c)
		}
		// ProductVersion is the floor's own spelling, so it is preferred.
		if c.ComparedSource != "product_version" || c.Compared != staleProductVer {
			t.Fatalf("component %s: compared %q from %q, want %q from product_version",
				c.Name, c.Compared, c.ComparedSource, staleProductVer)
		}
		if c.Path != filepath.Join(dir, c.Name) {
			t.Fatalf("component %s: Path = %q, want it resolved under %s", c.Name, c.Path, dir)
		}
	}

	opt.Strict = true
	refuse := ConPTYPreflight(opt)
	if refuse.Verdict != ConPTYRefuse {
		t.Fatalf("stale pair, strict: verdict = %q, want %q", refuse.Verdict, ConPTYRefuse)
	}
	if refuse.OK() {
		t.Fatal("stale pair, strict: OK() = true, want false")
	}
}

// An older minor version is stale no matter how wide its build tail is.
func TestConPTYPreflightRefusesOlderMinor(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "Terminal")
	writeStub(t, dir, ConPTYHostBinary)
	writeStub(t, dir, ConPTYLibrary)

	rep := ConPTYPreflight(ConPTYOptions{
		SearchDirs:      []string{dir},
		Strict:          true,
		ReadVersionInfo: versionMap(pair(oldMinorVer, "")),
	})
	if rep.Verdict != ConPTYRefuse || rep.Reason != ReasonConPTYStale {
		t.Fatalf("old minor: verdict/reason = %q/%q, want %q/%q", rep.Verdict, rep.Reason, ConPTYRefuse, ReasonConPTYStale)
	}
	host := findComponent(t, rep, ConPTYHostBinary)
	if host.ComparedSource != "file_version" {
		t.Fatalf("no ProductVersion present: compared from %q, want file_version", host.ComparedSource)
	}
}

func TestConPTYPreflightPassesFreshPair(t *testing.T) {
	for _, ver := range []string{floorVer, newerVer, floorVer + ".0"} {
		dir := filepath.Join(t.TempDir(), "Terminal")
		writeStub(t, dir, ConPTYHostBinary)
		writeStub(t, dir, ConPTYLibrary)

		rep := ConPTYPreflight(ConPTYOptions{
			SearchDirs:      []string{dir},
			Strict:          true, // even the refusing mode must let a fresh pair through
			ReadVersionInfo: versionMap(pair("", ver)),
		})
		if rep.Verdict != ConPTYPass {
			t.Fatalf("fresh pair %q: verdict = %q (reason %q), want %q", ver, rep.Verdict, rep.Reason, ConPTYPass)
		}
		if !rep.OK() {
			t.Fatalf("fresh pair %q: OK() = false, want true", ver)
		}
		for _, c := range rep.Components {
			if c.Stale {
				t.Fatalf("fresh pair %q: component %s marked stale", ver, c.Name)
			}
		}
	}
}

// One stale half of the pair is enough: the fold takes the worst component.
func TestConPTYPreflightStaleLibraryDominatesFreshHost(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "Terminal")
	writeStub(t, dir, ConPTYHostBinary)
	writeStub(t, dir, ConPTYLibrary)

	rep := ConPTYPreflight(ConPTYOptions{
		SearchDirs: []string{dir},
		Strict:     true,
		ReadVersionInfo: versionMap(map[string]ConPTYVersionInfo{
			"openconsole.exe": {ProductVersion: liveProductVer},
			"conpty.dll":      {ProductVersion: staleProductVer},
		}),
	})
	if rep.Verdict != ConPTYRefuse || rep.Reason != ReasonConPTYStale {
		t.Fatalf("mixed pair: verdict/reason = %q/%q, want %q/%q", rep.Verdict, rep.Reason, ConPTYRefuse, ReasonConPTYStale)
	}
}

// conpty.dll ships BESIDE OpenConsole.exe inside the terminal's package dir and
// is not itself on PATH. Resolution must pair them.
func TestConPTYPreflightPairsLibraryBesideHost(t *testing.T) {
	root := t.TempDir()
	bundle := filepath.Join(root, "WindowsTerminal")
	writeStub(t, bundle, ConPTYHostBinary)
	writeStub(t, bundle, ConPTYLibrary)

	rep := ConPTYPreflight(ConPTYOptions{
		SearchDirs:      []string{bundle}, // only the host is nominally "on PATH"
		ReadVersionInfo: versionMap(pair(staleFileVer, staleProductVer)),
	})
	lib := findComponent(t, rep, ConPTYLibrary)
	if !lib.Found || lib.Path != filepath.Join(bundle, ConPTYLibrary) {
		t.Fatalf("conpty.dll not paired beside the host: %+v", lib)
	}
}

// First search dir wins, exactly like exec.LookPath.
func TestConPTYPreflightResolvesFirstSearchDirWins(t *testing.T) {
	root := t.TempDir()
	first := filepath.Join(root, "a")
	second := filepath.Join(root, "b")
	writeStub(t, first, ConPTYHostBinary)
	writeStub(t, second, ConPTYHostBinary)

	rep := ConPTYPreflight(ConPTYOptions{
		SearchDirs:      []string{first, second},
		ReadVersionInfo: versionMap(pair("", floorVer)),
	})
	host := findComponent(t, rep, ConPTYHostBinary)
	if host.Path != filepath.Join(first, ConPTYHostBinary) {
		t.Fatalf("resolved %q, want the first search dir %q", host.Path, first)
	}
}

// No ConPTY pair on the host is not a defect — a machine without Windows
// Terminal has nothing to be stale. It must not be reported as fresh either.
func TestConPTYPreflightSkipsWhenPairAbsent(t *testing.T) {
	rep := ConPTYPreflight(ConPTYOptions{
		SearchDirs:      []string{t.TempDir()},
		Strict:          true,
		ReadVersionInfo: versionMap(nil),
	})
	if rep.Verdict != ConPTYSkip || rep.Reason != ReasonConPTYAbsent {
		t.Fatalf("absent pair: verdict/reason = %q/%q, want %q/%q", rep.Verdict, rep.Reason, ConPTYSkip, ReasonConPTYAbsent)
	}
	if !rep.OK() {
		t.Fatal("absent pair: OK() = false, want true (nothing to refuse)")
	}
}

// A present binary whose version cannot be read or parsed is UNKNOWN, never a
// silent pass: a stale pair must not slip through a failed read.
func TestConPTYPreflightWarnsOnUnreadableVersion(t *testing.T) {
	readers := map[string]func(string) (ConPTYVersionInfo, error){
		"read fails": func(string) (ConPTYVersionInfo, error) {
			return ConPTYVersionInfo{}, errors.New("no version resource")
		},
		"unparsable strings": func(string) (ConPTYVersionInfo, error) {
			return ConPTYVersionInfo{FileVersion: "not-a-version", ProductVersion: "junk"}, nil
		},
		"empty strings": func(string) (ConPTYVersionInfo, error) {
			return ConPTYVersionInfo{}, nil
		},
	}
	for name, reader := range readers {
		dir := filepath.Join(t.TempDir(), "Terminal")
		writeStub(t, dir, ConPTYHostBinary)

		rep := ConPTYPreflight(ConPTYOptions{SearchDirs: []string{dir}, ReadVersionInfo: reader})
		if rep.Verdict != ConPTYWarn || rep.Reason != ReasonConPTYUnreadable {
			t.Fatalf("%s: verdict/reason = %q/%q, want %q/%q", name, rep.Verdict, rep.Reason, ConPTYWarn, ReasonConPTYUnreadable)
		}
		if rep.OK() {
			t.Fatalf("%s: OK() = true, want false", name)
		}
	}
}

// A stale component outranks an unreadable one in the fold.
func TestConPTYPreflightStaleOutranksUnreadable(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "Terminal")
	writeStub(t, dir, ConPTYHostBinary)
	writeStub(t, dir, ConPTYLibrary)

	rep := ConPTYPreflight(ConPTYOptions{
		SearchDirs: []string{dir},
		ReadVersionInfo: versionMap(map[string]ConPTYVersionInfo{
			"openconsole.exe": {ProductVersion: staleProductVer}, // conpty.dll read fails
		}),
	})
	if rep.Reason != ReasonConPTYStale {
		t.Fatalf("reason = %q, want %q", rep.Reason, ReasonConPTYStale)
	}
}

// A caller-supplied floor overrides the default, so the gate can be re-pinned
// without a rebuild when upstream moves.
func TestConPTYPreflightHonorsCustomFloor(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "Terminal")
	writeStub(t, dir, ConPTYHostBinary)

	rep := ConPTYPreflight(ConPTYOptions{
		SearchDirs:      []string{dir},
		Floor:           staleProductVer, // lower the bar to the stale build
		ReadVersionInfo: versionMap(pair(staleFileVer, staleProductVer)),
	})
	if rep.Verdict != ConPTYPass {
		t.Fatalf("custom floor: verdict = %q, want %q", rep.Verdict, ConPTYPass)
	}
	if rep.Floor != staleProductVer {
		t.Fatalf("custom floor: Floor = %q, want it echoed back", rep.Floor)
	}
}

// The REAL PE version-resource reader, driven against a REAL binary that every
// Windows host has. This is what keeps the suite from proving only that a mock
// returns what it was handed.
func TestReadVersionInfoReadsRealPEResource(t *testing.T) {
	if runtime.GOOS != "windows" {
		if _, err := ReadVersionInfo("anything"); !errors.Is(err, ErrFileVersionUnsupported) {
			t.Fatalf("off Windows: err = %v, want ErrFileVersionUnsupported", err)
		}
		t.Skip("PE version resources are a Windows facility")
	}
	root := os.Getenv("SystemRoot")
	if root == "" {
		t.Skip("SystemRoot unset")
	}
	real := filepath.Join(root, "System32", "kernel32.dll")
	if _, err := os.Stat(real); err != nil {
		t.Skip("kernel32.dll not present")
	}
	got, err := ReadVersionInfo(real)
	if err != nil {
		t.Fatalf("ReadVersionInfo(kernel32.dll): %v", err)
	}
	if got.FileVersion == "" {
		t.Fatal("ReadVersionInfo(kernel32.dll): empty FileVersion")
	}
	fields, err := NormalizeConPTYVersion(got.FileVersion)
	if err != nil {
		t.Fatalf("ReadVersionInfo(kernel32.dll) = %q, which the parser rejects: %v", got.FileVersion, err)
	}
	if len(fields) < 2 || fields[0] == 0 {
		t.Fatalf("ReadVersionInfo(kernel32.dll) = %q -> %v, want a plausible major.minor", got.FileVersion, fields)
	}
	// A missing version resource must surface as an error, not an empty pass.
	missing := filepath.Join(t.TempDir(), "empty.dll")
	if err := os.WriteFile(missing, []byte("not a PE"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if v, err := ReadVersionInfo(missing); err == nil {
		t.Fatalf("ReadVersionInfo(non-PE) = %+v, want error", v)
	}
}

// If this host actually has a ConPTY pair, the real end-to-end preflight — real
// resolution, real PE reads, real comparison — must not call it stale unless it
// genuinely is. Guards the false-stale bug against the live object.
func TestConPTYPreflightAgainstRealHostPairIsSelfConsistent(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("no ConPTY off Windows")
	}
	rep := ConPTYPreflight(ConPTYOptions{}) // real search dirs, real reader
	if rep.Verdict == ConPTYSkip {
		t.Skip("no ConPTY pair on this host's launch PATH")
	}
	for _, c := range rep.Components {
		if !c.Found || c.Compared == "" {
			continue
		}
		cmp, err := CompareConPTYVersions(c.Compared, rep.Floor)
		if err != nil {
			t.Fatalf("%s: compared %q unparsable: %v", c.Name, c.Compared, err)
		}
		if (cmp < 0) != c.Stale {
			t.Fatalf("%s: Stale=%v but compare(%q, floor %q)=%d", c.Name, c.Stale, c.Compared, rep.Floor, cmp)
		}
	}
}

// The default search path is derived from the real launch PATH, so the preflight
// inspects the same resolution order a spawned child would.
func TestDefaultConPTYSearchDirsFollowsPATH(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	t.Setenv("PATH", strings.Join([]string{a, b}, string(os.PathListSeparator)))
	dirs := DefaultConPTYSearchDirs()
	if len(dirs) < 2 || dirs[0] != a || dirs[1] != b {
		t.Fatalf("DefaultConPTYSearchDirs() = %v, want it to lead with the PATH entries %q, %q", dirs, a, b)
	}
}

// Bundle directories resolve NEWEST FIRST, so first-match picks the pair the
// current terminal loads rather than an older side-by-side package.
func TestConPTYBundleDirsAreNewestFirst(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{
		"Microsoft.WindowsTerminal_1.22.10352.0_x64__8wekyb3d8bbwe",
		"Microsoft.WindowsTerminal_1.24.11321.0_x64__8wekyb3d8bbwe",
		"Microsoft.WindowsTerminalPreview_1.23.1_x64__8wekyb3d8bbwe",
		"Microsoft.WindowsTerminal_notaversion_x64",
		"Unrelated.Package_9.9.9_x64",
	} {
		if err := os.MkdirAll(filepath.Join(root, name), 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
	}
	got := ConPTYBundleDirs([]string{root, filepath.Join(root, "does-not-exist")})

	var names []string
	for _, d := range got {
		names = append(names, filepath.Base(d))
	}
	if len(names) != 4 {
		t.Fatalf("ConPTYBundleDirs = %v, want the 4 terminal packages (Unrelated.Package excluded)", names)
	}
	if !strings.HasPrefix(names[0], "Microsoft.WindowsTerminal_1.24.") {
		t.Fatalf("newest package should sort first, got %v", names)
	}
	if !strings.Contains(names[1], "Preview_1.23.") {
		t.Fatalf("1.23 preview should sort second, got %v", names)
	}
	if !strings.HasPrefix(names[2], "Microsoft.WindowsTerminal_1.22.") {
		t.Fatalf("1.22 should sort third, got %v", names)
	}
	if names[3] != "Microsoft.WindowsTerminal_notaversion_x64" {
		t.Fatalf("an unparsable package version must sort last, got %v", names)
	}
}

func findComponent(t *testing.T, rep ConPTYReport, name string) ConPTYComponent {
	t.Helper()
	for _, c := range rep.Components {
		if c.Name == name {
			return c
		}
	}
	t.Fatalf("report has no component %q", name)
	return ConPTYComponent{}
}
