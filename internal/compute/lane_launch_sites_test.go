package compute

import (
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

const launchSiteDir = "internal/compute"

// scannedLaunch is one triple-angle launch as written: the kernel with its template arguments, and the grid,
// threads and dynamic shared-memory expressions of its configuration with whitespace normalised. SmemExpr is
// empty when the configuration has no third argument.
type scannedLaunch struct {
	Kernel, ThreadsExpr, GridExpr, SmemExpr string
}

func (l scannedLaunch) String() string {
	return fmt.Sprintf("%s<<<%s, %s, %s>>>", l.Kernel, l.GridExpr, l.ThreadsExpr, l.SmemExpr)
}

// scanKernelNames returns the kernel of every triple-angle launch in src.
func scanKernelNames(src string) ([]string, error) {
	launches, err := scanLaunches(src)
	if err != nil {
		return nil, err
	}
	names := make([]string, len(launches))
	for i, l := range launches {
		names[i] = l.Kernel
	}
	return names, nil
}

// scanLaunches returns every triple-angle launch in src. Line continuations are joined and comments and
// string literals are blanked first, so launch syntax inside a comment is not counted.
func scanLaunches(src string) ([]scannedLaunch, error) {
	text := strings.ReplaceAll(strings.ReplaceAll(src, "\r\n", "\n"), "\\\n", " ")
	text = blankCommentsAndStrings(text)
	var out []scannedLaunch
	for pos := 0; ; {
		k := strings.Index(text[pos:], "<<<")
		if k < 0 {
			return out, nil
		}
		p := pos + k
		name, err := kernelBefore(text, p)
		if err != nil {
			return nil, err
		}
		args, end, err := launchConfigArgs(text, p+3)
		if err != nil {
			return nil, err
		}
		l := scannedLaunch{Kernel: name, GridExpr: args[0], ThreadsExpr: args[1]}
		if len(args) > 2 {
			l.SmemExpr = args[2]
		}
		out = append(out, l)
		pos = end
	}
}

// launchConfigArgs splits the configuration that starts at start, just after "<<<", into its top-level comma
// separated arguments, and returns them with the index just past the closing ">>>". Commas inside parentheses,
// brackets or braces belong to one argument.
func launchConfigArgs(text string, start int) ([]string, int, error) {
	depth, last := 0, start
	var args []string
	for i := start; i < len(text); i++ {
		switch c := text[i]; {
		case c == '(' || c == '[' || c == '{':
			depth++
		case c == ')' || c == ']' || c == '}':
			depth--
		case depth == 0 && c == ',':
			args = append(args, normalizeExpr(text[last:i]))
			last = i + 1
		case depth == 0 && strings.HasPrefix(text[i:], ">>>"):
			args = append(args, normalizeExpr(text[last:i]))
			if len(args) < 2 {
				return nil, 0, fmt.Errorf("launch at offset %d has no grid and threads arguments", start)
			}
			return args, i + 3, nil
		}
	}
	return nil, 0, fmt.Errorf("unterminated launch configuration at offset %d", start)
}

func normalizeExpr(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// kernelBefore reads the kernel identifier and its template arguments that end just before the launch at p.
func kernelBefore(text string, p int) (string, error) {
	q := skipSpaceBack(text, p)
	tmpl := ""
	if q > 0 && text[q-1] == '>' {
		r, depth := q-1, 0
		for ; r >= 0; r-- {
			if text[r] == '>' {
				depth++
			} else if text[r] == '<' {
				depth--
				if depth == 0 {
					break
				}
			}
		}
		if r < 0 {
			return "", fmt.Errorf("unbalanced template before launch at offset %d", p)
		}
		tmpl = strings.Join(strings.Fields(text[r:q]), " ")
		q = skipSpaceBack(text, r)
	}
	end := q
	for q > 0 && isKernelIdentByte(text[q-1]) {
		q--
	}
	if q == end {
		return "", fmt.Errorf("launch without a kernel identifier at offset %d", p)
	}
	return text[q:end] + tmpl, nil
}

func skipSpaceBack(text string, q int) int {
	for q > 0 && isSpaceByte(text[q-1]) {
		q--
	}
	return q
}

func isSpaceByte(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'
}

func isKernelIdentByte(c byte) bool {
	return c == '_' || c == ':' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

// blankCommentsAndStrings replaces comments with spaces and string or character literal contents with
// spaces, keeping newlines so the text keeps its line structure.
func blankCommentsAndStrings(text string) string {
	var b strings.Builder
	n := len(text)
	for i := 0; i < n; {
		switch {
		case strings.HasPrefix(text[i:], "//"):
			j := strings.IndexByte(text[i:], '\n')
			if j < 0 {
				j = n - i
			}
			b.WriteString(strings.Repeat(" ", j))
			i += j
		case strings.HasPrefix(text[i:], "/*"):
			end := n
			if j := strings.Index(text[i+2:], "*/"); j >= 0 {
				end = i + 2 + j + 2
			}
			b.WriteString(blankKeepNewlines(text[i:end]))
			i = end
		case text[i] == '"' || text[i] == '\'':
			end := stringLiteralEnd(text, i)
			b.WriteString(blankStringLiteral(text[i:end]))
			i = end
		default:
			b.WriteByte(text[i])
			i++
		}
	}
	return b.String()
}

// stringLiteralEnd returns the index just past the literal that opens at i, stopping at a newline if the
// literal is unterminated.
func stringLiteralEnd(text string, i int) int {
	q := text[i]
	j := i + 1
	for j < len(text) && text[j] != q && text[j] != '\n' {
		if text[j] == '\\' {
			j += 2
		} else {
			j++
		}
	}
	if j+1 > len(text) {
		return len(text)
	}
	return j + 1
}

func blankStringLiteral(lit string) string {
	if len(lit) < 2 || lit[len(lit)-1] != lit[0] {
		return lit
	}
	return lit[:1] + strings.Repeat(" ", len(lit)-2) + lit[:1]
}

func blankKeepNewlines(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '\n' {
			return r
		}
		return ' '
	}, s)
}

// cudaSourceFiles maps each .cu and .cuh file under the package directory to its repo-relative path.
func cudaSourceFiles(t *testing.T) map[string]string {
	t.Helper()
	files := map[string]string{}
	err := filepath.WalkDir(".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if ext := filepath.Ext(p); ext == ".cu" || ext == ".cuh" {
			files[path.Join(launchSiteDir, filepath.ToSlash(p))] = p
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", launchSiteDir, err)
	}
	return files
}

// launchSiteDataFile is the data file the inventory rows live in, named in failure messages.
const launchSiteDataFile = "internal/compute/lane_launch_sites.txt"

// launchInventory groups the data rows by source file, keeping each row's line for the failure messages.
func launchInventory(t *testing.T) map[string][]launchSiteRow {
	t.Helper()
	rows, err := launchSiteRowsChecked()
	if err != nil {
		t.Fatalf("%s: %v", launchSiteDataFile, err)
	}
	inv := map[string][]launchSiteRow{}
	for _, r := range rows {
		inv[r.site.File] = append(inv[r.site.File], r)
	}
	return inv
}

// dataRowText renders a launch as the tab-separated line lane_launch_sites.txt expects, so a missing-row
// failure can be pasted into the data file.
func dataRowText(file string, l scannedLaunch) string {
	tpb := 0
	if n, err := strconv.Atoi(l.ThreadsExpr); err == nil && n > 0 {
		tpb = n
	}
	return strings.Join([]string{file, l.Kernel, l.ThreadsExpr, strconv.Itoa(tpb), l.GridExpr, l.SmemExpr}, "\t")
}

func launchStrings(launches []scannedLaunch) []string {
	out := make([]string, len(launches))
	for i, l := range launches {
		out[i] = l.String()
	}
	return out
}

func scanFile(t *testing.T, local string) []scannedLaunch {
	t.Helper()
	src, err := os.ReadFile(local)
	if err != nil {
		t.Fatalf("read %s: %v", local, err)
	}
	launches, err := scanLaunches(string(src))
	if err != nil {
		t.Fatalf("scan %s: %v", local, err)
	}
	return launches
}

// compareLaunches reports the source launch texts with no inventory row and the inventory rows with no source
// launch text. The inventory holds one row per distinct launch text, so repeated identical launches compare as
// one. A second launch of a kernel with a different configuration is a distinct text and is reported.
func compareLaunches(t *testing.T, file string, got []scannedLaunch, want []launchSiteRow) {
	t.Helper()
	inSource := map[scannedLaunch]bool{}
	for _, l := range got {
		inSource[l] = true
	}
	inTable := map[scannedLaunch]launchSiteRow{}
	for _, r := range want {
		s := r.site
		inTable[scannedLaunch{Kernel: s.Kernel, ThreadsExpr: s.ThreadsExpr, GridExpr: s.GridExpr, SmemExpr: s.SmemExpr}] = r
	}
	var missing []scannedLaunch
	for l := range inSource {
		if _, ok := inTable[l]; !ok {
			missing = append(missing, l)
		}
	}
	sort.Slice(missing, func(i, j int) bool { return missing[i].String() < missing[j].String() })
	for _, l := range missing {
		t.Errorf("%s: source launch %s has no row; add this line to %s:\n%s",
			file, l, launchSiteDataFile, dataRowText(file, l))
	}
	var stale []launchSiteRow
	for l, r := range inTable {
		if !inSource[l] {
			stale = append(stale, r)
		}
	}
	sort.Slice(stale, func(i, j int) bool { return stale[i].line < stale[j].line })
	for _, r := range stale {
		t.Errorf("%s: %s line %d lists %s, which is no longer in source; delete that line",
			file, launchSiteDataFile, r.line, launchSiteRowLabel(r.site))
	}
}

func launchSiteRowLabel(s LaunchSite) string {
	return fmt.Sprintf("%s<<<%s, %s, %s>>>", s.Kernel, s.GridExpr, s.ThreadsExpr, s.SmemExpr)
}

// TestLaunchSitesCoverScan checks the inventory against the .cu and .cuh files on disk, comparing each file's
// launch configurations (kernel, grid, threads and shared-memory text) as a multiset. A listed file that is
// absent fails, since its launches would go unchecked. A file with launches but no inventory rows fails too.
func TestLaunchSitesCoverScan(t *testing.T) {
	inventory := launchInventory(t)
	onDisk := cudaSourceFiles(t)

	files := make([]string, 0, len(inventory))
	for file := range inventory {
		files = append(files, file)
	}
	sort.Strings(files)
	for _, file := range files {
		t.Run(file, func(t *testing.T) {
			local, ok := onDisk[file]
			if !ok {
				t.Fatalf("%s is in the launch inventory but absent on disk; its launch sites cannot be checked", file)
			}
			compareLaunches(t, file, scanFile(t, local), inventory[file])
		})
	}

	for file, local := range onDisk {
		if _, listed := inventory[file]; listed {
			continue
		}
		for _, l := range scanFile(t, local) {
			t.Errorf("%s: source launch %s has no row; add this line to %s:\n%s",
				file, l, launchSiteDataFile, dataRowText(file, l))
		}
	}
}

// TestLaunchSiteDataParses checks that the embedded data file parses, and that a malformed file is reported
// with the line that is wrong.
func TestLaunchSiteDataParses(t *testing.T) {
	rows, err := launchSiteRowsChecked()
	if err != nil {
		t.Fatalf("%s: %v", launchSiteDataFile, err)
	}
	if len(rows) == 0 {
		t.Fatalf("%s has no launch rows", launchSiteDataFile)
	}
	bad := []struct{ name, data, want string }{
		{"empty", "", "line 1 must be the header"},
		{"wrong header", "file\tkernel\n", "line 1 must be the header"},
		{"short row", launchSiteDataHeader + "\na\tb\t1\n", "line 2 has 3 tab-separated fields"},
		{"empty kernel", launchSiteDataHeader + "\na\t\t1\t1\tg\ts\n", "line 2 has an empty file or kernel"},
		{"bad threads", launchSiteDataHeader + "\na\tk\t1\tx\tg\ts\n", "line 2 threads_per_block \"x\""},
		{"blank line", launchSiteDataHeader + "\na\tk\t1\t1\tg\ts\n\n", "line 3 has 1 tab-separated fields"},
	}
	for _, tt := range bad {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseLaunchSiteRows(tt.data)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("parseLaunchSiteRows error = %v, want it to contain %q", err, tt.want)
			}
		})
	}
}

// TestLaunchSiteLiteralThreads checks that a row whose thread expression is an integer literal records that
// integer, so its partial-warp finding follows from the source text. Macro and identifier expressions are not
// resolved here.
func TestLaunchSiteLiteralThreads(t *testing.T) {
	literal := 0
	for _, s := range LaunchSites() {
		n, err := strconv.Atoi(s.ThreadsExpr)
		if err != nil {
			continue
		}
		literal++
		if s.ThreadsPerBlock != n {
			t.Errorf("%s %s: threads %q recorded as %d", s.File, s.Kernel, s.ThreadsExpr, s.ThreadsPerBlock)
		}
	}
	if literal == 0 {
		t.Fatal("no literal thread expressions; the check would run on nothing")
	}
}

func compareLaunchSites(a, b LaunchSite) int {
	pairs := [][2]string{
		{a.File, b.File},
		{a.Kernel, b.Kernel},
		{a.ThreadsExpr, b.ThreadsExpr},
		{a.GridExpr, b.GridExpr},
		{a.SmemExpr, b.SmemExpr},
	}
	for _, p := range pairs {
		if c := strings.Compare(p[0], p[1]); c != 0 {
			return c
		}
	}
	return 0
}

func TestLaunchSitesSortedUnique(t *testing.T) {
	sites := LaunchSites()
	if len(sites) == 0 {
		t.Fatal("launch inventory is empty")
	}
	for i := 1; i < len(sites); i++ {
		if compareLaunchSites(sites[i-1], sites[i]) >= 0 {
			t.Errorf("site %d (%s %s) does not sort strictly after site %d (%s %s)",
				i, sites[i].File, sites[i].Kernel, i-1, sites[i-1].File, sites[i-1].Kernel)
		}
	}
	sites[0].Kernel = "mutated"
	if LaunchSites()[0].Kernel == "mutated" {
		t.Error("LaunchSites shares storage with the inventory")
	}
}

func TestLaunchSitePartialWarpLanes(t *testing.T) {
	tests := []struct {
		name  string
		site  LaunchSite
		warp  int
		lanes int
		known bool
	}{
		{"whole warps", LaunchSite{ThreadsPerBlock: 128}, 32, 0, true},
		{"single thread", LaunchSite{ThreadsPerBlock: 1}, 32, 1, true},
		{"partial tail", LaunchSite{ThreadsPerBlock: 100}, 32, 4, true},
		{"one over a warp", LaunchSite{ThreadsPerBlock: 33}, 32, 1, true},
		{"symbolic threads", LaunchSite{ThreadsExpr: "block"}, 32, 0, false},
		{"non-positive warp size", LaunchSite{ThreadsPerBlock: 64}, 0, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lanes, known := tt.site.PartialWarpLanes(tt.warp)
			if lanes != tt.lanes || known != tt.known {
				t.Errorf("PartialWarpLanes(%d) = (%d, %v), want (%d, %v)", tt.warp, lanes, known, tt.lanes, tt.known)
			}
		})
	}
}

// launchSitePartialWarpPins names the launches whose literal thread count is a single thread, which
// leaves 31 idle lanes in the only warp of every block.
var launchSitePartialWarpPins = []string{
	"internal/compute/cuda/graph_capture_once.cuh k_graph_once_position",
	"internal/compute/cuda/inkling2_loop.cuh k_ink2_argkey_index",
	"internal/compute/cuda/inkling2_prefill_nvfp4.cuh k_ink2p_nvf4_probe",
	"internal/compute/cuda/model_candidates.cuh k_v41_bf16_publish_status",
	"internal/compute/cuda/paged_attention_vllm.cuh fak_vllm_pa::paged_store_int_kernel",
	"internal/compute/cuda_v41_routed_mmid.cu k_v41_bf16_publish_status",
}

func TestLaunchSiteFindings(t *testing.T) {
	const warp = 32
	var partial []string
	unjudged := 0
	for _, f := range LaunchSiteFindings(warp) {
		switch f.Kind {
		case LaunchFindingPartialWarp:
			if f.PartialLanes == 0 || f.PartialLanes != f.Site.ThreadsPerBlock%warp {
				t.Errorf("%s %s: partial lanes %d for %d threads", f.Site.File, f.Site.Kernel, f.PartialLanes, f.Site.ThreadsPerBlock)
			}
			partial = append(partial, f.Site.File+" "+f.Site.Kernel)
		case LaunchFindingUnjudged:
			if f.Site.ThreadsPerBlock != 0 {
				t.Errorf("%s %s: literal %d threads reported as unjudged", f.Site.File, f.Site.Kernel, f.Site.ThreadsPerBlock)
			}
			unjudged++
		default:
			t.Errorf("%s %s: unknown finding kind %q", f.Site.File, f.Site.Kernel, f.Kind)
		}
	}
	if strings.Join(partial, "\n") != strings.Join(launchSitePartialWarpPins, "\n") {
		t.Errorf("partial-warp launches = %v, want %v", partial, launchSitePartialWarpPins)
	}
	symbolic := 0
	for _, s := range LaunchSites() {
		if s.ThreadsPerBlock == 0 {
			symbolic++
		}
	}
	if unjudged != symbolic {
		t.Errorf("unjudged findings = %d, symbolic sites = %d", unjudged, symbolic)
	}
	if got := len(LaunchSiteFindings(0)); got != len(LaunchSites()) {
		t.Errorf("LaunchSiteFindings(0) = %d findings, want every site (%d) unjudged", got, len(LaunchSites()))
	}
}

// TestScanLaunchConfigs checks that the configuration arguments are split at top-level commas and normalised,
// so a changed grid, thread or shared-memory expression changes the compared text.
func TestScanLaunchConfigs(t *testing.T) {
	tests := []struct {
		name string
		src  string
		want scannedLaunch
	}{
		{"three arguments", "k<<<g, 128, 768>>>(x);", scannedLaunch{Kernel: "k", GridExpr: "g", ThreadsExpr: "128", SmemExpr: "768"}},
		{"no shared memory argument", "k<<<g, 1>>>(x);", scannedLaunch{Kernel: "k", GridExpr: "g", ThreadsExpr: "1"}},
		{"commas inside parentheses", "k<<<dim3(a,  b), (n + 255) / 256, 0, s>>>(x);", scannedLaunch{Kernel: "k", GridExpr: "dim3(a, b)", ThreadsExpr: "(n + 255) / 256", SmemExpr: "0"}},
		{"stream argument ignored", "k<<<g, 32, 0, s>>>(x);", scannedLaunch{Kernel: "k", GridExpr: "g", ThreadsExpr: "32", SmemExpr: "0"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := scanLaunches(tt.src)
			if err != nil {
				t.Fatalf("scanLaunches: %v", err)
			}
			if len(got) != 1 || got[0] != tt.want {
				t.Errorf("scanLaunches = %+v, want [%+v]", got, tt.want)
			}
		})
	}
}

func TestScanKernelNames(t *testing.T) {
	tests := []struct {
		name string
		src  string
		want []string
	}{
		{"plain", "k<<<g, 128, 0, s>>>(x);", []string{"k"}},
		{"template args", "k_flash<false> <<<g, 256>>>(x);", []string{"k_flash<false>"}},
		{"qualified", "ns::k<1, Dot><<<g, b>>>(x);", []string{"ns::k<1, Dot>"}},
		{"comment ignored", "// k<<<g, b>>>\n/* j<<<g, b>>> */ m<<<g, b>>>();", []string{"m"}},
		{"string ignored", "s(\"k<<<g, b>>>\"); n<<<g, b>>>();", []string{"n"}},
		{"continuation", "#define L(F) \\\n  k<F><<<g, \\\n  b>>>(x)", []string{"k<F>"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := scanKernelNames(tt.src)
			if err != nil {
				t.Fatalf("scanKernelNames: %v", err)
			}
			if strings.Join(got, "|") != strings.Join(tt.want, "|") {
				t.Errorf("scanKernelNames = %q, want %q", got, tt.want)
			}
		})
	}
}
