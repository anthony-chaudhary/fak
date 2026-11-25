package compute

import (
	_ "embed"
	"fmt"
	"strconv"
	"strings"
	"sync"
)

// lane_launch_sites.go is the static inventory of CUDA kernel launches under internal/compute. Each row of
// lane_launch_sites.txt is one distinct launch text in its file: the kernel with its template arguments and
// the grid, threads and dynamic shared-memory expressions. TestLaunchSitesCoverScan enforces that each file's
// launch texts match the rows, so an edit to a launch configuration fails until the data file changes.
//
// ThreadsPerBlock is a literal only when the threads expression resolves to a positive integer through
// integer arithmetic, dim3 products, and #define or constexpr integers whose every definition in
// internal/compute agrees. Zero means symbolic: the count depends on a runtime value, a struct member,
// or a name with conflicting definitions, so a symbolic site is unjudged rather than clean. The values come
// from a generator outside this repo; no test re-resolves a macro or identifier to its value.

// LaunchSite is one CUDA kernel launch as written in source.
type LaunchSite struct {
	File, Kernel, ThreadsExpr string
	ThreadsPerBlock           int
	GridExpr, SmemExpr        string
}

// LaunchSiteFinding flags one launch. Kind is LaunchFindingPartialWarp when the literal thread count
// leaves a partial warp in every block, or LaunchFindingUnjudged when the count is symbolic.
type LaunchSiteFinding struct {
	Site         LaunchSite
	Kind         string
	PartialLanes int
}

const (
	LaunchFindingPartialWarp = "partial_warp"
	LaunchFindingUnjudged    = "unjudged"
)

// LaunchSites returns a copy of the inventory in (File, Kernel, ThreadsExpr, GridExpr, SmemExpr) order.
func LaunchSites() []LaunchSite {
	rows := launchSiteRows()
	sites := make([]LaunchSite, len(rows))
	for i, r := range rows {
		sites[i] = r.site
	}
	return sites
}

// PartialWarpLanes returns the active lanes in the trailing warp of each block, or 0 when the block fills
// its warps exactly. known is false when the thread count is symbolic or warpSize is not positive.
func (s LaunchSite) PartialWarpLanes(warpSize int) (lanes int, known bool) {
	if warpSize <= 0 || s.ThreadsPerBlock <= 0 {
		return 0, false
	}
	return s.ThreadsPerBlock % warpSize, true
}

// LaunchSiteFindings flags every launch that leaves a partial warp in each block and every launch whose
// thread count is symbolic. Symbolic launches are reported as unjudged, never as clean.
func LaunchSiteFindings(warpSize int) []LaunchSiteFinding {
	var out []LaunchSiteFinding
	for _, r := range launchSiteRows() {
		s := r.site
		lanes, known := s.PartialWarpLanes(warpSize)
		switch {
		case !known:
			out = append(out, LaunchSiteFinding{Site: s, Kind: LaunchFindingUnjudged})
		case lanes != 0:
			out = append(out, LaunchSiteFinding{Site: s, Kind: LaunchFindingPartialWarp, PartialLanes: lanes})
		}
	}
	return out
}

// launchSiteDataHeader names the tab-separated columns of lane_launch_sites.txt, in order.
const launchSiteDataHeader = "file\tkernel\tthreads_expr\tthreads_per_block\tgrid_expr\tsmem_expr"

// launchSiteRow is one parsed data row with its 1-based line in the data file, so a maintainer can be sent
// to the exact line that needs editing.
type launchSiteRow struct {
	site LaunchSite
	line int
}

//go:embed lane_launch_sites.txt
var launchSiteData string

var (
	launchSiteOnce  sync.Once
	launchSiteCache []launchSiteRow
	launchSiteErr   error
)

// launchSiteRowsChecked parses the embedded data once and returns the error instead of panicking, so the
// tests can report a malformed table with its line number.
func launchSiteRowsChecked() ([]launchSiteRow, error) {
	launchSiteOnce.Do(func() {
		launchSiteCache, launchSiteErr = parseLaunchSiteRows(launchSiteData)
	})
	return launchSiteCache, launchSiteErr
}

// launchSiteRows panics on a malformed embedded table. The panic happens on first use, not at package init,
// and the test suite parses the same data, so a malformed table fails there before it reaches a caller.
func launchSiteRows() []launchSiteRow {
	rows, err := launchSiteRowsChecked()
	if err != nil {
		panic("compute: lane_launch_sites.txt: " + err.Error())
	}
	return rows
}

func parseLaunchSiteRows(data string) ([]launchSiteRow, error) {
	lines := strings.Split(strings.TrimSuffix(data, "\n"), "\n")
	if lines[0] != launchSiteDataHeader {
		return nil, fmt.Errorf("line 1 must be the header %q", launchSiteDataHeader)
	}
	var rows []launchSiteRow
	for i, text := range lines[1:] {
		line := i + 2
		fields := strings.Split(text, "\t")
		if len(fields) != 6 {
			return nil, fmt.Errorf("line %d has %d tab-separated fields, want 6", line, len(fields))
		}
		if fields[0] == "" || fields[1] == "" {
			return nil, fmt.Errorf("line %d has an empty file or kernel", line)
		}
		tpb, err := strconv.Atoi(fields[3])
		if err != nil || tpb < 0 {
			return nil, fmt.Errorf("line %d threads_per_block %q is not a non-negative integer", line, fields[3])
		}
		rows = append(rows, launchSiteRow{
			site: LaunchSite{
				File:            fields[0],
				Kernel:          fields[1],
				ThreadsExpr:     fields[2],
				ThreadsPerBlock: tpb,
				GridExpr:        fields[4],
				SmemExpr:        fields[5],
			},
			line: line,
		})
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("no launch rows after the header")
	}
	return rows, nil
}
