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
// launch texts match the rows, and TestLaunchSitesGenerated enforces that every row equals what the resolver
// in the package's tests derives from source, so an edit to a launch configuration fails until the table is
// regenerated.
//
// Resolution says how the thread count was judged. A literal or resolved-chain row carries its count in
// ThreadsPerBlock; a runtime or template row keeps the count at zero, so it is unjudged and never clean.

// LaunchSite is one CUDA kernel launch as written in source.
type LaunchSite struct {
	File, Kernel, ThreadsExpr string
	ThreadsPerBlock           int
	Resolution                string
	GridExpr, SmemExpr        string
}

// LaunchSiteFinding flags one launch. Kind is LaunchFindingPartialWarp when the resolved thread count leaves a
// partial warp in every block, or LaunchFindingUnjudged when the count is not resolved.
type LaunchSiteFinding struct {
	Site         LaunchSite
	Kind         string
	PartialLanes int
}

const (
	LaunchFindingPartialWarp = "partial_warp"
	LaunchFindingUnjudged    = "unjudged"
)

// Resolution classes of a launch's thread count. Literal is a bare integer; resolved-chain is an integer reached
// through constants, macros, dim3 products or single-return helpers; runtime and template are unresolved, the
// template class when the count depends on a template parameter.
const (
	LaunchGenResLiteral  = "literal"
	LaunchGenResChain    = "resolved-chain"
	LaunchGenResRuntime  = "runtime"
	LaunchGenResTemplate = "template"
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

// PartialWarpActiveLanes returns the active lanes in the trailing warp of each block, or 0 when the block fills
// its warps exactly. known is false when the thread count is unresolved or warpSize is not positive.
func (s LaunchSite) PartialWarpActiveLanes(warpSize int) (lanes int, known bool) {
	if warpSize <= 0 || s.ThreadsPerBlock <= 0 {
		return 0, false
	}
	return s.ThreadsPerBlock % warpSize, true
}

// LaunchSiteFindings flags every launch that leaves a partial warp in each block and every launch whose
// thread count is unresolved. Unresolved launches are reported as unjudged, never as clean.
func LaunchSiteFindings(warpSize int) []LaunchSiteFinding {
	var out []LaunchSiteFinding
	for _, r := range launchSiteRows() {
		s := r.site
		lanes, known := s.PartialWarpActiveLanes(warpSize)
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
const launchSiteDataHeader = "file\tkernel\tthreads_expr\tthreads_per_block\tthreads_resolution\tgrid_expr\tsmem_expr"

// launchSiteRow is one parsed data row with its 1-based line in the data file, so a maintainer can be sent
// to the exact line that needs editing.
type launchSiteRow struct {
	site LaunchSite
	line int
}

//go:embed lane_launch_sites.txt
var launchSiteData string

var (
	launchSiteOnce   sync.Once
	launchSiteParsed []launchSiteRow
	launchSiteErr    error
)

// launchSiteRowsChecked parses the embedded data once and returns the error instead of panicking, so the
// tests can report a malformed table with its line number.
func launchSiteRowsChecked() ([]launchSiteRow, error) {
	launchSiteOnce.Do(func() {
		launchSiteParsed, launchSiteErr = parseLaunchSiteRows(launchSiteData)
	})
	return launchSiteParsed, launchSiteErr
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
		row, err := launchGenParseRow(text, line)
		if err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("no launch rows after the header")
	}
	return rows, nil
}

func launchGenParseRow(text string, line int) (launchSiteRow, error) {
	fields := strings.Split(text, "\t")
	if len(fields) != 7 {
		return launchSiteRow{}, fmt.Errorf("line %d has %d tab-separated fields, want 7", line, len(fields))
	}
	if fields[0] == "" || fields[1] == "" {
		return launchSiteRow{}, fmt.Errorf("line %d has an empty file or kernel", line)
	}
	tpb, err := strconv.Atoi(fields[3])
	if err != nil || tpb < 0 {
		return launchSiteRow{}, fmt.Errorf("line %d threads_per_block %q is not a non-negative integer", line, fields[3])
	}
	res := fields[4]
	switch res {
	case LaunchGenResLiteral:
		if n, err := strconv.Atoi(fields[2]); err != nil || n != tpb || tpb <= 0 {
			return launchSiteRow{}, fmt.Errorf("line %d literal resolution needs threads_expr %q to equal a positive threads_per_block %d", line, fields[2], tpb)
		}
	case LaunchGenResChain:
		if tpb <= 0 {
			return launchSiteRow{}, fmt.Errorf("line %d resolved-chain needs a positive threads_per_block", line)
		}
	case LaunchGenResRuntime, LaunchGenResTemplate:
		if tpb != 0 {
			return launchSiteRow{}, fmt.Errorf("line %d %s is unresolved, so threads_per_block must be 0", line, res)
		}
	default:
		return launchSiteRow{}, fmt.Errorf("line %d threads_resolution %q is not one of literal, resolved-chain, runtime, template", line, res)
	}
	return launchSiteRow{
		site: LaunchSite{
			File:            fields[0],
			Kernel:          fields[1],
			ThreadsExpr:     fields[2],
			ThreadsPerBlock: tpb,
			Resolution:      res,
			GridExpr:        fields[5],
			SmemExpr:        fields[6],
		},
		line: line,
	}, nil
}
