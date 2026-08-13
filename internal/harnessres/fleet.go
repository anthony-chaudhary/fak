package harnessres

// Fleet rollup: the same meter, one level up. The Sampler above measures ONE guarded
// session (kernel half + wrapped child). A live fleet is N of those plus everything they
// spawned — broker/MCP hosts, tool runtimes — and until this file existed the only way to
// see that total was `Get-Process` in a shell, i.e. from outside fak. Rung 1 of epic
// #6552 (#6557) makes the fleet's own footprint readable BY the thing that spawned it,
// because every later rung of that epic is justified by a delta in this number.
//
// The split of responsibility is deliberate and mirrors host.go: this leaf still imports
// nothing internal and reads no census itself. The caller supplies the pid/ppid/cmdline
// rows (fak wires internal/procguard, which already owns the per-GOOS census); this file
// selects the fak-owned subtree from them, classifies each process, reads the per-PID
// resource axes through the same per-platform readers as the single-session path
// (sample_{unix,windows,other}.go), and folds one rollup. Absent axes stay behind
// presence bits exactly as they do for a Half — a process this user may not open reads
// "unreadable", never 0.
//
// This rung only READS. Nothing here kills, reaps, throttles, or admits.

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// FleetLedgerSchema tags each durable fleet-rollup JSONL row. It rides on the SAME
// ledger file as the per-session rows (DefaultLedgerRel) with its own schema tag, so a
// reader that filters on LedgerSchema is unaffected and a reader that wants waves
// compared over time finds both grains in one place.
const FleetLedgerSchema = "fak-harness-fleet-resources/1"

// Class is the role a walked process plays in the fleet. The three that carry meaning
// are the three the epic measured separately, because they have different fixes: seats
// are the agent harnesses (the per-seat cost pooling attacks), fak processes are fak's
// own binaries, brokers are the MCP/tool hosts (the duplicated-resident-tax row). The
// names match the rows of the hand measurement in #6552 (claude / fak / node).
type Class string

const (
	// ClassSeat is an agent harness process — one live seat.
	ClassSeat Class = "seat"
	// ClassFak is one of fak's own binaries: a guard, a broker daemon, a CLI verb.
	ClassFak Class = "fak"
	// ClassBroker is an MCP server or tool runtime spawned under a seat or a fak process.
	ClassBroker Class = "broker"
	// ClassOther is anything else inside the fak-owned tree (shells, git, compilers).
	ClassOther Class = "other"
)

// ClassOrder is the stable render/marshal order for the per-class rows.
var ClassOrder = []Class{ClassSeat, ClassFak, ClassBroker, ClassOther}

// seatNames / brokerNames classify by executable base name (lowercased, ".exe"
// stripped); fak's own binaries are recognized by isFakBinary rather than a table, since
// the `fak-` prefix covers the sibling binaries without enumerating them. Name matching
// is deliberately conservative: a bare `node` is only ever counted because the tree walk
// already proved it descends from a fak-owned root, never because "node" was seen on the
// box.
var (
	seatNames = map[string]bool{
		"claude": true, "claude-code": true, "codex": true, "cursor-agent": true,
	}
	brokerNames = map[string]bool{
		"node": true, "npm": true, "npx": true, "bun": true, "deno": true,
		"python": true, "python3": true, "uv": true, "uvx": true, "pipx": true,
	}
)

// ProcRef is one row of the caller's process census: enough to rebuild the tree and to
// name the process, and nothing more. AgeSec rides along because it is the only
// denominator a lifetime CPU total can honestly be divided by (see FleetRollup.Window)
// and because it makes the parent edge falsifiable on a PID-recycling OS.
type ProcRef struct {
	PID     int
	PPID    int
	Name    string // executable base name; ".exe" and directories are stripped by the caller or by Classify
	Cmdline string
	AgeSec  int
	HaveAge bool
}

// FleetProc is one walked process: its census row, the class it was assigned, and the
// per-PID resource reading. Read is false when the process could not be opened at all
// (it exited between census and sample, or this user may not query it) — that is a
// distinct state from "opened but this platform cannot read RSS".
type FleetProc struct {
	ProcRef
	Class        Class
	Read         bool
	CPUSeconds   float64
	HaveCPU      bool
	RSSBytes     uint64
	HaveRSS      bool
	PrivateBytes uint64
	HavePrivate  bool
}

// normalizeName lowercases an executable name and strips a trailing ".exe" so the same
// table classifies a Windows census and a POSIX one.
func normalizeName(name string) string {
	n := strings.ToLower(strings.TrimSpace(name))
	if i := strings.LastIndexAny(n, `/\`); i >= 0 {
		n = n[i+1:]
	}
	return strings.TrimSuffix(n, ".exe")
}

// isFakBinary reports whether an executable name is one of fak's own binaries. The
// `fak-` prefix covers the sibling binaries (fak-dev, fak-dos) without matching
// unrelated words that merely start with "fak".
func isFakBinary(name string) bool {
	return name == "fak" || strings.HasPrefix(name, "fak-")
}

// IsFleetRoot reports whether a census row is a root of the fak-owned tree: a fak binary
// or an agent seat. Everything else enters the walk only as a descendant of one of these.
func IsFleetRoot(p ProcRef) bool {
	n := normalizeName(p.Name)
	return isFakBinary(n) || seatNames[n]
}

// Classify assigns a walked process to a class. It is only meaningful for processes the
// walk already admitted: an unrecognized name inside the fak-owned tree is ClassOther
// (a shell, a git, a compiler the fleet spawned), which is real fleet cost and is
// counted, just not attributed to seats or brokers.
func Classify(p ProcRef) Class {
	n := normalizeName(p.Name)
	switch {
	case isFakBinary(n):
		return ClassFak
	case seatNames[n]:
		return ClassSeat
	case brokerNames[n] || strings.Contains(strings.ToLower(p.Cmdline), "mcp"):
		return ClassBroker
	default:
		return ClassOther
	}
}

// WalkFleet selects the fak-owned process tree out of a whole-host census: every root
// (IsFleetRoot) plus the transitive closure of its children, deduped by PID and returned
// in ascending PID order.
//
// The parent edge is only believed when it is not contradicted by age. On Windows a PID
// is recycled the moment it is free, so a long-lived unrelated process can name a dead
// fak guard's recycled PID as its parent and drag an arbitrary subtree into the rollup.
// A child cannot be older than its parent, so when both ages are known and the child is
// the older one the edge is dropped. Ages are best-effort, so an unknown age keeps the
// edge — this discards impossible parents, it does not demand proof of possible ones.
func WalkFleet(census []ProcRef) []ProcRef {
	byPID := make(map[int]ProcRef, len(census))
	for _, p := range census {
		byPID[p.PID] = p
	}
	children := make(map[int][]int, len(census))
	for _, p := range census {
		if p.PPID <= 0 || p.PPID == p.PID {
			continue
		}
		parent, ok := byPID[p.PPID]
		if !ok {
			continue
		}
		if p.HaveAge && parent.HaveAge && p.AgeSec > parent.AgeSec {
			continue // impossible edge: the "child" predates its parent (recycled PID)
		}
		children[p.PPID] = append(children[p.PPID], p.PID)
	}

	selected := make(map[int]bool, len(census))
	var queue []int
	for _, p := range census {
		if IsFleetRoot(p) && !selected[p.PID] {
			selected[p.PID] = true
			queue = append(queue, p.PID)
		}
	}
	// Breadth-first over a set that only ever grows and is guarded by `selected`, so a
	// cyclic parent chain (possible with recycled PIDs) terminates rather than spinning.
	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		for _, kid := range children[pid] {
			if selected[kid] {
				continue
			}
			selected[kid] = true
			queue = append(queue, kid)
		}
	}

	out := make([]ProcRef, 0, len(selected))
	for _, p := range census {
		if selected[p.PID] {
			out = append(out, p)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].PID < out[j].PID })
	return out
}

// SampleFleet reads the per-PID resource axes for each walked process through the same
// per-platform reader family the single-session path uses, and stamps each with its
// class. A process that cannot be opened comes back with Read false and every axis
// absent — counted in the process total, excluded from every byte and CPU total, and
// reported as `unreadable` so a partial read can never masquerade as a small fleet.
func SampleFleet(refs []ProcRef) []FleetProc {
	out := make([]FleetProc, 0, len(refs))
	for _, ref := range refs {
		fp := FleetProc{ProcRef: ref, Class: Classify(ref)}
		if ps, ok := readProcPID(ref.PID); ok {
			fp.Read = true
			if ps.haveCPU {
				fp.CPUSeconds, fp.HaveCPU = (ps.cpuUser + ps.cpuSys).Seconds(), true
			}
			if ps.haveRSS {
				fp.RSSBytes, fp.HaveRSS = ps.rss, true
			}
			if ps.havePrivate {
				fp.PrivateBytes, fp.HavePrivate = ps.private, true
			}
		}
		out = append(out, fp)
	}
	return out
}

// ClassRollup is one class's (or the whole fleet's) folded totals. The Have* bits mean
// "at least one process in this class reported this axis" — a class where nothing was
// readable renders n/a rather than a zero that reads as "free".
type ClassRollup struct {
	Class        Class
	Procs        int
	Sampled      int
	RSSBytes     uint64
	HaveRSS      bool
	PrivateBytes uint64
	HavePrivate  bool
	CPUSeconds   float64
	HaveCPU      bool
}

func (c *ClassRollup) add(p FleetProc) {
	c.Procs++
	if p.Read {
		c.Sampled++
	}
	if p.HaveRSS {
		c.RSSBytes, c.HaveRSS = c.RSSBytes+p.RSSBytes, true
	}
	if p.HavePrivate {
		c.PrivateBytes, c.HavePrivate = c.PrivateBytes+p.PrivateBytes, true
	}
	if p.HaveCPU {
		c.CPUSeconds, c.HaveCPU = c.CPUSeconds+p.CPUSeconds, true
	}
}

// FleetRollup is one folded reading of the whole fak-owned process tree.
type FleetRollup struct {
	Procs      int
	Sampled    int
	Unreadable int
	Classes    []ClassRollup // ClassOrder, empty classes dropped
	Total      ClassRollup
	NumCPU     int
	Host       Host
	// Window is the age of the OLDEST walked process — the only wall-clock span the
	// fleet's LIFETIME CPU totals can honestly be divided by. Unlike the per-session
	// Snapshot there is no single start instant here: processes come and go, so the
	// denominator is the span over which the fleet as a whole has been accumulating.
	// It over-states capacity for a fleet whose seats are much younger than its oldest
	// guard, which is the safe direction: the CPU fraction it yields is a floor.
	Window     time.Duration
	HaveWindow bool
}

// FoldFleet folds sampled processes into the per-class rollup. cores is the host's
// logical CPU count (the caller passes runtime.NumCPU, the same source Snapshot.NumCPU
// uses) and host is the box context from the SetHostProvider seam.
func FoldFleet(procs []FleetProc, host Host, cores int) FleetRollup {
	r := FleetRollup{NumCPU: cores, Host: host, Total: ClassRollup{Class: "total"}}
	byClass := make(map[Class]*ClassRollup, len(ClassOrder))
	for _, c := range ClassOrder {
		byClass[c] = &ClassRollup{Class: c}
	}
	for _, p := range procs {
		r.Procs++
		if p.Read {
			r.Sampled++
		} else {
			r.Unreadable++
		}
		if p.HaveAge && time.Duration(p.AgeSec)*time.Second > r.Window {
			r.Window, r.HaveWindow = time.Duration(p.AgeSec)*time.Second, true
		}
		cr, ok := byClass[p.Class]
		if !ok {
			cr = byClass[ClassOther]
		}
		cr.add(p)
		r.Total.add(p)
	}
	for _, c := range ClassOrder {
		if byClass[c].Procs > 0 {
			r.Classes = append(r.Classes, *byClass[c])
		}
	}
	return r
}

// RSSPercentOfHostRAM is the fleet's resident bytes as a percentage of host physical
// RAM — the host-relative number a seat budget can be written against. ok is false when
// either side is unread.
func (r FleetRollup) RSSPercentOfHostRAM() (pct float64, ok bool) {
	return pctOfRAM(r.Total.RSSBytes, r.Total.HaveRSS, r.Host)
}

// PrivatePercentOfHostRAM is the fleet's private (non-shared) bytes as a percentage of
// host physical RAM. Private is the honest number for a pooling argument: resident
// double-counts pages that N copies of the same binary already share, private does not.
func (r FleetRollup) PrivatePercentOfHostRAM() (pct float64, ok bool) {
	return pctOfRAM(r.Total.PrivateBytes, r.Total.HavePrivate, r.Host)
}

func pctOfRAM(bytes uint64, have bool, host Host) (float64, bool) {
	if !have || !host.HaveRAM || host.TotalRAMBytes == 0 {
		return 0, false
	}
	return float64(bytes) / float64(host.TotalRAMBytes) * 100, true
}

// HostCoreSeconds is the CPU capacity the host offered over the fleet's window:
// cores x the oldest walked process's age. ok is false when either is unknown.
func (r FleetRollup) HostCoreSeconds() (float64, bool) {
	if r.NumCPU <= 0 || !r.HaveWindow || r.Window <= 0 {
		return 0, false
	}
	return float64(r.NumCPU) * r.Window.Seconds(), true
}

// CPUPercentOfHost is the fleet's CPU-seconds as a percentage of the host core-seconds
// available over the window — 100% means the fleet kept every core busy for its whole
// life. ok is false when either side is unread.
func (r FleetRollup) CPUPercentOfHost() (pct float64, ok bool) {
	coreS, haveCore := r.HostCoreSeconds()
	if !r.Total.HaveCPU || !haveCore || coreS <= 0 {
		return 0, false
	}
	return r.Total.CPUSeconds / coreS * 100, true
}

// Report renders the human rollup: one line per class, a total line, and the host line
// that turns the byte counts into fractions of the box.
func (r FleetRollup) Report() string {
	var b strings.Builder
	fmt.Fprintf(&b, "fleet resources — %d processes", r.Procs)
	if r.Unreadable > 0 {
		fmt.Fprintf(&b, " (%d sampled, %d unreadable)", r.Sampled, r.Unreadable)
	}
	if r.HaveWindow {
		fmt.Fprintf(&b, ", oldest %s", humanDur(r.Window))
	}
	b.WriteString("\n")
	for _, c := range r.Classes {
		writeClassRow(&b, c)
	}
	writeClassRow(&b, r.Total)
	writeFleetHostRow(&b, r)
	return b.String()
}

func writeClassRow(b *strings.Builder, c ClassRollup) {
	fmt.Fprintf(b, "  %-7s %4d procs", string(c.Class), c.Procs)
	if c.HaveRSS {
		fmt.Fprintf(b, ", rss %s", humanBytes(c.RSSBytes))
	} else {
		b.WriteString(", rss n/a")
	}
	if c.HavePrivate {
		fmt.Fprintf(b, ", private %s", humanBytes(c.PrivateBytes))
	} else {
		b.WriteString(", private n/a")
	}
	if c.HaveCPU {
		fmt.Fprintf(b, ", cpu %.1fs", c.CPUSeconds)
	} else {
		b.WriteString(", cpu n/a")
	}
	b.WriteString("\n")
}

func writeFleetHostRow(b *strings.Builder, r FleetRollup) {
	fmt.Fprintf(b, "  host    %4d cores", r.NumCPU)
	if r.Host.HaveRAM {
		fmt.Fprintf(b, " / %s ram", humanBytes(r.Host.TotalRAMBytes))
		if r.Host.AvailRAMBytes > 0 {
			fmt.Fprintf(b, " (%s avail)", humanBytes(r.Host.AvailRAMBytes))
		}
	}
	if r.Host.HaveLoad {
		fmt.Fprintf(b, ", load %.2f", r.Host.Load1)
	}
	b.WriteString("\n  fleet/host ")
	if pct, ok := r.RSSPercentOfHostRAM(); ok {
		fmt.Fprintf(b, "rss %s of host ram", humanPercent(pct))
	} else {
		b.WriteString("rss n/a")
	}
	if pct, ok := r.PrivatePercentOfHostRAM(); ok {
		fmt.Fprintf(b, ", private %s of host ram", humanPercent(pct))
	} else {
		b.WriteString(", private n/a")
	}
	if coreS, ok := r.HostCoreSeconds(); ok {
		fmt.Fprintf(b, ", cpu %.1fs of %.1f core-s", r.Total.CPUSeconds, coreS)
		if pct, ok := r.CPUPercentOfHost(); ok {
			fmt.Fprintf(b, " (%s)", humanPercent(pct))
		}
	} else {
		b.WriteString(", cpu n/a")
	}
	b.WriteString("\n")
}

// fleetLedgerRow is the durable JSONL shape for one rollup, and the exact bytes `fak
// fleet res --json` prints. Pointer fields + omitempty keep an axis nothing could read
// OUT of the row rather than banking a misleading 0.
type fleetLedgerRow struct {
	Schema     string        `json:"schema"`
	TS         string        `json:"ts"`
	Procs      int           `json:"procs"`
	Sampled    int           `json:"sampled"`
	Unreadable int           `json:"unreadable"`
	WindowS    *float64      `json:"window_s,omitempty"`
	Classes    []classJSON   `json:"classes"`
	Total      classJSON     `json:"total"`
	Host       fleetHostJSON `json:"host"`
}

type classJSON struct {
	Class        string   `json:"class"`
	Procs        int      `json:"procs"`
	Sampled      int      `json:"sampled"`
	RSSBytes     *uint64  `json:"rss_bytes,omitempty"`
	PrivateBytes *uint64  `json:"private_bytes,omitempty"`
	CPUSeconds   *float64 `json:"cpu_s,omitempty"`
}

// fleetHostJSON is the box context plus the fleet-as-fraction-of-host numbers, so a
// banked row stays interpretable without knowing which machine wrote it.
type fleetHostJSON struct {
	Cores              int      `json:"cores"`
	RAMTotalBytes      *uint64  `json:"ram_total_bytes,omitempty"`
	RAMAvailBytes      *uint64  `json:"ram_avail_bytes,omitempty"`
	Load1              *float64 `json:"load1,omitempty"`
	CoreSeconds        *float64 `json:"core_seconds,omitempty"`
	FleetRSSPctRAM     *float64 `json:"fleet_rss_pct_of_host_ram,omitempty"`
	FleetPrivatePctRAM *float64 `json:"fleet_private_pct_of_host_ram,omitempty"`
	FleetCPUPct        *float64 `json:"fleet_cpu_pct_of_host_core_s,omitempty"`
}

func (c ClassRollup) toJSON() classJSON {
	j := classJSON{Class: string(c.Class), Procs: c.Procs, Sampled: c.Sampled}
	if c.HaveRSS {
		v := c.RSSBytes
		j.RSSBytes = &v
	}
	if c.HavePrivate {
		v := c.PrivateBytes
		j.PrivateBytes = &v
	}
	if c.HaveCPU {
		v := c.CPUSeconds
		j.CPUSeconds = &v
	}
	return j
}

// MarshalLedgerRow renders one durable JSONL rollup row (no trailing newline).
func (r FleetRollup) MarshalLedgerRow(now time.Time) ([]byte, error) {
	row := fleetLedgerRow{
		Schema:     FleetLedgerSchema,
		TS:         now.UTC().Format(time.RFC3339),
		Procs:      r.Procs,
		Sampled:    r.Sampled,
		Unreadable: r.Unreadable,
		Classes:    make([]classJSON, 0, len(r.Classes)),
		Total:      r.Total.toJSON(),
		Host:       fleetHostJSON{Cores: r.NumCPU},
	}
	if r.HaveWindow {
		w := r.Window.Seconds()
		row.WindowS = &w
	}
	for _, c := range r.Classes {
		row.Classes = append(row.Classes, c.toJSON())
	}
	if r.Host.HaveRAM {
		t := r.Host.TotalRAMBytes
		row.Host.RAMTotalBytes = &t
		if r.Host.AvailRAMBytes > 0 {
			a := r.Host.AvailRAMBytes
			row.Host.RAMAvailBytes = &a
		}
	}
	if r.Host.HaveLoad {
		l := r.Host.Load1
		row.Host.Load1 = &l
	}
	if coreS, ok := r.HostCoreSeconds(); ok {
		row.Host.CoreSeconds = &coreS
	}
	if pct, ok := r.RSSPercentOfHostRAM(); ok {
		row.Host.FleetRSSPctRAM = &pct
	}
	if pct, ok := r.PrivatePercentOfHostRAM(); ok {
		row.Host.FleetPrivatePctRAM = &pct
	}
	if pct, ok := r.CPUPercentOfHost(); ok {
		row.Host.FleetCPUPct = &pct
	}
	return json.Marshal(row)
}
