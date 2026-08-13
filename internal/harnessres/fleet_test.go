package harnessres

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// liveFleetCensus is a miniature of the shape epic #6552 measured on the real box: a fak
// guard with an agent seat under it, an MCP broker under the seat, a bare shell the seat
// spawned, plus host noise that must NOT be counted (an unrelated node, an unrelated
// explorer, and a second-order child of the unrelated node).
func liveFleetCensus() []ProcRef {
	return []ProcRef{
		{PID: 100, PPID: 1, Name: "fak.exe", Cmdline: "fak guard --agent a1", AgeSec: 900, HaveAge: true},
		{PID: 101, PPID: 100, Name: "claude", Cmdline: "claude --print", AgeSec: 880, HaveAge: true},
		{PID: 102, PPID: 101, Name: "node.exe", Cmdline: "node mcp-server.js", AgeSec: 870, HaveAge: true},
		{PID: 103, PPID: 101, Name: "pwsh", Cmdline: "pwsh -c git status", AgeSec: 10, HaveAge: true},
		{PID: 200, PPID: 1, Name: "node", Cmdline: "node unrelated-dev-server.js", AgeSec: 5000, HaveAge: true},
		{PID: 201, PPID: 200, Name: "esbuild", Cmdline: "esbuild", AgeSec: 4000, HaveAge: true},
		{PID: 300, PPID: 1, Name: "explorer.exe", Cmdline: "explorer.exe", AgeSec: 90000, HaveAge: true},
	}
}

func pidsOf(refs []ProcRef) []int {
	out := make([]int, 0, len(refs))
	for _, r := range refs {
		out = append(out, r.PID)
	}
	return out
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestWalkFleetSelectsOwnedTreeOnly(t *testing.T) {
	got := pidsOf(WalkFleet(liveFleetCensus()))
	want := []int{100, 101, 102, 103}
	if !sameInts(got, want) {
		t.Fatalf("walked PIDs = %v, want %v (the unrelated node tree and explorer must stay out)", got, want)
	}
}

// A bare `node` on the box is host noise; the SAME binary under a seat is a fleet broker.
// The distinction is the tree walk, not the name — this is the test that keeps the
// classifier from inflating the rollup with every Node process a developer happens to run.
func TestWalkFleetNeedsAFakOwnedRoot(t *testing.T) {
	orphanBroker := []ProcRef{{PID: 200, PPID: 1, Name: "node", Cmdline: "node mcp-server.js"}}
	if got := WalkFleet(orphanBroker); len(got) != 0 {
		t.Fatalf("a node with no fak-owned ancestor was walked: %v", pidsOf(got))
	}
}

// On Windows a PID is recycled the instant it is free, so a long-lived unrelated process
// can name a dead guard's recycled PID as its parent. A child cannot predate its parent,
// so that edge is dropped and the stranger's subtree stays out of the fleet's bill.
func TestWalkFleetRejectsRecycledParentEdge(t *testing.T) {
	census := []ProcRef{
		{PID: 100, PPID: 1, Name: "fak", Cmdline: "fak guard", AgeSec: 60, HaveAge: true},
		{PID: 400, PPID: 100, Name: "chrome", Cmdline: "chrome", AgeSec: 90000, HaveAge: true},
		{PID: 401, PPID: 400, Name: "chrome", Cmdline: "chrome --type=renderer", AgeSec: 89000, HaveAge: true},
	}
	if got := pidsOf(WalkFleet(census)); !sameInts(got, []int{100}) {
		t.Fatalf("walked PIDs = %v, want [100]: a 25-hour-old 'child' of a 60s-old guard is a recycled PID", got)
	}
	// The same edge with the age unknown is kept: this rule discards impossible parents,
	// it does not demand proof of possible ones.
	census[1].HaveAge = false
	if got := pidsOf(WalkFleet(census)); !sameInts(got, []int{100, 400, 401}) {
		t.Fatalf("walked PIDs = %v, want [100 400 401]: an unknown age must not drop the edge", got)
	}
}

// Recycled PIDs can also produce a parent cycle. The walk must terminate on one.
func TestWalkFleetTerminatesOnCycle(t *testing.T) {
	census := []ProcRef{
		{PID: 100, PPID: 1, Name: "fak", Cmdline: "fak guard"},
		{PID: 101, PPID: 100, Name: "claude"},
		{PID: 102, PPID: 103, Name: "node"},
		{PID: 103, PPID: 102, Name: "node"}, // 102 <-> 103 mutual parents, reachable from neither root
	}
	done := make(chan []int, 1)
	go func() { done <- pidsOf(WalkFleet(census)) }()
	select {
	case got := <-done:
		if !sameInts(got, []int{100, 101}) {
			t.Fatalf("walked PIDs = %v, want [100 101]", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("WalkFleet did not terminate on a cyclic parent chain")
	}
}

func TestClassify(t *testing.T) {
	cases := []struct {
		ref  ProcRef
		want Class
	}{
		{ProcRef{Name: "fak"}, ClassFak},
		{ProcRef{Name: "fak.exe"}, ClassFak},
		{ProcRef{Name: "FAK.EXE"}, ClassFak},
		{ProcRef{Name: "fak-dos"}, ClassFak},
		{ProcRef{Name: `C:\work\fak\fak.exe`}, ClassFak},
		{ProcRef{Name: "claude"}, ClassSeat},
		{ProcRef{Name: "node"}, ClassBroker},
		{ProcRef{Name: "uvx"}, ClassBroker},
		{ProcRef{Name: "dotnet", Cmdline: "dotnet run --mcp-stdio"}, ClassBroker},
		{ProcRef{Name: "pwsh", Cmdline: "pwsh -c git status"}, ClassOther},
		// "fake" is not a fak binary: the prefix rule must be `fak-`, not `fak`.
		{ProcRef{Name: "fakeroot"}, ClassOther},
	}
	for _, c := range cases {
		if got := Classify(c.ref); got != c.want {
			t.Errorf("Classify(%q/%q) = %q, want %q", c.ref.Name, c.ref.Cmdline, got, c.want)
		}
	}
}

// sampled builds a FleetProc with every axis present, as a readable process would.
func sampled(pid int, class Class, rss, private uint64, cpu float64, ageSec int) FleetProc {
	return FleetProc{
		ProcRef:    ProcRef{PID: pid, AgeSec: ageSec, HaveAge: true},
		Class:      class,
		Read:       true,
		CPUSeconds: cpu, HaveCPU: true,
		RSSBytes: rss, HaveRSS: true,
		PrivateBytes: private, HavePrivate: true,
	}
}

func TestFoldFleetTotalsAndHostFractions(t *testing.T) {
	procs := []FleetProc{
		sampled(100, ClassFak, 1<<30, 512<<20, 100, 900),
		sampled(101, ClassSeat, 2<<30, 1<<30, 200, 880),
		sampled(102, ClassBroker, 1<<29, 1<<28, 4, 870),
		// Unreadable: counted as a process, excluded from every byte and CPU total.
		{ProcRef: ProcRef{PID: 103, AgeSec: 10, HaveAge: true}, Class: ClassOther},
	}
	host := Host{TotalRAMBytes: 32 << 30, HaveRAM: true, AvailRAMBytes: 8 << 30}
	r := FoldFleet(procs, host, 8)

	if r.Procs != 4 || r.Sampled != 3 || r.Unreadable != 1 {
		t.Fatalf("procs/sampled/unreadable = %d/%d/%d, want 4/3/1", r.Procs, r.Sampled, r.Unreadable)
	}
	if len(r.Classes) != 4 {
		t.Fatalf("classes = %d, want 4 (one row per populated class)", len(r.Classes))
	}
	if r.Classes[0].Class != ClassSeat || r.Classes[1].Class != ClassFak {
		t.Fatalf("class order = %q,%q, want seat,fak (ClassOrder)", r.Classes[0].Class, r.Classes[1].Class)
	}
	wantRSS := uint64(1<<30 + 2<<30 + 1<<29)
	if r.Total.RSSBytes != wantRSS {
		t.Fatalf("total rss = %d, want %d", r.Total.RSSBytes, wantRSS)
	}
	if r.Total.CPUSeconds != 304 {
		t.Fatalf("total cpu = %v, want 304", r.Total.CPUSeconds)
	}
	// The unreadable process contributes a head count and nothing else.
	other := r.Classes[3]
	if other.Class != ClassOther || other.Procs != 1 || other.Sampled != 0 || other.HaveRSS || other.HaveCPU {
		t.Fatalf("unreadable class row = %+v, want 1 proc / 0 sampled / no axes", other)
	}
	// Window is the OLDEST walked process — the only span a lifetime CPU total divides by.
	if !r.HaveWindow || r.Window != 900*time.Second {
		t.Fatalf("window = %v (have=%v), want 900s", r.Window, r.HaveWindow)
	}

	pct, ok := r.RSSPercentOfHostRAM()
	if !ok || pct < 10.9 || pct > 11.0 {
		t.Fatalf("rss pct of host ram = %v (ok=%v), want ~10.94 (3.5 GiB of 32 GiB)", pct, ok)
	}
	// 512 MiB + 1 GiB + 256 MiB = 1.75 GiB private — below the 3.5 GiB resident total,
	// which is the whole point of carrying both axes.
	if pct, ok := r.PrivatePercentOfHostRAM(); !ok || pct != 5.46875 {
		t.Fatalf("private pct of host ram = %v (ok=%v), want 5.46875 (1.75 GiB of 32 GiB)", pct, ok)
	}
	coreS, ok := r.HostCoreSeconds()
	if !ok || coreS != 7200 { // 8 cores x 900s
		t.Fatalf("host core-seconds = %v (ok=%v), want 7200", coreS, ok)
	}
	if pct, ok := r.CPUPercentOfHost(); !ok || pct < 4.22 || pct > 4.23 {
		t.Fatalf("cpu pct of host = %v (ok=%v), want ~4.222 (304 of 7200 core-s)", pct, ok)
	}
}

// A fleet nothing could be read from must report n/a on every axis, never a 0 that reads
// as "this fleet is free". This is the same no-fabrication rule the per-session Half keeps.
func TestFoldFleetUnreadableFleetHasNoFabricatedZeros(t *testing.T) {
	procs := []FleetProc{
		{ProcRef: ProcRef{PID: 1}, Class: ClassSeat},
		{ProcRef: ProcRef{PID: 2}, Class: ClassFak},
	}
	r := FoldFleet(procs, Host{TotalRAMBytes: 32 << 30, HaveRAM: true}, 8)
	if r.Sampled != 0 || r.Unreadable != 2 {
		t.Fatalf("sampled/unreadable = %d/%d, want 0/2", r.Sampled, r.Unreadable)
	}
	if r.Total.HaveRSS || r.Total.HavePrivate || r.Total.HaveCPU {
		t.Fatalf("unread axes claim presence: %+v", r.Total)
	}
	if _, ok := r.RSSPercentOfHostRAM(); ok {
		t.Fatal("rss fraction reported for a fleet with no rss reading")
	}
	if _, ok := r.CPUPercentOfHost(); ok {
		t.Fatal("cpu fraction reported for a fleet with no cpu reading")
	}
	report := r.Report()
	if !strings.Contains(report, "2 processes") || !strings.Contains(report, "0 sampled, 2 unreadable") {
		t.Fatalf("report hides the unreadable count:\n%s", report)
	}
	if !strings.Contains(report, "rss n/a") {
		t.Fatalf("report renders an unread axis as a number:\n%s", report)
	}
}

// A host with no RAM reading must still produce a rollup — the byte counts stand, only
// the fractions drop out.
func TestFoldFleetWithoutHostRAM(t *testing.T) {
	r := FoldFleet([]FleetProc{sampled(1, ClassSeat, 1<<30, 1<<29, 10, 100)}, Host{}, 4)
	if !r.Total.HaveRSS || r.Total.RSSBytes != 1<<30 {
		t.Fatalf("total rss = %d (have=%v), want 1 GiB", r.Total.RSSBytes, r.Total.HaveRSS)
	}
	if _, ok := r.RSSPercentOfHostRAM(); ok {
		t.Fatal("host fraction reported with no host RAM reading")
	}
	if _, ok := r.CPUPercentOfHost(); !ok {
		t.Fatal("cpu fraction needs cores+window only, and both are known here")
	}
}

func TestReportRendersEveryClassAndTheHostLine(t *testing.T) {
	procs := []FleetProc{
		sampled(100, ClassFak, 1<<30, 512<<20, 100, 900),
		sampled(101, ClassSeat, 2<<30, 1<<30, 200, 880),
		sampled(102, ClassBroker, 1<<29, 1<<28, 4, 870),
	}
	got := FoldFleet(procs, Host{TotalRAMBytes: 32 << 30, HaveRAM: true, AvailRAMBytes: 8 << 30}, 8).Report()
	for _, want := range []string{
		"fleet resources — 3 processes",
		"seat", "fak", "broker", "total",
		"rss ", "private ", "cpu ",
		"host ", "cores", "fleet/host",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("report missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, " other ") {
		t.Fatalf("empty class rendered as a row:\n%s", got)
	}
}

func TestMarshalLedgerRow(t *testing.T) {
	procs := []FleetProc{
		sampled(100, ClassFak, 1<<30, 512<<20, 100, 900),
		sampled(101, ClassSeat, 2<<30, 1<<30, 200, 880),
		{ProcRef: ProcRef{PID: 102}, Class: ClassBroker},
	}
	r := FoldFleet(procs, Host{TotalRAMBytes: 32 << 30, HaveRAM: true, AvailRAMBytes: 8 << 30}, 8)
	line, err := r.MarshalLedgerRow(time.Unix(1700000000, 0).UTC())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(line), "\n") {
		t.Fatal("ledger row contains a newline: JSONL rows must be single-line")
	}

	var row struct {
		Schema     string `json:"schema"`
		TS         string `json:"ts"`
		Procs      int    `json:"procs"`
		Sampled    int    `json:"sampled"`
		Unreadable int    `json:"unreadable"`
		WindowS    *float64
		Classes    []struct {
			Class        string   `json:"class"`
			Procs        int      `json:"procs"`
			RSSBytes     *uint64  `json:"rss_bytes"`
			PrivateBytes *uint64  `json:"private_bytes"`
			CPUSeconds   *float64 `json:"cpu_s"`
		} `json:"classes"`
		Total struct {
			RSSBytes   *uint64  `json:"rss_bytes"`
			CPUSeconds *float64 `json:"cpu_s"`
		} `json:"total"`
		Host struct {
			Cores          int      `json:"cores"`
			RAMTotalBytes  *uint64  `json:"ram_total_bytes"`
			CoreSeconds    *float64 `json:"core_seconds"`
			FleetRSSPctRAM *float64 `json:"fleet_rss_pct_of_host_ram"`
			FleetCPUPct    *float64 `json:"fleet_cpu_pct_of_host_core_s"`
		} `json:"host"`
	}
	if err := json.Unmarshal(line, &row); err != nil {
		t.Fatalf("row does not round-trip: %v\n%s", err, line)
	}
	// The fleet row carries its OWN schema tag while sharing the session ledger file, so
	// a reader filtering on LedgerSchema is unaffected by these rows.
	if row.Schema != FleetLedgerSchema || row.Schema == LedgerSchema {
		t.Fatalf("schema = %q, want %q (distinct from the session schema)", row.Schema, FleetLedgerSchema)
	}
	if row.TS != "2023-11-14T22:13:20Z" {
		t.Fatalf("ts = %q, want the UTC RFC3339 stamp", row.TS)
	}
	if row.Procs != 3 || row.Sampled != 2 || row.Unreadable != 1 {
		t.Fatalf("counts = %d/%d/%d, want 3/2/1", row.Procs, row.Sampled, row.Unreadable)
	}
	if len(row.Classes) != 3 {
		t.Fatalf("classes = %d, want 3", len(row.Classes))
	}
	if row.Total.RSSBytes == nil || *row.Total.RSSBytes != 3<<30 {
		t.Fatalf("total rss_bytes = %v, want %d", row.Total.RSSBytes, uint64(3<<30))
	}
	if row.Host.Cores != 8 || row.Host.RAMTotalBytes == nil || *row.Host.RAMTotalBytes != 32<<30 {
		t.Fatalf("host block = %+v, want 8 cores / 32 GiB", row.Host)
	}
	if row.Host.CoreSeconds == nil || *row.Host.CoreSeconds != 7200 {
		t.Fatalf("host core_seconds = %v, want 7200", row.Host.CoreSeconds)
	}
	if row.Host.FleetRSSPctRAM == nil || row.Host.FleetCPUPct == nil {
		t.Fatalf("host fractions missing from a row that has both sides: %+v", row.Host)
	}
	// The broker class read nothing, so its axes must be ABSENT, not zero.
	for _, c := range row.Classes {
		if c.Class != string(ClassBroker) {
			continue
		}
		if c.Procs != 1 {
			t.Fatalf("broker procs = %d, want 1", c.Procs)
		}
		if c.RSSBytes != nil || c.PrivateBytes != nil || c.CPUSeconds != nil {
			t.Fatalf("unread broker axes banked as values: %+v", c)
		}
	}
}

// The whole pipeline over a synthetic census: walk, classify, fold. SampleFleet is not in
// this path because it reads real PIDs; the live-fleet reading is the CLI's witness.
func TestWalkThenClassifyOverACensus(t *testing.T) {
	walked := WalkFleet(liveFleetCensus())
	counts := map[Class]int{}
	for _, ref := range walked {
		counts[Classify(ref)]++
	}
	want := map[Class]int{ClassFak: 1, ClassSeat: 1, ClassBroker: 1, ClassOther: 1}
	for class, n := range want {
		if counts[class] != n {
			t.Errorf("class %q = %d, want %d (counts=%v)", class, counts[class], n, counts)
		}
	}
}
