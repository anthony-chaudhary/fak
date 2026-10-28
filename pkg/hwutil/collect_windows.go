//go:build windows

package hwutil

import (
	"context"
	"fmt"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

// Windows host collection without cgo: CPU utilization from kernel32
// GetSystemTimes deltas, DRAM levels from GlobalMemoryStatusEx plus
// page-fault/swap counters, and disk/net counters from OS-native surfaces
// reached through one inline powershell.exe expression per family (no .ps1
// files, per the repo script ban). The
// kernel32 handles follow the repo's zero-dependency lazy-DLL pattern.
// Anything unreadable stays nil.

var runPowerShell = func(ctx context.Context, command string) ([]byte, error) {
	return exec.CommandContext(ctx, "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command).Output()
}

func collectFamilies(ctx context.Context, o Options) familyResults {
	res := familyResults{}

	// CPU: kernel/idle time deltas over the window (pure Go, no cgo).
	util, cores, ok := windowsCPUUtil(ctx, o)
	if ok {
		res.cpu = &CPU{Cores: cores, UtilPct: util, IntervalS: round6(o.Interval.Seconds())}
	} else {
		res.errs = append(res.errs, "cpu: GetSystemTimes unreadable")
	}

	// DRAM: global memory status for the level fields; the page-fault rate
	// and in-use pagefile bytes come from OS-native counter surfaces. A
	// failing extras poll leaves those fields nil on the base row; bandwidth
	// has no Windows source and stays nil.
	if d, ok := windowsDRAM(); ok {
		res.dram = d
		windowsDRAMExtras(ctx, res.dram)
	} else {
		res.errs = append(res.errs, "dram: GlobalMemoryStatusEx unreadable")
	}

	// Disks + net: WMI identity plus standard performance-counter rates; a
	// failing poll degrades that family to nil with a named note.
	if disks, err := windowsDisks(ctx); err != nil {
		res.errs = append(res.errs, "disks: "+err.Error())
	} else {
		res.disks = disks
	}
	if nets, err := windowsNet(ctx, o); err != nil {
		res.errs = append(res.errs, "net: "+err.Error())
	} else {
		res.net = nets
	}
	return res
}

// windowsCPUUtil samples GetSystemTimes across the configured windows and
// returns the aggregate busy percentage. Steal/iowait have no Windows source
// and remain nil in the caller.
func windowsCPUUtil(ctx context.Context, o Options) (*float64, int, bool) {
	idle1, kern1, _, ok1 := windowsSysTimes()
	if !ok1 {
		return nil, 0, false
	}
	for i := 0; i < o.Samples; i++ {
		if !sleepCtx(ctx, o.Interval) {
			break
		}
	}
	idle2, kern2, _, ok2 := windowsSysTimes()
	if !ok2 {
		return nil, 0, false
	}
	// Kernel time includes idle; busy = kernel - idle over the window.
	if dk, ok := safeDelta(kern1, kern2); ok && dk > 0 {
		var di uint64
		if d, ok := safeDelta(idle1, idle2); ok {
			di = d
		}
		if di <= dk {
			return avg(float64(dk-di)*100, float64(dk)), runtime.NumCPU(), true
		}
	}
	return nil, runtime.NumCPU(), false
}

var (
	kernel32                 = syscall.NewLazyDLL("kernel32.dll")
	procGetSystemTimes       = kernel32.NewProc("GetSystemTimes")
	procGlobalMemoryStatusEx = kernel32.NewProc("GlobalMemoryStatusEx")
)

// winCallSuccess reports whether a BOOL-returning kernel32 call succeeded.
// Proc.Call surfaces GetLastError() as its error even on success (Errno(0)
// renders as "The operation completed successfully."), so the decision rests
// on the return value, not the error.
func winCallSuccess(r1 uintptr) bool { return r1 != 0 }

func windowsSysTimes() (idle, kernel, system uint64, ok bool) {
	var i, k, s syscall.Filetime
	r1, _, _ := procGetSystemTimes.Call(
		uintptr(unsafe.Pointer(&i)),
		uintptr(unsafe.Pointer(&k)),
		uintptr(unsafe.Pointer(&s)),
	)
	if !winCallSuccess(r1) {
		return 0, 0, 0, false
	}
	return filetimeUint64(i), filetimeUint64(k), filetimeUint64(s), true
}

func filetimeUint64(ft syscall.Filetime) uint64 {
	return uint64(ft.HighDateTime)<<32 | uint64(ft.LowDateTime)
}

// memoryStatusEx mirrors MEMORYSTATUSEX (the struct layout is fixed by the
// Win32 ABI; dwLength must be its size before the call).
type memoryStatusEx struct {
	length          uint32
	memoryLoad      uint32
	totalPhys       uint64
	availPhys       uint64
	totalPageFile   uint64
	availPageFile   uint64
	totalVirtual    uint64
	availVirtual    uint64
	availExtVirtual uint64
}

func windowsDRAM() (*DRAM, bool) {
	var st memoryStatusEx
	st.length = uint32(unsafe.Sizeof(st))
	r1, _, _ := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&st)))
	if !winCallSuccess(r1) {
		return nil, false
	}
	total := int64(st.totalPhys)
	avail := int64(st.availPhys)
	d := &DRAM{TotalBytes: i64(total), AvailableBytes: i64(avail)}
	if total >= avail {
		d.UsedBytes = i64(total - avail)
	}
	return d, true
}

// The formatted-data WMI classes (Win32_PerfFormattedData_*) are not
// registered on every Windows build ("Invalid class", HRESULT 0x80041010),
// so disk/net rates come from the standard performance counters through
// Get-Counter instead. Two samples at a one-second interval are taken and
// only the last sample set is consumed (the first is a since-boot average).
// A failing counter query degrades that family to identity-only rows; the
// poll itself failing degrades the whole family with a named note.
const windowsDiskPS = `
$drive = Get-CimInstance Win32_DiskDrive | Select-Object DeviceID,Size,BusType | ConvertTo-Json -Compress
$counters = Get-Counter -SampleInterval 1 -MaxSamples 2 -Counter @('\PhysicalDisk(*)\Disk Read Bytes/sec','\PhysicalDisk(*)\Disk Write Bytes/sec','\PhysicalDisk(*)\Disk Reads/sec','\PhysicalDisk(*)\Disk Writes/sec','\PhysicalDisk(*)\Current Disk Queue Length') | ForEach-Object { $_.CounterSamples } | Where-Object { $_.Path -notlike '*_total*' } | Select-Object Path,CookedValue | ConvertTo-Json -Compress
Write-Output 'DRIVE'; Write-Output $drive; Write-Output 'COUNTER'; Write-Output $counters
`

// windowsDRAMPS polls the two DRAM extras that have no kernel32 surface:
// the \Memory page-fault rate (standard performance counter, last sample;
// the first is a since-boot average) and the in-use pagefile bytes
// (Win32_PageFile; empty when the system runs without a paging file).
const windowsDRAMPS = `
$pf = Get-CimInstance Win32_PageFile | Select-Object AllocatedBaseSize,CurrentUsage | ConvertTo-Json -Compress
$counters = Get-Counter -SampleInterval 1 -MaxSamples 2 -Counter @('\Memory\Page Faults/sec') | ForEach-Object { $_.CounterSamples } | Select-Object Path,CookedValue | ConvertTo-Json -Compress
Write-Output 'PAGEFILE'; Write-Output $pf; Write-Output 'COUNTER'; Write-Output $counters
`

// windowsDRAMExtras fills PageFaultsPerSec and SwapUsedBytes on the base
// DRAM row. Any failing or empty source leaves the matching field nil; the
// base row survives an extras failure.
func windowsDRAMExtras(ctx context.Context, d *DRAM) {
	out, err := runPowerShell(ctx, windowsDRAMPS)
	if err != nil {
		return
	}
	sec := psSections(out)
	if v, ok := lastCounterByName(decodePSRows(sec["COUNTER"]), `\memory\page faults/sec`); ok {
		d.PageFaultsPerSec = f64(v)
	}
	if pf := decodePSRows(sec["PAGEFILE"]); len(pf) > 0 {
		if b, ok := pageFileSwapUsed(pf); ok {
			d.SwapUsedBytes = i64(b)
		}
	}
}

// windowsNetPSFmt polls Win32_NetworkAdapter twice, %d milliseconds apart,
// so InboundDiscards/OutboundDiscards deltas become drop rates; the second
// poll also carries the cumulative PacketErrors. Adapters that report null
// for these properties keep their fields nil.
const windowsNetPSFmt = `
$adapters = Get-CimInstance Win32_NetworkAdapter | Where-Object {$_.NetConnectionID -ne $null} | Select-Object Name,NetConnectionID,Speed,AdminStatus,InboundDiscards,OutboundDiscards,PacketErrors | ConvertTo-Json -Compress
Start-Sleep -Milliseconds %d
$adapters2 = Get-CimInstance Win32_NetworkAdapter | Where-Object {$_.NetConnectionID -ne $null} | Select-Object Name,NetConnectionID,Speed,AdminStatus,InboundDiscards,OutboundDiscards,PacketErrors | ConvertTo-Json -Compress
$counters = Get-Counter -SampleInterval 1 -MaxSamples 2 -Counter @('\Network Interface(*)\Bytes Received/sec','\Network Interface(*)\Bytes Sent/sec','\Network Interface(*)\Packets Received/sec','\Network Interface(*)\Packets Sent/sec') | ForEach-Object { $_.CounterSamples } | Select-Object Path,CookedValue | ConvertTo-Json -Compress
Write-Output 'ADAPTER'; Write-Output $adapters; Write-Output 'ADAPTER2'; Write-Output $adapters2; Write-Output 'COUNTER'; Write-Output $counters
`

// windowsDisks joins Win32_DiskDrive identities (size, bus kind) with
// per-drive performance-counter rates. Free bytes cannot be attributed to a
// physical disk reliably and stay nil; a missing counter section leaves the
// rate fields nil on identity-only rows.
func windowsDisks(ctx context.Context) ([]Disk, error) {
	out, err := runPowerShell(ctx, windowsDiskPS)
	if err != nil {
		return nil, fmt.Errorf("disk WMI query failed")
	}
	sec := psSections(out)
	drives := decodePSRows(sec["DRIVE"])
	rates := lastCounterValues(decodePSRows(sec["COUNTER"]))

	ratesByNum := map[int]map[string]float64{}
	for inst, m := range rates {
		if n, ok := counterInstanceNumber(inst); ok {
			if _, seen := ratesByNum[n]; !seen {
				ratesByNum[n] = m
			}
		}
	}

	type driveInfo struct {
		size    int64
		hasSize bool
		bus     string
	}
	byNum := map[int]driveInfo{}
	var nums []int
	for _, dv := range drives {
		dev := strings.ToLower(strings.TrimSpace(stringField(dv, "DeviceID")))
		dev = strings.TrimPrefix(dev, `\\.\`)
		if !strings.HasPrefix(dev, "physicaldrive") {
			continue
		}
		num, err := strconv.Atoi(strings.TrimPrefix(dev, "physicaldrive"))
		if err != nil || num < 0 {
			continue
		}
		info := driveInfo{bus: strings.TrimSpace(stringField(dv, "BusType"))}
		if v, ok := numberField(dv, "Size"); ok {
			info.size, info.hasSize = int64(v), true
		}
		if _, seen := byNum[num]; !seen {
			nums = append(nums, num)
		}
		byNum[num] = info
	}
	sort.Ints(nums)
	disks := make([]Disk, 0, len(nums))
	for _, num := range nums {
		info := byNum[num]
		d := Disk{Name: fmt.Sprintf("PhysicalDrive%d", num), Kind: windowsDiskKind(info.bus)}
		if info.hasSize {
			d.TotalBytes = i64(info.size)
		}
		if m, ok := ratesByNum[num]; ok {
			if v, ok := m["disk read bytes/sec"]; ok {
				d.ReadBytesPerS = f64(v)
			}
			if v, ok := m["disk write bytes/sec"]; ok {
				d.WriteBytesPerS = f64(v)
			}
			if v, ok := m["disk reads/sec"]; ok {
				d.IopsRead = f64(v)
			}
			if v, ok := m["disk writes/sec"]; ok {
				d.IopsWrite = f64(v)
			}
			if v, ok := m["current disk queue length"]; ok {
				d.QueueDepth = i64(int64(v))
			}
		}
		disks = append(disks, d)
	}
	return disks, nil
}

// adapterRow carries the identity fields of one Win32_NetworkAdapter that
// the frozen v1 wire shape can carry.
type adapterRow struct {
	name, connID string
}

// windowsNet emits one row per observed network interface: every
// performance-counter instance becomes a data row (enriched with adapter
// identity when the names line up), and every adapter that no counter
// instance claims stays visible as an identity-only row with nil rates.
// Drop rates and the cumulative error total come from the
// Win32_NetworkAdapter discard/error properties where the adapter exposes
// them; adapters reporting null keep those fields nil.
func windowsNet(ctx context.Context, o Options) ([]NetIfc, error) {
	out, err := runPowerShell(ctx, fmt.Sprintf(windowsNetPSFmt, o.Interval.Milliseconds()))
	if err != nil {
		return nil, fmt.Errorf("network WMI query failed")
	}
	sec := psSections(out)
	adapters := decodePSRows(sec["ADAPTER"])
	drops := netDropRates(adapters, decodePSRows(sec["ADAPTER2"]), o.Interval.Seconds())
	rates := lastCounterValues(decodePSRows(sec["COUNTER"]))

	// Each adapter contributes exactly one row; its name and connection id
	// are both lookup keys, but only the first unseen key enters the order.
	byKey := map[string]adapterRow{}
	seenAdapter := map[string]bool{}
	var order []string
	for _, a := range adapters {
		name := strings.TrimSpace(stringField(a, "Name"))
		if name == "" || seenAdapter[strings.ToLower(name)] {
			continue
		}
		seenAdapter[strings.ToLower(name)] = true
		row := adapterRow{name: name, connID: strings.TrimSpace(stringField(a, "NetConnectionID"))}
		// The name is the row's order key (unique per adapter via
		// seenAdapter); the connection id registers only as a lookup alias.
		nameKey := strings.ToLower(name)
		order = append(order, nameKey)
		byKey[nameKey] = row
		if cid := strings.ToLower(row.connID); cid != "" {
			if _, taken := byKey[cid]; !taken {
				byKey[cid] = row
			}
		}
	}

	matched := map[string]bool{}
	nets := make([]NetIfc, 0, len(adapters)+len(rates))
	for _, key := range order {
		row := byKey[key]
		keys := []string{key}
		if cid := strings.ToLower(row.connID); cid != "" && cid != key {
			keys = append(keys, cid)
		}
		n := netIfcIdentity(row)
		if _, m, ok := matchCounterInstance(keys, rates, matched); ok {
			fillNetRates(&n, m)
		}
		if dr, ok := drops[key]; ok {
			applyNetDrops(&n, dr)
		}
		nets = append(nets, n)
	}
	for inst, m := range rates {
		if matched[inst] {
			continue
		}
		n := NetIfc{Iface: inst}
		fillNetRates(&n, m)
		nets = append(nets, n)
	}
	return nets, nil
}

// netIfcIdentity builds the identity row; interface link speed has no member
// in the frozen v1 wire shape, so adapter speed is not carried on NetIfc.
func netIfcIdentity(row adapterRow) NetIfc {
	return NetIfc{Iface: row.name}
}

func fillNetRates(n *NetIfc, m map[string]float64) {
	if v, ok := m["bytes received/sec"]; ok {
		n.RxBytesPerS = f64(v)
	}
	if v, ok := m["bytes sent/sec"]; ok {
		n.TxBytesPerS = f64(v)
	}
	if v, ok := m["packets received/sec"]; ok {
		n.RxPacketsPerS = f64(v)
	}
	if v, ok := m["packets sent/sec"]; ok {
		n.TxPacketsPerS = f64(v)
	}
}

// applyNetDrops copies the derived drop rates and cumulative errors onto the
// row; a nil-derived field leaves the wire field nil.
func applyNetDrops(n *NetIfc, dr netDropRow) {
	n.RxDropsPerS = dr.rxPerS
	n.TxDropsPerS = dr.txPerS
	n.ErrorsTotal = dr.errorsTotal
}
