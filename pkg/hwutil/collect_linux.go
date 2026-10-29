//go:build linux

package hwutil

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"
)

// Linux host collection reads stdlib-only sources: /proc/stat, /proc/meminfo,
// /proc/vmstat, /proc/diskstats, /proc/loadavg, /proc/mounts, /sys/class/net,
// /sys/class/block, and the Thunderbolt/USB sysfs classes. Every unread source
// degrades its field to nil; nothing is zero-filled.

// linuxRaw is one poll of every Linux counter source. Per-source presence
// flags carry the null-on-error state forward.
type linuxRaw struct {
	cpu        cpuJiffies
	hasCPU     bool
	mem        meminfoFields
	hasMem     bool
	pgfault    uint64
	hasPGFault bool
	disks      map[string]diskSample
	net        map[string]netStats
	netPresent map[string][]bool
}

func readLinuxRaw() linuxRaw {
	var r linuxRaw
	if b, err := os.ReadFile("/proc/stat"); err == nil {
		r.cpu, r.hasCPU = parseProcStat(string(b))
	}
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		r.mem, r.hasMem = parseMeminfo(string(b))
	}
	if b, err := os.ReadFile("/proc/vmstat"); err == nil {
		if v, ok := vmstatCounter(string(b), "pgfault"); ok {
			r.pgfault, r.hasPGFault = v, true
		}
	}
	if b, err := os.ReadFile("/proc/diskstats"); err == nil {
		r.disks = parseDiskstats(string(b))
	}
	r.net = make(map[string]netStats)
	r.netPresent = make(map[string][]bool)
	if ifaces, err := os.ReadDir("/sys/class/net"); err == nil {
		for _, ent := range ifaces {
			name := ent.Name()
			if name == "lo" {
				continue
			}
			var lines []string
			for _, k := range netStatKeys {
				b, err := os.ReadFile(filepath.Join("/sys/class/net", name, "statistics", k))
				if err != nil {
					lines = append(lines, "")
					continue
				}
				lines = append(lines, string(b))
			}
			s, present := parseNetStatsText(strings.Join(lines, "\n"))
			r.net[name] = s
			r.netPresent[name] = present
		}
	}
	return r
}

func collectFamilies(ctx context.Context, o Options) familyResults {
	res := familyResults{}
	first := readLinuxRaw()
	started := time.Now()
	last := first
	for i := 1; i <= o.Samples; i++ {
		if !sleepCtx(ctx, o.Interval) {
			break
		}
		last = readLinuxRaw()
	}
	dt := time.Since(started).Seconds()
	if dt <= 0 {
		dt = o.Interval.Seconds()
	}

	// CPU: jiffies delta plus loadavg from the last poll.
	if last.hasCPU && first.hasCPU {
		c := cpuDelta(first.cpu, last.cpu, dt)
		if b, err := os.ReadFile("/proc/loadavg"); err == nil {
			if l, ok := parseLoadavg(string(b)); ok {
				c.Load1 = l
			}
		}
		res.cpu = c
	} else {
		res.errs = append(res.errs, "cpu: /proc/stat unreadable")
	}

	// DRAM: meminfo levels plus vmstat page-fault rate.
	if last.hasMem {
		d := dramFromMeminfo(last.mem)
		if first.hasPGFault && last.hasPGFault && last.pgfault >= first.pgfault {
			d.PageFaultsPerSec = avg(float64(last.pgfault-first.pgfault), dt)
		}
		res.dram = d
	} else {
		res.errs = append(res.errs, "dram: /proc/meminfo unreadable")
	}

	// Disks: diskstats deltas plus sysfs kind and mount-point capacity.
	if last.disks != nil {
		names := make([]string, 0, len(last.disks))
		for n := range last.disks {
			names = append(names, n)
		}
		sort.Strings(names)
		for _, name := range names {
			if !keepDisk(name, last.disks) {
				continue
			}
			rd, wr, ir, iw, lat, qd := diskDelta(first.disks[name], last.disks[name], dt)
			free, total := diskCapacity(name)
			res.disks = append(res.disks, Disk{
				Name:           name,
				Kind:           diskKind(name),
				ReadBytesPerS:  rd,
				WriteBytesPerS: wr,
				IopsRead:       ir,
				IopsWrite:      iw,
				QueueDepth:     qd,
				LatencyMSAvg:   lat,
				FreeBytes:      free,
				TotalBytes:     total,
			})
		}
	} else {
		res.errs = append(res.errs, "disks: /proc/diskstats unreadable")
	}

	// Network: sysfs statistics deltas.
	if len(last.net) > 0 {
		ifaces := make([]string, 0, len(last.net))
		for n := range last.net {
			ifaces = append(ifaces, n)
		}
		sort.Strings(ifaces)
		for _, name := range ifaces {
			present := last.netPresent[name]
			if present == nil {
				present = make([]bool, len(netStatKeys))
			}
			res.net = append(res.net, *netDelta(name, first.net[name], last.net[name], present, dt))
		}
	} else {
		res.errs = append(res.errs, "net: /sys/class/net unreadable")
	}

	// Host fabric links: Thunderbolt bridges, else USB devices, else nil.
	res.links = collectHostLinks()
	return res
}

// diskKind classifies the transport from the device name and sysfs markers.
func diskKind(name string) DiskKind {
	usbMarker := probePath("/sys/class/block/"+name+"/device/busnum") == nil
	var modelHasUSB bool
	if b, err := os.ReadFile("/sys/class/block/" + name + "/device/model"); err == nil {
		modelHasUSB = strings.Contains(strings.ToUpper(string(b)), "USB")
	}
	rot, hasRot := parseFloatFile("/sys/class/block/" + name + "/queue/rotational")
	return classifyDisk(name, usbMarker, modelHasUSB, rot, hasRot)
}

func probePath(p string) error {
	_, err := os.Lstat(p)
	return err
}

// diskCapacity attributes a mount point to the device via /proc/mounts and
// reports its free/total bytes; devices without a direct mount stay nil.
func diskCapacity(name string) (free, total *int64) {
	mount, ok := mountPointFor(name)
	if !ok {
		return nil, nil
	}
	var st syscall.Statfs_t
	if err := syscall.Statfs(mount, &st); err != nil {
		return nil, nil
	}
	bsize := uint64(st.Bsize)
	if bsize == 0 {
		return nil, nil
	}
	free = i64(int64(st.Bfree) * int64(bsize))
	total = i64(int64(st.Blocks) * int64(bsize))
	return free, total
}

func mountPointFor(devName string) (string, bool) {
	b, err := os.ReadFile("/proc/mounts")
	if err != nil {
		return "", false
	}
	want := "/dev/" + devName
	for _, line := range strings.Split(string(b), "\n") {
		f := strings.Fields(line)
		if len(f) >= 2 && f[0] == want {
			return f[1], true
		}
	}
	return "", false
}

// collectHostLinks emits Thunderbolt bridge rows when the class exists,
// otherwise USB device rows, otherwise nil (no host fabric link surface).
func collectHostLinks() []Link {
	if entries, err := os.ReadDir("/sys/class/thunderbolt"); err == nil {
		var out []Link
		for _, ent := range entries {
			if !ent.IsDir() || !strings.HasPrefix(ent.Name(), "br") {
				continue
			}
			dir := filepath.Join("/sys/class/thunderbolt", ent.Name())
			stateRaw := ""
			if b, err := os.ReadFile(filepath.Join(dir, "state")); err == nil {
				stateRaw = strings.TrimSpace(string(b))
			}
			bw, hasBW := parseUintFile(filepath.Join(dir, "bandwidth"))
			out = append(out, thunderboltLinkRow(ent.Name(), stateRaw, bw, hasBW))
		}
		if len(out) > 0 {
			return out
		}
	}
	if entries, err := os.ReadDir("/sys/bus/usb/devices"); err == nil {
		var out []Link
		for _, ent := range entries {
			if !ent.IsDir() {
				continue
			}
			name := ent.Name()
			if !strings.Contains(name, "-") {
				continue // bus roots, not links
			}
			port := name[strings.LastIndex(name, "-")+1:]
			if port == "0" {
				continue // root hub
			}
			dir := filepath.Join("/sys/bus/usb/devices", name)
			speedMbps, ok := parseUintFile(filepath.Join(dir, "speed"))
			if !ok || speedMbps == 0 {
				continue
			}
			authorized := true
			if b, err := os.ReadFile(filepath.Join(dir, "authorized")); err == nil {
				authorized = strings.TrimSpace(string(b)) == "1"
			}
			out = append(out, usbLinkRow(name, speedMbps, authorized))
		}
		if len(out) > 0 {
			return out
		}
	}
	return nil
}
