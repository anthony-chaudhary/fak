package macobs

import (
	"context"
	"encoding/xml"
	"io"
	"strconv"
	"strings"
	"time"
)

func runIsolated(ctx context.Context, runner CommandRunner, timeout time.Duration, name string, args ...string) ([]byte, error) {
	if runner == nil {
		runner = DefaultCommandRunner
	}
	tctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	type res struct {
		out []byte
		err error
	}
	ch := make(chan res, 1)
	go func() {
		out, err := runner(tctx, name, args...)
		ch <- res{out: out, err: err}
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-tctx.Done():
		return nil, tctx.Err()
	case r := <-ch:
		return r.out, r.err
	}
}

// CollectHardwareWithRunner collects hardware telemetry using the provided command runner.
func CollectHardwareWithRunner(ctx context.Context, runner CommandRunner) HardwareTelemetry {
	hw := HardwareTelemetry{
		ThermalState: ThermalNominal,
		PowerSource:  PowerUnknown,
	}

	const cmdTimeout = 2 * time.Second

	// 1. IORegistry IOAccelerator counters
	if out, err := runIsolated(ctx, runner, cmdTimeout, "/usr/sbin/ioreg", "-a", "-r", "-d", "1", "-c", "IOAccelerator"); err == nil && len(out) > 0 {
		if alloc, inUse, devUtil, rendUtil, recov, ok := ParseIORegXML(out); ok {
			hw.AllocSystemMemoryBytes = alloc
			hw.InUseSystemMemoryBytes = inUse
			hw.DeviceUtilizationPct = devUtil
			hw.RendererUtilizationPct = rendUtil
			hw.RecoveryCount = recov
			hw.Available = true
		}
	} else if out, err := runIsolated(ctx, runner, cmdTimeout, "ioreg", "-a", "-r", "-d", "1", "-c", "IOAccelerator"); err == nil && len(out) > 0 {
		if alloc, inUse, devUtil, rendUtil, recov, ok := ParseIORegXML(out); ok {
			hw.AllocSystemMemoryBytes = alloc
			hw.InUseSystemMemoryBytes = inUse
			hw.DeviceUtilizationPct = devUtil
			hw.RendererUtilizationPct = rendUtil
			hw.RecoveryCount = recov
			hw.Available = true
		}
	}

	// 2. Sysctl memory size
	if out, err := runIsolated(ctx, runner, cmdTimeout, "/usr/sbin/sysctl", "-n", "hw.memsize"); err == nil {
		if mem, ok := ParseSysctlMemsize(string(out)); ok {
			hw.TotalSystemMemoryBytes = mem
			hw.Available = true
		}
	} else if out, err := runIsolated(ctx, runner, cmdTimeout, "sysctl", "-n", "hw.memsize"); err == nil {
		if mem, ok := ParseSysctlMemsize(string(out)); ok {
			hw.TotalSystemMemoryBytes = mem
			hw.Available = true
		}
	}

	// 3. Sysctl wired limits (iogpu.wired_limit_mb then iogpu.wired_mem_limit)
	if out, err := runIsolated(ctx, runner, cmdTimeout, "/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"); err == nil {
		if limit, ok := ParseSysctlWiredLimitMB(string(out)); ok && limit > 0 {
			hw.WiredMemoryLimitBytes = limit
		}
	}
	if hw.WiredMemoryLimitBytes == 0 {
		if out, err := runIsolated(ctx, runner, cmdTimeout, "/usr/sbin/sysctl", "-n", "iogpu.wired_mem_limit"); err == nil {
			if limit, ok := ParseSysctlMemsize(string(out)); ok && limit > 0 {
				hw.WiredMemoryLimitBytes = limit
			}
		}
	}
	// Default fallback: 75% of total system memory if limit not explicitly configured
	if hw.WiredMemoryLimitBytes == 0 && hw.TotalSystemMemoryBytes > 0 {
		hw.WiredMemoryLimitBytes = (hw.TotalSystemMemoryBytes * 3) / 4
	}

	// 4. Sysctl swap usage
	if out, err := runIsolated(ctx, runner, cmdTimeout, "/usr/sbin/sysctl", "-n", "vm.swapusage"); err == nil {
		if total, used, ok := ParseSysctlSwapUsage(string(out)); ok {
			hw.SwapTotalBytes = total
			hw.SwapUsedBytes = used
		}
	} else if out, err := runIsolated(ctx, runner, cmdTimeout, "sysctl", "-n", "vm.swapusage"); err == nil {
		if total, used, ok := ParseSysctlSwapUsage(string(out)); ok {
			hw.SwapTotalBytes = total
			hw.SwapUsedBytes = used
		}
	}

	// 5. Sysctl memorystatus level
	if out, err := runIsolated(ctx, runner, cmdTimeout, "/usr/sbin/sysctl", "-n", "kern.memorystatus_level"); err == nil {
		if lvl, ok := ParseSysctlMemoryStatusLevel(string(out)); ok {
			hw.MemoryStatusLevel = lvl
		}
	}

	// 6. vm_stat paging and memory residency
	if out, err := runIsolated(ctx, runner, cmdTimeout, "/usr/bin/vm_stat"); err == nil {
		if free, wired, compressed, pageins, pageouts, ok := ParseVMStat(string(out)); ok {
			hw.FreeBytes = free
			hw.WiredBytes = wired
			hw.CompressedBytes = compressed
			hw.PageIns = pageins
			hw.PageOuts = pageouts
			hw.Available = true
		}
	} else if out, err := runIsolated(ctx, runner, cmdTimeout, "vm_stat"); err == nil {
		if free, wired, compressed, pageins, pageouts, ok := ParseVMStat(string(out)); ok {
			hw.FreeBytes = free
			hw.WiredBytes = wired
			hw.CompressedBytes = compressed
			hw.PageIns = pageins
			hw.PageOuts = pageouts
			hw.Available = true
		}
	}

	// 7. pmset therm
	if out, err := runIsolated(ctx, runner, cmdTimeout, "/usr/bin/pmset", "-g", "therm"); err == nil {
		state, cpuLvl, gpuLvl := ParsePMSetTherm(string(out))
		hw.ThermalState = state
		hw.CPUThermalLevel = cpuLvl
		hw.GPUThermalLevel = gpuLvl
	}

	// 8. pmset ps
	if out, err := runIsolated(ctx, runner, cmdTimeout, "/usr/bin/pmset", "-g", "ps"); err == nil {
		if pwr, batt, ok := ParsePMSetPower(string(out)); ok {
			hw.PowerSource = pwr
			hw.BatteryLevelPct = batt
		}
	}

	return hw
}

// ParseIORegXML parses XML output from `ioreg -a -r -d 1 -c IOAccelerator`.
func ParseIORegXML(data []byte) (allocMem uint64, inUseMem uint64, devUtil float64, rendUtil float64, recoveryCount uint64, ok bool) {
	if len(data) == 0 {
		return 0, 0, 0, 0, 0, false
	}
	decoder := xml.NewDecoder(strings.NewReader(string(data)))
	inPerfStats := false
	perfDepth := 0
	depth := 0
	var currentKey string
	var textBuf strings.Builder

	for {
		tok, err := decoder.Token()
		if err != nil {
			if err == io.EOF {
				break
			}
			break
		}
		switch el := tok.(type) {
		case xml.StartElement:
			depth++
			textBuf.Reset()
		case xml.CharData:
			textBuf.Write(el)
		case xml.EndElement:
			text := strings.TrimSpace(textBuf.String())
			switch el.Name.Local {
			case "key":
				if text == "PerformanceStatistics" {
					inPerfStats = true
					perfDepth = depth
				} else if inPerfStats && depth == perfDepth+1 {
					currentKey = text
				}
			case "integer", "real":
				if inPerfStats && depth == perfDepth+1 && currentKey != "" {
					switch currentKey {
					case "Alloc system memory":
						if v, err := strconv.ParseUint(text, 10, 64); err == nil {
							allocMem = v
							ok = true
						}
					case "In use system memory":
						if v, err := strconv.ParseUint(text, 10, 64); err == nil {
							inUseMem = v
							ok = true
						}
					case "Device Utilization %", "GPU Activity(%)":
						if v, err := strconv.ParseFloat(text, 64); err == nil {
							devUtil = v
							ok = true
						}
					case "Renderer Utilization %":
						if v, err := strconv.ParseFloat(text, 64); err == nil {
							rendUtil = v
							ok = true
						}
					case "recoveryCount":
						if v, err := strconv.ParseUint(text, 10, 64); err == nil {
							recoveryCount = v
							ok = true
						}
					}
					currentKey = ""
				}
			case "dict":
				if inPerfStats && depth == perfDepth {
					inPerfStats = false
					perfDepth = 0
					currentKey = ""
				}
			}
			textBuf.Reset()
			depth--
		}
	}
	return allocMem, inUseMem, devUtil, rendUtil, recoveryCount, ok
}

// ParseSysctlMemsize parses uint64 sysctl values like hw.memsize.
func ParseSysctlMemsize(out string) (uint64, bool) {
	fields := strings.Fields(strings.TrimSpace(out))
	if len(fields) == 0 {
		return 0, false
	}
	v, err := strconv.ParseUint(fields[0], 10, 64)
	if err != nil || v == 0 {
		return 0, false
	}
	return v, true
}

// ParseSysctlWiredLimitMB parses iogpu.wired_limit_mb into bytes.
func ParseSysctlWiredLimitMB(out string) (uint64, bool) {
	mb, ok := ParseSysctlMemsize(out)
	if !ok || mb == 0 {
		return 0, false
	}
	return mb * 1024 * 1024, true
}

// ParseSysctlMemoryStatusLevel parses kern.memorystatus_level.
func ParseSysctlMemoryStatusLevel(out string) (uint64, bool) {
	fields := strings.Fields(strings.TrimSpace(out))
	if len(fields) == 0 {
		return 0, false
	}
	v, err := strconv.ParseUint(fields[0], 10, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// ParseSysctlSwapUsage parses `vm.swapusage`:
// `total = 24576.00M used = 23070.06M free = 1505.94M (encrypted)`
func ParseSysctlSwapUsage(out string) (total uint64, used uint64, ok bool) {
	fields := strings.Fields(out)
	var hasTotal, hasUsed bool
	for i := 0; i < len(fields); i++ {
		key := strings.ToLower(strings.TrimSuffix(fields[i], ":"))
		if key != "total" && key != "used" {
			continue
		}
		valIdx := i + 1
		if valIdx < len(fields) && fields[valIdx] == "=" {
			valIdx++
		}
		if valIdx >= len(fields) {
			continue
		}
		bytes, parsed := parseByteQuantity(fields[valIdx])
		if !parsed {
			continue
		}
		if key == "total" {
			total = bytes
			hasTotal = true
		} else if key == "used" {
			used = bytes
			hasUsed = true
		}
	}
	return total, used, hasTotal && hasUsed
}

func parseByteQuantity(raw string) (uint64, bool) {
	raw = strings.TrimSpace(raw)
	if len(raw) < 2 {
		return 0, false
	}
	unit := raw[len(raw)-1]
	multiplier := uint64(1)
	switch unit {
	case 'K', 'k':
		multiplier = 1024
	case 'M', 'm':
		multiplier = 1024 * 1024
	case 'G', 'g':
		multiplier = 1024 * 1024 * 1024
	case 'T', 't':
		multiplier = 1024 * 1024 * 1024 * 1024
	case 'B', 'b':
		multiplier = 1
	default:
		// Try parsing as plain uint64
		if v, err := strconv.ParseUint(raw, 10, 64); err == nil {
			return v, true
		}
		return 0, false
	}
	numStr := raw[:len(raw)-1]
	val, err := strconv.ParseFloat(numStr, 64)
	if err != nil || val < 0 {
		return 0, false
	}
	return uint64(val * float64(multiplier)), true
}

// ParseVMStat parses `vm_stat` output for free, wired, compressed, and paging counters.
func ParseVMStat(out string) (free, wired, compressed, pageins, pageouts uint64, ok bool) {
	lines := strings.Split(out, "\n")
	pageSize := uint64(16384) // standard Apple Silicon page size default
	for _, l := range lines {
		lower := strings.ToLower(l)
		const marker = "page size of "
		if idx := strings.Index(lower, marker); idx >= 0 {
			fields := strings.Fields(lower[idx+len(marker):])
			if len(fields) >= 1 {
				if ps, err := strconv.ParseUint(fields[0], 10, 64); err == nil && ps > 0 {
					pageSize = ps
				}
			}
		}
	}

	for _, l := range lines {
		k, v, found := strings.Cut(l, ":")
		if !found {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(k))
		valStr := strings.TrimSuffix(strings.TrimSpace(v), ".")
		val, err := strconv.ParseUint(valStr, 10, 64)
		if err != nil {
			continue
		}

		switch key {
		case "pages free":
			free = val * pageSize
			ok = true
		case "pages wired down":
			wired = val * pageSize
			ok = true
		case "pages occupied by compressor":
			compressed = val * pageSize
			ok = true
		case "pageins":
			pageins = val
			ok = true
		case "pageouts":
			pageouts = val
			ok = true
		}
	}
	return free, wired, compressed, pageins, pageouts, ok
}

// ParsePMSetTherm parses `pmset -g therm` output.
func ParsePMSetTherm(out string) (state ThermalState, cpuLevel, gpuLevel uint64) {
	lower := strings.ToLower(out)
	if strings.Contains(lower, "no thermal warning level has been recorded") {
		return ThermalNominal, 0, 0
	}

	for _, l := range strings.Split(out, "\n") {
		line := strings.TrimSpace(l)
		if strings.Contains(line, "CPU_Thermal_Level") {
			parts := strings.Split(line, "=")
			if len(parts) >= 2 {
				if v, err := strconv.ParseUint(strings.TrimSpace(parts[1]), 10, 64); err == nil {
					cpuLevel = v
				}
			}
		} else if strings.Contains(line, "GPU_Thermal_Level") {
			parts := strings.Split(line, "=")
			if len(parts) >= 2 {
				if v, err := strconv.ParseUint(strings.TrimSpace(parts[1]), 10, 64); err == nil {
					gpuLevel = v
				}
			}
		} else if strings.Contains(line, "Thermal warning level") {
			parts := strings.Split(line, "=")
			if len(parts) >= 2 {
				if v, err := strconv.ParseUint(strings.TrimSpace(parts[1]), 10, 64); err == nil {
					if v > cpuLevel {
						cpuLevel = v
					}
				}
			}
		}
	}

	maxLvl := cpuLevel
	if gpuLevel > maxLvl {
		maxLvl = gpuLevel
	}

	switch {
	case maxLvl == 0:
		state = ThermalNominal
	case maxLvl == 1:
		state = ThermalFair
	case maxLvl == 2:
		state = ThermalSerious
	case maxLvl >= 3:
		state = ThermalCritical
	default:
		state = ThermalUnknown
	}

	return state, cpuLevel, gpuLevel
}

// ParsePMSetPower parses `pmset -g ps` output for power source and battery %.
func ParsePMSetPower(out string) (source PowerSource, batteryPct int, ok bool) {
	if strings.Contains(out, "'AC Power'") {
		source = PowerAC
		ok = true
	} else if strings.Contains(out, "'Battery Power'") {
		source = PowerBattery
		ok = true
	} else {
		source = PowerUnknown
	}

	// Parse battery percentage: e.g., "100%;" or "85%"
	for _, token := range strings.Fields(out) {
		token = strings.TrimSuffix(token, ";")
		token = strings.TrimSuffix(token, ",")
		if strings.HasSuffix(token, "%") {
			pctStr := strings.TrimSuffix(token, "%")
			if pct, err := strconv.Atoi(pctStr); err == nil && pct >= 0 && pct <= 100 {
				batteryPct = pct
				ok = true
				break
			}
		}
	}

	return source, batteryPct, ok
}
