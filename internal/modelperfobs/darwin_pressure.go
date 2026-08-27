package modelperfobs

import (
	"context"
	"math"
	"math/big"
	"strconv"
	"strings"
	"time"
)

const darwinAvailableSemantics = "darwin-vm-stat-free-plus-inactive-pages"

func collectDarwinHostSnapshot(ctx context.Context, pid int, run func(context.Context, string, ...string) (string, error), now func() time.Time) hostSnapshot {
	s := hostSnapshot{collector: "darwin-vm-stat-sysctl"}
	vmStatStarted := now()
	vmStat, vmStatErr := run(ctx, "/usr/bin/vm_stat")
	vmStatFinished := now()
	s.at = darwinObservationMidpoint(vmStatStarted, vmStatFinished)
	if vmStatErr == nil {
		parseDarwinVMStat(vmStat, &s.host)
	}
	if out, err := run(ctx, "/usr/sbin/sysctl", "-n", "hw.memsize"); err == nil {
		parseDarwinPhysicalTotal(out, &s.host)
	}
	if out, err := run(ctx, "/usr/sbin/sysctl", "-n", "vm.swapusage"); err == nil {
		parseDarwinSwapUsage(out, &s.host)
	}
	if out, err := run(ctx, "/bin/ps", "-o", "rss=", "-p", strconv.Itoa(pid)); err == nil {
		parseDarwinPSRSS(out, &s.host)
	}

	s.availability.PhysicalMemory = s.host.PhysicalTotalBytes != nil ||
		s.host.PhysicalAvailableBytes != nil || s.host.PhysicalFreeBytes != nil ||
		s.host.MemoryWiredResidentBytes != nil || s.host.MemoryCompressedResidentBytes != nil
	s.availability.ProcessMemory = s.host.ProcessResidentBytes != nil
	s.availability.MemoryPressure = s.host.MemoryWiredResidentBytes != nil ||
		s.host.MemoryCompressedResidentBytes != nil || s.host.SwapTotalBytes != nil ||
		s.host.SwapUsedBytes != nil || s.host.MemoryPageInPagesTotal != nil ||
		s.host.MemoryPageOutPagesTotal != nil || s.host.MemorySwapInPagesTotal != nil ||
		s.host.MemorySwapOutPagesTotal != nil
	return s
}

func darwinObservationMidpoint(started, finished time.Time) time.Time {
	if finished.Before(started) {
		return started
	}
	return started.Add(finished.Sub(started) / 2)
}

// parseDarwinVMStat retains independently valid rows. Byte-valued residency
// fields require a valid page-size header; cumulative paging counters do not.
func parseDarwinVMStat(input string, host *HostSignals) bool {
	if host == nil {
		return false
	}
	pageSize, pageSizeOK := parseDarwinVMStatPageSize(input)
	values := make(map[string]uint64)
	for _, line := range strings.Split(input, "\n") {
		key, raw, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		fields := strings.Fields(strings.TrimSpace(raw))
		if len(fields) == 0 {
			continue
		}
		value, err := strconv.ParseUint(strings.TrimSuffix(fields[0], "."), 10, 64)
		if err != nil {
			continue
		}
		values[strings.ToLower(strings.TrimSpace(key))] = value
	}

	available := false
	setCounter := func(destination **uint64, keys ...string) {
		if value, ok := firstDarwinValue(values, keys...); ok {
			*destination = cloneU64(&value)
			available = true
		}
	}
	setCounter(&host.MemoryPageInPagesTotal, "pageins")
	setCounter(&host.MemoryPageOutPagesTotal, "pageouts")
	setCounter(&host.MemorySwapInPagesTotal, "swapins")
	setCounter(&host.MemorySwapOutPagesTotal, "swapouts")

	if !pageSizeOK {
		return available
	}
	freePages, freeOK := firstDarwinValue(values, "pages free", "free pages")
	inactivePages, inactiveOK := firstDarwinValue(values, "pages inactive", "inactive pages")
	if freeOK {
		if bytes, ok := darwinPagesToBytes(freePages, pageSize); ok {
			host.PhysicalFreeBytes = cloneU64(&bytes)
			available = true
		}
	}
	if freeOK && inactiveOK && freePages <= math.MaxUint64-inactivePages {
		if bytes, ok := darwinPagesToBytes(freePages+inactivePages, pageSize); ok {
			host.PhysicalAvailableBytes = cloneU64(&bytes)
			host.PhysicalAvailableSemantics = darwinAvailableSemantics
			available = true
		}
	}
	setResidentBytes := func(destination **uint64, keys ...string) {
		pages, ok := firstDarwinValue(values, keys...)
		if !ok {
			return
		}
		bytes, ok := darwinPagesToBytes(pages, pageSize)
		if !ok {
			return
		}
		*destination = cloneU64(&bytes)
		available = true
	}
	setResidentBytes(&host.MemoryWiredResidentBytes, "pages wired down", "pages wired")
	setResidentBytes(&host.MemoryCompressedResidentBytes, "pages occupied by compressor")
	return available
}

func parseDarwinVMStatPageSize(input string) (uint64, bool) {
	const marker = "page size of "
	for _, line := range strings.Split(input, "\n") {
		lower := strings.ToLower(line)
		start := strings.Index(lower, marker)
		if start < 0 {
			continue
		}
		fields := strings.Fields(lower[start+len(marker):])
		if len(fields) < 2 || !strings.HasPrefix(fields[1], "byte") {
			return 0, false
		}
		pageSize, err := strconv.ParseUint(fields[0], 10, 64)
		return pageSize, err == nil && pageSize > 0
	}
	return 0, false
}

func parseDarwinPhysicalTotal(input string, host *HostSignals) bool {
	if host == nil {
		return false
	}
	fields := strings.Fields(input)
	if len(fields) != 1 {
		return false
	}
	total, err := strconv.ParseUint(fields[0], 10, 64)
	if err != nil || total == 0 {
		return false
	}
	host.PhysicalTotalBytes = cloneU64(&total)
	return true
}

func parseDarwinSwapUsage(input string, host *HostSignals) bool {
	if host == nil {
		return false
	}
	fields := strings.Fields(input)
	var total, used *uint64
	for i := 0; i < len(fields); i++ {
		key := strings.ToLower(strings.TrimSuffix(fields[i], ":"))
		if key != "total" && key != "used" {
			continue
		}
		valueIndex := i + 1
		if valueIndex < len(fields) && fields[valueIndex] == "=" {
			valueIndex++
		}
		if valueIndex >= len(fields) {
			continue
		}
		value, ok := parseDarwinByteQuantity(fields[valueIndex])
		if !ok {
			continue
		}
		if key == "total" && total == nil {
			total = cloneU64(&value)
		}
		if key == "used" && used == nil {
			used = cloneU64(&value)
		}
	}
	if total != nil {
		host.SwapTotalBytes = total
	}
	if used != nil && (total == nil || *used <= *total) {
		host.SwapUsedBytes = used
	}
	return host.SwapTotalBytes != nil || host.SwapUsedBytes != nil
}

func parseDarwinPSRSS(input string, host *HostSignals) bool {
	if host == nil {
		return false
	}
	fields := strings.Fields(input)
	if len(fields) != 1 {
		return false
	}
	kib, err := strconv.ParseUint(fields[0], 10, 64)
	if err != nil || kib > math.MaxUint64/1024 {
		return false
	}
	bytes := kib * 1024
	host.ProcessResidentBytes = cloneU64(&bytes)
	return true
}

func parseDarwinByteQuantity(raw string) (uint64, bool) {
	raw = strings.TrimSpace(raw)
	if len(raw) < 2 {
		return 0, false
	}
	unit := raw[len(raw)-1]
	var multiplier uint64
	switch unit {
	case 'K', 'k':
		multiplier = 1 << 10
	case 'M', 'm':
		multiplier = 1 << 20
	case 'G', 'g':
		multiplier = 1 << 30
	case 'T', 't':
		multiplier = 1 << 40
	default:
		return 0, false
	}
	number := raw[:len(raw)-1]
	if !isDarwinDecimal(number) {
		return 0, false
	}
	quantity, ok := new(big.Rat).SetString(number)
	if !ok || quantity.Sign() < 0 {
		return 0, false
	}
	quantity.Mul(quantity, new(big.Rat).SetInt(new(big.Int).SetUint64(multiplier)))
	bytes := new(big.Int).Quo(quantity.Num(), quantity.Denom())
	if !bytes.IsUint64() {
		return 0, false
	}
	return bytes.Uint64(), true
}

func isDarwinDecimal(value string) bool {
	if value == "" {
		return false
	}
	dot := -1
	for i := range len(value) {
		switch {
		case value[i] >= '0' && value[i] <= '9':
		case value[i] == '.' && dot < 0:
			dot = i
		default:
			return false
		}
	}
	return dot != 0 && dot != len(value)-1
}

func firstDarwinValue(values map[string]uint64, keys ...string) (uint64, bool) {
	for _, key := range keys {
		if value, ok := values[key]; ok {
			return value, true
		}
	}
	return 0, false
}

func darwinPagesToBytes(pages, pageSize uint64) (uint64, bool) {
	if pageSize == 0 || pages > math.MaxUint64/pageSize {
		return 0, false
	}
	return pages * pageSize, true
}
