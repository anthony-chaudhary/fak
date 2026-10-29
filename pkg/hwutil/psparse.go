package hwutil

// Pure parsers for the Windows collector's PowerShell output. They take raw
// command text and produce normalized rows so the null-on-error contract is
// testable without a WMI surface (the fake-probe pattern).

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// psSections splits a sectioned PowerShell output ("MARKER" line followed by
// its JSON payload, repeated) into marker→payload strings.
func psSections(out []byte) map[string]string {
	lines := strings.Split(strings.ReplaceAll(string(out), "\r\n", "\n"), "\n")
	sections := map[string][]string{}
	var cur string
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if l == "" {
			continue
		}
		switch l {
		case "DRIVE", "PHYS", "ADAPTER", "ADAPTER2", "PERF", "PAGEFILE", "COUNTER":
			cur = l
			sections[cur] = nil
		default:
			if cur != "" {
				sections[cur] = append(sections[cur], l)
			}
		}
	}
	outMap := make(map[string]string, len(sections))
	for k, v := range sections {
		outMap[k] = strings.Join(v, "\n")
	}
	return outMap
}

// numberField reads one numeric field from a decoded PowerShell object; JSON
// numbers arrive as float64, WMI sometimes as strings. Absent or null fields
// report ok=false (the null-on-error edge).
func numberField(m map[string]any, key string) (float64, bool) {
	v, ok := m[key]
	if !ok || v == nil {
		return 0, false
	}
	switch n := v.(type) {
	case float64:
		return n, true
	case string:
		var f float64
		if _, err := fmt.Sscanf(n, "%g", &f); err == nil {
			return f, true
		}
	}
	return 0, false
}

// stringField reads one string field from a decoded PowerShell object.
func stringField(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

// decodePSRows unmarshals a PowerShell ConvertTo-Json payload into row maps;
// a single-row payload (an object, not an array) normalizes to one row.
func decodePSRows(raw string) []map[string]any {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	var rows []map[string]any
	if err := json.Unmarshal([]byte(raw), &rows); err == nil {
		return rows
	}
	var one map[string]any
	if err := json.Unmarshal([]byte(raw), &one); err == nil {
		return []map[string]any{one}
	}
	return nil
}

// windowsDiskKind maps a Win32 BusType onto the closed disk-kind set. SAS and
// RAID controllers have no honest member of the set and stay unknown.
func windowsDiskKind(busType string) DiskKind {
	switch strings.ToUpper(strings.TrimSpace(busType)) {
	case "NVME":
		return DiskNVMe
	case "USB":
		return DiskUSB
	case "IDE":
		return DiskSATA
	default:
		return DiskUnknown
	}
}

// driveByNumber matches a Win32_DiskDrive DeviceID against a physical-disk
// number, case-insensitively.
func driveByNumber(drives []map[string]any, num int) (map[string]any, bool) {
	want := fmt.Sprintf("\\\\.\\physicaldrive%d", num)
	for _, d := range drives {
		if strings.EqualFold(strings.TrimSpace(stringField(d, "DeviceID")), want) {
			return d, true
		}
	}
	return nil, false
}

// splitCounterPath splits a performance-counter path of the form
// \\host\object(instance)\counter into object, instance, and counter. The
// instance keeps its original case (it becomes a wire identity); the object
// and counter come back lowercased for lookup.
func splitCounterPath(path string) (object, instance, counter string, ok bool) {
	rest := strings.TrimSpace(path)
	for strings.HasPrefix(rest, `\\`) {
		rest = rest[2:]
		i := strings.IndexByte(rest, '\\')
		if i < 0 {
			break
		}
		rest = rest[i+1:]
	}
	parts := strings.Split(rest, `\`)
	if len(parts) != 2 {
		return "", "", "", false
	}
	objPart, counterName := parts[0], strings.TrimSpace(parts[1])
	open := strings.IndexByte(objPart, '(')
	close := strings.LastIndexByte(objPart, ')')
	if open < 0 || close <= open || counterName == "" {
		return "", "", "", false
	}
	return strings.ToLower(objPart[:open]), objPart[open+1 : close], strings.ToLower(counterName), true
}

// normalizeCounterInstance lowercases and trims a counter instance name so
// adapter identity can be matched without case games.
func normalizeCounterInstance(inst string) string {
	return strings.ToLower(strings.TrimSpace(inst))
}

// counterInstanceNumber extracts the leading drive number from a
// \\PhysicalDisk(*) instance name such as "0 c:".
func counterInstanceNumber(inst string) (int, bool) {
	s := strings.ToLower(strings.TrimSpace(inst))
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i == 0 {
		return 0, false
	}
	n, err := strconv.Atoi(s[:i])
	if err != nil || n < 0 {
		return 0, false
	}
	return n, true
}

// matchCounterInstance finds the unclaimed counter instance whose normalized
// name equals one of the accepted keys (adapter name or connection id).
func matchCounterInstance(keys []string, rates map[string]map[string]float64, matched map[string]bool) (string, map[string]float64, bool) {
	want := make(map[string]bool, len(keys))
	for _, k := range keys {
		want[k] = true
	}
	best := ""
	for inst := range rates {
		if matched[inst] || !want[normalizeCounterInstance(inst)] {
			continue
		}
		best = inst
	}
	if best == "" {
		return "", nil, false
	}
	matched[best] = true
	return best, rates[best], true
}

// lastCounterByName returns the cooked value of the LAST row whose counter
// path ends with the given lowercased "\\object\\counter" suffix. Instance-
// free objects (such as \\Memory) carry no parenthesized instance, which
// splitCounterPath deliberately rejects, so they are matched by suffix.
// Rows arrive in sample order and later samples overwrite earlier ones (the
// first sample of a /sec counter is a since-boot average); null or missing
// CookedValues stay unread.
func lastCounterByName(rows []map[string]any, suffix string) (float64, bool) {
	var v float64
	seen := false
	for _, row := range rows {
		p := strings.ToLower(strings.TrimSpace(stringField(row, "Path")))
		if !strings.HasSuffix(p, suffix) {
			continue
		}
		if n, ok := numberField(row, "CookedValue"); ok {
			v, seen = n, true
		}
	}
	return v, seen
}

// pageFileSwapUsed sums in-use pagefile bytes across Win32_PageFile rows:
// CurrentUsage when present, otherwise AllocatedBaseSize; WMI reports both
// in KiB. A row exposing neither usable value contributes nothing, and if no
// row does, ok=false (no pagefile surface means nil downstream, never an
// invented zero).
func pageFileSwapUsed(rows []map[string]any) (int64, bool) {
	var total int64
	any := false
	for _, row := range rows {
		v, ok := numberField(row, "CurrentUsage")
		if !ok {
			v, ok = numberField(row, "AllocatedBaseSize")
		}
		if ok && v >= 0 {
			total += int64(int64(v) * int64(kilobyte))
			any = true
		}
	}
	return total, any
}

// netDropRow carries one adapter's drop rates and cumulative error count
// derived from two Win32_NetworkAdapter polls. Every field stays nil until a
// usable source value exists (never an invented zero).
type netDropRow struct {
	rxPerS      *float64
	txPerS      *float64
	errorsTotal *int64
}

// netDropRates reduces two Win32_NetworkAdapter polls (before and after,
// dt seconds apart) into per-adapter drop rates plus the cumulative error
// total. Rates are safeDelta over InboundDiscards/OutboundDiscards; a
// backwards step (provider reset) keeps its rate nil. errors_total is the
// LATEST PacketErrors. An adapter missing from either poll, or one whose
// property is null or absent (common on virtual adapters), leaves the
// matching field nil.
func netDropRates(before, after []map[string]any, dt float64) map[string]netDropRow {
	prev := map[string]map[string]any{}
	for _, row := range before {
		key := normalizeCounterInstance(stringField(row, "Name"))
		if key == "" {
			continue
		}
		if _, seen := prev[key]; !seen {
			prev[key] = row
		}
	}
	out := map[string]netDropRow{}
	for _, row := range after {
		key := normalizeCounterInstance(stringField(row, "Name"))
		if key == "" {
			continue
		}
		if _, seen := out[key]; seen {
			continue
		}
		dr := netDropRow{}
		if e, ok := numberField(row, "PacketErrors"); ok {
			dr.errorsTotal = i64(int64(e))
		}
		if b, ok := prev[key]; ok {
			rate := func(dst **float64, prop string) {
				a, okA := numberField(b, prop)
				c, okC := numberField(row, prop)
				if !okA || !okC {
					return
				}
				if d, ok := safeDelta(uint64(a), uint64(c)); ok {
					*dst = avg(float64(d), dt)
				}
			}
			rate(&dr.rxPerS, "InboundDiscards")
			rate(&dr.txPerS, "OutboundDiscards")
		}
		out[key] = dr
	}
	return out
}

// lastCounterValues reduces a flat list of {Path, CookedValue} rows emitted
// by Get-Counter to instance -> counter -> value, keeping only the LAST
// sample set: samples arrive in order and earlier ones are since-boot
// averages, so later values overwrite earlier ones per (instance, counter).
func lastCounterValues(rows []map[string]any) map[string]map[string]float64 {
	out := map[string]map[string]float64{}
	for _, row := range rows {
		_, inst, counter, ok := splitCounterPath(stringField(row, "Path"))
		if !ok {
			continue
		}
		v, ok := numberField(row, "CookedValue")
		if !ok {
			continue
		}
		m := out[inst]
		if m == nil {
			m = map[string]float64{}
			out[inst] = m
		}
		m[counter] = v
	}
	return out
}
