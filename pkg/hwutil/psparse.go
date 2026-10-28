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
		case "DRIVE", "PHYS", "ADAPTER", "PERF", "COUNTER":
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
