//go:build darwin

package macobs

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestParseIORegXML(t *testing.T) {
	mockXML := []byte(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
	<dict>
		<key>PerformanceStatistics</key>
		<dict>
			<key>Alloc system memory</key>
			<integer>22620471296</integer>
			<key>Allocated PB Size</key>
			<integer>113508352</integer>
			<key>Device Utilization %</key>
			<integer>86</integer>
			<key>In use system memory</key>
			<integer>1334083584</integer>
			<key>Renderer Utilization %</key>
			<integer>83</integer>
			<key>recoveryCount</key>
			<integer>2</integer>
		</dict>
	</dict>
</array>
</plist>`)

	alloc, inUse, devUtil, rendUtil, recov, ok := ParseIORegXML(mockXML)
	if !ok {
		t.Fatalf("expected ParseIORegXML ok=true")
	}
	if alloc != 22620471296 {
		t.Errorf("got alloc %d, want 22620471296", alloc)
	}
	if inUse != 1334083584 {
		t.Errorf("got inUse %d, want 1334083584", inUse)
	}
	if devUtil != 86.0 {
		t.Errorf("got devUtil %f, want 86.0", devUtil)
	}
	if rendUtil != 83.0 {
		t.Errorf("got rendUtil %f, want 83.0", rendUtil)
	}
	if recov != 2 {
		t.Errorf("got recov %d, want 2", recov)
	}
}

func TestParseIORegXMLEmpty(t *testing.T) {
	_, _, _, _, _, ok := ParseIORegXML(nil)
	if ok {
		t.Errorf("expected ok=false for empty ioreg XML")
	}
}

func TestParseSysctlValues(t *testing.T) {
	// hw.memsize
	mem, ok := ParseSysctlMemsize("38654705664\n")
	if !ok || mem != 38654705664 {
		t.Errorf("ParseSysctlMemsize got (%d, %v), want (38654705664, true)", mem, ok)
	}

	// iogpu.wired_limit_mb
	wiredBytes, ok := ParseSysctlWiredLimitMB("24576\n")
	if !ok || wiredBytes != 24576*1024*1024 {
		t.Errorf("ParseSysctlWiredLimitMB got (%d, %v), want (%d, true)", wiredBytes, ok, 24576*1024*1024)
	}

	// kern.memorystatus_level
	lvl, ok := ParseSysctlMemoryStatusLevel("37\n")
	if !ok || lvl != 37 {
		t.Errorf("ParseSysctlMemoryStatusLevel got (%d, %v), want (37, true)", lvl, ok)
	}

	// vm.swapusage
	swapOut := "total = 24576.00M  used = 1024.50M  free = 23551.50M  (encrypted)\n"
	total, used, ok := ParseSysctlSwapUsage(swapOut)
	if !ok {
		t.Fatalf("ParseSysctlSwapUsage failed")
	}
	if total != 24576*1024*1024 {
		t.Errorf("got total swap %d, want %d", total, 24576*1024*1024)
	}
	wantUsed := uint64(1024.50 * 1024 * 1024)
	if used != wantUsed {
		t.Errorf("got used swap %d, want %d", used, wantUsed)
	}
}

func TestParseVMStat(t *testing.T) {
	mockVMStat := `Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    12500.
Pages active:                                 400000.
Pages inactive:                               300000.
Pages wired down:                             200000.
Pages occupied by compressor:                 100000.
Pageins:                                      5000000.
Pageouts:                                        1234.
`
	free, wired, comp, ins, outs, ok := ParseVMStat(mockVMStat)
	if !ok {
		t.Fatalf("ParseVMStat failed")
	}
	if free != 12500*16384 {
		t.Errorf("got free %d, want %d", free, 12500*16384)
	}
	if wired != 200000*16384 {
		t.Errorf("got wired %d, want %d", wired, 200000*16384)
	}
	if comp != 100000*16384 {
		t.Errorf("got comp %d, want %d", comp, 100000*16384)
	}
	if ins != 5000000 {
		t.Errorf("got pageins %d, want 5000000", ins)
	}
	if outs != 1234 {
		t.Errorf("got pageouts %d, want 1234", outs)
	}
}

func TestParsePMSetTherm(t *testing.T) {
	// Nominal
	state, cpu, gpu := ParsePMSetTherm("Note: No thermal warning level has been recorded\n")
	if state != ThermalNominal || cpu != 0 || gpu != 0 {
		t.Errorf("got (%s, %d, %d), want (NOMINAL, 0, 0)", state, cpu, gpu)
	}

	// Throttled CPU
	out := `CPU_Thermal_Level = 2
GPU_Thermal_Level = 1
IOPlatformThermalProfile = 0`
	state, cpu, gpu = ParsePMSetTherm(out)
	if state != ThermalSerious || cpu != 2 || gpu != 1 {
		t.Errorf("got (%s, %d, %d), want (SERIOUS, 2, 1)", state, cpu, gpu)
	}

	// Critical GPU
	outCrit := `CPU_Thermal_Level = 1
GPU_Thermal_Level = 3`
	state, cpu, gpu = ParsePMSetTherm(outCrit)
	if state != ThermalCritical || cpu != 1 || gpu != 3 {
		t.Errorf("got (%s, %d, %d), want (CRITICAL, 1, 3)", state, cpu, gpu)
	}
}

func TestParsePMSetPower(t *testing.T) {
	acOut := `Now drawing from 'AC Power'
 -InternalBattery-0 (id=22020195)	100%; charged; 0:00 remaining present: true`
	src, pct, ok := ParsePMSetPower(acOut)
	if !ok || src != PowerAC || pct != 100 {
		t.Errorf("got (%s, %d, %v), want (AC, 100, true)", src, pct, ok)
	}

	battOut := `Now drawing from 'Battery Power'
 -InternalBattery-0 (id=22020195)	85%; discharging; 4:20 remaining present: true`
	src, pct, ok = ParsePMSetPower(battOut)
	if !ok || src != PowerBattery || pct != 85 {
		t.Errorf("got (%s, %d, %v), want (BATTERY, 85, true)", src, pct, ok)
	}
}

func TestCollectHardwareWithRunner(t *testing.T) {
	mockRunner := func(ctx context.Context, name string, args ...string) ([]byte, error) {
		cmdStr := name + " " + strings.Join(args, " ")
		switch {
		case strings.Contains(cmdStr, "hw.memsize"):
			return []byte("34359738368\n"), nil // 32GB
		case strings.Contains(cmdStr, "iogpu.wired_limit_mb"):
			return []byte("24576\n"), nil // 24GB
		case strings.Contains(cmdStr, "vm.swapusage"):
			return []byte("total = 1024.00M used = 0.00M free = 1024.00M\n"), nil
		case strings.Contains(cmdStr, "kern.memorystatus_level"):
			return []byte("65\n"), nil
		case strings.Contains(cmdStr, "vm_stat"):
			return []byte("page size of 16384 bytes\nPages free: 10000.\nPages wired down: 20000.\nPages occupied by compressor: 5000.\nPageins: 100.\nPageouts: 0.\n"), nil
		case strings.Contains(cmdStr, "therm"):
			return []byte("Note: No thermal warning level has been recorded\n"), nil
		case strings.Contains(cmdStr, "ps"):
			return []byte("Now drawing from 'AC Power'\n -InternalBattery-0 100%;\n"), nil
		default:
			return nil, errors.New("unknown mock command")
		}
	}

	ctx := context.Background()
	hw := CollectHardwareWithRunner(ctx, mockRunner)
	if !hw.Available {
		t.Fatalf("expected hw.Available to be true")
	}
	if hw.TotalSystemMemoryBytes != 34359738368 {
		t.Errorf("got total sys mem %d, want 34359738368", hw.TotalSystemMemoryBytes)
	}
	if hw.WiredMemoryLimitBytes != 24576*1024*1024 {
		t.Errorf("got wired limit %d, want %d", hw.WiredMemoryLimitBytes, 24576*1024*1024)
	}
	if hw.ThermalState != ThermalNominal {
		t.Errorf("got thermal state %s, want NOMINAL", hw.ThermalState)
	}
	if hw.PowerSource != PowerAC {
		t.Errorf("got power source %s, want AC", hw.PowerSource)
	}
}

func TestCollectHardwareLiveDarwin(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	hw := CollectHardware(ctx)
	if !hw.Available {
		t.Logf("CollectHardware: live hardware counters not available in current environment (permitted fail-soft)")
		return
	}
	if hw.TotalSystemMemoryBytes == 0 {
		t.Errorf("expected non-zero total system memory on Darwin")
	}
}

func TestParseIORegXML_MalformedAndCorruptedStress(t *testing.T) {
	cases := []struct {
		name string
		data []byte
	}{
		{"binary garbage", []byte{0x00, 0xff, 0xfe, 0x12, 0x34, 0xaa, 0xbb}},
		{"truncated at preamble", []byte(`<?xml version="1.0"?><plist><array><dict><key>PerformanceStatistics`)},
		{"truncated at dict", []byte(`<?xml version="1.0"?><plist><array><dict><key>PerformanceStatistics</key><dict>`)},
		{"truncated at key", []byte(`<?xml version="1.0"?><plist><array><dict><key>PerformanceStatistics</key><dict><key>Alloc system memory`)},
		{"truncated at integer open", []byte(`<?xml version="1.0"?><plist><array><dict><key>PerformanceStatistics</key><dict><key>Alloc system memory</key><integer>`)},
		{"truncated at integer value", []byte(`<?xml version="1.0"?><plist><array><dict><key>PerformanceStatistics</key><dict><key>Alloc system memory</key><integer>12345`)},
		{"valid plist without PerformanceStatistics", []byte(`<?xml version="1.0"?><plist><dict><key>CFBundleName</key><string>fak</string></dict></plist>`)},
		{"empty PerformanceStatistics dict", []byte(`<?xml version="1.0"?><plist><dict><key>PerformanceStatistics</key><dict></dict></dict></plist>`)},
		{"scalar PerformanceStatistics", []byte(`<?xml version="1.0"?><plist><dict><key>PerformanceStatistics</key><string>corrupted</string></dict></plist>`)},
		{"non-numeric integer", []byte(`<?xml version="1.0"?><plist><dict><key>PerformanceStatistics</key><dict><key>Alloc system memory</key><integer>not_a_number</integer></dict></dict></plist>`)},
		{"negative integer value", []byte(`<?xml version="1.0"?><plist><dict><key>PerformanceStatistics</key><dict><key>In use system memory</key><integer>-500</integer></dict></dict></plist>`)},
		{"non-numeric real value", []byte(`<?xml version="1.0"?><plist><dict><key>PerformanceStatistics</key><dict><key>Device Utilization %</key><real>nan_or_bad</real></dict></dict></plist>`)},
		{"overflow integer value", []byte(`<?xml version="1.0"?><plist><dict><key>PerformanceStatistics</key><dict><key>recoveryCount</key><integer>9999999999999999999999999999999999999999999999999999999999999</integer></dict></dict></plist>`)},
		{"mismatched unclosed tags", []byte(`<dict><key><integer><unclosed><foo>bar`)},
		{"whitespace only", []byte("   \n\t  \r\n ")},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("ParseIORegXML panicked on %s: %v", tc.name, r)
				}
			}()
			alloc, inUse, devUtil, rendUtil, recov, ok := ParseIORegXML(tc.data)
			if ok {
				t.Logf("[%s] yielded partial parse: alloc=%d inUse=%d devUtil=%f rendUtil=%f recov=%d",
					tc.name, alloc, inUse, devUtil, rendUtil, recov)
			}
		})
	}
}

func TestParseSysctl_MalformedAndTruncatedStress(t *testing.T) {
	// 1. Memsize
	memCases := []string{
		"",
		"   ",
		"\n\t",
		"hw.memsize: unknown oid 'hw.memsize'",
		"38654abc",
		"-1024",
		"0",
		"99999999999999999999999999999999999999999999999999999999999999999999999",
	}
	for _, c := range memCases {
		val, ok := ParseSysctlMemsize(c)
		if ok {
			t.Errorf("expected ParseSysctlMemsize(%q) ok=false, got %d", c, val)
		}
	}

	// 2. WiredLimitMB
	wiredCases := []string{
		"",
		"0",
		"-100",
		"iogpu: error 1",
		"99999999999999999999999999999999999999999999999999999999999999999999999",
	}
	for _, c := range wiredCases {
		val, ok := ParseSysctlWiredLimitMB(c)
		if ok {
			t.Errorf("expected ParseSysctlWiredLimitMB(%q) ok=false, got %d", c, val)
		}
	}

	// 3. MemoryStatusLevel
	lvlCases := []string{
		"",
		"   ",
		"invalid",
		"-5",
		"99999999999999999999999999999999999999999999999999999999999999999999999",
	}
	for _, c := range lvlCases {
		val, ok := ParseSysctlMemoryStatusLevel(c)
		if ok {
			t.Errorf("expected ParseSysctlMemoryStatusLevel(%q) ok=false, got %d", c, val)
		}
	}

	// 4. SwapUsage
	swapCases := []string{
		"",
		"total =",
		"total = ",
		"total = 1024.00M used =",
		"total",
		"used",
		"total = notanumberM used = corruptG",
		"total = 1024.00X used = 500.00Z",
		"total = M used = G",
		"total = -100M used = -50M",
		"vm.swapusage: sysctl failed",
	}
	for _, c := range swapCases {
		total, used, ok := ParseSysctlSwapUsage(c)
		if ok {
			t.Errorf("expected ParseSysctlSwapUsage(%q) ok=false, got (%d, %d)", c, total, used)
		}
	}

	// Test units support: K, M, G, T, B, and bare number
	validUnits := []struct {
		input     string
		wantTotal uint64
		wantUsed  uint64
	}{
		{"total = 1024.00K used = 512.00K free = 512.00K", 1024 * 1024, 512 * 1024},
		{"total = 16.00M used = 8.00M free = 8.00M", 16 * 1024 * 1024, 8 * 1024 * 1024},
		{"total = 2.00G used = 1.00G free = 1.00G", 2 * 1024 * 1024 * 1024, 1 * 1024 * 1024 * 1024},
		{"total = 1.00T used = 0.50T free = 0.50T", 1024 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024},
		{"total = 1048576B used = 524288B free = 524288B", 1048576, 524288},
		{"total = 2000 used = 1000", 2000, 1000},
	}
	for _, tc := range validUnits {
		total, used, ok := ParseSysctlSwapUsage(tc.input)
		if !ok {
			t.Errorf("ParseSysctlSwapUsage(%q) failed unexpectedly", tc.input)
		}
		if total != tc.wantTotal {
			t.Errorf("[%s] total = %d, want %d", tc.input, total, tc.wantTotal)
		}
		if used != tc.wantUsed {
			t.Errorf("[%s] used = %d, want %d", tc.input, used, tc.wantUsed)
		}
	}
}

func TestParseVMStat_MalformedAndTruncatedStress(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		{"empty string", ""},
		{"header only", "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"},
		{"truncated page size header", "Mach Virtual Memory Statistics: (page size of \nPages free: 100.\n"},
		{"non-numeric page size", "Mach Virtual Memory Statistics: (page size of notanumber bytes)\nPages free: 100.\n"},
		{"zero page size", "Mach Virtual Memory Statistics: (page size of 0 bytes)\nPages free: 100.\n"},
		{"missing colons", "Pages free 12500\nPages wired 2000\n"},
		{"non-numeric values", "Pages free: notanumber.\nPageins: corrupt.\n"},
		{"negative values", "Pages free: -50.\n"},
		{"overflow values", "Pages free: 99999999999999999999999999999999999999999999999999999999999999999999999.\n"},
		{"random garbage lines", "unexpected daemon message\nwarning: vm_stat deprecated\n"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("ParseVMStat panicked on %s: %v", tc.name, r)
				}
			}()
			ParseVMStat(tc.input)
		})
	}
}

func TestParsePMSet_MalformedAndTruncatedStress(t *testing.T) {
	// 1. pmset -g therm
	thermCases := []struct {
		name      string
		input     string
		wantState ThermalState
	}{
		{"empty string", "", ThermalNominal},
		{"unrelated error", "pmset: error reading therm", ThermalNominal},
		{"missing equals", "CPU_Thermal_Level\nGPU_Thermal_Level", ThermalNominal},
		{"non-numeric levels", "CPU_Thermal_Level = high\nGPU_Thermal_Level = bad", ThermalNominal},
		{"extreme high level", "CPU_Thermal_Level = 99", ThermalCritical},
		{"GPU extreme high level", "GPU_Thermal_Level = 4", ThermalCritical},
		{"thermal warning level key", "Thermal warning level = 2", ThermalSerious},
	}
	for _, tc := range thermCases {
		t.Run(tc.name, func(t *testing.T) {
			state, _, _ := ParsePMSetTherm(tc.input)
			if state != tc.wantState {
				t.Errorf("[%s] state = %s, want %s", tc.name, state, tc.wantState)
			}
		})
	}

	// 2. pmset -g ps
	powerCases := []struct {
		name       string
		input      string
		wantSource PowerSource
		wantPct    int
		wantOk     bool
	}{
		{"empty string", "", PowerUnknown, 0, false},
		{"no battery line", "Now drawing from 'AC Power'\n no internal battery", PowerAC, 0, true},
		{"no power drawing string", " -InternalBattery-0 80%;", PowerUnknown, 80, true},
		{"corrupted percentage", "Now drawing from 'Battery Power'\n -InternalBattery-0 bad%;", PowerBattery, 0, true},
		{"out of bounds high percentage", "Now drawing from 'Battery Power'\n -InternalBattery-0 150%;", PowerBattery, 0, true},
		{"negative percentage", "Now drawing from 'Battery Power'\n -InternalBattery-0 -20%;", PowerBattery, 0, true},
		{"zero percent battery", "Now drawing from 'Battery Power'\n -InternalBattery-0 0%;", PowerBattery, 0, true},
		{"battery charging with punctuation", "Now drawing from 'AC Power'\n -InternalBattery-0 98%, charging;", PowerAC, 98, true},
	}
	for _, tc := range powerCases {
		t.Run(tc.name, func(t *testing.T) {
			src, pct, ok := ParsePMSetPower(tc.input)
			if src != tc.wantSource {
				t.Errorf("[%s] source = %s, want %s", tc.name, src, tc.wantSource)
			}
			if pct != tc.wantPct {
				t.Errorf("[%s] pct = %d, want %d", tc.name, pct, tc.wantPct)
			}
			if ok != tc.wantOk {
				t.Errorf("[%s] ok = %v, want %v", tc.name, ok, tc.wantOk)
			}
		})
	}
}
