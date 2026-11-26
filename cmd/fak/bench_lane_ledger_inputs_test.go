package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const benchLaneProps = "testdata/bench_lane_ledger/rtx5080_props.json"

func TestBenchLaneResolveHardwareFromFlags(t *testing.T) {
	hw, err := benchLaneResolveHardware("sm_120", 84, "")
	if err != nil {
		t.Fatalf("resolve from flags: %v", err)
	}
	if hw.Arch != "sm_120" || hw.SMs != 84 || hw.Check != nil {
		t.Errorf("got arch=%q sms=%d check=%v, want sm_120, 84, no check", hw.Arch, hw.SMs, hw.Check)
	}
}

func TestBenchLaneResolveHardwareFromProps(t *testing.T) {
	hw, err := benchLaneResolveHardware("", 0, benchLaneProps)
	if err != nil {
		t.Fatalf("resolve from props: %v", err)
	}
	if hw.Arch != "sm_120" || hw.SMs != 84 {
		t.Errorf("got arch=%q sms=%d, want sm_120 and 84 from the file", hw.Arch, hw.SMs)
	}
	if hw.Check == nil || hw.Check.Verdict != compute.DevicePropsConfirmed {
		t.Fatalf("check = %+v, want a confirmed verdict", hw.Check)
	}
	if hw.Check.Capability != compute.DevicePropsMatch {
		t.Errorf("capability = %q, want MATCH", hw.Check.Capability)
	}
}

func TestBenchLaneResolveHardwareAgreeingFlags(t *testing.T) {
	hw, err := benchLaneResolveHardware("sm_120", 84, benchLaneProps)
	if err != nil {
		t.Fatalf("agreeing flags refused: %v", err)
	}
	if hw.Arch != "sm_120" || hw.SMs != 84 {
		t.Errorf("got arch=%q sms=%d, want sm_120 and 84", hw.Arch, hw.SMs)
	}
}

func TestBenchLaneResolveHardwareRefusals(t *testing.T) {
	cases := []struct {
		name     string
		arch     string
		sms      int
		props    string
		sentinel error
		wantText string
	}{
		{name: "missing arch without props", arch: "", sms: 84, sentinel: benchLaneErrNoHardware, wantText: "--arch is missing"},
		{name: "zero sms without props", arch: "sm_120", sms: 0, sentinel: benchLaneErrNoHardware, wantText: "--sms must be positive"},
		{name: "unknown arch without props", arch: "sm_999", sms: 84, wantText: `unknown --arch "sm_999"`},
		{name: "arch disagrees with file", arch: "sm_89", props: benchLaneProps, sentinel: benchLaneErrArchDisagree, wantText: "file reports sm_120"},
		{name: "sms disagrees with file", arch: "", sms: 80, props: benchLaneProps, sentinel: benchLaneErrSMsDisagree, wantText: "file reports 84"},
		{name: "refuted row", props: "testdata/bench_lane_ledger/rtx5080_props_refuted.json", sentinel: benchLaneErrRefuted, wantText: "arch sm_120"},
		{name: "missing props file", props: "testdata/bench_lane_ledger/no_such_file.json", wantText: "--device-props"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := benchLaneResolveHardware(tc.arch, tc.sms, tc.props)
			if err == nil {
				t.Fatalf("resolved, want refusal containing %q", tc.wantText)
			}
			if tc.sentinel != nil && !errors.Is(err, tc.sentinel) {
				t.Errorf("err = %v, want errors.Is %v", err, tc.sentinel)
			}
			if !strings.Contains(err.Error(), tc.wantText) {
				t.Errorf("err = %q, want it to contain %q", err.Error(), tc.wantText)
			}
		})
	}
}

// The refusal sentinels must stay distinct so a caller can branch on the cause.
func TestBenchLaneHardwareSentinelsDistinct(t *testing.T) {
	all := []error{benchLaneErrNoHardware, benchLaneErrArchDisagree, benchLaneErrSMsDisagree, benchLaneErrRefuted}
	for i, a := range all {
		for j, b := range all {
			if i != j && errors.Is(a, b) {
				t.Errorf("sentinel %d matches sentinel %d", i, j)
			}
		}
	}
}

func TestBenchLaneRefutedCheckIsPrinted(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBenchLaneLedger(&stdout, &stderr, []string{
		"--device-props", "testdata/bench_lane_ledger/rtx5080_props_refuted.json",
		"--threads", "256", "--regs", "137", "--grid", "84",
	})
	if code == 0 {
		t.Fatalf("exit 0 for a refuted device, want a refusal")
	}
	for _, want := range []string{"verdict=refuted", "MISMATCH", "MaxThreadsPerSM"} {
		if !strings.Contains(stderr.String(), want) {
			t.Errorf("stderr missing %q:\n%s", want, stderr.String())
		}
	}
	if stdout.Len() != 0 {
		t.Errorf("refusal wrote to stdout: %q", stdout.String())
	}
}

func TestBenchLaneKeyName(t *testing.T) {
	cases := map[string]string{
		"ink2pa::k_ink2p_attend_v2": "k_ink2p_attend_v2",
		"k_copyrow":                 "k_copyrow",
		"(anonymous namespace)::k":  "k",
		"a::b::k_deep":              "k_deep",
	}
	for in, want := range cases {
		if got := benchLaneKeyName(in); got != want {
			t.Errorf("benchLaneKeyName(%q) = %q, want %q", in, got, want)
		}
	}
	if got := benchLaneSiteBase("k_flash<false>"); got != "k_flash" {
		t.Errorf("benchLaneSiteBase = %q, want k_flash", got)
	}
}

// benchLaneTwoSourceFixture is a small ptxas report and inventory where each key is decided by a different rule.
func benchLaneTwoSourceFixture() (compute.PtxasReport, []compute.LaunchSite) {
	report := compute.PtxasReport{Entries: []compute.PtxasEntry{
		// Two namespaces disagree on registers: the key k must not be keyed.
		{BaseName: "a::k", HasRegisters: true, Registers: 40, SmemBytes: 512},
		{BaseName: "b::k", HasRegisters: true, Registers: 56, SmemBytes: 512},
		// One source each: d is fully keyed, except shared memory which the launch makes dynamic.
		{BaseName: "d", HasRegisters: true, Registers: 30, SmemBytes: 0},
		// No measured registers and a symbolic thread count: e is keyed for shared memory only.
		{BaseName: "e", HasRegisters: false, SmemBytes: 0},
	}}
	sites := []compute.LaunchSite{
		{Kernel: "k<Ti>", ThreadsPerBlock: 256, SmemExpr: "0"},
		{Kernel: "d", ThreadsPerBlock: 128, SmemExpr: "shmem"},
		{Kernel: "e", ThreadsPerBlock: 0, SmemExpr: "0"},
	}
	return report, sites
}

func TestBenchLaneObservedInputsKeysOnlyAgreeingSources(t *testing.T) {
	arch, ok := laneLedgerArchByLabel("sm_120")
	if !ok {
		t.Fatal("sm_120 row not registered")
	}
	report, sites := benchLaneTwoSourceFixture()
	in, conflicts := benchLaneObservedInputs(report, sites, arch, 84)

	if _, ok := in.Regs["k"]; ok {
		t.Errorf("regs for k keyed though its namespaces disagree: %d", in.Regs["k"])
	}
	if got := in.Threads["k"]; got != 256 {
		t.Errorf("threads[k] = %d, want 256 from the inventory", got)
	}
	if got := in.SmemBytes["k"]; got != 512 {
		t.Errorf("smem[k] = %d, want 512 (both namespaces agree)", got)
	}
	if got := in.Regs["d"]; got != 30 {
		t.Errorf("regs[d] = %d, want 30", got)
	}
	if _, ok := in.SmemBytes["d"]; ok {
		t.Error("smem[d] keyed though the launch's dynamic shared memory is not literal 0")
	}
	if _, ok := in.Regs["e"]; ok {
		t.Error("regs[e] keyed though no measured register count exists")
	}
	if _, ok := in.Threads["e"]; ok {
		t.Error("threads[e] keyed though the thread count is symbolic")
	}
	if got, ok := in.SmemBytes["e"]; !ok || got != 0 {
		t.Errorf("smem[e] = %d present=%t, want 0 present", got, ok)
	}

	want := map[string]string{
		"k/regs":       "sources disagree",
		"d/smem_bytes": "a source has no measured value",
		"e/regs":       "a source has no measured value",
		"e/threads":    "a source has no measured value",
	}
	got := map[string]string{}
	for _, c := range conflicts {
		got[c.Name+"/"+c.Field] = c.Reason
	}
	for key, reason := range want {
		if got[key] != reason {
			t.Errorf("conflict %s = %q, want %q (all: %v)", key, got[key], reason, got)
		}
	}
	if len(got) != len(want) {
		t.Errorf("got %d conflicts, want exactly %d: %v", len(got), len(want), got)
	}
}

func TestBenchLaneObservedInputsAbsentKernelHasNoInputs(t *testing.T) {
	arch, _ := laneLedgerArchByLabel("sm_120")
	in, conflicts := benchLaneObservedInputs(compute.PtxasReport{}, nil, arch, 84)
	if len(in.Regs)+len(in.Threads)+len(in.SmemBytes) != 0 || len(conflicts) != 0 {
		t.Errorf("empty sources produced inputs %v or conflicts %v", in, conflicts)
	}
}

// benchLaneWriteTemp writes a file into the test's own temp directory and returns its path.
func benchLaneWriteTemp(t *testing.T, name, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	return path
}

func TestBenchLaneObservedRejectsMalformedProfile(t *testing.T) {
	bad := benchLaneWriteTemp(t, "bad.json", `[{"name": "k", "gx": 0, "gy": 1, "calls": 1, "total_ns": 5}]`)
	var stdout, stderr bytes.Buffer
	code := runBenchLaneLedger(&stdout, &stderr, []string{
		"observed", "--profile", bad, "--ptxas-log", "testdata/bench_lane_ledger/ptxas_sm120_excerpt.txt",
		"--device-props", benchLaneProps,
	})
	if code == 0 {
		t.Fatal("malformed profile accepted")
	}
	if !strings.Contains(stderr.String(), "grid") && !strings.Contains(stderr.String(), "row") {
		t.Errorf("stderr = %q, want it to name the bad row", stderr.String())
	}
}
