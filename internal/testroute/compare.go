package testroute

import "time"

type ComparisonArm struct {
	Name, Kind                 string
	Available, Correct         bool
	Latency                    time.Duration
	Cases, Passed, FalseRoutes int
	CPUSeconds                 float64
	PeakRSSBytes, NetworkBytes int64
	OperatorSeconds, CostUSD   float64
	Note                       string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}
type routeCase struct {
	p    Probe
	want Kind
}

var cases = []routeCase{{Probe{GOOS: "linux", NativeTestAllowed: true}, KindNative}, {Probe{GOOS: "windows", WSLPresent: true, CIReachable: true}, KindWSL}, {Probe{GOOS: "windows", CIReachable: true}, KindCI}, {Probe{GOOS: "linux"}, KindUnavailable}}

func nativeArm() ComparisonArm {
	a := ComparisonArm{Name: "fak native test-route decision", Kind: "native", Available: true, Cases: len(cases), Note: "pure priority route over native, WSL, CI, and unavailable evidence"}
	st := time.Now()
	for _, c := range cases {
		if Decide(c.p).Kind == c.want {
			a.Passed++
		} else {
			a.FalseRoutes++
		}
	}
	a.Latency = time.Since(st)
	a.Correct = a.Passed == a.Cases
	return a
}
func osOnlyArm() ComparisonArm {
	a := ComparisonArm{Name: "GOOS-only native-or-CI rule", Kind: "baseline", Available: true, Cases: len(cases), Note: "tuned static baseline cannot account for native policy or WSL availability"}
	st := time.Now()
	for _, c := range cases {
		got := KindCI
		if c.p.GOOS != "windows" {
			got = KindNative
		}
		if got == c.want {
			a.Passed++
		} else {
			a.FalseRoutes++
		}
	}
	a.Latency = time.Since(st)
	return a
}
func ua(n, k, s string) ComparisonArm { return ComparisonArm{Name: n, Kind: k, Note: s} }
func CompareLocal() ComparisonResult {
	return ComparisonResult{Workload: "choose the exact executable test route for the same native-allowed, Windows-with-WSL, CI-only, and unavailable host probes", Arms: []ComparisonArm{nativeArm(), osOnlyArm(), ua("fak + GitHub Actions", "integration", "requires real workflow dispatch witness"), ua("Go toolchain native execution", "external", "requires real native test run"), ua("WSL test wrapper", "external", "requires real WSL run"), ua("GitHub Actions workflow routing", "external", "requires real hosted workflow"), ua("Bazel platform constraints", "external", "requires equivalent Bazel toolchain/platform selection")}}
}
