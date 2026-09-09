package hil

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestProbeHardware(t *testing.T) {
	hw := ProbeHardware()
	if hw.Architecture == "" {
		t.Errorf("expected architecture to be non-empty, got %q", hw.Architecture)
	}
	if hw.Platform == "" {
		t.Errorf("expected platform to be non-empty, got %q", hw.Platform)
	}
	if hw.Kind == "" || hw.Kind == HardwareUnknown {
		t.Errorf("expected known hardware kind, got %q", hw.Kind)
	}
}

func TestRunMicroDosesExecution(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	report := RunMicroDoses(ctx)
	if report.Schema != ReportSchema {
		t.Errorf("expected schema %q, got %q", ReportSchema, report.Schema)
	}
	if len(report.MicroDoses) == 0 {
		t.Fatalf("expected at least 1 micro-dose executed, got 0")
	}

	// Invariant: each micro-dose must be sub-second (micro-dose definition < 1,000,000 micros)
	for _, md := range report.MicroDoses {
		if md.DurationMicros > 1_000_000 {
			t.Errorf("micro-dose %q took %d micros (> 1s), exceeding micro-dose budget", md.Name, md.DurationMicros)
		}
		if !md.Passed {
			t.Errorf("micro-dose %q failed: %s", md.Name, md.Detail)
		}
		if !md.OutputVerified {
			t.Errorf("micro-dose %q output not verified", md.Name)
		}
	}

	if !report.AllPassed {
		t.Errorf("expected AllPassed to be true, report guidance: %s", report.Guidance)
	}
}

func TestAuditComparisonRealHardware(t *testing.T) {
	cand := ComparisonArm{
		Name:              "fak-native-metal",
		IsPhysicalSilicon: true,
		HardwareTarget:    "Apple M3 Pro",
		EvidenceType:      "hardware_measurement",
		Metric:            "tok/s",
		Value:             15.22,
		Unit:              "tok/s",
		SampleCount:       20,
	}
	base := ComparisonArm{
		Name:              "llama.cpp-metal",
		IsPhysicalSilicon: true,
		HardwareTarget:    "Apple M3 Pro",
		EvidenceType:      "hardware_measurement",
		Metric:            "tok/s",
		Value:             12.14,
		Unit:              "tok/s",
		SampleCount:       20,
	}

	audit := AuditComparison("fak-native vs llama.cpp MTP Decode", cand, base, false)
	if audit.Verdict != VerdictRealHardwareVerified {
		t.Errorf("expected verdict %s, got %s", VerdictRealHardwareVerified, audit.Verdict)
	}
	if !audit.AllowedAsAchievedWin {
		t.Errorf("expected AllowedAsAchievedWin to be true")
	}
	if audit.IsEarlyIndicator {
		t.Errorf("expected IsEarlyIndicator to be false")
	}
	if audit.Speedup < 1.25 || audit.Speedup > 1.26 {
		t.Errorf("expected speedup ~1.253, got %.4f", audit.Speedup)
	}
}

func TestAuditComparisonRejectsSimulation(t *testing.T) {
	cand := ComparisonArm{
		Name:              "fak-simulated-moe",
		IsPhysicalSilicon: false,
		HardwareTarget:    "Analytical Roofline (unmeasured)",
		EvidenceType:      "analytical_bound",
		Metric:            "tok/s",
		Value:             318.8,
		Unit:              "tok/s",
		SampleCount:       0,
	}
	base := ComparisonArm{
		Name:              "llama.cpp-baseline",
		IsPhysicalSilicon: true,
		HardwareTarget:    "Apple M3 Pro",
		EvidenceType:      "hardware_measurement",
		Metric:            "tok/s",
		Value:             52.4,
		Unit:              "tok/s",
		SampleCount:       10,
	}

	audit := AuditComparison("fak-simulated vs llama.cpp", cand, base, false)
	if audit.Verdict != VerdictEarlyIndicatorOnly {
		t.Errorf("expected verdict %s, got %s", VerdictEarlyIndicatorOnly, audit.Verdict)
	}
	if audit.AllowedAsAchievedWin {
		t.Errorf("expected AllowedAsAchievedWin to be false for simulated comparison")
	}
	if !audit.IsEarlyIndicator {
		t.Errorf("expected IsEarlyIndicator to be true")
	}
}

func TestAuditComparisonRejectsInvalidMetrics(t *testing.T) {
	cand := ComparisonArm{
		Name:  "cand",
		Value: 0,
	}
	base := ComparisonArm{
		Name:  "base",
		Value: 10,
	}

	audit := AuditComparison("invalid", cand, base, false)
	if audit.Verdict != VerdictInvalidComparison {
		t.Errorf("expected verdict %s, got %s", VerdictInvalidComparison, audit.Verdict)
	}
}

func TestProbeLANNodeHTTPHealthz(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	info := ProbeLANNode(ctx, ts.Listener.Addr().String())
	if info.Status != LANNodeOnline {
		t.Fatalf("expected status %s, got %s (error: %s)", LANNodeOnline, info.Status, info.Error)
	}
	if !info.Reachable {
		t.Errorf("expected reachable true")
	}
	if info.Transport != "http_healthz" {
		t.Errorf("expected transport http_healthz, got %q", info.Transport)
	}
	if info.Appliance != "AMD Ryzen AI MAX+ 395" {
		t.Errorf("expected appliance AMD Ryzen AI MAX+ 395, got %q", info.Appliance)
	}
	if info.GPU != "AMD Radeon 8060S (40 CUs, gfx1151, 64 GB UMA)" {
		t.Errorf("expected GPU AMD Radeon 8060S (40 CUs, gfx1151, 64 GB UMA), got %q", info.GPU)
	}
	if info.LatencyMicros <= 0 {
		t.Errorf("expected latency > 0, got %d", info.LatencyMicros)
	}
	if info.Host != "<LAN_IP>" {
		t.Errorf("expected host to be scrubbed to <LAN_IP>, got %q", info.Host)
	}
	if strings.Contains(info.Endpoint, "127.0.0.1") {
		t.Errorf("expected endpoint to be sanitized, got %q", info.Endpoint)
	}
}

func TestProbeLANNodeAutoDiscoversCanonicalStrix(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	t.Setenv("FAK_STRIX_HOST", "")
	t.Setenv("FAK_LAN_HOST", "")
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	info := probeLANNode(ctx, "", []string{ts.Listener.Addr().String()})
	if info.Status != LANNodeOnline || !info.Reachable {
		t.Fatalf("empty-config canonical discovery = %+v, want reachable ONLINE", info)
	}
	if info.Transport != "http_healthz" {
		t.Fatalf("transport = %q, want http_healthz", info.Transport)
	}
	if info.Host != "<LAN_IP>" || strings.Contains(info.Endpoint, "127.0.0.1") {
		t.Fatalf("auto-discovered telemetry was not scrubbed: host=%q endpoint=%q", info.Host, info.Endpoint)
	}
}

func TestProbeLANNodeSSHReachable(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen error: %v", err)
	}
	defer l.Close()

	go func() {
		for {
			conn, aErr := l.Accept()
			if aErr != nil {
				return
			}
			_ = conn.Close()
		}
	}()

	_, sshPort, err := net.SplitHostPort(l.Addr().String())
	if err != nil {
		t.Fatalf("split host port: %v", err)
	}

	t.Setenv("FAK_LAN_SSH_PORT", sshPort)
	t.Setenv("FAK_LAN_GATEWAY_PORT", "59997") // gateway port unreachable

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	info := ProbeLANNode(ctx, "127.0.0.1")
	if info.Status != LANNodeOnline {
		t.Fatalf("expected status %s, got %s (error: %s)", LANNodeOnline, info.Status, info.Error)
	}
	if !info.Reachable {
		t.Errorf("expected reachable true")
	}
	if info.Transport != "tcp_ssh" {
		t.Errorf("expected transport tcp_ssh, got %q", info.Transport)
	}
	if info.Appliance != "strix-halo (SSH port 22 reachable; gateway offline)" {
		t.Errorf("expected SSH appliance message, got %q", info.Appliance)
	}
	if info.Host != "<LAN_IP>" {
		t.Errorf("expected host scrubbed, got %q", info.Host)
	}
}

func TestProbeLANNodeUnreachable(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Direct probe to test network IP that is non-routable / packet-dropping
	start := time.Now()
	info := ProbeLANNode(ctx, "192.0.2.1")
	elapsed := time.Since(start)

	if info.Status != LANNodeOffline {
		t.Errorf("expected status %s, got %s", LANNodeOffline, info.Status)
	}
	if info.Reachable {
		t.Errorf("expected reachable false")
	}
	if info.Transport != "offline" {
		t.Errorf("expected transport offline, got %q", info.Transport)
	}
	if info.Error == "" {
		t.Errorf("expected non-empty error on unreachable")
	}
	if strings.Contains(info.Error, "192.0.2.1") {
		t.Errorf("expected raw IP to be scrubbed from error, got %q", info.Error)
	}
	// Fail closed within timeout (<60ms)
	if elapsed > 100*time.Millisecond {
		t.Errorf("expected probe to fail closed quickly (<60ms target), took %v", elapsed)
	}
}

func TestProbeLANNodeUnconfigured(t *testing.T) {
	t.Setenv("FAK_STRIX_HOST", "")
	t.Setenv("FAK_LAN_HOST", "")

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	info := probeLANNode(ctx, "", nil)
	if info.Status != LANNodeUnconfigured {
		t.Errorf("expected status %s, got %s", LANNodeUnconfigured, info.Status)
	}
	if info.Reachable {
		t.Errorf("expected reachable false")
	}
	if info.Transport != "offline" {
		t.Errorf("expected transport offline, got %q", info.Transport)
	}
	if info.Host != "" {
		t.Errorf("expected empty host for unconfigured, got %q", info.Host)
	}
}

func TestProbeInventory(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	rep := ProbeInventory(ctx, ts.Listener.Addr().String())
	if rep.Schema != InventorySchema {
		t.Errorf("expected schema %q, got %q", InventorySchema, rep.Schema)
	}
	if !rep.Sanitized {
		t.Errorf("expected sanitized true")
	}
	if rep.Local.Architecture == "" {
		t.Errorf("expected local architecture to be non-empty")
	}
	if rep.Local.LANNode == nil {
		t.Errorf("expected local.LANNode to be populated")
	}
	if rep.LAN.Status != LANNodeOnline {
		t.Errorf("expected LAN status %s, got %s", LANNodeOnline, rep.LAN.Status)
	}
	if rep.NextAction == "" {
		t.Errorf("expected non-empty next action")
	}

	// Unconfigured inventory check
	t.Setenv("FAK_STRIX_HOST", "")
	t.Setenv("FAK_LAN_HOST", "")
	repUnconf := ProbeInventory(ctx, "")
	if repUnconf.LAN.Status != LANNodeUnconfigured {
		t.Errorf("expected unconfigured LAN status, got %s", repUnconf.LAN.Status)
	}
	if !strings.Contains(repUnconf.NextAction, "unconfigured") {
		t.Errorf("expected next action to mention unconfigured, got %q", repUnconf.NextAction)
	}
}

func TestScrubLANTelemetry(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{
			input:    "http://192.168.1.50:8080/healthz",
			expected: "http://<LAN_IP>:8080/healthz",
		},
		{
			input:    "dial tcp 10.0.0.1:22: connection refused",
			expected: "dial tcp <LAN_IP>:22: connection refused",
		},
		{
			input:    "target: 172.16.5.10",
			expected: "target: <LAN_IP>",
		},
		{
			input:    "127.0.0.1:8080",
			expected: "<LAN_IP>:8080",
		},
		{
			input:    "strix1",
			expected: "strix1",
		},
		{
			input:    "strix-halo-fak.local",
			expected: "strix-halo-fak.local",
		},
		{
			input:    "AMD Radeon 8060S (40 CUs, gfx1151, 64 GB UMA)",
			expected: "AMD Radeon 8060S (40 CUs, gfx1151, 64 GB UMA)",
		},
		{
			input:    "IPv6 loopback ::1:8080",
			expected: "IPv6 loopback <LAN_IP>:8080",
		},
	}

	for i, tc := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			got := ScrubLANTelemetry(tc.input)
			if got != tc.expected {
				t.Errorf("ScrubLANTelemetry(%q) = %q, expected %q", tc.input, got, tc.expected)
			}
		})
	}
}

func TestConcurrentInvocations(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	const concurrency = 16
	var wg sync.WaitGroup
	wg.Add(concurrency)

	for i := 0; i < concurrency; i++ {
		go func(idx int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			// 1. ProbeHardware
			hw := ProbeHardware()
			if hw.Platform == "" {
				t.Errorf("worker %d: empty platform", idx)
			}

			// 2. RunMicroDoses
			rep := RunMicroDoses(ctx, DoseLiveness, DoseNumericParity)
			if rep.Schema != ReportSchema {
				t.Errorf("worker %d: bad report schema %q", idx, rep.Schema)
			}

			// 3. AuditComparison
			cand := ComparisonArm{
				Name:              "cand",
				IsPhysicalSilicon: true,
				EvidenceType:      "physical_silicon",
				Metric:            "tok/s",
				Value:             20,
				Unit:              "tok/s",
				SampleCount:       10,
			}
			base := ComparisonArm{
				Name:              "base",
				IsPhysicalSilicon: true,
				EvidenceType:      "hardware_measurement",
				Metric:            "tok/s",
				Value:             10,
				Unit:              "tok/s",
				SampleCount:       10,
			}
			audit := AuditComparison("conc", cand, base, false)
			if audit.Verdict != VerdictRealHardwareVerified {
				t.Errorf("worker %d: unexpected verdict %s", idx, audit.Verdict)
			}

			// 4. ProbeLANNode
			lan := ProbeLANNode(ctx, ts.Listener.Addr().String())
			if lan.Status != LANNodeOnline {
				t.Errorf("worker %d: unexpected lan status %s", idx, lan.Status)
			}

			// 5. ProbeInventory
			inv := ProbeInventory(ctx, ts.Listener.Addr().String())
			if inv.Schema != InventorySchema {
				t.Errorf("worker %d: bad inv schema %q", idx, inv.Schema)
			}
		}(i)
	}

	wg.Wait()
}

func TestSchemaConformance(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 1. ReportSchema (fak.hil.report.v1)
	rep := RunMicroDoses(ctx, DoseLiveness)
	repBytes, err := json.Marshal(rep)
	if err != nil {
		t.Fatalf("marshal report: %v", err)
	}
	var repMap map[string]any
	if err := json.Unmarshal(repBytes, &repMap); err != nil {
		t.Fatalf("unmarshal report: %v", err)
	}
	if s, ok := repMap["schema"].(string); !ok || s != "fak.hil.report.v1" {
		t.Errorf("expected schema 'fak.hil.report.v1', got %v", repMap["schema"])
	}
	for _, field := range []string{"schema", "timestamp", "hardware", "bias_toward_hardware", "micro_doses", "total_duration_micros", "all_passed", "guidance"} {
		if _, ok := repMap[field]; !ok {
			t.Errorf("report missing field %q", field)
		}
	}

	// 2. MicroDoseSchema (fak.hil.microdose.v1)
	if len(rep.MicroDoses) == 0 {
		t.Fatalf("expected microdoses in report")
	}
	mdBytes, err := json.Marshal(rep.MicroDoses[0])
	if err != nil {
		t.Fatalf("marshal microdose: %v", err)
	}
	var mdMap map[string]any
	if err := json.Unmarshal(mdBytes, &mdMap); err != nil {
		t.Fatalf("unmarshal microdose: %v", err)
	}
	if s, ok := mdMap["schema"].(string); !ok || s != "fak.hil.microdose.v1" {
		t.Errorf("expected schema 'fak.hil.microdose.v1', got %v", mdMap["schema"])
	}
	for _, field := range []string{"schema", "name", "kind", "hardware", "duration_micros", "duration_formatted", "operations", "output_verified", "passed", "detail"} {
		if _, ok := mdMap[field]; !ok {
			t.Errorf("microdose missing field %q", field)
		}
	}

	// 3. ComparisonSchema (fak.hil.comparison.v1)
	cand := ComparisonArm{
		Name:              "cand",
		IsPhysicalSilicon: true,
		EvidenceType:      "physical_silicon",
		Metric:            "tok/s",
		Value:             100,
		Unit:              "tok/s",
		SampleCount:       5,
	}
	base := ComparisonArm{
		Name:              "base",
		IsPhysicalSilicon: true,
		EvidenceType:      "hardware_measurement",
		Metric:            "tok/s",
		Value:             50,
		Unit:              "tok/s",
		SampleCount:       5,
	}
	audit := AuditComparison("test", cand, base, false)
	auditBytes, err := json.Marshal(audit)
	if err != nil {
		t.Fatalf("marshal audit: %v", err)
	}
	var auditMap map[string]any
	if err := json.Unmarshal(auditBytes, &auditMap); err != nil {
		t.Fatalf("unmarshal audit: %v", err)
	}
	if s, ok := auditMap["schema"].(string); !ok || s != "fak.hil.comparison.v1" {
		t.Errorf("expected schema 'fak.hil.comparison.v1', got %v", auditMap["schema"])
	}
	for _, field := range []string{"schema", "timestamp", "headline", "candidate", "baseline", "speedup", "verdict", "allowed_as_achieved_win", "is_early_indicator", "reason", "enforcement_action"} {
		if _, ok := auditMap[field]; !ok {
			t.Errorf("audit missing field %q", field)
		}
	}

	// 4. InventorySchema (fak.hil.inventory.v1)
	inv := ProbeInventory(ctx, "")
	invBytes, err := json.Marshal(inv)
	if err != nil {
		t.Fatalf("marshal inventory: %v", err)
	}
	var invMap map[string]any
	if err := json.Unmarshal(invBytes, &invMap); err != nil {
		t.Fatalf("unmarshal inventory: %v", err)
	}
	if s, ok := invMap["schema"].(string); !ok || s != "fak.hil.inventory.v1" {
		t.Errorf("expected schema 'fak.hil.inventory.v1', got %v", invMap["schema"])
	}
	for _, field := range []string{"schema", "timestamp", "local", "lan", "sanitized", "next_action"} {
		if _, ok := invMap[field]; !ok {
			t.Errorf("inventory missing field %q", field)
		}
	}
}
