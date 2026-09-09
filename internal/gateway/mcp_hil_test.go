package gateway

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hil"
)

func TestMCPHIL(t *testing.T) {
	srv := &Server{disableMCPDefer: true}

	t.Run("tools/list contains HIL tools", func(t *testing.T) {
		res, rerr := srv.handleMethod(context.Background(), "tools/list", nil)
		if rerr != nil {
			t.Fatalf("tools/list failed: %v", rerr)
		}
		resMap, ok := res.(map[string]any)
		if !ok {
			t.Fatalf("tools/list response not map: %T", res)
		}
		toolsRaw, ok := resMap["tools"].([]map[string]any)
		if !ok {
			t.Fatalf("tools/list missing tools array: %T", resMap["tools"])
		}

		expected := map[string]bool{
			"fak_hil_probe":            false,
			"fak_hil_microdose":        false,
			"fak_hil_audit_comparison": false,
		}

		for _, tool := range toolsRaw {
			name, _ := tool["name"].(string)
			if _, ok := expected[name]; ok {
				expected[name] = true
				desc, _ := tool["description"].(string)
				if desc == "" {
					t.Errorf("tool %q missing description", name)
				}
				schema, ok := tool["inputSchema"].(json.RawMessage)
				if !ok || len(schema) == 0 {
					t.Errorf("tool %q missing inputSchema", name)
				}
				ann, ok := tool["annotations"].(map[string]any)
				if !ok || ann["readOnly"] != true {
					t.Errorf("tool %q should have readOnly annotation", name)
				}
			}
		}

		for name, found := range expected {
			if !found {
				t.Errorf("expected tool %q not found in tools/list", name)
			}
		}
	})

	t.Run("CoreToolsPalette contains HIL capabilities", func(t *testing.T) {
		for _, name := range []string{"fak_hil_probe", "fak_hil_microdose", "fak_hil_audit_comparison"} {
			tool, ok := LookupCoreTool(name)
			if !ok {
				t.Errorf("expected tool %q in CoreToolsPalette", name)
				continue
			}
			if tool.DestructiveHint {
				t.Errorf("tool %q should have DestructiveHint: false", name)
			}
			if tool.OpenWorldHint {
				t.Errorf("tool %q should have OpenWorldHint: false", name)
			}
		}
	})

	t.Run("tools/call fak_hil_probe", func(t *testing.T) {
		params, _ := json.Marshal(map[string]any{
			"name":      "fak_hil_probe",
			"arguments": map[string]any{},
		})
		res, rerr := srv.handleMethod(context.Background(), "tools/call", params)
		if rerr != nil {
			t.Fatalf("tools/call fak_hil_probe error: %v", rerr)
		}
		var hw hil.HardwareInfo
		decodeMCPText(t, res, &hw)
		if hw.Platform == "" {
			t.Error("expected non-empty Platform in HardwareInfo")
		}
		if hw.Architecture == "" {
			t.Error("expected non-empty Architecture in HardwareInfo")
		}
	})

	t.Run("tools/call fak_hil_microdose", func(t *testing.T) {
		report := callMCPTool[hil.Report](t, srv, "fak_hil_microdose", map[string]any{
			"kinds": []string{"liveness"},
		})
		if report.Schema != hil.ReportSchema {
			t.Errorf("expected schema %q, got %q", hil.ReportSchema, report.Schema)
		}
		if len(report.MicroDoses) != 1 {
			t.Fatalf("expected 1 microdose result, got %d", len(report.MicroDoses))
		}
		if report.MicroDoses[0].Kind != hil.DoseLiveness {
			t.Errorf("expected dose kind %q, got %q", hil.DoseLiveness, report.MicroDoses[0].Kind)
		}
		if !report.MicroDoses[0].Passed {
			t.Errorf("expected liveness microdose to pass")
		}

		// Empty/default arguments
		defaultReport := callMCPTool[hil.Report](t, srv, "fak_hil_microdose", map[string]any{})
		if defaultReport.Schema != hil.ReportSchema {
			t.Errorf("expected schema %q, got %q", hil.ReportSchema, defaultReport.Schema)
		}
		if len(defaultReport.MicroDoses) == 0 {
			t.Fatalf("expected default microdoses, got 0")
		}
	})

	t.Run("tools/call fak_hil_audit_comparison verified physical silicon", func(t *testing.T) {
		audit := callMCPTool[hil.ComparisonAudit](t, srv, "fak_hil_audit_comparison", map[string]any{
			"headline": "Metal vs CPU",
			"candidate": hil.ComparisonArm{
				Name:              "fak-metal",
				IsPhysicalSilicon: true,
				HardwareTarget:    "Apple Silicon",
				EvidenceType:      "physical_silicon",
				Metric:            "tok/s",
				Value:             50.0,
				Unit:              "tok/s",
				SampleCount:       10,
			},
			"baseline": hil.ComparisonArm{
				Name:              "fak-cpu",
				IsPhysicalSilicon: true,
				HardwareTarget:    "Apple Silicon CPU",
				EvidenceType:      "hardware_measurement",
				Metric:            "tok/s",
				Value:             25.0,
				Unit:              "tok/s",
				SampleCount:       10,
			},
			"lower_is_better": false,
		})

		if audit.Verdict != hil.VerdictRealHardwareVerified {
			t.Fatalf("expected verdict %s, got %s", hil.VerdictRealHardwareVerified, audit.Verdict)
		}
		if !audit.AllowedAsAchievedWin {
			t.Error("expected AllowedAsAchievedWin == true")
		}
		if audit.IsEarlyIndicator {
			t.Error("expected IsEarlyIndicator == false")
		}
		if audit.Speedup != 2.0 {
			t.Errorf("expected speedup 2.0, got %f", audit.Speedup)
		}
	})

	t.Run("tools/call fak_hil_audit_comparison simulated arm gates as early indicator", func(t *testing.T) {
		audit := callMCPTool[hil.ComparisonAudit](t, srv, "fak_hil_audit_comparison", map[string]any{
			"headline": "Simulated vs Physical",
			"candidate": hil.ComparisonArm{
				Name:              "sim-candidate",
				IsPhysicalSilicon: false,
				EvidenceType:      "analytical_bound",
				Metric:            "tok/s",
				Value:             100.0,
				Unit:              "tok/s",
				SampleCount:       1,
			},
			"baseline": hil.ComparisonArm{
				Name:              "real-base",
				IsPhysicalSilicon: true,
				HardwareTarget:    "Apple Silicon",
				EvidenceType:      "hardware_measurement",
				Metric:            "tok/s",
				Value:             25.0,
				Unit:              "tok/s",
				SampleCount:       10,
			},
			"lower_is_better": false,
		})

		if audit.Verdict != hil.VerdictEarlyIndicatorOnly {
			t.Fatalf("expected verdict %s, got %s", hil.VerdictEarlyIndicatorOnly, audit.Verdict)
		}
		if audit.AllowedAsAchievedWin {
			t.Error("expected AllowedAsAchievedWin == false")
		}
		if !audit.IsEarlyIndicator {
			t.Error("expected IsEarlyIndicator == true")
		}
	})
}
