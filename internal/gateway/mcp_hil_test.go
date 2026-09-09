package gateway

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"

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

func TestMCPHardwareInventoryResource(t *testing.T) {
	srv := &Server{disableMCPDefer: true}

	t.Run("resources/list contains fak://hardware/inventory", func(t *testing.T) {
		res, rerr := srv.handleMethod(context.Background(), "resources/list", nil)
		if rerr != nil {
			t.Fatalf("resources/list failed: %v", rerr)
		}
		resMap, ok := res.(map[string]any)
		if !ok {
			t.Fatalf("resources/list response not map: %T", res)
		}
		resources, ok := resMap["resources"].([]map[string]any)
		if !ok {
			t.Fatalf("resources/list missing resources array: %T", resMap["resources"])
		}
		var found bool
		for _, r := range resources {
			if r["uri"] == "fak://hardware/inventory" {
				found = true
				if r["name"] != "fak hardware inventory" {
					t.Errorf("expected name %q, got %q", "fak hardware inventory", r["name"])
				}
				if r["mimeType"] != "application/json" {
					t.Errorf("expected mimeType %q, got %q", "application/json", r["mimeType"])
				}
				desc, _ := r["description"].(string)
				if desc == "" {
					t.Error("expected non-empty description for fak://hardware/inventory")
				}
			}
		}
		if !found {
			t.Fatal("fak://hardware/inventory not found in resources/list")
		}
	})

	t.Run("resources/read returns valid InventoryReport with application/json", func(t *testing.T) {
		params, _ := json.Marshal(map[string]any{"uri": "fak://hardware/inventory"})
		res, rerr := srv.handleMethod(context.Background(), "resources/read", params)
		if rerr != nil {
			t.Fatalf("resources/read failed: %v", rerr)
		}
		resMap, ok := res.(map[string]any)
		if !ok {
			t.Fatalf("resources/read response not map: %T", res)
		}
		contents, ok := resMap["contents"].([]map[string]any)
		if !ok || len(contents) != 1 {
			t.Fatalf("expected 1 content entry, got: %v", resMap["contents"])
		}
		item := contents[0]
		if item["uri"] != "fak://hardware/inventory" {
			t.Errorf("expected uri fak://hardware/inventory, got %q", item["uri"])
		}
		if item["mimeType"] != "application/json" {
			t.Errorf("expected mimeType application/json, got %q", item["mimeType"])
		}
		text, _ := item["text"].(string)
		if text == "" {
			t.Fatal("expected non-empty text in contents")
		}
		var rep hil.InventoryReport
		if err := json.Unmarshal([]byte(text), &rep); err != nil {
			t.Fatalf("unmarshal InventoryReport: %v\ntext: %s", err, text)
		}
		if rep.Schema != hil.InventorySchema {
			t.Errorf("expected schema %q, got %q", hil.InventorySchema, rep.Schema)
		}
		if !rep.Sanitized {
			t.Error("expected report.Sanitized == true")
		}
	})

	t.Run("privacy boundary scrubbing replaces internal IPs and sensitive hostnames", func(t *testing.T) {
		leakyReport := hil.InventoryReport{
			Schema:    hil.InventorySchema,
			Timestamp: time.Now().UTC(),
			Local: hil.HardwareInfo{
				Kind:              hil.HardwareMetal,
				DeviceName:        "anthony-mbp.local",
				Architecture:      "arm64",
				Platform:          "darwin",
				PhysicalAvailable: true,
				Details: map[string]string{
					"internal_host": "node-box.example.internal",
					"ip":            "192.168.1.100",
				},
			},
			LAN: hil.LANNodeInfo{
				Status:        hil.LANNodeOnline,
				Host:          "192.168.1.50",
				Reachable:     true,
				LatencyMicros: 42000,
				Transport:     "http_healthz",
				Endpoint:      "http://192.168.1.50:8080",
				Appliance:     "AMD Ryzen AI MAX+ 395",
				GPU:           "AMD Radeon 8060S (40 CUs, gfx1151, 64 GB UMA)",
				Error:         "dial tcp 10.0.0.1:22: connection refused",
			},
			Sanitized:  false,
			NextAction: "dispatch offload to 192.168.1.50 or strix-halo-fak.local",
		}
		srv.SetHardwareInventory(leakyReport)

		params, _ := json.Marshal(map[string]any{"uri": "fak://hardware/inventory"})
		res, rerr := srv.handleMethod(context.Background(), "resources/read", params)
		if rerr != nil {
			t.Fatalf("resources/read failed: %v", rerr)
		}
		contents := res.(map[string]any)["contents"].([]map[string]any)
		text := contents[0]["text"].(string)

		var scrubbed hil.InventoryReport
		if err := json.Unmarshal([]byte(text), &scrubbed); err != nil {
			t.Fatalf("unmarshal scrubbed report: %v", err)
		}

		if scrubbed.Local.DeviceName != "local-silicon" {
			t.Errorf("expected Local.DeviceName to be local-silicon, got %q", scrubbed.Local.DeviceName)
		}
		if scrubbed.LAN.Host != "strix1" {
			t.Errorf("expected LAN.Host to be strix1, got %q", scrubbed.LAN.Host)
		}
		if scrubbed.LAN.Endpoint != "http://strix1:8080" {
			t.Errorf("expected LAN.Endpoint to be http://strix1:8080, got %q", scrubbed.LAN.Endpoint)
		}
		if strings.Contains(scrubbed.LAN.Error, "10.0.0.1") {
			t.Errorf("LAN.Error leaked internal IP 10.0.0.1: %q", scrubbed.LAN.Error)
		}
		if !strings.Contains(scrubbed.LAN.Error, "strix1") {
			t.Errorf("LAN.Error missing strix1 alias: %q", scrubbed.LAN.Error)
		}
		if strings.Contains(scrubbed.NextAction, "192.168.1.50") || strings.Contains(scrubbed.NextAction, "strix-halo-fak.local") {
			t.Errorf("NextAction leaked sensitive host or IP: %q", scrubbed.NextAction)
		}
		if !strings.Contains(scrubbed.NextAction, "strix1") {
			t.Errorf("NextAction missing strix1 alias: %q", scrubbed.NextAction)
		}
		if !scrubbed.Sanitized {
			t.Error("expected scrubbed.Sanitized == true")
		}
	})

	t.Run("resources/subscribe and resources/unsubscribe with update notifications", func(t *testing.T) {
		subSrv := &Server{disableMCPDefer: true}

		baseReport := hil.InventoryReport{
			Schema:    hil.InventorySchema,
			Timestamp: time.Now().UTC(),
			LAN: hil.LANNodeInfo{
				Status:    hil.LANNodeOnline,
				Host:      "strix1",
				Reachable: true,
			},
			Sanitized: true,
		}
		subSrv.SetHardwareInventory(baseReport)

		// Parameter validations
		_, errSub := subSrv.handleMethod(context.Background(), "resources/subscribe", json.RawMessage(`{}`))
		if errSub == nil || errSub.Code != rpcInvalidParams {
			t.Errorf("expected rpcInvalidParams on empty uri subscribe, got: %v", errSub)
		}
		_, errSubUnknown := subSrv.handleMethod(context.Background(), "resources/subscribe", json.RawMessage(`{"uri":"fak://unknown/resource"}`))
		if errSubUnknown == nil || errSubUnknown.Code != rpcInvalidParams {
			t.Errorf("expected rpcInvalidParams on unknown uri subscribe, got: %v", errSubUnknown)
		}

		ctx := withMCPPeer(context.Background(), "test-client-1")
		subRes, rerr := subSrv.handleMethod(ctx, "resources/subscribe", json.RawMessage(`{"uri":"fak://hardware/inventory"}`))
		if rerr != nil {
			t.Fatalf("resources/subscribe failed: %v", rerr)
		}
		if subRes == nil {
			t.Fatal("resources/subscribe returned nil result")
		}

		notifCh := make(chan map[string]any, 10)
		unreg := subSrv.RegisterMCPNotificationSink("test-client-1", func(method string, params any) {
			if method == "notifications/resources/updated" {
				if pMap, ok := params.(map[string]any); ok {
					notifCh <- pMap
				}
			}
		})
		defer unreg()

		// Same status update -> no notification
		transitioned := subSrv.UpdateHardwareInventory(baseReport)
		if transitioned {
			t.Error("expected transitioned == false for identical report")
		}
		select {
		case n := <-notifCh:
			t.Fatalf("unexpected notification for unchanged inventory: %v", n)
		case <-time.After(50 * time.Millisecond):
		}

		// Hardware transition (LAN goes offline) -> notification emitted
		offlineReport := baseReport
		offlineReport.LAN.Status = hil.LANNodeOffline
		offlineReport.LAN.Reachable = false

		transitioned = subSrv.UpdateHardwareInventory(offlineReport)
		if !transitioned {
			t.Error("expected transitioned == true for offline status transition")
		}

		select {
		case n := <-notifCh:
			if n["uri"] != "fak://hardware/inventory" {
				t.Errorf("expected notification uri fak://hardware/inventory, got %v", n["uri"])
			}
		case <-time.After(1 * time.Second):
			t.Fatal("timed out waiting for notifications/resources/updated")
		}

		// Unsubscribe -> no further notifications
		unsubRes, rerr := subSrv.handleMethod(ctx, "resources/unsubscribe", json.RawMessage(`{"uri":"fak://hardware/inventory"}`))
		if rerr != nil {
			t.Fatalf("resources/unsubscribe failed: %v", rerr)
		}
		if unsubRes == nil {
			t.Fatal("resources/unsubscribe returned nil result")
		}

		onlineReport := offlineReport
		onlineReport.LAN.Status = hil.LANNodeOnline
		onlineReport.LAN.Reachable = true
		subSrv.UpdateHardwareInventory(onlineReport)

		select {
		case n := <-notifCh:
			t.Fatalf("received unexpected notification after unsubscribe: %v", n)
		case <-time.After(50 * time.Millisecond):
		}
	})

	t.Run("ServeStdio end-to-end JSON-RPC subscription and update streaming", func(t *testing.T) {
		stdioSrv := &Server{disableMCPDefer: true}
		inR, inW := io.Pipe()
		outR, outW := io.Pipe()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		serveDone := make(chan error, 1)
		go func() {
			serveDone <- stdioSrv.ServeStdio(ctx, inR, outW)
		}()

		inEnc := json.NewEncoder(inW)
		outDec := json.NewDecoder(outR)

		// 1. Initialize
		_ = inEnc.Encode(map[string]any{
			"jsonrpc": "2.0",
			"id":      1,
			"method":  "initialize",
			"params":  map[string]any{"protocolVersion": "2024-11-05"},
		})
		var initResp rpcResponse
		if err := outDec.Decode(&initResp); err != nil {
			t.Fatalf("decode initialize response: %v", err)
		}
		resMap, ok := initResp.Result.(map[string]any)
		if !ok {
			t.Fatalf("malformed initialize result: %v", initResp.Result)
		}
		caps, _ := resMap["capabilities"].(map[string]any)
		resCaps, _ := caps["resources"].(map[string]any)
		if resCaps["subscribe"] != true {
			t.Errorf("expected capabilities.resources.subscribe == true, got %v", resCaps["subscribe"])
		}

		// 2. Subscribe to fak://hardware/inventory
		_ = inEnc.Encode(map[string]any{
			"jsonrpc": "2.0",
			"id":      2,
			"method":  "resources/subscribe",
			"params":  map[string]any{"uri": "fak://hardware/inventory"},
		})
		var subResp rpcResponse
		if err := outDec.Decode(&subResp); err != nil {
			t.Fatalf("decode subscribe response: %v", err)
		}
		if subResp.Error != nil {
			t.Fatalf("subscribe returned error: %+v", subResp.Error)
		}

		// 3. Trigger hardware transition
		transReport := hil.InventoryReport{
			Schema:    hil.InventorySchema,
			Timestamp: time.Now().UTC(),
			LAN: hil.LANNodeInfo{
				Status:    hil.LANNodeOnline,
				Host:      "strix1",
				Reachable: true,
			},
			Sanitized: true,
		}
		stdioSrv.UpdateHardwareInventory(transReport)

		// 4. Read notification frame from stdout
		var notifFrame struct {
			JSONRPC string         `json:"jsonrpc"`
			Method  string         `json:"method"`
			Params  map[string]any `json:"params"`
		}
		if err := outDec.Decode(&notifFrame); err != nil {
			t.Fatalf("decode notification frame: %v", err)
		}
		if notifFrame.Method != "notifications/resources/updated" {
			t.Errorf("expected method notifications/resources/updated, got %q", notifFrame.Method)
		}
		if notifFrame.Params["uri"] != "fak://hardware/inventory" {
			t.Errorf("expected notification uri fak://hardware/inventory, got %v", notifFrame.Params["uri"])
		}

		cancel()
		inW.Close()
		outR.Close()
	})
}
