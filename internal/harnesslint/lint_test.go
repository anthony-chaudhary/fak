package harnesslint

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestRuleHL001_SecretPlaintext(t *testing.T) {
	// Case 1: Plaintext secret in assets
	lockWithPlaintextSecret := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"id": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
		"assets": [
			{
				"kind": "secret",
				"id": "anthropic-api-key",
				"value": "sk-ant-secret-key-12345"
			}
		]
	}`

	report := LintLock([]byte(lockWithPlaintextSecret))
	if report.Valid {
		t.Fatalf("expected Valid=false for plaintext secret leak, got true")
	}
	if report.ErrorCount < 1 {
		t.Fatalf("expected ErrorCount >= 1, got %d", report.ErrorCount)
	}

	found := false
	for _, d := range report.Diagnostics {
		if d.Rule == HL001_SECRET_PLAINTEXT {
			found = true
			if d.Severity != SeverityError {
				t.Errorf("expected severity ERROR, got %q", d.Severity)
			}
			if d.Field != "assets[0].value" {
				t.Errorf("expected field assets[0].value, got %q", d.Field)
			}
		}
	}
	if !found {
		t.Fatalf("diagnostic for %s not found in report", HL001_SECRET_PLAINTEXT)
	}

	// Case 2: Secret in top-level secrets array
	lockWithSecretList := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"secrets": [
			{
				"id": "db-token",
				"value": "plaintext_token"
			}
		]
	}`
	reportList := LintLock([]byte(lockWithSecretList))
	if reportList.Valid || reportList.ErrorCount < 1 {
		t.Fatalf("expected reportList to fail closed for secrets list plaintext, got valid=%v", reportList.Valid)
	}

	// Case 3: Secret with ref only (value empty)
	lockWithSecretRef := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"assets": [
			{
				"kind": "secret",
				"id": "safe-key",
				"value": "",
				"ref": "env:SAFE_API_KEY"
			}
		]
	}`
	reportSafe := LintLock([]byte(lockWithSecretRef))
	for _, d := range reportSafe.Diagnostics {
		if d.Rule == HL001_SECRET_PLAINTEXT {
			t.Errorf("unexpected HL001 diagnostic for secret with empty value and ref: %+v", d)
		}
	}
}

func TestRuleHL002_CRLFLineEndings(t *testing.T) {
	// Case 1: CRLF in asset value
	lockWithCRLFAsset := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"assets": [
			{
				"kind": "instruction",
				"id": "system-prompt",
				"value": "line1\r\nline2"
			}
		]
	}`

	report := LintLock([]byte(lockWithCRLFAsset))
	found := false
	for _, d := range report.Diagnostics {
		if d.Rule == HL002_CRLF_LINE_ENDINGS {
			found = true
			if d.Severity != SeverityError {
				t.Errorf("expected severity ERROR, got %q", d.Severity)
			}
			if d.Field != "assets[0].value" {
				t.Errorf("expected field assets[0].value, got %q", d.Field)
			}
		}
	}
	if !found {
		t.Fatalf("diagnostic for %s not found in report", HL002_CRLF_LINE_ENDINGS)
	}

	// Case 2: Raw CRLF bytes in document
	rawCRLF := "{\r\n  \"schema\": \"fak.harness-product-lock/v1alpha2\"\r\n}"
	reportRaw := LintLock([]byte(rawCRLF))
	foundRaw := false
	for _, d := range reportRaw.Diagnostics {
		if d.Rule == HL002_CRLF_LINE_ENDINGS {
			foundRaw = true
		}
	}
	if !foundRaw {
		t.Fatalf("diagnostic for %s not found on raw CRLF bytes", HL002_CRLF_LINE_ENDINGS)
	}

	// Case 3: Pure LF line endings
	lockWithLF := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"assets": [
			{
				"kind": "instruction",
				"id": "system-prompt",
				"value": "line1\nline2"
			}
		]
	}`
	reportLF := LintLock([]byte(lockWithLF))
	for _, d := range reportLF.Diagnostics {
		if d.Rule == HL002_CRLF_LINE_ENDINGS {
			t.Errorf("unexpected HL002 diagnostic for LF-only content: %+v", d)
		}
	}
}

func TestRuleHL003_SinglePlatform(t *testing.T) {
	// Case 1: Single platform without opt-in
	lockSinglePlatformTrap := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"platforms": ["linux/amd64"]
	}`
	report := LintLock([]byte(lockSinglePlatformTrap))
	if !report.Valid {
		t.Errorf("warning should not invalidate lock; expected Valid=true, got false")
	}
	if report.WarningCount < 1 {
		t.Errorf("expected WarningCount >= 1, got %d", report.WarningCount)
	}
	found := false
	for _, d := range report.Diagnostics {
		if d.Rule == HL003_SINGLE_PLATFORM {
			found = true
			if d.Severity != SeverityWarn {
				t.Errorf("expected severity WARN, got %q", d.Severity)
			}
			if d.Field != "platforms" {
				t.Errorf("expected field platforms, got %q", d.Field)
			}
		}
	}
	if !found {
		t.Fatalf("diagnostic for %s not found in report", HL003_SINGLE_PLATFORM)
	}

	// Case 2: Single platform with explicit single_platform opt-in
	lockWithOptIn := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"platforms": ["linux/amd64"],
		"single_platform": true
	}`
	reportOptIn := LintLock([]byte(lockWithOptIn))
	for _, d := range reportOptIn.Diagnostics {
		if d.Rule == HL003_SINGLE_PLATFORM {
			t.Errorf("unexpected HL003 diagnostic when opt-in present: %+v", d)
		}
	}

	// Case 3: Multiple platforms
	lockMultiPlatform := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"platforms": ["linux/amd64", "darwin/arm64", "windows/amd64"]
	}`
	reportMulti := LintLock([]byte(lockMultiPlatform))
	for _, d := range reportMulti.Diagnostics {
		if d.Rule == HL003_SINGLE_PLATFORM {
			t.Errorf("unexpected HL003 diagnostic for multi-platform lock: %+v", d)
		}
	}
}

func TestRuleHL004_UnpinnedMCPTools(t *testing.T) {
	// Case 1: MCP tool without schema fingerprint
	lockUnpinnedMCP := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"assets": [
			{
				"kind": "tool",
				"id": "mcp-filesystem",
				"source": "mcp"
			}
		]
	}`
	report := LintLock([]byte(lockUnpinnedMCP))
	if !report.Valid {
		t.Errorf("expected Valid=true for warning-only report, got false")
	}
	if report.WarningCount < 1 {
		t.Errorf("expected WarningCount >= 1, got %d", report.WarningCount)
	}
	found := false
	for _, d := range report.Diagnostics {
		if d.Rule == HL004_UNPINNED_MCP_TOOLS {
			found = true
			if d.Severity != SeverityWarn {
				t.Errorf("expected severity WARN, got %q", d.Severity)
			}
			if d.Field != "assets[0].schema_sha256" {
				t.Errorf("expected field assets[0].schema_sha256, got %q", d.Field)
			}
		}
	}
	if !found {
		t.Fatalf("diagnostic for %s not found in report", HL004_UNPINNED_MCP_TOOLS)
	}

	// Case 2: MCP tool in mcp_tools array unpinned
	lockUnpinnedArray := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"mcp_tools": [
			{
				"id": "weather-service"
			}
		]
	}`
	reportArray := LintLock([]byte(lockUnpinnedArray))
	foundArray := false
	for _, d := range reportArray.Diagnostics {
		if d.Rule == HL004_UNPINNED_MCP_TOOLS {
			foundArray = true
		}
	}
	if !foundArray {
		t.Fatalf("diagnostic for %s not found for unpinned mcp_tools array", HL004_UNPINNED_MCP_TOOLS)
	}

	// Case 3: Pinned MCP tool with valid SHA-256 fingerprint
	lockPinnedMCP := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"assets": [
			{
				"kind": "tool",
				"id": "mcp-filesystem",
				"source": "mcp",
				"schema_sha256": "sha256:7c2348ecbdc353d4b1f2a5ac7f583b4116a7baecfd7cc7aa071434a705989283"
			}
		]
	}`
	reportPinned := LintLock([]byte(lockPinnedMCP))
	for _, d := range reportPinned.Diagnostics {
		if d.Rule == HL004_UNPINNED_MCP_TOOLS {
			t.Errorf("unexpected HL004 diagnostic for pinned MCP tool: %+v", d)
		}
	}

	// Case 4: Non-MCP tool without fingerprint should not trigger warning
	lockDefaultTool := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"assets": [
			{
				"kind": "tool",
				"id": "search_kb",
				"source": "default"
			}
		]
	}`
	reportDefault := LintLock([]byte(lockDefaultTool))
	for _, d := range reportDefault.Diagnostics {
		if d.Rule == HL004_UNPINNED_MCP_TOOLS {
			t.Errorf("unexpected HL004 diagnostic for standard non-MCP tool: %+v", d)
		}
	}
}

func TestRuleHL005_UnknownFields(t *testing.T) {
	// Case 1: Unknown field in JSON
	lockUnknownField := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"unexpected_rogue_key": 42
	}`
	report := LintLock([]byte(lockUnknownField))
	if report.Valid {
		t.Fatalf("expected Valid=false for unknown fields, got true")
	}
	found := false
	for _, d := range report.Diagnostics {
		if d.Rule == HL005_UNKNOWN_FIELDS {
			found = true
			if d.Severity != SeverityError {
				t.Errorf("expected severity ERROR, got %q", d.Severity)
			}
			if d.Field != "unexpected_rogue_key" {
				t.Errorf("expected field unexpected_rogue_key, got %q", d.Field)
			}
		}
	}
	if !found {
		t.Fatalf("diagnostic for %s not found in report", HL005_UNKNOWN_FIELDS)
	}

	// Case 2: Malformed JSON syntax
	lockMalformed := `{"schema": "fak.harness-product-lock/v1alpha2", invalid_json`
	reportMalformed := LintLock([]byte(lockMalformed))
	if reportMalformed.Valid || reportMalformed.ErrorCount < 1 {
		t.Fatalf("expected report to fail closed on malformed JSON")
	}
	foundMalformed := false
	for _, d := range reportMalformed.Diagnostics {
		if d.Rule == HL005_UNKNOWN_FIELDS {
			foundMalformed = true
		}
	}
	if !foundMalformed {
		t.Fatalf("diagnostic for malformed JSON not found under %s", HL005_UNKNOWN_FIELDS)
	}

	// Case 3: Empty input
	reportEmpty := LintLock([]byte(""))
	if reportEmpty.Valid || reportEmpty.ErrorCount < 1 {
		t.Fatalf("expected report to fail closed on empty input")
	}

	// Case 4: Unanchored schema (missing schema field)
	lockUnanchored := `{
		"id": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
	}`
	reportUnanchored := LintLock([]byte(lockUnanchored))
	if reportUnanchored.Valid || reportUnanchored.ErrorCount < 1 {
		t.Fatalf("expected report to fail closed on missing schema")
	}
	foundUnanchored := false
	for _, d := range reportUnanchored.Diagnostics {
		if d.Rule == HL005_UNKNOWN_FIELDS && d.Field == "schema" {
			foundUnanchored = true
		}
	}
	if !foundUnanchored {
		t.Fatalf("diagnostic for missing schema not found under %s", HL005_UNKNOWN_FIELDS)
	}

	// Case 5: Trailing data after JSON
	lockTrailing := `{"schema": "fak.harness-product-lock/v1alpha2"}{"extra": 1}`
	reportTrailing := LintLock([]byte(lockTrailing))
	if reportTrailing.Valid || reportTrailing.ErrorCount < 1 {
		t.Fatalf("expected report to fail closed on trailing JSON data")
	}
}

func TestCleanValidV2Lock(t *testing.T) {
	cleanLock := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"id": "sha256:518381f886ec4719d8a21647b2e4f5a31b54e5bd711b7e3d7a9afe00e4774d1d",
		"environment": {
			"os": "linux",
			"arch": "amd64",
			"contract": "v1"
		},
		"budget": {
			"context_tokens": 16384,
			"memory_mib": 4096,
			"workers": 4
		},
		"platforms": [
			"linux/amd64",
			"darwin/arm64",
			"windows/amd64"
		],
		"components": [
			{
				"id": "kernel",
				"version": "1.0.0",
				"digest": "sha256:7c2348ecbdc353d4b1f2a5ac7f583b4116a7baecfd7cc7aa071434a705989283",
				"source": "registry/kernel",
				"reason": "selected root",
				"provider": "manifest",
				"provides": [
					"runtime"
				],
				"compatibility": {
					"os": ["linux", "darwin", "windows"],
					"arch": ["amd64", "arm64"],
					"contract": "v1"
				},
				"cost": {},
				"adapters": ["native"]
			}
		],
		"assets": [
			{
				"kind": "instruction",
				"id": "response-style",
				"value": "concise\naccurate",
				"source": "default"
			},
			{
				"kind": "policy",
				"id": "tools",
				"grants": ["search_kb"],
				"denies": ["shell"],
				"source": "default"
			},
			{
				"kind": "tool",
				"id": "search_kb",
				"value": "available",
				"source": "default"
			},
			{
				"kind": "tool",
				"id": "mcp-filesystem",
				"source": "mcp",
				"schema_sha256": "sha256:7c2348ecbdc353d4b1f2a5ac7f583b4116a7baecfd7cc7aa071434a705989283"
			},
			{
				"kind": "secret",
				"id": "db-token",
				"value": "",
				"ref": "env:DB_TOKEN",
				"source": "default"
			}
		],
		"asset_trace": [
			{
				"layer": "default",
				"kind": "instruction",
				"id": "response-style",
				"action": "add",
				"reason": "instruction uses explicit add semantics"
			}
		],
		"decisions": null
	}`

	report := LintLock([]byte(cleanLock))
	if !report.Valid {
		var messages []string
		for _, d := range report.Diagnostics {
			messages = append(messages, d.Rule+": "+d.Message+" ("+d.Field+")")
		}
		t.Fatalf("expected clean lock to be Valid=true, got Valid=false, errors: %s", strings.Join(messages, "; "))
	}
	if report.ErrorCount != 0 {
		t.Fatalf("expected ErrorCount=0, got %d", report.ErrorCount)
	}
	if report.WarningCount != 0 {
		var warnings []string
		for _, d := range report.Diagnostics {
			warnings = append(warnings, d.Rule+": "+d.Message)
		}
		t.Fatalf("expected WarningCount=0, got %d (%s)", report.WarningCount, strings.Join(warnings, "; "))
	}
	if len(report.Diagnostics) != 0 {
		t.Fatalf("expected 0 diagnostics, got %d", len(report.Diagnostics))
	}
}

func TestPerformanceUnder50ms(t *testing.T) {
	cleanLock := `{
		"schema": "fak.harness-product-lock/v1alpha2",
		"id": "sha256:518381f886ec4719d8a21647b2e4f5a31b54e5bd711b7e3d7a9afe00e4774d1d",
		"platforms": ["linux/amd64", "darwin/arm64"],
		"components": [
			{
				"id": "kernel",
				"version": "1.0.0",
				"digest": "sha256:7c2348ecbdc353d4b1f2a5ac7f583b4116a7baecfd7cc7aa071434a705989283",
				"source": "study/default-control",
				"provider": "manifest"
			}
		],
		"assets": [
			{
				"kind": "instruction",
				"id": "response-style",
				"value": "concise",
				"source": "default"
			},
			{
				"kind": "tool",
				"id": "mcp-eval",
				"source": "mcp",
				"schema_sha256": "sha256:abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef"
			}
		]
	}`

	data := []byte(cleanLock)

	// Measure single invocation.
	start := time.Now()
	report := LintLock(data)
	elapsed := time.Since(start)

	if !report.Valid {
		t.Fatalf("expected report to be valid in perf test")
	}
	if elapsed >= 50*time.Millisecond {
		t.Fatalf("LintLock took %v, must execute in <50ms", elapsed)
	}

	// Stress loop: 100 iterations must easily finish under 50ms total.
	loopStart := time.Now()
	for i := 0; i < 100; i++ {
		r := LintLock(data)
		if !r.Valid {
			t.Fatalf("iteration %d: expected valid lock", i)
		}
	}
	loopElapsed := time.Since(loopStart)
	if loopElapsed >= 50*time.Millisecond {
		t.Fatalf("100 iterations of LintLock took %v, must be <50ms", loopElapsed)
	}
}

func TestReportSerialization(t *testing.T) {
	report := LintReport{
		Valid: true,
		Diagnostics: []Diagnostic{
			{
				Rule:     HL003_SINGLE_PLATFORM,
				Severity: SeverityWarn,
				Message:  "test warning",
				Field:    "platforms",
			},
		},
		ErrorCount:   0,
		WarningCount: 1,
	}

	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	var parsed LintReport
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if parsed.Valid != report.Valid || parsed.WarningCount != 1 || len(parsed.Diagnostics) != 1 {
		t.Fatalf("round-trip mismatch: got %+v", parsed)
	}
}
