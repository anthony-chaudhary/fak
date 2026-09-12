package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"
	"time"

	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

func TestTurnkeyMTPQualificationCatalogSelectsOnlyExactReviewedContext(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)
	qualification := turnkeyMTPQualificationKey{
		ModelArtifactSHA256: strings.Repeat("ab", 32),
		ModelFamily:         "Qwen3.8",
		MTPFormat:           fakmodel.Qwen38MTPFormatQ4K,
		DeviceCompatibility: "apple/m3-pro-36gb/v1",
		MetalCompatibility:  "fak-metal/qwen38-mtp/v1",
		EngineCompatibility: "fak-native/qwen38-hybrid/v1",
		NativeConfigSHA256:  strings.Repeat("cd", 32),
		Workload: turnkeyMTPWorkloadEnvelope{
			Greedy: true, ContextTokens: 4096, DraftDepth: 3,
		},
	}
	context := turnkeyMTPQualificationContext{Qualification: qualification, AvailableHeadroomBytes: 4 << 30}
	evidence := fakmodel.Qwen38MTPCanaryEvidence{
		Receipt: fakmodel.Qwen38MTPCanaryReceipt{
			SchemaVersion: fakmodel.Qwen38MTPCanaryReceiptSchema,
			ReceiptID:     "qwen38-m3pro-reviewed-1", DefaultOn: true,
			Engine: fakmodel.Qwen38EngineMTP,
			Envelope: fakmodel.Qwen38CanaryEnvelope{
				ModelFamily: "Qwen3.8", Format: fakmodel.Qwen38MTPFormatQ4K,
				Backend: fakmodel.Qwen38MTPBackendMetal, HeadroomBytes: 3 << 30,
				ArtifactHash: qualification.ModelArtifactSHA256, DraftDepth: 3,
			},
			Speedup: 1.1, TokensProduced: 4, TokensProposed: 3, TokensAccepted: 3,
			DowngradeReason: fakmodel.Qwen38MTPEligible, CircuitStatus: fakmodel.CanaryCircuitClosed,
			LatencyNS:   fakmodel.Qwen38MTPLatencyNS{Setup: 1, Draft: 1, Verify: 1, Total: 3},
			MemoryBytes: fakmodel.Qwen38MTPMemoryBytes{DraftWorkspace: 1, VerifyWorkspace: 1, Peak: 1},
		},
		ObservedAt: now.Add(-time.Hour), ValidUntil: now.Add(time.Hour),
	}
	revision := strings.Repeat("1a", 20)
	catalog := turnkeyMTPTestCatalog(t, evidence, context, revision, false)

	result := selectTurnkeyMTPQualificationCatalog(catalog, now, context)
	if result.Refusal != turnkeyMTPQualificationAvailable || result.Selection == nil {
		t.Fatalf("exact reviewed context result = %+v", result)
	}
	if result.Selection.ReceiptID != evidence.Receipt.ReceiptID || result.Selection.Qualification != qualification || result.Selection.RuntimeContext != context || result.Selection.RequiredHeadroomBytes != 3<<30 {
		t.Fatalf("selection lost bound identity/context: %+v", result.Selection)
	}
	mgr := fakmodel.NewQwen38MTPCanaryManager()
	if err := mgr.RegisterCanaryEvidence(result.Selection.Evidence); err != nil {
		t.Fatal(err)
	}
	decision := mgr.EvaluateCanary(fakmodel.Qwen38CanaryRequest{
		Envelope:          result.Selection.Evidence.Receipt.Envelope,
		EvidenceReceiptID: result.Selection.ReceiptID,
		ModelReady:        true,
	})
	if decision.Engine != fakmodel.Qwen38EngineMTP || !decision.CanaryDefaultOn {
		t.Fatalf("selected evidence did not interoperate with canary evaluator: %+v", decision)
	}

	for name, mutate := range map[string]func(*turnkeyMTPQualificationContext){
		"artifact": func(c *turnkeyMTPQualificationContext) {
			c.Qualification.ModelArtifactSHA256 = strings.Repeat("ef", 32)
		},
		"model family": func(c *turnkeyMTPQualificationContext) { c.Qualification.ModelFamily = "qwen3.8" },
		"device":       func(c *turnkeyMTPQualificationContext) { c.Qualification.DeviceCompatibility = "apple/m3-max-36gb/v1" },
		"metal": func(c *turnkeyMTPQualificationContext) {
			c.Qualification.MetalCompatibility = "fak-metal/qwen38-mtp/v2"
		},
		"engine": func(c *turnkeyMTPQualificationContext) {
			c.Qualification.EngineCompatibility = "fak-native/qwen38-hybrid/v2"
		},
		"native config": func(c *turnkeyMTPQualificationContext) { c.Qualification.NativeConfigSHA256 = strings.Repeat("12", 32) },
		"workload":      func(c *turnkeyMTPQualificationContext) { c.Qualification.Workload.ContextTokens++ },
		"headroom":      func(c *turnkeyMTPQualificationContext) { c.AvailableHeadroomBytes = 2 << 30 },
	} {
		t.Run(name+" mismatch", func(t *testing.T) {
			gotContext := context
			mutate(&gotContext)
			got := selectTurnkeyMTPQualificationCatalog(catalog, now, gotContext)
			if got.Refusal != turnkeyMTPNoEligibleContext || got.Selection != nil {
				t.Fatalf("mismatch result = %+v", got)
			}
		})
	}

	missingRuntime := context
	missingRuntime.Qualification.DeviceCompatibility = ""
	if got := selectTurnkeyMTPQualificationCatalog(catalog, now, missingRuntime); got.Refusal != turnkeyMTPNoEligibleContext || got.Selection != nil {
		t.Fatalf("missing runtime context = %+v", got)
	}
	if got := selectTurnkeyMTPQualificationCatalog(turnkeyMTPTestCatalog(t, evidence, context, revision, true), now, context); got.Refusal != turnkeyMTPNoEligibleContext || got.Selection != nil {
		t.Fatalf("revoked evidence = %+v", got)
	}
	if got := selectTurnkeyMTPQualificationCatalog(catalog, evidence.ValidUntil, context); got.Refusal != turnkeyMTPNoEligibleContext || got.Selection != nil {
		t.Fatalf("expired evidence = %+v", got)
	}
	if got := selectTurnkeyMTPQualificationCatalog(catalog, evidence.ObservedAt.Add(-time.Second), context); got.Refusal != turnkeyMTPNoEligibleContext || got.Selection != nil {
		t.Fatalf("future-observed evidence = %+v", got)
	}
	if got := embeddedTurnkeyMTPQualification(now, context); got.Refusal != turnkeyMTPNoEligibleContext || got.Selection != nil {
		t.Fatalf("empty production catalog = %+v", got)
	}

	var decoded map[string]any
	if err := json.Unmarshal(catalog, &decoded); err != nil {
		t.Fatal(err)
	}
	decoded["unexpected"] = true
	malformed, _ := json.Marshal(decoded)
	if got := selectTurnkeyMTPQualificationCatalog(malformed, now, context); got.Refusal != turnkeyMTPMalformedCatalog || got.Selection != nil {
		t.Fatalf("unknown-field catalog = %+v", got)
	}
	for name, duplicate := range map[string][]byte{
		"top level":  []byte(`{"schema":"fak.up.mtp-evidence-catalog/1","records":[],"records":[],"revocations":[]}`),
		"case alias": []byte(`{"schema":"fak.up.mtp-evidence-catalog/1","records":[],"revocations":[],"Revocations":[]}`),
		"context":    []byte(strings.Replace(string(catalog), `"greedy":true`, `"greedy":true,"greedy":true`, 1)),
		"evidence":   []byte(strings.Replace(string(catalog), `"receipt_id":"qwen38-m3pro-reviewed-1"`, `"receipt_id":"qwen38-m3pro-reviewed-1","receipt_id":"qwen38-m3pro-reviewed-1"`, 1)),
	} {
		t.Run("duplicate key "+name, func(t *testing.T) {
			if got := selectTurnkeyMTPQualificationCatalog(duplicate, now, context); got.Refusal != turnkeyMTPMalformedCatalog || got.Selection != nil {
				t.Fatalf("duplicate-key catalog = %+v", got)
			}
		})
	}
	var validCatalog turnkeyMTPEvidenceCatalog
	if err := json.Unmarshal(catalog, &validCatalog); err != nil {
		t.Fatal(err)
	}
	if valid := selectTurnkeyMTPQualificationCatalog(catalog, now, context); valid.Selection == nil {
		t.Fatalf("valid prefix precondition = %+v", valid)
	}
	validCatalog.Records = append(validCatalog.Records, turnkeyMTPEvidenceRecord{Evidence: json.RawMessage(`{}`)})
	malformed, _ = json.Marshal(validCatalog)
	if got := selectTurnkeyMTPQualificationCatalog(malformed, now, context); got.Refusal != turnkeyMTPMalformedCatalog || got.Selection != nil {
		t.Fatalf("partially malformed catalog was not atomic: %+v", got)
	}
}

func turnkeyMTPTestCatalog(t *testing.T, evidence fakmodel.Qwen38MTPCanaryEvidence, context turnkeyMTPQualificationContext, revision string, revoked bool) []byte {
	t.Helper()
	evidenceRaw, err := json.Marshal(evidence)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(evidenceRaw)
	digest := hex.EncodeToString(sum[:])
	record := turnkeyMTPEvidenceRecord{
		Evidence: evidenceRaw, EvidenceSHA256: digest,
		SourceURL:            "https://github.com/anthony-chaudhary/fak/blob/" + revision + "/experiments/benchmark/mtp/receipt.json",
		SourceRevision:       revision,
		TestedSourceRevision: strings.Repeat("2b", 20),
		Qualification:        context.Qualification,
	}
	catalog := turnkeyMTPEvidenceCatalog{Schema: turnkeyMTPEvidenceCatalogSchema, Records: []turnkeyMTPEvidenceRecord{record}}
	if revoked {
		catalog.Revocations = []turnkeyMTPEvidenceRevocation{{ReceiptID: evidence.Receipt.ReceiptID, EvidenceSHA256: digest}}
	} else {
		catalog.Revocations = []turnkeyMTPEvidenceRevocation{}
	}
	raw, err := json.Marshal(catalog)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
