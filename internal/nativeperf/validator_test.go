package nativeperf

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
)

// validTestReceipt returns a pristine valid benchmark receipt for testing.
func validTestReceipt() *BenchmarkReceipt {
	return &BenchmarkReceipt{
		ReceiptID:                    "rcpt-strix-valid-001",
		Device:                       "gfx1151",
		ModelArchitecture:            "qwen38",
		TheoreticalPeakBandwidthGBps: 273.056,
		AchievedBandwidthGBps:        225.0, // 225.0 / 273.056 = 0.8240 (82.40% >= 80%)
		LogitCosineSimilarity:        0.999948,
		Simulated:                    false,
		ExecutionTimeMs:              12.45,
		TokensPerSecond:              184.10,
		PromptTokens:                 512,
		DecodeTokens:                 128,
		TotalTokens:                  640,
		PeakMemoryBytes:              8 * 1024 * 1024 * 1024,
		Timestamp:                    "2026-09-05T08:00:00Z",
		GitRevision:                  "c5a3b411f",
		Environment: map[string]string{
			"OS":       "linux",
			"ROCM_REV": "7.14.0",
		},
		Metadata: map[string]any{
			"cu_count": 40,
			"backend":  "vulkan",
		},
	}
}

// TestValidateRooflineAttainment_ValidPassesWithCleanStatus verifies valid receipt passes with exit status clean.
func TestValidateRooflineAttainment_ValidPassesWithCleanStatus(t *testing.T) {
	receipt := validTestReceipt()

	res, err := ValidateRooflineAttainment(receipt)
	if err != nil {
		t.Fatalf("expected ValidateRooflineAttainment to pass, got error: %v", err)
	}
	if res == nil {
		t.Fatalf("expected non-nil VerificationResult")
	}
	if !res.Passed {
		t.Fatalf("expected res.Passed to be true, got false with violations: %v", res.Violations)
	}
	if !res.Clean() {
		t.Fatalf("expected res.Clean() to be true")
	}
	if len(res.Violations) != 0 {
		t.Errorf("expected 0 violations, got %d: %v", len(res.Violations), res.Violations)
	}
	if res.AttainmentRatio < MinimumRooflineAttainmentRatio {
		t.Errorf("attainment ratio %.4f < 0.80", res.AttainmentRatio)
	}
	if res.LogitCosineSimilarity < MinimumLogitCosineSimilarity {
		t.Errorf("logit cosine similarity %.6f < 0.999900", res.LogitCosineSimilarity)
	}

	// Test valid simulated run with proof token passes cleanly as well
	simReceipt := validTestReceipt()
	simReceipt.Simulated = true
	simReceipt.ProofToken = "fak-proof:simulated:sha256:7e78da5d7e3ae28d"
	simRes, simErr := ValidateRooflineAttainment(simReceipt)
	if simErr != nil {
		t.Fatalf("expected valid simulated receipt with proof token to pass, got: %v", simErr)
	}
	if !simRes.Passed || !simRes.Clean() {
		t.Fatalf("expected simulated receipt to pass cleanly: %v", simRes.Violations)
	}
}

// TestValidateRooflineAttainment_FourNegativeFixtures verifies the 4 mandatory negative rejection fixtures.
func TestValidateRooflineAttainment_FourNegativeFixtures(t *testing.T) {
	// Negative Fixture 1: sub-80% attainment fails
	t.Run("sub-80% attainment fails", func(t *testing.T) {
		r := validTestReceipt()
		// 180.0 / 273.056 = ~0.6592 (< 0.80)
		r.AchievedBandwidthGBps = 180.0

		res, err := ValidateRooflineAttainment(r)
		if err == nil {
			t.Fatalf("expected error for sub-80%% attainment, got nil")
		}
		if res == nil || res.Passed {
			t.Fatalf("expected res.Passed to be false")
		}
		if !errors.Is(err, ErrSub80PercentAttainment) {
			t.Errorf("expected ErrSub80PercentAttainment, got %v", err)
		}
		if !strings.Contains(err.Error(), "below strict 80% acceptance gate") {
			t.Errorf("expected error message to mention 80%% acceptance gate, got: %s", err.Error())
		}
	})

	// Negative Fixture 2: logit divergence fails
	t.Run("logit divergence fails", func(t *testing.T) {
		r := validTestReceipt()
		r.LogitCosineSimilarity = 0.999850 // < 0.999900

		res, err := ValidateRooflineAttainment(r)
		if err == nil {
			t.Fatalf("expected error for logit divergence, got nil")
		}
		if res == nil || res.Passed {
			t.Fatalf("expected res.Passed to be false")
		}
		if !errors.Is(err, ErrLogitDivergence) {
			t.Errorf("expected ErrLogitDivergence, got %v", err)
		}
		if !strings.Contains(err.Error(), "below numerical accuracy gate") {
			t.Errorf("expected error message to mention numerical accuracy gate, got: %s", err.Error())
		}
	})

	// Negative Fixture 3: simulated run without proof fails
	t.Run("simulated run without proof fails", func(t *testing.T) {
		r := validTestReceipt()
		r.Simulated = true
		r.ProofToken = ""
		r.Signature = ""

		res, err := ValidateRooflineAttainment(r)
		if err == nil {
			t.Fatalf("expected error for simulated run without proof, got nil")
		}
		if res == nil || res.Passed {
			t.Fatalf("expected res.Passed to be false")
		}
		if !errors.Is(err, ErrSimulatedWithoutProof) {
			t.Errorf("expected ErrSimulatedWithoutProof, got %v", err)
		}
		if !strings.Contains(err.Error(), "simulated-only run without proof token or signature") {
			t.Errorf("expected error message to mention proof token, got: %s", err.Error())
		}
	})

	// Negative Fixture 4: missing denominator fails
	t.Run("missing denominator fails", func(t *testing.T) {
		r := validTestReceipt()
		r.TheoreticalPeakBandwidthGBps = 0.0 // Missing denominator

		res, err := ValidateRooflineAttainment(r)
		if err == nil {
			t.Fatalf("expected error for missing denominator, got nil")
		}
		if res == nil || res.Passed {
			t.Fatalf("expected res.Passed to be false")
		}
		if !errors.Is(err, ErrMissingDenominator) {
			t.Errorf("expected ErrMissingDenominator, got %v", err)
		}
		if !strings.Contains(err.Error(), "missing denominator") {
			t.Errorf("expected error message to mention missing denominator, got: %s", err.Error())
		}
	})
}

// TestValidateRooflineAttainment_AdditionalRejectionCriteria verifies supplementary safety invariants.
func TestValidateRooflineAttainment_AdditionalRejectionCriteria(t *testing.T) {
	t.Run("zero achieved bandwidth fails", func(t *testing.T) {
		r := validTestReceipt()
		r.AchievedBandwidthGBps = 0.0
		res, err := ValidateRooflineAttainment(r)
		if err == nil {
			t.Fatal("expected error for zero achieved bandwidth")
		}
		if res.Passed {
			t.Fatal("expected res.Passed to be false")
		}
	})

	t.Run("mismatched target device fails", func(t *testing.T) {
		r := validTestReceipt()
		r.Device = "nvidia-h100-sxm5"
		res, err := ValidateRooflineAttainment(r)
		if err == nil {
			t.Fatal("expected error for mismatched target device")
		}
		if res.Passed {
			t.Fatal("expected res.Passed to be false")
		}
		if !errors.Is(err, ErrMismatchedTargetDevice) {
			t.Errorf("expected ErrMismatchedTargetDevice, got %v", err)
		}
	})

	t.Run("missing or invalid model architecture fails", func(t *testing.T) {
		r := validTestReceipt()
		r.ModelArchitecture = ""
		res, err := ValidateRooflineAttainment(r)
		if err == nil {
			t.Fatal("expected error for empty model architecture")
		}
		if res.Passed {
			t.Fatal("expected res.Passed to be false")
		}
		if !errors.Is(err, ErrMismatchedModelArch) {
			t.Errorf("expected ErrMismatchedModelArch, got %v", err)
		}

		r.ModelArchitecture = "unknown"
		_, err2 := ValidateRooflineAttainment(r)
		if err2 == nil || !errors.Is(err2, ErrMismatchedModelArch) {
			t.Errorf("expected ErrMismatchedModelArch for unknown model, got %v", err2)
		}
	})

	t.Run("negative timing or counter metrics fails", func(t *testing.T) {
		r := validTestReceipt()
		r.ExecutionTimeMs = -10.0
		res, err := ValidateRooflineAttainment(r)
		if err == nil {
			t.Fatal("expected error for negative execution time")
		}
		if res.Passed {
			t.Fatal("expected res.Passed to be false")
		}
		if !errors.Is(err, ErrInvalidMetrics) {
			t.Errorf("expected ErrInvalidMetrics, got %v", err)
		}

		r2 := validTestReceipt()
		r2.TokensPerSecond = -5.0
		_, errTokens := ValidateRooflineAttainment(r2)
		if errTokens == nil || !errors.Is(errTokens, ErrInvalidMetrics) {
			t.Errorf("expected ErrInvalidMetrics for negative tokens per second, got %v", errTokens)
		}

		r3 := validTestReceipt()
		r3.PromptTokens = -1
		_, errCount := ValidateRooflineAttainment(r3)
		if errCount == nil || !errors.Is(errCount, ErrInvalidMetrics) {
			t.Errorf("expected ErrInvalidMetrics for negative token count, got %v", errCount)
		}
	})

	t.Run("nil receipt returns ErrNilReceipt", func(t *testing.T) {
		res, err := ValidateRooflineAttainment(nil)
		if !errors.Is(err, ErrNilReceipt) {
			t.Fatalf("expected ErrNilReceipt, got %v", err)
		}
		if res != nil {
			t.Fatalf("expected nil result for nil receipt")
		}
	})
}

// TestReproducibilityPacket_GenerationAndChecksumDeterminism verifies packet generation,
// schema compliance, deterministic checksums, and tamper detection.
func TestReproducibilityPacket_GenerationAndChecksumDeterminism(t *testing.T) {
	receipt := validTestReceipt()

	// 1. Generation from valid receipt
	p1, err := GenerateReproducibilityPacket(receipt)
	if err != nil {
		t.Fatalf("GenerateReproducibilityPacket failed: %v", err)
	}
	if p1 == nil {
		t.Fatalf("expected non-nil ReproducibilityPacket")
	}
	if p1.Schema != ReproducibilityPacketSchema {
		t.Errorf("schema = %q, want %q", p1.Schema, ReproducibilityPacketSchema)
	}
	if p1.AttainmentRatio < MinimumRooflineAttainmentRatio {
		t.Errorf("attainment ratio = %.4f, want >= 0.80", p1.AttainmentRatio)
	}
	if !p1.Attainment.GatePassed {
		t.Errorf("expected attainment gate to pass")
	}
	if !p1.NumericalAccuracy.GatePassed {
		t.Errorf("expected numerical accuracy gate to pass")
	}
	if p1.Checksum == "" {
		t.Fatalf("expected non-empty packet Checksum")
	}

	// Verify checksum integrity
	if err := p1.VerifyChecksum(); err != nil {
		t.Fatalf("p1.VerifyChecksum() failed: %v", err)
	}

	// 2. Determinism: generating from identical receipt produces identical packet and checksum
	p2, err := GenerateReproducibilityPacket(receipt)
	if err != nil {
		t.Fatalf("second GenerateReproducibilityPacket failed: %v", err)
	}
	if p1.Checksum != p2.Checksum {
		t.Fatalf("checksum indeterminism detected: p1=%s, p2=%s", p1.Checksum, p2.Checksum)
	}

	json1, err := p1.JSON()
	if err != nil {
		t.Fatalf("failed to serialize p1 JSON: %v", err)
	}
	json2, err := p2.JSON()
	if err != nil {
		t.Fatalf("failed to serialize p2 JSON: %v", err)
	}
	if string(json1) != string(json2) {
		t.Fatalf("JSON serialization indeterminism detected:\np1:\n%s\np2:\n%s", string(json1), string(json2))
	}

	// 3. Tamper detection: mutating any field in the packet causes checksum verification to fail
	tampered := *p1
	tampered.AttainmentRatio = 0.50
	if err := tampered.VerifyChecksum(); err == nil {
		t.Fatalf("expected VerifyChecksum to fail on tampered packet, got nil")
	}

	// 4. Scrubbing audit: verify private credentials, paths, and host details are scrubbed
	dirtyReceipt := validTestReceipt()
	dirtyReceipt.Environment["SECRET_KEY"] = "sk-ant-api03-abcdef1234567890abcdef1234567890"
	dirtyReceipt.Environment["USER_PROFILE_PATH"] = `C:\Users\devuser\secret\keys`
	dirtyReceipt.Environment["SAFE_FLAG"] = "enabled"
	dirtyReceipt.Metadata["run_path"] = `/home/anthony/private/bench`
	dirtyReceipt.GitRevision = `c5a3b411f-C:\Users\devuser\repo`

	scrubbedPacket, err := GenerateReproducibilityPacket(dirtyReceipt)
	if err != nil {
		t.Fatalf("failed to generate packet with dirty receipt: %v", err)
	}

	rawScrubbed, err := scrubbedPacket.JSON()
	if err != nil {
		t.Fatalf("failed to serialize scrubbed packet: %v", err)
	}
	scrubbedStr := string(rawScrubbed)

	// Check that secrets and user paths are absent
	if strings.Contains(scrubbedStr, `C:\Users\`) || strings.Contains(scrubbedStr, `C:/Users/`) {
		t.Errorf("scrubbed packet leaked Windows user path:\n%s", scrubbedStr)
	}
	if strings.Contains(scrubbedStr, `/home/anthony`) {
		t.Errorf("scrubbed packet leaked Unix user path:\n%s", scrubbedStr)
	}
	if strings.Contains(scrubbedStr, "sk-ant-") {
		t.Errorf("scrubbed packet leaked secret key:\n%s", scrubbedStr)
	}
	if !scrubbedPacket.Provenance.Scrubbed {
		t.Errorf("expected Provenance.Scrubbed to be true")
	}

	// 5. Witness writing to directory
	tmpDir := t.TempDir()
	outPath, err := scrubbedPacket.WriteWitness(tmpDir)
	if err != nil {
		t.Fatalf("WriteWitness failed: %v", err)
	}
	if _, err := os.Stat(outPath); err != nil {
		t.Fatalf("witness file was not written: %v", err)
	}

	// Verify the written file round-trips cleanly
	writtenBytes, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("failed to read witness file: %v", err)
	}
	var roundTrip ReproducibilityPacket
	if err := json.Unmarshal(writtenBytes, &roundTrip); err != nil {
		t.Fatalf("failed to unmarshal written witness: %v", err)
	}
	if err := roundTrip.VerifyChecksum(); err != nil {
		t.Fatalf("written witness failed checksum verification: %v", err)
	}
}

// TestGenerateReproducibilityPacket_RejectsInvalidReceipt verifies generator refuses invalid inputs.
func TestGenerateReproducibilityPacket_RejectsInvalidReceipt(t *testing.T) {
	// Sub-80% receipt rejected by packet generator
	badReceipt := validTestReceipt()
	badReceipt.AchievedBandwidthGBps = 150.0 // sub-80%

	_, err := GenerateReproducibilityPacket(badReceipt)
	if err == nil {
		t.Fatal("expected GenerateReproducibilityPacket to reject sub-80% receipt")
	}

	// Nil receipt rejected
	_, errNil := GenerateReproducibilityPacket(nil)
	if !errors.Is(errNil, ErrNilReceipt) {
		t.Fatalf("expected ErrNilReceipt, got %v", errNil)
	}
}
