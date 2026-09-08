package qwen38quantrun

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
)

func TestStrixComparisonCellManifestSeal(t *testing.T) {
	m := validStrixComparisonCellManifest(t)
	sealed, err := SealStrixComparisonCellManifest(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyStrixComparisonCellManifest(sealed, sealed.Digest); err != nil {
		t.Fatal(err)
	}
	again, err := SealStrixComparisonCellManifest(m)
	if err != nil {
		t.Fatal(err)
	}
	if again.Digest != sealed.Digest {
		t.Fatalf("nondeterministic manifest digest: %s != %s", again.Digest, sealed.Digest)
	}
	m.Challenge.PairIDs[0] = "caller-mutated-after-seal"
	if err := VerifyStrixComparisonCellManifest(sealed, sealed.Digest); err != nil {
		t.Fatalf("caller-owned slice mutation changed sealed manifest: %v", err)
	}

	raw, err := ExportStrixComparisonCellManifest(sealed, sealed.Digest)
	if err != nil {
		t.Fatal(err)
	}
	imported, err := ImportStrixComparisonCellManifest(raw, sealed.Digest)
	if err != nil {
		t.Fatal(err)
	}
	if imported.Digest != sealed.Digest {
		t.Fatalf("round-trip digest = %q want %q", imported.Digest, sealed.Digest)
	}
	for _, forbidden := range [][]byte{[]byte("physical_identity_verified"), []byte("strix_absolute_eligible"), []byte("overall_win"), []byte("tok_per_sec_observed")} {
		if bytes.Contains(raw, forbidden) {
			t.Fatalf("integrity-only manifest exposed authority/credit field %q", forbidden)
		}
	}

	t.Run("tamper", func(t *testing.T) {
		changed := sealed
		changed.Capture.ReplayKey = "changed-replay-key"
		if err := VerifyStrixComparisonCellManifest(changed, sealed.Digest); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
			t.Fatalf("tamper was not rejected: %v", err)
		}
	})

	t.Run("recomputed-unapproved", func(t *testing.T) {
		changed := cloneStrixComparisonCellManifestForTest(t, sealed)
		changed.Capture.ReplayKey = "changed-replay-key"
		resealed, err := SealStrixComparisonCellManifest(changed)
		if err != nil {
			t.Fatal(err)
		}
		if resealed.Digest == sealed.Digest {
			t.Fatal("changed manifest retained approved identity")
		}
		if err := VerifyStrixComparisonCellManifest(resealed, sealed.Digest); err == nil || !strings.Contains(err.Error(), "not the approved challenge") {
			t.Fatalf("recomputed unapproved challenge was not rejected: %v", err)
		}
	})

	t.Run("strict-json", func(t *testing.T) {
		unknown := append([]byte(nil), raw[:len(raw)-1]...)
		unknown = append(unknown, []byte(",\n  \"unapproved\": true\n}")...)
		if _, err := ImportStrixComparisonCellManifest(unknown, sealed.Digest); err == nil || !strings.Contains(err.Error(), "unknown field") {
			t.Fatalf("unknown field was not rejected: %v", err)
		}
		duplicate := bytes.Replace(raw, []byte(`"schema": "`+StrixComparisonCellManifestSchema+`"`), []byte(`"schema": "`+StrixComparisonCellManifestSchema+`", "schema": "`+StrixComparisonCellManifestSchema+`"`), 1)
		if _, err := ImportStrixComparisonCellManifest(duplicate, sealed.Digest); err == nil || !strings.Contains(err.Error(), "duplicate JSON field") {
			t.Fatalf("duplicate field was not rejected: %v", err)
		}
		caseAlias := bytes.Replace(raw, []byte(`"schema":`), []byte(`"Schema":`), 1)
		if _, err := ImportStrixComparisonCellManifest(caseAlias, sealed.Digest); err == nil || !strings.Contains(err.Error(), "non-canonical JSON field") {
			t.Fatalf("case-aliased field was not rejected: %v", err)
		}
		unicodeAlias := bytes.Replace(raw, []byte(`"schema":`), []byte(`"ſchema":`), 1)
		if _, err := ImportStrixComparisonCellManifest(unicodeAlias, sealed.Digest); err == nil || !strings.Contains(err.Error(), "non-canonical JSON field") {
			t.Fatalf("Unicode simple-fold alias was not rejected: %v", err)
		}
	})

	t.Run("packet-duplicate-json", func(t *testing.T) {
		changed := cloneStrixComparisonCellManifestForTest(t, sealed)
		changed.Workload.PromptPacketBytes = bytes.Replace(changed.Workload.PromptPacketBytes, []byte(`"schema":`), []byte(`"schema":"`+PromptTokenPacketSchema+`","schema":`), 1)
		if _, err := SealStrixComparisonCellManifest(changed); err == nil || !strings.Contains(err.Error(), "duplicate JSON field") {
			t.Fatalf("duplicate prompt-packet field was not rejected: %v", err)
		}
	})

	mutations := []struct {
		name string
		mut  func(*StrixComparisonCellManifest)
	}{
		{"schema", func(x *StrixComparisonCellManifest) { x.Schema += ".other" }},
		{"source row", func(x *StrixComparisonCellManifest) { x.Challenge.SourceRowID += "-other" }},
		{"bar relabel", func(x *StrixComparisonCellManifest) { x.Challenge.Concurrency = 4 }},
		{"statistics", func(x *StrixComparisonCellManifest) { x.Challenge.ConfidenceRule = "two-sided-95-percent" }},
		{"warmups", func(x *StrixComparisonCellManifest) { x.Challenge.Warmups-- }},
		{"order", func(x *StrixComparisonCellManifest) {
			x.Challenge.AlternatingOrder[0], x.Challenge.AlternatingOrder[1] = x.Challenge.AlternatingOrder[1], x.Challenge.AlternatingOrder[0]
		}},
		{"device", func(x *StrixComparisonCellManifest) { x.Platform.GPU = "RX 7600" }},
		{"observation", func(x *StrixComparisonCellManifest) { x.Platform.Observations[0].Value = "observed-different" }},
		{"artifact", func(x *StrixComparisonCellManifest) { x.Workload.ArtifactSHA256 = strings.Repeat("a", 64) }},
		{"reasoning", func(x *StrixComparisonCellManifest) { x.Workload.ReasoningEnabled = true }},
		{"cache", func(x *StrixComparisonCellManifest) { x.Memory.PrimaryCacheState = "warm-prefix" }},
		{"candidate fallback", func(x *StrixComparisonCellManifest) { x.Candidate.Fallback = true }},
		{"reference revision", func(x *StrixComparisonCellManifest) { x.Reference.SourceRevision = strings.Repeat("b", 40) }},
		{"capture count", func(x *StrixComparisonCellManifest) { x.Capture.TrialCount++ }},
		{"capture binding", func(x *StrixComparisonCellManifest) { x.Capture.ObservationBindings[0] = "claimed-timing" }},
	}
	for _, tc := range mutations {
		t.Run("mutation/"+tc.name, func(t *testing.T) {
			changed := cloneStrixComparisonCellManifestForTest(t, sealed)
			tc.mut(&changed)
			if err := VerifyStrixComparisonCellManifest(changed, sealed.Digest); err == nil {
				t.Fatal("single-field mutation was accepted")
			}
		})
	}

	nonFinite := m
	nonFinite.Challenge.AcceptedOutputTPS = math.NaN()
	if _, err := SealStrixComparisonCellManifest(nonFinite); err == nil {
		t.Fatal("non-finite challenge bar was accepted")
	}
}

func TestStrixComparisonCellManifestAlternatesPairOrder(t *testing.T) {
	pairIDs := make([]string, 0, StrixComparisonMinimumMeasuredPairs)
	order := make([]string, 0, 2*StrixComparisonMinimumMeasuredPairs)
	input := AMDScoreboardInput{}
	for i := range StrixComparisonMinimumMeasuredPairs {
		pairID, first, second, candidateSequence, referenceSequence := strixComparisonPairOrder(i)
		pairIDs = append(pairIDs, pairID)
		order = append(order, first, second)
		input.Candidate.Trials = append(input.Candidate.Trials, AMDScoreboardTrial{Repetition: i + 1, Sequence: candidateSequence})
		input.Reference.Trials = append(input.Reference.Trials, AMDScoreboardTrial{Repetition: i + 1, Sequence: referenceSequence})
	}

	m := validStrixComparisonCellManifest(t)
	m.Challenge.PairIDs = slices.Clone(pairIDs)
	m.Challenge.AlternatingOrder = slices.Clone(order)
	m.Capture.PairIDs = slices.Clone(pairIDs)
	m.Capture.AlternatingOrder = slices.Clone(order)
	if _, err := SealStrixComparisonCellManifest(m); err != nil {
		t.Fatalf("shared AB/BA schedule rejected by manifest: %v", err)
	}
	if reasons := validateAMDScoreboard(input); slices.Contains(reasons, "alternating-paired-trial-order-required") {
		t.Fatalf("shared AB/BA schedule rejected by scoreboard: %v", reasons)
	}

	rejectManifest := func(name string, mutate func(*StrixComparisonCellManifest)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			changed := cloneStrixComparisonCellManifestForTest(t, m)
			mutate(&changed)
			if _, err := SealStrixComparisonCellManifest(changed); err == nil {
				t.Fatal("non-canonical pair schedule was accepted")
			}
		})
	}
	rejectManifest("all-ab", func(changed *StrixComparisonCellManifest) {
		for i, pairID := range changed.Challenge.PairIDs {
			changed.Challenge.AlternatingOrder[2*i], changed.Challenge.AlternatingOrder[2*i+1] = "candidate:"+pairID, "reference:"+pairID
		}
		changed.Capture.AlternatingOrder = slices.Clone(changed.Challenge.AlternatingOrder)
	})
	rejectManifest("all-ba", func(changed *StrixComparisonCellManifest) {
		for i, pairID := range changed.Challenge.PairIDs {
			changed.Challenge.AlternatingOrder[2*i], changed.Challenge.AlternatingOrder[2*i+1] = "reference:"+pairID, "candidate:"+pairID
		}
		changed.Capture.AlternatingOrder = slices.Clone(changed.Challenge.AlternatingOrder)
	})
	rejectManifest("swapped-pair-ids", func(changed *StrixComparisonCellManifest) {
		changed.Challenge.PairIDs[0], changed.Challenge.PairIDs[1] = changed.Challenge.PairIDs[1], changed.Challenge.PairIDs[0]
		changed.Capture.PairIDs = slices.Clone(changed.Challenge.PairIDs)
	})
	rejectManifest("duplicate-pair-id", func(changed *StrixComparisonCellManifest) {
		changed.Challenge.PairIDs[1] = changed.Challenge.PairIDs[0]
		changed.Capture.PairIDs = slices.Clone(changed.Challenge.PairIDs)
	})
	rejectManifest("missing-pair", func(changed *StrixComparisonCellManifest) {
		changed.Challenge.MeasuredPairs--
		changed.Challenge.PairIDs = changed.Challenge.PairIDs[:4]
		changed.Challenge.AlternatingOrder = changed.Challenge.AlternatingOrder[:8]
		changed.Capture.PairIDs = slices.Clone(changed.Challenge.PairIDs)
		changed.Capture.AlternatingOrder = slices.Clone(changed.Challenge.AlternatingOrder)
		changed.Capture.TrialCount = 8
	})
	rejectManifest("appended-pair", func(changed *StrixComparisonCellManifest) {
		changed.Challenge.MeasuredPairs++
		changed.Challenge.PairIDs = append(changed.Challenge.PairIDs, "pair-06")
		changed.Challenge.AlternatingOrder = append(changed.Challenge.AlternatingOrder, "reference:pair-06", "candidate:pair-06")
		changed.Capture.PairIDs = slices.Clone(changed.Challenge.PairIDs)
		changed.Capture.AlternatingOrder = slices.Clone(changed.Challenge.AlternatingOrder)
		changed.Capture.TrialCount = 12
	})

	parityChanged := cloneStrixComparisonCellManifestForTest(t, m)
	parityChanged.Challenge.AlternatingOrder[2], parityChanged.Challenge.AlternatingOrder[3] = "candidate:pair-02", "reference:pair-02"
	parityChanged.Capture.AlternatingOrder = slices.Clone(parityChanged.Challenge.AlternatingOrder)
	if _, err := SealStrixComparisonCellManifest(parityChanged); err == nil {
		t.Fatal("manifest accepted an all-AB parity mutation")
	}
	input.Candidate.Trials[1].Sequence, input.Reference.Trials[1].Sequence = 3, 4
	if reasons := validateAMDScoreboard(input); !slices.Contains(reasons, "alternating-paired-trial-order-required") {
		t.Fatalf("scoreboard accepted the same all-AB parity mutation: %v", reasons)
	}
}

func validStrixComparisonCellManifest(t *testing.T) StrixComparisonCellManifest {
	t.Helper()
	packet := PromptTokenPacket{
		Schema: PromptTokenPacketSchema, PacketID: "strix-cell-c1-packet",
		ArtifactSHA256: StrixComparisonArtifactSHA256, TokenizerIdentity: GGUFTokenizerIdentity,
		TokenizerDigest: StrixComparisonTokenizerSHA256, TemplateDigest: StrixComparisonTemplateSHA256,
		PromptTokenIDs: []int{
			151644, 872, 198, 2610, 525, 264, 25, 13, 151645, 198, 151644, 77091, 198,
			151667, 198, 16, 17, 18, 19, 20, 21, 22, 23, 24, 151645, 198,
		},
		ContextBudget:      ContextBudget{ContextTokens: StrixComparisonContextTokens, ContextBudgetBytes: 48 << 30},
		GenerationControls: GenerationControls{Temperature: 0, TopP: 1, TopK: 1, MaxOutputTokens: StrixComparisonOutputTokens, IgnoreEOS: true},
	}
	packet, err := FreezePromptPacket(packet)
	if err != nil {
		t.Fatal(err)
	}
	packetBytes, err := ExportPromptPacket(packet)
	if err != nil {
		t.Fatal(err)
	}
	pairs := make([]string, 0, StrixComparisonMinimumMeasuredPairs)
	order := make([]string, 0, 2*len(pairs))
	for i := range StrixComparisonMinimumMeasuredPairs {
		pairID, first, second, _, _ := strixComparisonPairOrder(i)
		pairs = append(pairs, pairID)
		order = append(order, first, second)
	}
	observations := make([]StrixComparisonObservation, len(strixComparisonPlatformObservationNames))
	for i, name := range strixComparisonPlatformObservationNames {
		value := "observed-" + name
		observations[i] = StrixComparisonObservation{Name: name, Value: value, ValueSHA256: fmt.Sprintf("%x", sha256.Sum256([]byte(value)))}
	}
	return StrixComparisonCellManifest{
		Schema: StrixComparisonCellManifestSchema,
		Challenge: StrixComparisonChallenge{
			SourceRowID: StrixComparisonSourceRowID, SourceDate: StrixComparisonSourceDate, SourceRevision: StrixComparisonSourceRevision,
			Concurrency: 1, AcceptedOutputTPS: 16.65, ConfidenceRule: "one-sided-95-percent",
			PairedRatioRule: "paired-ratio-lcb95>1", SamplingAssumption: "iid-approximately-normal-paired-ratios",
			Warmups: StrixComparisonWarmups, MeasuredPairs: len(pairs), AlternatingOrder: order, PairIDs: pairs,
		},
		Platform: StrixComparisonPlatform{
			ApplianceID: "strix1", CPU: "AMD Ryzen AI MAX+ 395", GPU: "Radeon 8060S", GPUArchitecture: "gfx1151",
			ComputeUnits: 40, UMAClassBytes: StrixComparisonUMAClassBytes, PhysicalRAMBytes: 64 << 30,
			LeaseIdentity: "lease-strix-cell-c1", Observations: observations,
		},
		Workload: StrixComparisonWorkload{
			Model: "Qwen3.8-27B", Quantization: "Q4_K_M", ArtifactSHA256: StrixComparisonArtifactSHA256,
			PromptPacketBytes: packetBytes, PromptPacketDigest: packet.PacketDigest,
			TokenizerSHA256: StrixComparisonTokenizerSHA256, TemplateSHA256: StrixComparisonTemplateSHA256,
			RenderedPromptSHA256: StrixComparisonRenderedPromptSHA256, PromptTokenIDs: slices.Clone(packet.PromptTokenIDs),
			ContextTokens: StrixComparisonContextTokens, AcceptedOutputTokens: StrixComparisonOutputTokens,
		},
		Memory: StrixComparisonMemoryEnvelope{
			ContextBudgetBytes: 48 << 30, KVTypeK: "f16", KVTypeV: "f16", KVOffload: "gpu",
			FlashAttention: true, GPUUMABudgetBytes: 56 << 30, CandidateBudgetBytes: 56 << 30, ReferenceBudgetBytes: 56 << 30, HostSpillPolicy: "forbid",
			PrimaryCacheState: "cold-no-prefix", MemoryAccountingPolicy: "uma-overlap-not-summed",
			ResidentModelBytes: 17_106_775_008, CandidatePeakMethod: "authoritative-peak-uma",
			ReferencePeakMethod: "authoritative-peak-uma",
		},
		Candidate: StrixComparisonCandidatePin{
			CampaignClass: "fak-native", Runtime: "native", Owner: "fak", Planner: "inkernel", Backend: "vulkan",
			SourceRevision: "internal/compute@r1+gabcdef0", SourceArchiveSHA256: testManifestHash(20),
			BuildManifestSHA256: testManifestHash(21), ToolchainSHA256: testManifestHash(22),
			ExecutableSHA256: testManifestHash(23), ShaderBundleSHA256: testManifestHash(24),
			ModelSHA256: StrixComparisonArtifactSHA256, ForwardPath: "modelbench/raw-decode/vulkan",
		},
		Reference: StrixComparisonReferencePin{
			CampaignClass: "llama.cpp-comparator-only", SourceRevision: StrixComparisonLlamaSourceRevision,
			SourceTreeSHA256: StrixComparisonLlamaTreeSHA256, BuildType: "Release", GGMLVulkan: true,
			SourceArchiveSHA256: testManifestHash(30), BuildManifestSHA256: testManifestHash(31),
			ToolchainSHA256: testManifestHash(32), ServerBinarySHA256: testManifestHash(33),
			BenchBinarySHA256: testManifestHash(34), LoaderSHA256: testManifestHash(35), DependencySHA256: testManifestHash(36),
			RADVDeviceIdentity: "radv-gfx1151-pci-observed", ModelSHA256: StrixComparisonArtifactSHA256,
			PromptPacketDigest: packet.PacketDigest,
		},
		Capture: StrixComparisonCapturePlan{
			CellNonce: "fresh-cell-nonce-c1", AuthoritySchema: "fak.qwen38.capture-authority.v1",
			ArmRoles: []string{"candidate", "reference"}, PairIDs: slices.Clone(pairs), AlternatingOrder: slices.Clone(order),
			TrialCount: 2 * len(pairs), MonotonicClockID: "clock-boottime-session-1", SessionIdentity: "boot-session-observed",
			ReplayKey: "replay-key-c1-session-1", ObservationBindings: slices.Clone(strixComparisonCaptureBindings),
		},
	}
}

func testManifestHash(n int) string { return fmt.Sprintf("%064x", n) }

func cloneStrixComparisonCellManifestForTest(t *testing.T, in StrixComparisonCellManifest) StrixComparisonCellManifest {
	t.Helper()
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var out StrixComparisonCellManifest
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}
