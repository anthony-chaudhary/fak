package newmodel

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestPinnedReleaseReplayLedger(t *testing.T) {
	corpusRaw := replayFixture(t, "corpus.json")
	corpus, err := ParseReplayCorpus(corpusRaw)
	if err != nil {
		t.Fatal(err)
	}
	ledger, err := Replay(corpus)
	if err != nil {
		t.Fatal(err)
	}
	again, err := Replay(corpus)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := MarshalReplayLedger(ledger)
	if err != nil {
		t.Fatal(err)
	}
	encodedAgain, err := MarshalReplayLedger(again)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded, encodedAgain) {
		t.Fatal("same offline corpus produced different ledger bytes")
	}
	if want := replayFixture(t, "ledger.json"); !bytes.Equal(encoded, want) {
		t.Fatalf("committed ledger drifted; got:\n%s", encoded)
	}
	if ledger.Summary.Manifests != len(corpus.Cases) || ledger.Summary.ArchitectureFamilies < 3 || !ledger.Summary.ByteIdentical {
		t.Fatalf("replay summary does not cover the required corpus: %+v", ledger.Summary)
	}
	if ledger.Summary.Packets == 0 || ledger.Summary.Refusals == 0 {
		t.Fatalf("ledger must preserve both packets and typed refusals: %+v", ledger.Summary)
	}

	wantPins := map[string]string{
		"Qwen/Qwen3.6-27B":                   "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
		"Qwen/Qwen3.6-35B-A3B":               "995ad96eacd98c81ed38be0c5b274b04031597b0",
		"Qwen/Qwen3.5-397B-A17B":             "8472618112abcbd45acbcdc58436aff4233c23f7",
		"Qwen/Qwen3.5-9B":                    "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
		"deepseek-ai/DeepSeek-V3.2":          "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6",
		"deepseek-ai/DeepSeek-V3.1-Terminus": "19510d6dc61f79dbd925bd51ee8a9081c509a4b6",
		"deepseek-ai/DeepSeek-V3-0324":       "e9b33add76883f293d6bf61f6bd89b497e80e335",
		"zai-org/GLM-5.3-Flash":              "04c4e9e95c5da8862dced7e5056455116f83a7e0",
		"zai-org/GLM-5.2":                    "b4734de4facf877f85769a911abafc5283eab3d9",
		"zai-org/GLM-5.1":                    "26e1bd6e011feb778d25ae34b09b07074139d92d",
	}
	seen := map[string]bool{}
	for _, row := range ledger.Rows {
		if wantPins[row.Repository] != row.Revision {
			t.Fatalf("unwitnessed source pin %s@%s", row.Repository, row.Revision)
		}
		seen[row.Repository] = true
		if len(row.SourceConfigSHA256) != 64 || len(row.OutcomeSHA256) != 64 || !row.ByteIdentical { //boundarylint:ignore CHANGE_DETECTOR_TEST both fields are SHA-256 hex digests, whose algorithm-defined width is exactly 64 characters
			t.Fatalf("row lost config/outcome digest: %+v", row)
		}
		if row.Execution != "not-run" || row.ModelExecuted || row.SupportClaim || row.PerformanceClaim {
			t.Fatalf("row crossed the evidence boundary: %+v", row)
		}
		manifest := replayManifestForCase(t, corpus, row.ID)
		if row.LicenseDisposition != ReplayLicenseDisposition || !rowCoversManifestObligations(row, manifest) || len(row.SemanticGaps) == 0 {
			t.Fatalf("row lost license, obligation, or gap evidence: %+v", row)
		}
		if strings.Contains(row.Repository, "Qwen3.6") && (row.CompatibilityException == "" || row.ManualCorrections != 1) {
			t.Fatalf("Qwen3.6 compatibility exception was hidden: %+v", row)
		}
	}
	if len(seen) != len(wantPins) {
		t.Fatalf("source repositories = %d, want %d", len(seen), len(wantPins))
	}
}

func TestReplayRecordsTypedSemanticRefusal(t *testing.T) {
	corpus, err := ParseReplayCorpus(replayFixture(t, "corpus.json"))
	if err != nil {
		t.Fatal(err)
	}
	ledger, err := Replay(corpus)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range ledger.Rows {
		if row.ID != "deepseek-v32" {
			continue
		}
		if row.Outcome != "refusal" || row.Reason != RefusalUnknownSemanticDelta || row.Axis != "attention" {
			t.Fatalf("typed refusal was not retained: %+v", row)
		}
		manifest := replayManifestForCase(t, corpus, row.ID)
		if !contains(row.SemanticGaps, "attention:deepseek-sparse-attention") || !rowCoversManifestObligations(row, manifest) {
			t.Fatalf("refusal lost semantic gap or open obligations: %+v", row)
		}
		return
	}
	t.Fatal("deepseek-v32 replay row missing")
}

func TestReplayRejectsDishonestPinOrExecutionArtifact(t *testing.T) {
	corpus, err := ParseReplayCorpus(replayFixture(t, "corpus.json"))
	if err != nil {
		t.Fatal(err)
	}
	corpus.Cases[0].Revision = strings.Repeat("0", 40)
	if _, err := Replay(corpus); err == nil {
		t.Fatal("corpus/manifest revision disagreement admitted")
	}
	corpus, _ = ParseReplayCorpus(replayFixture(t, "corpus.json"))
	var manifest ReleaseManifest
	if err := json.Unmarshal(corpus.Cases[0].Manifest, &manifest); err != nil {
		t.Fatal(err)
	}
	manifest.Artifact.URI = "https://huggingface.co/weights.safetensors"
	corpus.Cases[0].Manifest, _ = json.Marshal(manifest)
	if _, err := Replay(corpus); err == nil {
		t.Fatal("runnable artifact URI crossed the replay fence")
	}
}

func replayFixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile("testdata/replay/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func replayManifestForCase(t *testing.T, corpus ReplayCorpus, id string) ReleaseManifest {
	t.Helper()
	for _, replayCase := range corpus.Cases {
		if replayCase.ID != id {
			continue
		}
		var manifest ReleaseManifest
		if err := json.Unmarshal(replayCase.Manifest, &manifest); err != nil {
			t.Fatalf("parse replay manifest %q: %v", id, err)
		}
		normalizeManifest(&manifest)
		return manifest
	}
	t.Fatalf("replay corpus missing case %q", id)
	return ReleaseManifest{}
}

func rowCoversManifestObligations(row ReplayRow, manifest ReleaseManifest) bool {
	if len(row.Obligations) != len(manifest.Obligations) {
		return false
	}
	for _, obligation := range manifest.Obligations {
		if !contains(row.Obligations, obligation.Kind+":"+obligation.ID) {
			return false
		}
	}
	return true
}
