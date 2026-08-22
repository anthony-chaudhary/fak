package macmetalwitness_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
)

var pinnedFiles = map[string]string{
	"archive.json":         "ecf32c1f01e64f838b55cdc139ecef13138b840b47f41afa300918cdbae57905",
	"platform.json":        "97c6c50b766f753fa3bd7f7a41bc930f3d48d58e714fc12d170d3e7d491fe8ac",
	"campaign-report.json": "58638d810d4d849f279121034cabd7e5b7c0c089638d0952210fe44096248d75",
	"summary.json":         "88e20a8df3abf817ef92505c700626003cd674669d9757d44e9df2f28b04fefe",
}

func TestMacMetalCampaignWitness(t *testing.T) {
	for name, want := range pinnedFiles {
		body, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(body)
		if got := hex.EncodeToString(sum[:]); got != want {
			t.Fatalf("%s hash=%s want=%s", name, got, want)
		}
	}
	var report struct {
		Verdict string `json:"Verdict"`
		Trials  []struct {
			Quality string `json:"quality"`
		} `json:"trials"`
		RawArchiveSHA256 string `json:"raw_archive_sha256"`
	}
	body, err := os.ReadFile("campaign-report.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(body, &report); err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "PROMOTE" || len(report.Trials) != 18 {
		t.Fatalf("verdict=%q trials=%d", report.Verdict, len(report.Trials))
	}
	for i, trial := range report.Trials {
		if trial.Quality != "PASS" {
			t.Fatalf("trial %d quality=%q", i, trial.Quality)
		}
	}
	if report.RawArchiveSHA256 != pinnedFiles["archive.json"] {
		t.Fatalf("raw archive hash=%q", report.RawArchiveSHA256)
	}
}
