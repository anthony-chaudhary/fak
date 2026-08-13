package memview

import (
	"strings"
	"testing"
)

func TestSkillManifestBudgetAdmissionAndInvalidation(t *testing.T) {
	entries := []SkillManifestEntry{
		{Name: "best", Version: "r3", Provenance: "rsiloop:keep-a", Value: 9, Active: true, Witnessed: true, Admitted: true},
		{Name: "low", Version: "r1", Provenance: "rsiloop:keep-b-with-a-long-witness-digest-that-costs-budget", Value: 1, Active: true, Witnessed: true, Admitted: true},
		{Name: "quarantined-secret", Version: "r8", Provenance: "rsiloop:q", Value: 99, Active: true, Witnessed: true, Admitted: true, Quarantined: true},
		{Name: "unwitnessed", Version: "r2", Provenance: "skillenv", Value: 8, Active: true, Admitted: true},
	}
	full, err := BuildSkillManifest(entries, FormatJSON, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if full.Included != 2 || full.Dropped != 0 {
		t.Fatalf("full counts = %d/%d", full.Included, full.Dropped)
	}
	if strings.Contains(string(full.Bytes), "quarantined-secret") || strings.Contains(string(full.Bytes), "unwitnessed") {
		t.Fatalf("inadmissible skill leaked: %s", full.Bytes)
	}
	if full.SourceDigest == "" || full.ContentDigest != Digest(full.Bytes) || !full.IsValid(entries) {
		t.Fatal("missing provenance/digest binding")
	}
	changed := append([]SkillManifestEntry(nil), entries...)
	changed[0].Version = "r4"
	if full.IsValid(changed) {
		t.Fatal("source change did not invalidate derived view")
	}
	one, err := BuildSkillManifest(entries, FormatJSON, len(full.Bytes)-1)
	if err != nil {
		t.Fatal(err)
	}
	if one.Dropped == 0 || !strings.Contains(string(one.Bytes), "OVERFLOW") || !strings.Contains(string(one.Bytes), "best") || strings.Contains(string(one.Bytes), `"low"`) {
		t.Fatalf("bad budget projection: %s", one.Bytes)
	}
}

func TestSkillManifestUsesEncoderTable(t *testing.T) {
	entries := []SkillManifestEntry{{Name: "repair", Version: "r1", Provenance: "keep:abc", Value: 2, Active: true, Witnessed: true, Admitted: true}}
	for _, f := range []Format{FormatMarkdown, FormatJSON, FormatTOON} {
		got, err := BuildSkillManifest(entries, f, 4096)
		if err != nil {
			t.Fatal(err)
		}
		direct, err := Encode(f, got.Surface)
		if err != nil {
			t.Fatal(err)
		}
		if string(direct) != string(got.Bytes) {
			t.Fatalf("%s bypassed encoder", f)
		}
	}
	manifest, err := BuildSkillManifest(entries, FormatJSON, 4096)
	if err != nil {
		t.Fatal(err)
	}
	measurements, err := SweepFormats(manifest.Surface, nil)
	if err != nil || len(measurements) != len(KnownFormats()) {
		t.Fatalf("sweep skill-manifest: measurements=%d err=%v", len(measurements), err)
	}
}
