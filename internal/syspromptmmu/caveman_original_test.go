package syspromptmmu

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

func TestCavemanOriginalPinnedArtifacts(t *testing.T) {
	tests := []struct{ path, want string }{
		{"third_party/caveman/SKILL.md", CavemanOriginalSourceDigest},
		{"third_party/caveman/LICENSE", CavemanOriginalLicenseDigest},
	}
	for _, tt := range tests {
		got, err := cavemanOriginalFiles.ReadFile(tt.path)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(got)
		if digest := hex.EncodeToString(sum[:]); digest != tt.want {
			t.Fatalf("%s digest = %s, want %s", tt.path, digest, tt.want)
		}
	}
	license, _ := cavemanOriginalFiles.ReadFile("third_party/caveman/LICENSE")
	if !bytes.Contains(license, []byte("MIT License")) || !bytes.Contains(license, []byte("Julius Brussee")) {
		t.Fatal("vendored license lost MIT grant or attribution")
	}
}

func TestCavemanOriginalProfilesAreDistinctAndAttributed(t *testing.T) {
	seen := map[string]bool{}
	for _, intensity := range []string{"low", "medium", "high"} {
		name := "caveman:original:" + intensity
		got := DescribeStyle(name)
		if !got.Known || !got.Applied || got.Style != name || got.Family != "caveman:original" || got.Intensity != intensity {
			t.Fatalf("DescribeStyle(%q) = %+v", name, got)
		}
		if got.SourceRevision != CavemanOriginalRevision || got.SourceDigest != CavemanOriginalSourceDigest || got.ActivationSource != StyleEnvVar || got.DisableCommand != CavemanOriginalDisableCommand {
			t.Fatalf("DescribeStyle(%q) provenance = %+v", name, got)
		}
		if !strings.Contains(got.Segment, "juliusbrussee/caveman@"+CavemanOriginalRevision) || !strings.Contains(got.Segment, "system safety, explicit user formatting instructions, and preservation rules") {
			t.Fatalf("DescribeStyle(%q) missing attribution or precedence envelope", name)
		}
		if seen[got.Witness] {
			t.Fatalf("%q reused another intensity witness", name)
		}
		seen[got.Witness] = true
		native := DescribeStyle("caveman:native:" + intensity)
		if got.Witness == native.Witness || got.Segment == native.Segment {
			t.Fatalf("original %s is indistinguishable from native", intensity)
		}
	}
}

func TestCavemanOriginalUnknownIntensityFailsClosed(t *testing.T) {
	for _, name := range []string{"caveman:original", "caveman:original:lite", "caveman:original:auto", "caveman:original:ultra"} {
		got := DescribeStyle(name)
		if got.Known || got.Applied || got.Style != StyleFull || got.Segment != "" {
			t.Fatalf("DescribeStyle(%q) = %+v; want fail-closed full", name, got)
		}
		if _, ok := StyleSegment(name); ok {
			t.Fatalf("StyleSegment(%q) unexpectedly emitted bytes", name)
		}
	}
}

func TestCavemanOriginalCapturedContextPlan(t *testing.T) {
	native := DescribeStyle("caveman:native:medium")
	original := DescribeStyle("caveman:original:medium")
	plan := func(r StyleReadout) string {
		return r.Style + "|" + r.Family + "|" + r.Intensity + "|" + r.Witness + "|" + r.SourceRevision + "|" + r.SourceDigest + "|" + r.DisableCommand
	}
	nativePlan, originalPlan := plan(native), plan(original)
	if nativePlan == originalPlan {
		t.Fatalf("captured plans collide: %q", nativePlan)
	}
	for _, want := range []string{"caveman:original:medium", CavemanOriginalRevision, CavemanOriginalSourceDigest, CavemanOriginalDisableCommand} {
		if !strings.Contains(originalPlan, want) {
			t.Fatalf("original context plan %q missing %q", originalPlan, want)
		}
	}
	if strings.Contains(nativePlan, CavemanOriginalRevision) || strings.Contains(nativePlan, CavemanOriginalSourceDigest) {
		t.Fatalf("native context plan claims original provenance: %q", nativePlan)
	}
}
