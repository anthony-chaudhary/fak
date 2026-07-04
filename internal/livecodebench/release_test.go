package livecodebench

import (
	"reflect"
	"testing"
)

func TestResolveReleaseSingle(t *testing.T) {
	got, err := ResolveRelease("release_v2")
	if err != nil {
		t.Fatalf("ResolveRelease(release_v2): unexpected error %v", err)
	}
	want := ReleaseSelection{Selector: "release_v2", Releases: []string{"release_v2"}, Resolved: "release_v2"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ResolveRelease(release_v2) = %+v, want %+v", got, want)
	}
}

// The default is upstream's release_latest and it is recorded explicitly: an
// empty selector and the alias both resolve to the newest known release, and the
// Selector field never comes back as "" (an implicit default a reader can't pin).
func TestResolveReleaseLatestDefaultIsExplicit(t *testing.T) {
	for _, sel := range []string{"", "release_latest"} {
		got, err := ResolveRelease(sel)
		if err != nil {
			t.Fatalf("ResolveRelease(%q): unexpected error %v", sel, err)
		}
		if got.Selector != ReleaseLatest {
			t.Errorf("ResolveRelease(%q).Selector = %q, want %q (recorded explicitly)", sel, got.Selector, ReleaseLatest)
		}
		if got.Resolved != LatestRelease() {
			t.Errorf("ResolveRelease(%q).Resolved = %q, want %q", sel, got.Resolved, LatestRelease())
		}
		if len(got.Releases) != 1 || got.Releases[0] != LatestRelease() {
			t.Errorf("ResolveRelease(%q).Releases = %v, want [%s]", sel, got.Releases, LatestRelease())
		}
	}
}

func TestResolveReleaseRangeV1V3(t *testing.T) {
	got, err := ResolveRelease("v1_v3")
	if err != nil {
		t.Fatalf("ResolveRelease(v1_v3): unexpected error %v", err)
	}
	want := ReleaseSelection{
		Selector: "v1_v3",
		Releases: []string{"release_v1", "release_v2", "release_v3"},
		Resolved: "release_v3",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ResolveRelease(v1_v3) = %+v, want %+v", got, want)
	}
}

// A single release with the full "release_v" prefix also carries an underscore,
// so the range parser must not mistake "release_v2" for the range "release_v2".
func TestResolveReleaseSinglePrefixNotRange(t *testing.T) {
	got, err := ResolveRelease("release_v4")
	if err != nil {
		t.Fatalf("ResolveRelease(release_v4): unexpected error %v", err)
	}
	if len(got.Releases) != 1 || got.Releases[0] != "release_v4" {
		t.Fatalf("ResolveRelease(release_v4).Releases = %v, want [release_v4]", got.Releases)
	}
}

func TestResolveReleaseErrors(t *testing.T) {
	for _, sel := range []string{"release_v9", "v3_v1", "banana", "v0_v2", "v1_v9", "v1_v2_v3"} {
		if _, err := ResolveRelease(sel); err == nil {
			t.Errorf("ResolveRelease(%q): want a clear error, got nil", sel)
		}
	}
}

// The report header pins the resolved release alongside the problem count so the
// two can never drift; the acceptance calls out the v1_v3 range specifically.
func TestPinReleaseHeaderV1V3(t *testing.T) {
	hdr, err := PinRelease("v1_v3", 42)
	if err != nil {
		t.Fatalf("PinRelease(v1_v3, 42): unexpected error %v", err)
	}
	if hdr.Problems != 42 {
		t.Errorf("PinRelease count = %d, want 42", hdr.Problems)
	}
	if hdr.Selection.Resolved != "release_v3" || len(hdr.Selection.Releases) != 3 {
		t.Errorf("PinRelease selection = %+v, want v1..v3 resolved to release_v3", hdr.Selection)
	}
}

func TestPinReleaseRejectsUnknown(t *testing.T) {
	if _, err := PinRelease("release_v42", 1); err == nil {
		t.Errorf("PinRelease(release_v42): want error, got nil")
	}
}
