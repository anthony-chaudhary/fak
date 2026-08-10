package toolplugin

import "testing"

func TestResolvePinnedFailsClosed(t *testing.T) {
	p := BuiltinProfiles()[0]
	if _, err := ResolvePinned(p.ID, p.Version, p.Digest); err != nil {
		t.Fatal(err)
	}
	for _, tc := range [][3]string{{p.ID, p.Version, ""}, {"unknown", "1", "sha256:x"}, {p.ID, "2", p.Digest}} {
		if _, err := ResolvePinned(tc[0], tc[1], tc[2]); err == nil {
			t.Fatalf("accepted %+v", tc)
		}
	}
}
