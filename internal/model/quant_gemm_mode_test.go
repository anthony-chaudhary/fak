package model

import "testing"

func TestResolveQGemmModeDefaultsArm64(t *testing.T) {
	want := qgemmModeTile
	if qgemmAccelDefault() {
		want = qgemmModeAccel
	}
	if got := resolveQGemmMode("", false, "arm64"); got != want {
		t.Fatalf("unset FAK_QGEMM on arm64 = %q, want %q", got, want)
	}
}

func TestResolveQGemmModeDefaultsNonArm64ToTile(t *testing.T) {
	for _, goarch := range []string{"amd64", "386", "wasm"} {
		if got := resolveQGemmMode("", false, goarch); got != qgemmModeTile {
			t.Fatalf("unset FAK_QGEMM on %s = %q, want %q", goarch, got, qgemmModeTile)
		}
	}
}

func TestResolveQGemmModeEnvOverridesArchDefault(t *testing.T) {
	tests := []struct {
		name   string
		env    string
		goarch string
		want   string
	}{
		{name: "force tile on arm64", env: qgemmModeTile, goarch: "arm64", want: qgemmModeTile},
		{name: "force legacy on amd64", env: qgemmModeLegacy, goarch: "amd64", want: qgemmModeLegacy},
		{name: "force accel on arm64", env: qgemmModeAccel, goarch: "arm64", want: qgemmModeAccel},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveQGemmMode(tt.env, true, tt.goarch); got != tt.want {
				t.Fatalf("resolveQGemmMode(%q, true, %q) = %q, want %q", tt.env, tt.goarch, got, tt.want)
			}
		})
	}
}
