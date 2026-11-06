package sessionimage

import (
	"archive/tar"
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// fak-test:runtime fast est=25ms lane=default
func TestUnpackConfinesDestinationLinks(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name     string
		link     bool
		outside  bool
		absolute bool
	}{
		{name: "replace regular file"},
		{name: "relative link inside", link: true},
		{name: "relative link outside", link: true, outside: true},
		{name: "absolute link outside", link: true, outside: true, absolute: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := t.TempDir()
			dir := filepath.Join(base, "bundle")
			if err := os.Mkdir(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			destination := filepath.Join(dir, ImageFile)
			target := destination
			if tc.link {
				target = filepath.Join(dir, "target.json")
				if tc.outside {
					target = filepath.Join(base, "outside.json")
				}
			}
			const original = "keep existing contents"
			if err := os.WriteFile(target, []byte(original), 0o644); err != nil {
				t.Fatal(err)
			}
			if tc.link {
				linkTarget := target
				if !tc.absolute {
					var err error
					linkTarget, err = filepath.Rel(dir, target)
					if err != nil {
						t.Fatal(err)
					}
				}
				if err := os.Symlink(linkTarget, destination); err != nil {
					if runtime.GOOS == "windows" || runtime.GOOS == "plan9" {
						t.Skipf("symlink unavailable: %v", err)
					}
					t.Fatal(err)
				}
			}

			const payload = "replacement"
			var archive bytes.Buffer
			tw := tar.NewWriter(&archive)
			if err := tw.WriteHeader(&tar.Header{Name: ImageFile, Typeflag: tar.TypeReg, Mode: 0o644, Size: int64(len(payload))}); err != nil {
				t.Fatal(err)
			}
			if _, err := tw.Write([]byte(payload)); err != nil {
				t.Fatal(err)
			}
			if err := tw.Close(); err != nil {
				t.Fatal(err)
			}
			err := Unpack(&archive, dir)
			if (err != nil) != tc.outside {
				t.Errorf("Unpack error = %v, want refusal = %v", err, tc.outside)
			}
			got, err := os.ReadFile(target)
			if err != nil {
				t.Fatal(err)
			}
			want := payload
			if tc.outside {
				want = original
			}
			if string(got) != want {
				t.Errorf("target contents = %q, want %q", got, want)
			}
			if tc.link {
				info, err := os.Lstat(destination)
				if err != nil {
					t.Fatal(err)
				}
				if info.Mode()&os.ModeSymlink == 0 {
					t.Error("Unpack replaced the destination symlink")
				}
			}
		})
	}
}
