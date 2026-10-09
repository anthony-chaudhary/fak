package compute

import (
	"errors"
	"path/filepath"
	"testing"
)

func fakeDRM(files map[string]string) (func(string) ([]string, error), func(string) ([]byte, error)) {
	cards := map[string]bool{}
	for p := range files {
		cards[filepath.Dir(filepath.Dir(p))] = true
	}
	glob := func(string) ([]string, error) {
		out := make([]string, 0, len(cards))
		for c := range cards {
			out = append(out, c)
		}
		return out, nil
	}
	read := func(p string) ([]byte, error) {
		if v, ok := files[p]; ok {
			return []byte(v), nil
		}
		return nil, errors.New("absent")
	}
	return glob, read
}

func TestDedicatedVRAMCarveoutFromSysfs(t *testing.T) {
	root := filepath.Join("sys", "class", "drm")
	card := func(name, file string) string { return filepath.Join(root, name, "device", file) }
	cases := map[string]struct {
		files     map[string]string
		want      int64
		wantKnown bool
	}{
		"one-amd-card": {files: map[string]string{
			card("card1", "vendor"): "0x1002\n", card("card1", "mem_info_vram_total"): "103079215104\n",
			card("card1-DP-1", "vendor"): "0x1002\n",
		}, want: 103079215104, wantKnown: true},
		"two-amd-cards": {files: map[string]string{
			card("card0", "vendor"): "0x1002", card("card0", "mem_info_vram_total"): "1",
			card("card1", "vendor"): "0x1002", card("card1", "mem_info_vram_total"): "2",
		}},
		"non-amd":     {files: map[string]string{card("card0", "vendor"): "0x10de", card("card0", "mem_info_vram_total"): "8"}},
		"unreadable":  {files: map[string]string{card("card0", "vendor"): "0x1002"}},
		"nonpositive": {files: map[string]string{card("card0", "vendor"): "0x1002", card("card0", "mem_info_vram_total"): "0"}},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			glob, read := fakeDRM(c.files)
			got, known := dedicatedVRAMCarveoutFromSysfs(root, glob, read)
			if got != c.want || known != c.wantKnown {
				t.Fatalf("got (%d, %v), want (%d, %v)", got, known, c.want, c.wantKnown)
			}
		})
	}
}
