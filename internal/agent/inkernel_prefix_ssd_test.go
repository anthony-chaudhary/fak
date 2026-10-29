package agent

import (
	"bytes"
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/l3kv"
)

func ssdEnv(vals map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) {
		v, ok := vals[k]
		return v, ok
	}
}

func ssdDisk(free int64, known bool) func(string) (int64, int64, bool) {
	return func(string) (int64, int64, bool) { return free * 2, free, known }
}

// fak-test:runtime fast est=1ms lane=default
func TestRadixSSDSpillBudgetResolution(t *testing.T) {
	t.Setenv(l3kv.EnvSpec, t.TempDir())
	const gib = int64(1) << 30
	cases := []struct {
		name   string
		env    map[string]string
		free   int64
		known  bool
		want   int64
		source string
		err    error
	}{
		{name: "default 25% of free", free: 100 * gib, known: true, want: 25 * gib, source: InKernelRadixLocalSSDSourceDefault},
		{name: "default capped at 64 GiB", free: 1024 * gib, known: true, want: 64 * gib, source: InKernelRadixLocalSSDSourceDefault},
		{name: "unknown disk stays off", known: false, want: 0, source: InKernelRadixLocalSSDSourceUnknown},
		{name: "explicit 0 is the OFF arm", env: map[string]string{inKernelRadixLocalSSDBytesEnv: "0"}, free: 100 * gib, known: true, want: 0, source: InKernelRadixLocalSSDSourceEnv},
		{name: "explicit bytes honoured", env: map[string]string{inKernelRadixLocalSSDBytesEnv: "12345"}, free: 100 * gib, known: true, want: 12345, source: InKernelRadixLocalSSDSourceEnv},
		{name: "garbage refuses to off", env: map[string]string{inKernelRadixLocalSSDBytesEnv: "lots"}, free: 100 * gib, known: true, want: 0, source: InKernelRadixLocalSSDSourceRefused, err: ErrInKernelRadixLocalSSDConfig},
		{name: "negative refuses to off", env: map[string]string{inKernelRadixLocalSSDBytesEnv: "-1"}, free: 100 * gib, known: true, want: 0, source: InKernelRadixLocalSSDSourceRefused, err: ErrInKernelRadixLocalSSDConfig},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := resolveInKernelRadixLocalSSD(ssdEnv(tc.env), ssdDisk(tc.free, tc.known))
			if got.Bytes != tc.want || got.Source != tc.source || !errors.Is(got.Err, tc.err) {
				t.Fatalf("resolve = %+v, want bytes=%d source=%q err=%v", got, tc.want, tc.source, tc.err)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestRadixSSDSpillDirFollowsL3Root(t *testing.T) {
	root := t.TempDir()
	t.Setenv(l3kv.EnvSpec, root)
	if got, want := localSSDDir(), filepath.Join(root, "ssd-l3"); got != want {
		t.Fatalf("localSSDDir = %q, want %q (sibling of the warm-disk slot)", got, want)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRadixSSDSpillStoreRoundTripsAndPrunesToBudget(t *testing.T) {
	disk, err := l3kv.NewDiskStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	payload := bytes.Repeat([]byte{0xA5}, 1024)
	store := newLocalSSDSnapshotStore(disk, int64(len(payload))+512)
	ctx := context.Background()
	first := cachemeta.DigestBytes([]byte("first"))
	second := cachemeta.DigestBytes([]byte("second"))
	if err := store.Put(ctx, first, payload); err != nil {
		t.Fatal(err)
	}
	got, found, err := store.Get(ctx, first)
	if err != nil || !found || !bytes.Equal(got, payload) {
		t.Fatalf("round trip found=%v err=%v equal=%v", found, err, bytes.Equal(got, payload))
	}
	if err := store.Put(ctx, second, payload); err != nil {
		t.Fatal(err)
	}
	if _, found, err := store.Get(ctx, second); err != nil || !found {
		t.Fatalf("newest record pruned found=%v err=%v, want kept", found, err)
	}
	if _, found, _ := store.Get(ctx, first); found {
		t.Fatal("older record survived past the byte budget, want pruned")
	}
}
