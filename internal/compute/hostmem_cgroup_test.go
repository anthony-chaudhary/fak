package compute

import (
	"testing"
	"testing/fstest"
)

const gib = int64(1) << 30

func file(s string) *fstest.MapFile { return &fstest.MapFile{Data: []byte(s)} }

// TestCgroupMemoryAvailable pins the cgroup clamp on the host-memory fit check: inside a
// memory-limited cgroup the allocatable host bytes are limit - usage, not /proc/meminfo
// MemAvailable (which reports the whole box and over-admits the CPU-offload fit).
func TestCgroupMemoryAvailable(t *testing.T) {
	cases := []struct {
		name        string
		fs          fstest.MapFS
		wantAvail   int64
		wantLimited bool
	}{
		{
			name: "v2 limited",
			fs: fstest.MapFS{
				"proc/self/cgroup":                   file("0::/job\n"),
				"sys/fs/cgroup/job/memory.max":       file("8589934592\n"),
				"sys/fs/cgroup/job/memory.current":   file("2147483648\n"),
				"sys/fs/cgroup/unrelated/memory.max": file("1\n"),
			},
			wantAvail: 6 * gib, wantLimited: true,
		},
		{
			name: "v2 max is unlimited",
			fs: fstest.MapFS{
				"proc/self/cgroup":                 file("0::/job\n"),
				"sys/fs/cgroup/job/memory.max":     file("max\n"),
				"sys/fs/cgroup/job/memory.current": file("2147483648\n"),
			},
			wantAvail: 0, wantLimited: false,
		},
		{
			name: "v2 nested ancestor limit is tighter",
			fs: fstest.MapFS{
				"proc/self/cgroup":                       file("0::/slice/job\n"),
				"sys/fs/cgroup/slice/memory.max":         file("4294967296\n"),
				"sys/fs/cgroup/slice/memory.current":     file("3221225472\n"),
				"sys/fs/cgroup/slice/job/memory.max":     file("max\n"),
				"sys/fs/cgroup/slice/job/memory.current": file("1073741824\n"),
			},
			wantAvail: 1 * gib, wantLimited: true,
		},
		{
			name: "v2 namespaced path falls back to mount root",
			fs: fstest.MapFS{
				"proc/self/cgroup":             file("0::/host/only/path\n"),
				"sys/fs/cgroup/memory.max":     file("3221225472\n"),
				"sys/fs/cgroup/memory.current": file("1073741824\n"),
			},
			wantAvail: 2 * gib, wantLimited: true,
		},
		{
			name: "v2 usage over limit clamps to zero",
			fs: fstest.MapFS{
				"proc/self/cgroup":                 file("0::/job\n"),
				"sys/fs/cgroup/job/memory.max":     file("1073741824\n"),
				"sys/fs/cgroup/job/memory.current": file("2147483648\n"),
			},
			wantAvail: 0, wantLimited: true,
		},
		{
			name: "v1 limited",
			fs: fstest.MapFS{
				"proc/self/cgroup": file("12:cpu,cpuacct:/job\n4:memory:/job\n"),
				"sys/fs/cgroup/memory/job/memory.limit_in_bytes": file("4294967296\n"),
				"sys/fs/cgroup/memory/job/memory.usage_in_bytes": file("1073741824\n"),
			},
			wantAvail: 3 * gib, wantLimited: true,
		},
		{
			name: "v1 page-rounded max int64 is unlimited",
			fs: fstest.MapFS{
				"proc/self/cgroup": file("4:memory:/job\n"),
				"sys/fs/cgroup/memory/job/memory.limit_in_bytes": file("9223372036854771712\n"),
				"sys/fs/cgroup/memory/job/memory.usage_in_bytes": file("1073741824\n"),
			},
			wantAvail: 0, wantLimited: false,
		},
		{
			name:      "no cgroup info",
			fs:        fstest.MapFS{},
			wantAvail: 0, wantLimited: false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, limited := cgroupMemoryAvailable(c.fs)
			if got != c.wantAvail || limited != c.wantLimited {
				t.Fatalf("cgroupMemoryAvailable = (%d, %v), want (%d, %v)", got, limited, c.wantAvail, c.wantLimited)
			}
		})
	}
}

// TestCgroupMemoryAvailableClampsHostFree pins the composition hostSystemMemory uses: the
// cgroup figure only ever lowers the meminfo figure, never raises it.
func TestCgroupMemoryAvailableClampsHostFree(t *testing.T) {
	limited := fstest.MapFS{
		"proc/self/cgroup":                 file("0::/job\n"),
		"sys/fs/cgroup/job/memory.max":     file("8589934592\n"),
		"sys/fs/cgroup/job/memory.current": file("0\n"),
	}
	if got := clampFreeToCgroup(28*gib, limited); got != 8*gib {
		t.Fatalf("clamp(28GiB) = %d, want %d", got, 8*gib)
	}
	if got := clampFreeToCgroup(4*gib, limited); got != 4*gib {
		t.Fatalf("clamp(4GiB) = %d, want %d (cgroup must not raise free)", got, 4*gib)
	}
	if got := clampFreeToCgroup(28*gib, fstest.MapFS{}); got != 28*gib {
		t.Fatalf("clamp with no cgroup = %d, want unchanged %d", got, 28*gib)
	}
}
