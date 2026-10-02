// Default suites: go test ./cmd/fak and go test ./... (also under -race).
// The production-loader qualification skips without FAK_TEST_F32_ARENA_ARTIFACT.
// Opt-in suite: FAK_TEST_F32_ARENA_ARTIFACT=<audited-gguf> go test ./cmd/fak
// -run '^TestServeF32ArenaArtifactQualification$' -count=1 -v -timeout=180s.
package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"runtime"
	"runtime/metrics"
	"sort"
	"testing"

	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

type serveF32Interval struct {
	offset int64
	nbytes int64
}

// These are slice and manifest quantities, not allocation-size or RSS estimates.
// A is repeated manifest coverage, which can include legitimate tensor aliases.
type serveF32ArenaAccounting struct {
	R     int64 `json:"R_raw_len_bytes"`
	C     int64 `json:"C_raw_cap_bytes"`
	S     int64 `json:"S_manifest_sum_bytes"`
	U     int64 `json:"U_manifest_union_bytes"`
	H     int64 `json:"H_unreferenced_len_bytes"`
	A     int64 `json:"A_repeated_coverage_bytes"`
	Slack int64 `json:"capacity_slack_bytes"`
}

func accountServeF32Intervals(rawLen, rawCap int64, intervals []serveF32Interval) (serveF32ArenaAccounting, error) {
	result := serveF32ArenaAccounting{R: rawLen, C: rawCap}
	if rawLen < 0 || rawCap < rawLen {
		return result, fmt.Errorf("invalid raw length/capacity: %d/%d", rawLen, rawCap)
	}
	ordered := append([]serveF32Interval(nil), intervals...)
	const maxInt64 = int64(^uint64(0) >> 1)
	for _, interval := range ordered {
		// Subtraction checks bounds before an endpoint addition can overflow.
		if interval.offset < 0 || interval.nbytes < 0 || interval.offset > rawLen || interval.nbytes > rawLen-interval.offset {
			return result, fmt.Errorf("manifest interval offset=%d nbytes=%d outside raw length=%d", interval.offset, interval.nbytes, rawLen)
		}
		if result.S > maxInt64-interval.nbytes {
			return result, fmt.Errorf("manifest byte sum overflows int64")
		}
		result.S += interval.nbytes
	}
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].offset != ordered[j].offset {
			return ordered[i].offset < ordered[j].offset
		}
		return ordered[i].nbytes < ordered[j].nbytes
	})
	var end int64
	for _, interval := range ordered {
		if interval.nbytes == 0 {
			continue
		}
		start := interval.offset
		if start < end {
			start = end
		}
		if nextEnd := interval.offset + interval.nbytes; nextEnd > start {
			result.U += nextEnd - start
			end = nextEnd
		}
	}
	result.H = result.R - result.U
	result.A = result.S - result.U
	result.Slack = result.C - result.R
	return result, nil
}

func inspectServeF32Arena(m *fakmodel.Model) (serveF32ArenaAccounting, int, bool, error) {
	value := reflect.ValueOf(m).Elem()
	raw := value.FieldByName("raw")
	manifest := value.FieldByName("manifest")
	if raw.Kind() != reflect.Slice || raw.Type().Elem().Kind() != reflect.Uint8 || manifest.Kind() != reflect.Map {
		return serveF32ArenaAccounting{}, 0, false, fmt.Errorf("model raw/manifest storage contract unavailable")
	}
	// Pointer is read without Interface or unsafe. Emit presence only, never an address.
	hasBacking := raw.Pointer() != 0
	if raw.Len() > 0 && !hasBacking {
		return serveF32ArenaAccounting{}, 0, false, fmt.Errorf("nonempty model raw slice has no backing pointer")
	}
	intervals := make([]serveF32Interval, 0, manifest.Len())
	iter := manifest.MapRange()
	for iter.Next() {
		meta := iter.Value()
		dtype := meta.FieldByName("Dtype")
		offset := meta.FieldByName("Offset")
		nbytes := meta.FieldByName("Nbytes")
		if dtype.Kind() != reflect.String || offset.Kind() != reflect.Int || nbytes.Kind() != reflect.Int {
			return serveF32ArenaAccounting{}, 0, hasBacking, fmt.Errorf("tensor metadata storage contract unavailable")
		}
		if dtype.String() != "f32" {
			return serveF32ArenaAccounting{}, 0, hasBacking, fmt.Errorf("raw tensor %q has dtype %q, want f32", iter.Key().String(), dtype.String())
		}
		intervals = append(intervals, serveF32Interval{offset: offset.Int(), nbytes: nbytes.Int()})
	}
	accounting, err := accountServeF32Intervals(int64(raw.Len()), int64(raw.Cap()), intervals)
	runtime.KeepAlive(m)
	return accounting, len(intervals), hasBacking, err
}

// fak-test:runtime fast est=1ms lane=default
func TestServeF32ArenaIntervalAccounting(t *testing.T) {
	for _, test := range []struct {
		name      string
		rawLen    int64
		rawCap    int64
		intervals []serveF32Interval
		want      serveF32ArenaAccounting
	}{
		{
			name: "overlap with leading and trailing gaps", rawLen: 64, rawCap: 80,
			intervals: []serveF32Interval{{32, 16}, {8, 16}, {16, 24}},
			want:      serveF32ArenaAccounting{R: 64, C: 80, S: 56, U: 40, H: 24, A: 16, Slack: 16},
		},
		{
			name: "adjacency gap and empty endpoint", rawLen: 32, rawCap: 40,
			intervals: []serveF32Interval{{24, 8}, {8, 8}, {32, 0}, {0, 8}},
			want:      serveF32ArenaAccounting{R: 32, C: 40, S: 24, U: 24, H: 8, Slack: 8},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := accountServeF32Intervals(test.rawLen, test.rawCap, test.intervals)
			if err != nil || got != test.want {
				t.Fatalf("accounting=%+v err=%v, want %+v", got, err, test.want)
			}
		})
	}
	for _, interval := range []serveF32Interval{{-1, 4}, {0, -1}, {9, 0}, {7, 2}, {int64(^uint64(0) >> 1), 1}} {
		if _, err := accountServeF32Intervals(8, 8, []serveF32Interval{interval}); err == nil {
			t.Fatalf("accepted interval outside raw length: %+v", interval)
		}
	}
}

// fak-test:runtime integration est=150s lane=optin
func TestServeF32ArenaArtifactQualification(t *testing.T) {
	path := os.Getenv("FAK_TEST_F32_ARENA_ARTIFACT")
	if path == "" {
		t.Skip("set FAK_TEST_F32_ARENA_ARTIFACT to the audited GGUF for production-loader qualification")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal("open opt-in F32 arena artifact failed")
	}
	t.Cleanup(func() { _ = f.Close() })
	stat, err := f.Stat()
	const expectedBytes = 532517120
	const expectedSHA256 = "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"
	if err != nil || stat.Size() != expectedBytes {
		t.Fatal("opt-in artifact does not match the audited byte length")
	}
	hash := sha256.New()
	if _, err := io.Copy(hash, f); err != nil {
		t.Fatal("hash opt-in F32 arena artifact failed")
	}
	artifactSHA256 := fmt.Sprintf("%x", hash.Sum(nil))
	if artifactSHA256 != expectedSHA256 {
		t.Fatalf("opt-in artifact SHA256=%s, want %s", artifactSHA256, expectedSHA256)
	}
	if err := f.Close(); err != nil {
		t.Fatal("close opt-in F32 arena artifact reader failed")
	}
	for _, key := range []string{"FAK_STREAM_Q4K", "FAK_METAL_STREAM_Q4K", "FAK_GGUF_MMAP"} {
		t.Setenv(key, "")
	}
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_Q4K_FREE_CPU", "0")
	original := serveMetalAvailable
	serveMetalAvailable = func() bool { return false }
	t.Cleanup(func() { serveMetalAvailable = original })
	m, q4k, profile, phase := loadServeInKernelModel(path, nil, false, 0, nil, 16, nil)
	if m == nil || !q4k || profile == nil || profile.Mode != "gguf-resident-q4k" || phase.Name != "model-load" {
		t.Fatal("artifact did not load through the resident CPU serve entrypoint")
	}
	t.Cleanup(func() { _ = m.CloseWeights() })
	accounting, tensors, hasBacking, err := inspectServeF32Arena(m)
	if err != nil {
		t.Fatal(err)
	}
	resident := m.ResidentReport()
	if resident.F32Bytes != accounting.S || resident.F32Tensors != tensors {
		t.Fatalf("raw manifest enumeration disagrees with logical resident report: sum=%d tensors=%d report=%d/%d", accounting.S, tensors, resident.F32Bytes, resident.F32Tensors)
	}
	// The full-GC snapshot includes the live model. Neither it nor raw.Cap measures
	// allocator rounding, process RSS, physical footprint, or the loading peak.
	samples := []metrics.Sample{{Name: "/memory/classes/heap/objects:bytes"}}
	runtime.GC()
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	metrics.Read(samples)
	runtime.KeepAlive(m)
	if samples[0].Value.Kind() != metrics.KindUint64 {
		t.Fatal("runtime heap objects metric unavailable")
	}
	receipt := struct {
		ArtifactSHA256 string                  `json:"artifact_sha256"`
		ArtifactBytes  int64                   `json:"artifact_bytes"`
		Mode           string                  `json:"production_load_mode"`
		Accounting     serveF32ArenaAccounting `json:"f32_arena"`
		F32Tensors     int                     `json:"f32_tensors"`
		HasBacking     bool                    `json:"raw_backing_pointer_present"`
		HeapObjects    uint64                  `json:"heap_objects_bytes_after_gc"`
		HeapAlloc      uint64                  `json:"heap_alloc_bytes_after_gc"`
	}{artifactSHA256, stat.Size(), profile.Mode, accounting, tensors, hasBacking, samples[0].Value.Uint64(), memory.HeapAlloc}
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("FAK_F32_ARENA_RECEIPT %s", encoded)
	runtime.KeepAlive(m)
}
