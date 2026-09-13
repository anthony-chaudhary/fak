package model

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// v4_flash_grouped_wo_test.go — witnesses for the grouped-geometry derivation of
// DeepSeek V4 Flash's `attn.wo_a.weight`. The official pipinned config admits
// o_groups=8 and o_lora_rank=1024; the published flat weight is [8192, 4096],
// whose grouped view is [8, 1024, 4096].

const flashGroupedWoName = "layers.0.attn.wo_a.weight"

func TestV4GroupedWoAOfficialShapeYieldsGroupedGeometry(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	got, err := v4GroupedWoAFromShape(cfg, 8192, 4096)
	if err != nil {
		t.Fatalf("official wo_a [8192,4096] refused: %v", err)
	}
	if got.Groups != 8 || got.GroupRows != 1024 || got.Hidden != 4096 {
		t.Fatalf("official wo_a geometry = %+v, want {8 1024 4096}", got)
	}
	for _, dim := range []struct {
		name  string
		value int
	}{
		{"groups", got.Groups}, {"group_rows", got.GroupRows}, {"hidden", got.Hidden},
	} {
		if dim.value <= 0 {
			t.Fatalf("official wo_a %s = %d, want positive", dim.name, dim.value)
		}
	}
}

func TestV4GroupedWoARefusesInconsistentShapes(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	cases := []struct {
		name string
		rows int
		cols int
	}{
		{"one_column_short", 8192, 4095},
		{"pro_hidden_rows", 7168, 4096},
		{"rows_not_groups_times_rank", 8191, 4096},
		{"zero_rows", 0, 4096},
		{"zero_cols", 8192, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := v4GroupedWoAFromShape(cfg, tc.rows, tc.cols)
			if !errors.Is(err, ErrV4GroupedWoAGeometry) {
				t.Fatalf("wo_a [%d,%d] err = %v, want ErrV4GroupedWoAGeometry", tc.rows, tc.cols, err)
			}
		})
	}
}

func TestV4GroupedWoARefusesConfigWithoutBlockAlignedRank(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	cases := map[string]func(Config) Config{
		"rank_not_divisible_by_qblk": func(c Config) Config { c.OLoraRank = 1000; return c },
		"rank_one_over_block":        func(c Config) Config { c.OLoraRank = 1025; return c },
		"zero_groups":                func(c Config) Config { c.OGroups = 0; return c },
		"zero_rank":                  func(c Config) Config { c.OLoraRank = 0; return c },
		"negative_groups":            func(c Config) Config { c.OGroups = -1; return c },
		"negative_rank":              func(c Config) Config { c.OLoraRank = -32; return c },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			bad := mutate(cfg)
			if _, err := v4GroupedWoAFromShape(bad, 8192, 4096); !errors.Is(err, ErrV4GroupedWoAGeometry) {
				t.Fatalf("cfg o_groups=%d o_lora_rank=%d err = %v, want ErrV4GroupedWoAGeometry", bad.OGroups, bad.OLoraRank, err)
			}
		})
	}
}

func TestV4GroupedWoARefusesBeforeAnyWeightRead(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	for _, malformed := range []struct {
		name string
		rows int
	}{
		{"rows_too_few", 7168},
		{"rows_too_many", 8191},
	} {
		t.Run(malformed.name, func(t *testing.T) {
			scaleName := strings.TrimSuffix(flashGroupedWoName, ".weight") + ".scale"
			hdr := map[string]json.RawMessage{
				flashGroupedWoName: flashDenseRawEntry(t, stEntry{Dtype: "F8_E4M3", Shape: []int{malformed.rows, 4096}}),
				scaleName:          flashDenseRawEntry(t, stEntry{Dtype: "F8_E8M0", Shape: []int{(malformed.rows-1)/fp8BlockDim + 1, 32}}),
			}
			readerSentinel := errors.New("tensor reader reached")
			reads := 0
			m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
			var raw []byte
			off := 0
			err := quantizeNamedTensorsInto([]string{scaleName, flashGroupedWoName}, hdr, func(stEntry) ([]byte, error) {
				reads++
				return nil, readerSentinel
			}, nil, m, false, &raw, &off)
			if !errors.Is(err, ErrV4GroupedWoAGeometry) {
				t.Fatalf("malformed wo_a rows=%d error = %v, want ErrV4GroupedWoAGeometry", malformed.rows, err)
			}
			if reads != 0 {
				t.Fatalf("malformed wo_a rows=%d read %d payloads, want 0 (refused before weight read)", malformed.rows, reads)
			}
			if m.q8w[flashGroupedWoName] != nil {
				t.Fatalf("malformed wo_a was installed in the Q8 store despite the geometry refusal")
			}
		})
	}
}

func TestV4GroupedWoAOfficialPairStillLoadsThroughPairValidation(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	scaleName := strings.TrimSuffix(flashGroupedWoName, ".weight") + ".scale"
	hdr := map[string]json.RawMessage{
		flashGroupedWoName: flashDenseRawEntry(t, stEntry{Dtype: "F8_E4M3", Shape: []int{8192, 4096}}),
		scaleName:          flashDenseRawEntry(t, stEntry{Dtype: "F8_E8M0", Shape: []int{64, 32}}),
	}
	readerSentinel := errors.New("tensor reader reached")
	reads := 0
	m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
	var raw []byte
	off := 0
	err := quantizeNamedTensorsInto([]string{scaleName, flashGroupedWoName}, hdr, func(stEntry) ([]byte, error) {
		reads++
		return nil, readerSentinel
	}, nil, m, false, &raw, &off)
	if !errors.Is(err, readerSentinel) || reads != 1 {
		t.Fatalf("valid official wo_a metadata error=%v reads=%d, want one sentinel read", err, reads)
	}
}

func TestV4GroupedWoAExtractsGroupRowsDeterministically(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	geo, err := v4GroupedWoAFromShape(cfg, 8192, 4096)
	if err != nil {
		t.Fatal(err)
	}
	flat := make([]float32, geo.flatRows()*geo.Hidden)
	for i := range flat {
		flat[i] = float32(i)
	}
	for group := 0; group < geo.Groups; group++ {
		got := geo.extractGroup(flat, group)
		wantLen := geo.GroupRows * geo.Hidden
		if len(got) != wantLen {
			t.Fatalf("group %d len = %d, want %d", group, len(got), wantLen)
		}
		start := group * geo.GroupRows * geo.Hidden
		for i := 0; i < wantLen; i++ {
			if got[i] != float32(start+i) {
				t.Fatalf("group %d value[%d] = %v, want %v (row-major offset %d)", group, i, got[i], float32(start+i), start)
			}
		}
		// The sub-slice must alias the flat payload, never copy it.
		if len(got) > 0 && &got[0] != &flat[start] {
			t.Fatalf("group %d did not alias flat[%d]", group, start)
		}
	}
}

func TestV4GroupedWoAExtractGroupPanicsOutOfRange(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	geo, err := v4GroupedWoAFromShape(cfg, 8192, 4096)
	if err != nil {
		t.Fatal(err)
	}
	flat := make([]float32, geo.flatRows()*geo.Hidden)
	for _, tc := range []struct {
		name  string
		group int
	}{
		{"negative", -1},
		{"at_groups", geo.Groups},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatalf("extractGroup(%d) did not panic", tc.group)
				}
			}()
			geo.extractGroup(flat, tc.group)
		})
	}
	t.Run("short_flat", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Fatal("extractGroup on a short flat slice did not panic")
			}
		}()
		geo.extractGroup(flat[:len(flat)-1], 0)
	})
}
