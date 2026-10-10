package model

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=30ms
func TestSafetensorsRejectDuplicateJSONKeys(t *testing.T) {
	t.Parallel()
	const entry = `{"dtype":"F32","shape":[1],"data_offsets":[0,4]}`
	blob := func(header string) []byte {
		b := make([]byte, 8+len(header)+8)
		binary.LittleEndian.PutUint64(b, uint64(len(header)))
		copy(b[8:], header)
		return b
	}
	for name, header := range map[string]string{
		"tensor-name":    `{"w":` + entry + `,"w":` + entry + `}`,
		"escaped-name":   `{"w":` + entry + `,"\u0077":` + entry + `}`,
		"nested-dtype":   `{"w":{"dtype":"F16","dtyp\u0065":"F32","shape":[1],"data_offsets":[0,4]}}`,
		"nested-offsets": `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,2],"data_offsets":[0,4]}}`,
		"metadata":       `{"w":` + entry + `,"__metadata__":{"format":"pt","format":"pt"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			data := blob(header)
			hdr, offset, err := parseSafetensorsHeader(data)
			if err == nil || !strings.Contains(err.Error(), "duplicate") || hdr != nil || offset != 0 {
				t.Fatalf("blob accepted duplicate: hdr=%v offset=%d err=%v", hdr, offset, err)
			}
			for _, mapped := range []bool{false, true} {
				closed := 0
				closer := closerFunc(func() error { closed++; return nil })
				var sf *safetensorsFile
				if mapped {
					sf, err = newSafetensorsFileMmap(data, closer)
				} else {
					sf, err = newSafetensorsFile(bytes.NewReader(data), int64(len(data)), closer)
				}
				if err == nil || !strings.Contains(err.Error(), "duplicate") || sf != nil || closed != 1 {
					t.Fatalf("mapped=%v sf=%v closed=%d err=%v", mapped, sf, closed, err)
				}
			}
		})
	}
	t.Run("valid-header", func(t *testing.T) {
		// Reusing field names in different objects is legal. Ignored metadata
		// numbers need not fit float64, and strings that resemble JSON are opaque.
		header := `{"a":` + entry + `,"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"__metadata__":{"huge":1e1000,"nested":[{"x":1},{"x":2}],"text":"{\"x\":1,\"x\":2}"}}   `
		data := blob(header)
		var want map[string]json.RawMessage
		if err := json.Unmarshal([]byte(header), &want); err != nil {
			t.Fatal(err)
		}
		got, offset, err := parseSafetensorsHeader(data)
		if err != nil || offset != 8+len(header) || !reflect.DeepEqual(got, want) {
			t.Fatalf("valid blob changed: %d %v", offset, err)
		}
		closed := 0
		sf, err := newSafetensorsFileMmap(data, closerFunc(func() error { closed++; return nil }))
		if err != nil {
			t.Fatal(err)
		}
		if closed != 0 || !reflect.DeepEqual(sf.hdr, want) {
			t.Fatal("valid mapped header changed or closed early")
		}
		if err := sf.Close(); err != nil || closed != 1 {
			t.Fatalf("valid Close: count=%d err=%v", closed, err)
		}
	})
	for name, raw := range map[string]string{
		"duplicate-map":            `{"weight_map":{"w":"a.safetensors"},"weight_map":{"w":"b.safetensors"}}`,
		"duplicate-weight":         `{"weight_map":{"w":"a.safetensors","w":"b.safetensors"}}`,
		"escaped-weight":           `{"weight_map":{"w":"a.safetensors","\u0077":"b.safetensors"}}`,
		"duplicate-index-metadata": `{"weight_map":{"w":"a.safetensors"},"metadata":{"total_size":4,"total_size":8}}`,
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "model.safetensors.index.json")
			if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
				t.Fatal(err)
			}
			shards, weights, err := safetensorsIndexShards(path)
			if err == nil || !strings.Contains(err.Error(), "duplicate") || shards != nil || weights != nil {
				t.Fatalf("index accepted duplicate: %v %v %v", shards, weights, err)
			}
			opened := 0
			open := func(string) (*safetensorsFile, error) { opened++; return nil, fmt.Errorf("unexpected shard open") }
			for _, quant := range []bool{false, true} {
				var m *Model
				if quant {
					m, err = loadSafetensorsQuantDir(dir, Config{}, open)
				} else {
					m, err = loadSafetensorsDir(dir, Config{}, open)
				}
				if err == nil || !strings.Contains(err.Error(), "duplicate") || m != nil || opened != 0 {
					t.Fatalf("quant=%v model=%v opened=%d err=%v", quant, m, opened, err)
				}
			}
		})
	}
	t.Run("valid-index", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "model.safetensors.index.json")
		raw := `{"metadata":{"huge":1e1000},"weight_map":{"a":"z.safetensors","b":"a.safetensors","c":"z.safetensors"}}`
		if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
			t.Fatal(err)
		}
		shards, weights, err := safetensorsIndexShards(path)
		if err != nil || !reflect.DeepEqual(shards, []string{"a.safetensors", "z.safetensors"}) || !reflect.DeepEqual(weights, map[string]string{"a": "z.safetensors", "b": "a.safetensors", "c": "z.safetensors"}) {
			t.Fatalf("valid index changed: %v %v %v", shards, weights, err)
		}
	})
	t.Run("syntax-and-depth", func(t *testing.T) {
		for _, raw := range []string{`{"a":1} {"b":2}`, `{"a":[1,]}`, strings.Repeat("[", 10001) + "0" + strings.Repeat("]", 10001)} {
			var want, got any
			oldErr := json.Unmarshal([]byte(raw), &want)
			err := unmarshalUniqueJSON([]byte(raw), &got)
			if oldErr == nil || err == nil || err.Error() != oldErr.Error() {
				t.Fatalf("syntax/depth behavior changed: %v vs %v", oldErr, err)
			}
		}
	})
}
