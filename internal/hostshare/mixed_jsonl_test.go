package hostshare

import (
	"os"
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestMixedJSONLRetainsMalformedCensus(t *testing.T) {
	const good = `{"host_share":0.25}`
	for _, tc := range []struct {
		name, data   string
		readings     int
		malformed    bool
		evidenceFree int
	}{
		{"valid_object", good, 1, false, 0},
		{"pretty_object", "{\n  \"host_share\": 0.25\n}\n", 1, false, 0},
		{"valid_jsonl", good + "\n" + good + "\n", 2, false, 0},
		{"broken_tail", good + "\n{broken}\n", 1, true, 0},
		{"broken_head", "{broken}\n" + good, 1, true, 0},
		{"multiple_broken_rows", good + "\n{broken}\n" + good + "\n{unfinished", 2, true, 0},
		{"foreign_and_broken", "{}\n{broken}\n", 0, true, 0},
		{"foreign_only", "{}\n{}\n", 0, false, 1},
		{"entirely_broken", "{broken}\n", 0, true, 0},
		{"nonobject_row", "null\n" + good, 1, true, 0},
		{"blank_crlf_rows", "\r\n" + good + "\r\n \r\n", 1, false, 0},
		{"empty_file", "", 0, true, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "readings.jsonl")
			if err := os.WriteFile(path, []byte(tc.data), 0o600); err != nil {
				t.Fatal(err)
			}
			rep, err := ScanReport([]string{path})
			if err != nil {
				t.Fatal(err)
			}
			if rep.FilesScanned != 1 || len(rep.Readings) != tc.readings || rep.EvidenceFree != tc.evidenceFree {
				t.Fatalf("census = %+v; want readings=%d evidence_free=%d", rep, tc.readings, tc.evidenceFree)
			}
			wantMalformed := 0
			if tc.malformed {
				wantMalformed = 1
			}
			if len(rep.MalformedFiles) != wantMalformed {
				t.Fatalf("malformed files = %v, want count %d", rep.MalformedFiles, wantMalformed)
			}
			if tc.malformed && rep.MalformedFiles[0] != filepath.ToSlash(path) {
				t.Fatalf("malformed path = %q, want %q", rep.MalformedFiles[0], filepath.ToSlash(path))
			}
			for _, reading := range rep.Readings {
				if reading.HostShare == nil || *reading.HostShare != 0.25 {
					t.Fatalf("recovered reading changed: %+v", reading)
				}
			}
		})
	}
}
