package dispatch

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// fak-test:runtime fast est=10ms lane=default
// Estimate only: parsing plus tiny zero-shard snapshots in temporary directories.
func TestSnapshotRestoreRejectInvalidJSON(t *testing.T) {
	t.Parallel()
	handlers := []struct {
		name string
		op   uint8
		call func(*Dispatcher, protocol.Message) protocol.Message
	}{
		{"snapshot", protocol.OpSnapshot, (*Dispatcher).HandleSnapshot},
		{"restore", protocol.OpRestore, (*Dispatcher).HandleRestore},
	}
	for _, handler := range handlers {
		t.Run(handler.name, func(t *testing.T) {
			for _, tc := range []struct {
				name string
				body string
			}{
				{"syntax", "{"},
				{"whitespace", " \n"},
				{"wrong_shape", "[]"},
				{"wrong_dir_type", `{"dir":7}`},
				{"partially_decoded_dir", `{"dir":"explicit","dir":7}`},
				{"trailing_value", `{"dir":"explicit"} {}`},
			} {
				t.Run(tc.name, func(t *testing.T) {
					// A configured fallback and nil manager make falling through
					// to snapshot/restore work fail immediately, before filesystem IO.
					d := &Dispatcher{SnapshotDir: "configured"}
					resp := handler.call(d, protocol.Message{
						Header: protocol.Header{OpCode: handler.op, RequestID: 71},
						Body:   []byte(tc.body),
					})
					if resp.Header.OpCode != protocol.RespError || resp.Header.RequestID != 71 || !strings.HasPrefix(string(resp.Body), "invalid "+handler.name+" JSON: ") {
						t.Fatalf("invalid JSON response: header=%+v body=%q", resp.Header, resp.Body)
					}
				})
			}
			for _, body := range []string{"", `{}`, `null`, `{"dir":""}`, `{"ignored":true}`} {
				// Existing optional-body, null, empty-directory and unknown-field
				// behavior stays valid and reaches the unchanged configuration check.
				resp := handler.call(&Dispatcher{}, protocol.Message{
					Header: protocol.Header{OpCode: handler.op, RequestID: 72},
					Body:   []byte(body),
				})
				if resp.Header.OpCode != protocol.RespError || resp.Header.RequestID != 72 || string(resp.Body) != "no snapshot directory configured" {
					t.Fatalf("valid body %q refused by parser: header=%+v body=%q", body, resp.Header, resp.Body)
				}
			}
		})
	}

	// Preserve actual zero-shard snapshot/restore success for both explicit and
	// configured directories. No serving manager, shard allocation or goroutine.
	for _, explicit := range []bool{false, true} {
		dir := t.TempDir()
		d := &Dispatcher{Manager: &shard.Manager{}}
		var body []byte
		if explicit {
			var err error
			body, err = json.Marshal(struct {
				Dir string `json:"dir"`
			}{Dir: dir})
			if err != nil {
				t.Fatal(err)
			}
		} else {
			d.SnapshotDir = dir
		}
		for _, handler := range handlers {
			resp := handler.call(d, protocol.Message{
				Header: protocol.Header{OpCode: handler.op, RequestID: 73},
				Body:   body,
			})
			var result struct {
				Dir  string `json:"dir"`
				Keys int64  `json:"keys"`
			}
			if err := json.Unmarshal(resp.Body, &result); err != nil {
				t.Fatal(err)
			}
			if resp.Header.OpCode != protocol.RespOK || resp.Header.RequestID != 73 || result.Dir != dir || result.Keys != 0 {
				t.Fatalf("%s explicit=%v: header=%+v result=%+v", handler.name, explicit, resp.Header, result)
			}
		}
	}
}
