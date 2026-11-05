package dispatch

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// fak-test:runtime fast est=50ms lane=default
// Estimate only: two tiny shard fixtures and two deliberate 5ms timeouts.
func TestSingleKeyReadsPreserveShardErrors(t *testing.T) {
	t.Parallel()
	for _, running := range []bool{false, true} {
		name := "local_timeout"
		if running {
			name = "successful_hit_and_miss"
		}
		t.Run(name, func(t *testing.T) {
			timeoutMs := int64(5)
			if running {
				timeoutMs = 1000
			}
			mgr, err := shard.NewManager(shard.ManagerConfig{
				NumShards: 1, MaxMemoryGB: 0, IndexCapacity: 1024, DispatchTimeoutMs: timeoutMs,
				EvictionPolicy: "wtinylfu", SlabDistribution: "dedicated", ModelPageBytes: 256,
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(mgr.Stop)
			if running {
				mgr.Start()
				if !mgr.Set([]byte("present"), []byte("value"), 0) {
					t.Fatal("could not populate the live control")
				}
			}
			// The unstarted shard injects a Submit failure directly into the
			// handlers; the public startup gate normally rejects this state.
			d := &Dispatcher{Manager: mgr}
			for _, read := range []struct {
				name string
				op   uint8
				run  func(protocol.Message) protocol.Message
			}{
				{"GET", protocol.OpGet, d.HandleGet},
				{"TEST", protocol.OpTest, d.HandleTest},
			} {
				keys := []string{"present"}
				if running {
					keys = append(keys, "absent")
				}
				for _, key := range keys {
					resp := read.run(protocol.Message{
						Header: protocol.Header{OpCode: read.op, RequestID: 37},
						Body:   protocol.EncodeKeyBody([]byte(key)),
					})
					if !running {
						if resp.Header.OpCode != protocol.RespError || resp.Header.RequestID != 37 || string(resp.Body) != "shard dispatch timeout" {
							t.Fatalf("%s failure: header=%+v body=%q", read.name, resp.Header, resp.Body)
						}
						continue
					}
					wantFound := key == "present"
					if read.op == protocol.OpGet {
						value, found, err := protocol.DecodeValueResponse(resp.Body)
						wantValue := ""
						if wantFound {
							wantValue = "value"
						}
						if resp.Header.OpCode != protocol.RespValue || err != nil || found != wantFound || string(value) != wantValue {
							t.Fatalf("GET %q: header=%+v found=%t value=%q err=%v", key, resp.Header, found, value, err)
						}
					} else {
						wantBody := string([]byte{0})
						if wantFound {
							wantBody = string([]byte{1})
						}
						if resp.Header.OpCode != protocol.RespOK || string(resp.Body) != wantBody {
							t.Fatalf("TEST %q: header=%+v body=%q", key, resp.Header, resp.Body)
						}
					}
				}
			}
		})
	}
}
