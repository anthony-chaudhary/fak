package tcp_test

import (
	"bytes"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/client"
	"github.com/anthony-chaudhary/fak/internal/l3server/cluster"
	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/tcp"
)

// fak-test:runtime fast est=1s
func TestTCPProxyHandshakeAdmission(t *testing.T) {
	peer, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	var valid atomic.Bool
	var hellos, gets atomic.Int32
	var workers sync.WaitGroup
	var connections sync.Map
	workers.Add(1)
	go func() {
		defer workers.Done()
		for {
			conn, err := peer.Accept()
			if err != nil {
				return
			}
			connections.Store(conn, struct{}{})
			workers.Add(1)
			go func() {
				defer workers.Done()
				defer connections.Delete(conn)
				defer conn.Close()
				_ = conn.SetDeadline(time.Now().Add(3 * time.Second))
				codec := protocol.NewCodec(conn)
				for {
					request, err := codec.ReadMessage()
					if err != nil {
						return
					}
					response := protocol.Message{Header: protocol.Header{RequestID: request.Header.RequestID}}
					switch request.Header.OpCode {
					case protocol.OpHandshake:
						hellos.Add(1)
						response.Header.OpCode = protocol.RespOK
						response.Body = []byte(`{"status":"incompatible","api_version":1,"protocol_version":1,"server_version":"1.5.7"}`)
						if valid.Load() {
							response.Body = []byte(`{"status":"ok","api_version":1,"protocol_version":1,"server_version":"1.5.7"}`)
						}
					case protocol.OpGet:
						gets.Add(1)
						response.Header.OpCode = protocol.RespValue
						response.Body = protocol.EncodeValueResponse([]byte("remote-value"), true)
					default:
						return
					}
					if err := codec.WriteMessage(response); err != nil {
						return
					}
				}
			}()
		}
	}()
	t.Cleanup(func() {
		_ = peer.Close()
		connections.Range(func(key, _ any) bool {
			_ = key.(net.Conn).Close()
			return true
		})
		workers.Wait()
	})

	manager, err := shard.NewManager(shard.ManagerConfig{NumShards: 1, MaxMemoryGB: 1, EvictionPolicy: "wtinylfu", ModelPageBytes: 5242880, MaxLeaseDurMs: 30000, IndexCapacity: 256})
	if err != nil {
		t.Fatal(err)
	}
	manager.Start()
	t.Cleanup(manager.Stop)
	frontend := tcp.NewServer("127.0.0.1:0", manager, metrics.NewConnRegistry(), time.Now())
	ring := cluster.NewRing()
	ring.AddNode(&cluster.Node{ID: "remote", Addr: peer.Addr().String(), Alive: true})
	frontend.SetCluster(ring, nil, "frontend", "")
	if err := frontend.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(frontend.Stop)
	c, err := client.New(frontend.Addr())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	got, err := c.Get([]byte("remote-key"))
	if err != nil {
		t.Fatalf("frontend did not survive refused peer: %v", err)
	}
	if got != nil || gets.Load() != 0 || hellos.Load() != 1 {
		t.Fatalf("unadmitted peer received data operation: value=%q GET=%d HELLO=%d, want nil/0/1", got, gets.Load(), hellos.Load())
	}
	valid.Store(true)
	got, err = c.Get([]byte("remote-key"))
	if err != nil || !bytes.Equal(got, []byte("remote-value")) || gets.Load() != 1 || hellos.Load() != 2 {
		t.Fatalf("same endpoint did not recover through new handshake: value=%q err=%v GET=%d HELLO=%d, want remote-value/nil/1/2", got, err, gets.Load(), hellos.Load())
	}
}
