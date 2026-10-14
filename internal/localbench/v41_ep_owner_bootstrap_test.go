package localbench

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=20ms lane=default
func TestV41EPOwnerBootstrapContract(t *testing.T) {
	req := v41EPOwnerRequestFixture()
	newTransport := func(t *testing.T) *v41EPOwnerTransport {
		t.Helper()
		now := time.Unix(100, 0)
		transport, err := newV41EPOwnerTransport(now, now.Add(req.Budgets.TotalNS), now.Add(req.Budgets.JoinNS))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { transport.stop(nil) })
		return transport
	}
	newBootstrap := func(t *testing.T, rank int, transport *v41EPOwnerTransport) *v41EPOwnerBootstrap {
		t.Helper()
		bootstrap, err := newV41EPOwnerBootstrap(req, rank, transport)
		if err != nil {
			t.Fatal(err)
		}
		return bootstrap
	}
	admit := func(t *testing.T, transport *v41EPOwnerTransport, raw *v41EPFakeConn) *v41EPOwnerConn {
		t.Helper()
		conn, err := transport.ownConn(raw)
		if err != nil {
			t.Fatal(err)
		}
		return conn.(*v41EPOwnerConn)
	}
	peerStream := func(rank int, tail []byte) []byte {
		data := v41EPTestPreamble(t, req, rank)
		var join [4]byte
		binary.LittleEndian.PutUint32(join[:], uint32(rank))
		return append(append(data, join[:]...), tail...)
	}
	t.Run("constructor-and-exact-frame-cap", func(t *testing.T) {
		transport := newTransport(t)
		for _, rank := range []int{-1, 3} {
			if _, err := newV41EPOwnerBootstrap(req, rank, transport); err == nil {
				t.Fatal("invalid local rank accepted")
			}
		}
		if _, err := newV41EPOwnerBootstrap(req, 0, nil); err == nil {
			t.Fatal("nil registry accepted")
		}
		if _, err := newV41EPOwnerBootstrap(req, 0, &v41EPOwnerTransport{}); err == nil {
			t.Fatal("uninitialized registry accepted")
		}
		bad := req
		bad.SourceSHA256 = strings.Repeat("a", 40)
		if _, err := newV41EPOwnerBootstrap(bad, 0, transport); err == nil {
			t.Fatal("unvalidated request accepted")
		}
		max := req
		max.RunID = strings.Repeat("x", V41EPOwnerMaxRunIDBytes)
		bootstrap, err := newV41EPOwnerBootstrap(max, 1, transport)
		if err != nil {
			t.Fatal(err)
		}
		var written bytes.Buffer
		reader := bytes.NewReader(v41EPTestPreamble(t, max, 0))
		conn := admit(t, transport, &v41EPFakeConn{readFn: reader.Read, writeFn: written.Write})
		if err := bootstrap.exchangeWithRoot(conn); err != nil {
			t.Fatal(err)
		}
		if written.Len() != 240 || v41EPOwnerPeerMaxFrame != 240 || v41EPOwnerPeerMaxPayload != 236 || !bytes.Equal(written.Bytes(), v41EPTestPreamble(t, max, 1)) {
			t.Fatal("canonical maximum frame changed")
		}
		cause := errors.New("stopped")
		transport.stop(cause)
		if _, err := newV41EPOwnerBootstrap(req, 0, transport); !errors.Is(err, cause) {
			t.Fatalf("stopped registry accepted: %v", err)
		}
	})
	t.Run("distinct-root-ranks-replay-and-counters", func(t *testing.T) {
		transport := newTransport(t)
		bootstrap := newBootstrap(t, 0, transport)
		var wantRead, wantWritten uint64
		for _, rank := range []int{2, 1} {
			tail := []byte{10, 20, 30}
			stream := peerStream(rank, tail)
			reader := bytes.NewReader(stream)
			var written bytes.Buffer
			raw := &v41EPFakeConn{readFn: func(p []byte) (int, error) {
				// Deterministic one-byte fragmentation exercises bounded reads.
				return reader.Read(p[:1])
			}, writeFn: written.Write}
			conn := admit(t, transport, raw)
			replay, gotRank, err := bootstrap.admitFollower(conn)
			if err != nil || gotRank != rank {
				t.Fatalf("rank admission %d: %d %v", rank, gotRank, err)
			}
			if !bytes.Equal(written.Bytes(), v41EPTestPreamble(t, req, 0)) {
				t.Fatal("root reply changed exact request identity")
			}
			if n, err := replay.Read(nil); n != 0 || err != nil {
				t.Fatalf("zero read consumed prefix: %d %v", n, err)
			}
			got := make([]byte, 4+len(tail))
			if _, err := io.ReadFull(replay, got[:2]); err != nil {
				t.Fatal(err)
			}
			if _, err := io.ReadFull(replay, got[2:]); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, stream[len(stream)-len(got):]) || reader.Len() != 0 {
				t.Fatalf("Join prefix or following bytes changed: %v", got)
			}
			for _, setter := range []struct {
				kind string
				set  func(time.Time) error
			}{{"both", replay.SetDeadline}, {"read", replay.SetReadDeadline}, {"write", replay.SetWriteDeadline}} {
				if err := setter.set(time.Time{}); err != nil {
					t.Fatal(err)
				}
				raw.wantDeadline(t, setter.kind, transport.phaseDeadline)
			}
			if replay.LocalAddr().String() != "local" || replay.RemoteAddr().String() != "remote" {
				t.Fatal("addresses not forwarded")
			}
			replay.Close()
			replay.Close()
			if _, err := replay.Read(make([]byte, 1)); !errors.Is(err, net.ErrClosed) || raw.closes.Load() != 1 {
				t.Fatal("replay bypassed owned close")
			}
			wantRead += uint64(len(stream))
			wantWritten += uint64(written.Len())
		}
		transport.stop(nil)
		got, available := transport.snapshotAfterJoin()
		want := V41EPOwnerSocketCounters{ReadBytes: wantRead, WrittenBytes: wantWritten}
		if !available || got != want || !bootstrap.seen[1] || !bootstrap.seen[2] {
			t.Fatalf("replay double-counted or missing rank: %+v %v", got, available)
		}
	})
	t.Run("follower-identity-snapshot-and-single-use", func(t *testing.T) {
		for _, rank := range []int{1, 2} {
			transport := newTransport(t)
			localReq := req
			bootstrap, err := newV41EPOwnerBootstrap(localReq, rank, transport)
			if err != nil {
				t.Fatal(err)
			}
			localReq.RunID, localReq.SourceSHA256 = "changed", strings.Repeat("d", 64)
			var written bytes.Buffer
			reader := bytes.NewReader(append(v41EPTestPreamble(t, req, 0), 99))
			conn := admit(t, transport, &v41EPFakeConn{readFn: reader.Read, writeFn: written.Write})
			if err := bootstrap.exchangeWithRoot(conn); err != nil {
				t.Fatal(err)
			}
			if reader.Len() != 1 || !bytes.Equal(written.Bytes(), v41EPTestPreamble(t, req, rank)) {
				t.Fatal("snapshot changed, Join was emitted, or following bytes were consumed")
			}
			other := &v41EPFakeConn{}
			if err := bootstrap.exchangeWithRoot(admit(t, transport, other)); err == nil || other.closes.Load() != 1 {
				t.Fatalf("follower exchange reused: %v", err)
			}
		}
	})
	t.Run("malformed-or-mismatched-peer", func(t *testing.T) {
		for name, mutate := range map[string]func([]byte) []byte{
			"magic":               func(p []byte) []byte { p[4] ^= 1; return p },
			"version":             func(p []byte) []byte { p[11] ^= 1; return p },
			"world":               func(p []byte) []byte { p[13] = 2; return p },
			"root-as-follower":    func(p []byte) []byte { p[12] = 0; return p },
			"rank-out-of-range":   func(p []byte) []byte { p[12] = 3; return p },
			"request":             func(p []byte) []byte { p[16] ^= 1; return p },
			"source":              func(p []byte) []byte { p[48] ^= 1; return p },
			"shard-set":           func(p []byte) []byte { p[80] ^= 1; return p },
			"run-ID":              func(p []byte) []byte { p[112] ^= 1; return p },
			"zero-length":         func(p []byte) []byte { binary.BigEndian.PutUint32(p[:4], 0); return p },
			"fixed-only-length":   func(p []byte) []byte { binary.BigEndian.PutUint32(p[:4], 108); return p },
			"oversize":            func(p []byte) []byte { binary.BigEndian.PutUint32(p[:4], 237); return p },
			"length-overflow":     func(p []byte) []byte { binary.BigEndian.PutUint32(p[:4], ^uint32(0)); return p },
			"zero-run-length":     func(p []byte) []byte { binary.BigEndian.PutUint16(p[14:16], 0); return p },
			"oversize-run-length": func(p []byte) []byte { binary.BigEndian.PutUint16(p[14:16], 129); return p },
			"inexact-run-length":  func(p []byte) []byte { binary.BigEndian.PutUint16(p[14:16], uint16(len(req.RunID)+1)); return p },
			"trailing-payload":    func(p []byte) []byte { p = append(p, 0); binary.BigEndian.PutUint32(p[:4], uint32(len(p)-4)); return p },
			"truncated-header":    func(p []byte) []byte { return p[:3] },
			"truncated-payload":   func(p []byte) []byte { return p[:len(p)-1] },
		} {
			t.Run(name, func(t *testing.T) {
				transport := newTransport(t)
				bootstrap := newBootstrap(t, 0, transport)
				reader := bytes.NewReader(mutate(v41EPTestPreamble(t, req, 1)))
				raw := &v41EPFakeConn{readFn: reader.Read}
				if replay, _, err := bootstrap.admitFollower(admit(t, transport, raw)); err == nil || replay != nil || raw.closes.Load() != 1 || bootstrap.failed == nil {
					t.Fatalf("bad peer admitted: %v", err)
				}
			})
		}
	})
	t.Run("rank-and-registry-admission", func(t *testing.T) {
		for _, mode := range []string{"duplicate-rank", "duplicate-connection", "wrong-registry", "Join-rank", "truncated-Join", "follower-requires-root", "wrong-local-role", "after-stop", "nil"} {
			t.Run(mode, func(t *testing.T) {
				transport := newTransport(t)
				bootstrap := newBootstrap(t, 0, transport)
				stream := peerStream(1, nil)
				if mode == "Join-rank" {
					binary.LittleEndian.PutUint32(stream[len(stream)-4:], 2)
				}
				if mode == "truncated-Join" {
					stream = stream[:len(stream)-2]
				}
				reader := bytes.NewReader(stream)
				raw := &v41EPFakeConn{readFn: reader.Read}
				owner := transport
				if mode == "wrong-registry" {
					owner = newTransport(t)
				}
				conn := admit(t, owner, raw)
				if mode == "duplicate-rank" || mode == "duplicate-connection" {
					if _, _, err := bootstrap.admitFollower(conn); err != nil {
						t.Fatal(err)
					}
					if mode == "duplicate-rank" {
						reader = bytes.NewReader(peerStream(1, nil))
						raw = &v41EPFakeConn{readFn: reader.Read}
						conn = admit(t, transport, raw)
					}
				}
				if mode == "after-stop" {
					transport.stop(nil)
				}
				if mode == "nil" {
					conn = nil
				}
				var err error
				if mode == "follower-requires-root" || mode == "wrong-local-role" {
					if mode == "follower-requires-root" {
						bootstrap = newBootstrap(t, 1, transport)
					}
					err = bootstrap.exchangeWithRoot(conn)
				} else {
					_, _, err = bootstrap.admitFollower(conn)
				}
				if err == nil || bootstrap.failed == nil {
					t.Fatal("invalid admission succeeded")
				}
				if mode == "wrong-registry" {
					if raw.closes.Load() != 0 || reader.Len() != len(stream) {
						t.Fatal("foreign registry handle was touched")
					}
				} else if mode != "nil" && raw.closes.Load() != 1 {
					t.Fatal("failed owned connection not retired once")
				}
			})
		}
	})
	t.Run("io-faults-are-terminal", func(t *testing.T) {
		for _, mode := range []string{"short-write", "zero-write", "partial-write-error", "full-write-error", "read-zero-progress", "partial-read-error", "full-read-error", "panic"} {
			t.Run(mode, func(t *testing.T) {
				transport := newTransport(t)
				bootstrap := newBootstrap(t, 1, transport)
				failure := errors.New("raw I/O failure")
				reader := bytes.NewReader(v41EPTestPreamble(t, req, 0))
				writes, reads := 0, 0
				raw := &v41EPFakeConn{writeFn: func(p []byte) (int, error) {
					writes++
					switch mode {
					case "short-write":
						return len(p) - 1, nil
					case "zero-write":
						return 0, nil
					case "partial-write-error":
						return 1, failure
					case "full-write-error":
						return len(p), failure
					}
					return len(p), nil
				}, readFn: func(p []byte) (int, error) {
					reads++
					switch mode {
					case "read-zero-progress":
						return 0, nil
					case "partial-read-error":
						return 1, failure
					case "full-read-error":
						n, _ := reader.Read(p)
						return n, failure
					case "panic":
						panic(failure)
					}
					return reader.Read(p)
				}}
				conn := admit(t, transport, raw)
				var err error
				if mode == "panic" {
					func() {
						defer func() {
							if recover() != failure {
								t.Fatal("raw panic identity lost")
							}
						}()
						err = bootstrap.exchangeWithRoot(conn)
					}()
				} else {
					err = bootstrap.exchangeWithRoot(conn)
					if err == nil {
						t.Fatal("I/O fault admitted")
					}
					if strings.HasSuffix(mode, "error") && !errors.Is(err, failure) {
						t.Fatalf("raw cause lost: %v", err)
					}
				}
				if writes != 1 || reads > 1 || raw.closes.Load() != 1 || bootstrap.failed == nil {
					t.Fatalf("I/O retried or failure not terminal: writes=%d reads=%d", writes, reads)
				}
				later := &v41EPFakeConn{readFn: reader.Read}
				if err := bootstrap.exchangeWithRoot(admit(t, transport, later)); err == nil || later.closes.Load() != 1 {
					t.Fatal("terminal bootstrap reused")
				}
				transport.stop(nil)
				got, available := transport.snapshotAfterJoin()
				if mode == "panic" {
					if available {
						t.Fatal("panicking raw I/O acquired exact counters")
					}
					return
				}
				want := V41EPOwnerSocketCounters{WrittenBytes: uint64(len(v41EPTestPreamble(t, req, 1)))}
				switch mode {
				case "short-write":
					want.WrittenBytes--
				case "zero-write":
					want.WrittenBytes = 0
				case "partial-write-error":
					want.WrittenBytes, want.WriteErrors = 1, 1
				case "full-write-error":
					want.WriteErrors = 1
				case "partial-read-error":
					want.ReadBytes, want.ReadErrors = 1, 1
				case "full-read-error":
					want.ReadBytes, want.ReadErrors = 4, 1
				}
				if !available || got != want {
					t.Fatalf("failed exchange counters: %+v want %+v available=%v", got, want, available)
				}
			})
		}
	})
	t.Run("stop-during-final-operation-refuses-admission", func(t *testing.T) {
		transport := newTransport(t)
		bootstrap := newBootstrap(t, 0, transport)
		reader := bytes.NewReader(peerStream(1, nil))
		cause := errors.New("stop during Join read")
		raw := &v41EPFakeConn{readFn: func(p []byte) (int, error) {
			n, err := reader.Read(p)
			if reader.Len() == 0 {
				transport.stop(cause)
			}
			return n, err
		}}
		if replay, _, err := bootstrap.admitFollower(admit(t, transport, raw)); replay != nil || !errors.Is(err, cause) || bootstrap.seen[1] || raw.closes.Load() != 1 {
			t.Fatalf("accepted stopped final operation: %v", err)
		}
	})
}

// Independent wire fixture: hashes the canonical request directly and spells out
// the fixed offsets, without calling either production preamble codec method.
func v41EPTestPreamble(t *testing.T, req V41EPOwnerRequest, rank int) []byte {
	t.Helper()
	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(raw)
	frame := make([]byte, 112+len(req.RunID))
	binary.BigEndian.PutUint32(frame[:4], uint32(len(frame)-4))
	copy(frame[4:12], "FK41EP01")
	frame[12], frame[13] = byte(rank), 3
	binary.BigEndian.PutUint16(frame[14:16], uint16(len(req.RunID)))
	copy(frame[16:48], digest[:])
	for _, item := range []struct {
		value  string
		offset int
	}{{req.SourceSHA256, 48}, {req.ExpectedShardSetSHA256, 80}} {
		decoded, err := hex.DecodeString(item.value)
		if err != nil {
			t.Fatal(err)
		}
		copy(frame[item.offset:item.offset+32], decoded)
	}
	copy(frame[112:], req.RunID)
	return frame
}
