package model

import (
	"encoding/binary"
	"errors"
	"math"
	"net"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// Record only bytes accepted by the real TCP connection. Read after joining.
type epOwnerCompletionConn struct {
	net.Conn
	written []byte
}

func (c *epOwnerCompletionConn) Write(p []byte) (int, error) {
	n, err := c.Conn.Write(p)
	c.written = append(c.written, p[:n]...)
	return n, err
}
func epOwnerCompletionFrame(t *testing.T, c *epOwnerCompletionConn, digest [32]byte, failed bool, words int) []float32 {
	t.Helper()
	b := c.written
	if len(b) < 9 || int(binary.LittleEndian.Uint32(b[:4])) != len(b)-4 || distOp(b[4]) != opV41EPReduce {
		t.Fatalf("incomplete or unexpected frame: %x", b)
	}
	row, n, err := decodeF32(b[5:])
	if err != nil || n != len(b)-5 || len(row) != words {
		t.Fatalf("frame words=%d consumed=%d err=%v", len(row), n, err)
	}
	header := make([]float32, v41EPHeaderWords)
	v41EPHeader(header, digest, failed)
	for i, w := range header {
		if math.Float32bits(row[i]) != math.Float32bits(w) {
			t.Fatalf("frame header[%d]=%x want %x", i, math.Float32bits(row[i]), math.Float32bits(w))
		}
	}
	return row
}

// fak-test:runtime medium est=2s lane=default
func TestV41EPOwnerSumPreservesFailureStatus(t *testing.T) {
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, faultRank := range []int{0, 1} {
				for _, fault := range []string{"nil", "error", "value", "selected", "returned", "goexit", "healthy"} {
					t.Run("owner="+itoa(faultRank)+"/"+fault, func(t *testing.T) {
						ln, err := net.Listen("tcp", "127.0.0.1:0")
						if err != nil {
							t.Fatal(err)
						}
						defer ln.Close()
						if err := ln.(*net.TCPListener).SetDeadline(time.Now().Add(2 * time.Second)); err != nil {
							t.Fatal(err)
						}
						type formation struct {
							group *DistComm
							err   error
						}
						formed := make(chan formation, 1)
						go func() { g, e := Coordinate(ln, 2); formed <- formation{g, e} }()
						conn, err := net.DialTimeout("tcp", ln.Addr().String(), time.Second)
						if err != nil {
							ln.Close()
							<-formed
							t.Fatal(err)
						}
						defer conn.Close()
						if err := conn.SetDeadline(time.Now().Add(2 * time.Second)); err != nil {
							conn.Close()
							ln.Close()
							<-formed
							t.Fatal(err)
						}
						follower, err := Join(conn, 1, 2)
						if err != nil {
							conn.Close()
							ln.Close()
							<-formed
							t.Fatal(err)
						}
						f := <-formed
						if f.err != nil {
							t.Fatal(f.err)
						}
						groups := [2]*DistComm{f.group, follower}
						defer groups[0].Close()
						defer groups[1].Close()
						raw := [2]net.Conn{groups[0].workers[1], groups[1].coord}
						records := [2]*epOwnerCompletionConn{{Conn: raw[0]}, {Conn: raw[1]}}
						groups[0].workers[1] = records[0]
						groups[1].coord = records[1]
						closeSockets := func() {
							for _, c := range raw {
								_ = c.Close()
							}
						}
						defer closeSockets()
						owners := TPPlan{Dim: 2, Shards: []TPShard{{Rank: 0, Lo: 0, Hi: 1}, {Rank: 1, Lo: 1, Hi: 2}}}
						picks := []routePick{{expert: 0, weight: 1}, {expert: 1, weight: 1}}
						digest := [32]byte{1, 3, 5, 7}
						primary := errors.New("owner expert primary")
						selected := v41ExpertOperationErr(0, "down", &BackendForwardOperationError{Backend: "fixture", Path: "v41-routed-expert", Stage: "down", Cause: primary})
						var calls [2]int
						var sums [2][]float32
						var causes [2]error
						var payloads [2]any
						var returned, boundaryReturned [2]bool
						var wg sync.WaitGroup
						for rank, g := range groups {
							wg.Add(1)
							go func(rank int, g *DistComm) {
								defer wg.Done()
								// The fixture owns teardown on Goexit. This is not a production verdict.
								defer g.Close()
								func() {
									defer func() { payloads[rank] = recover() }()
									m := &Model{Cfg: Config{HiddenSize: 2, MoEIntermediateSize: 2}}
									st := &v41ForwardState{}
									st.routedExpert = func(layer int, stem string, x []float32, weight float32) ([]float32, v41RoutedExpertOutcome, error) {
										calls[rank]++
										if layer != 0 || stem != "ffn.experts."+itoa(rank) || len(x) != 2 || weight != 1 {
											panic("wrong owner callback")
										}
										if rank == faultRank {
											switch fault {
											case "nil":
												panic(nil)
											case "error":
												panic(primary)
											case "value":
												panic("owner value")
											case "selected":
												panic(selected)
											case "returned":
												return nil, v41RoutedExpertError, selected
											case "goexit":
												runtime.Goexit()
											}
										}
										return []float32{float32(rank + 1), float32(2 * (rank + 1))}, v41RoutedExpertHandled, nil
									}
									c := &v41ExpertParallelCollective{g: g, timeout: 2 * time.Second}
									sums[rank], causes[rank] = m.v41EPOwnerSum(c, owners, digest, 0, picks, []float32{1, 2}, &v41ProjScratch{}, st)
									returned[rank] = true
								}()
								boundaryReturned[rank] = true
							}(rank, g)
						}
						done := make(chan struct{})
						go func() { wg.Wait(); close(done) }()
						select {
						case <-done:
						case <-time.After(4 * time.Second):
							closeSockets()
							<-done
							t.Fatal("owner round timed out; sockets closed and workers joined")
						}
						for rank := range groups {
							if calls[rank] != 1 {
								t.Fatalf("owner %d callback count=%d", rank, calls[rank])
							}
							if fault == "goexit" && rank == faultRank {
								if returned[rank] || boundaryReturned[rank] || payloads[rank] != nil {
									t.Fatalf("Goexit produced return: %t/%t %v", returned[rank], boundaryReturned[rank], payloads[rank])
								}
								continue
							}
							if !returned[rank] || !boundaryReturned[rank] || payloads[rank] != nil {
								t.Fatalf("owner %d returned=%t boundary=%t panic=%v", rank, returned[rank], boundaryReturned[rank], payloads[rank])
							}
							if fault == "healthy" {
								if causes[rank] != nil || len(sums[rank]) != 2 || sums[rank][0] != 3 || sums[rank][1] != 6 {
									t.Fatalf("healthy owner %d sum=%v cause=%v", rank, sums[rank], causes[rank])
								}
							} else {
								if !errors.Is(causes[rank], ErrV41ExpertParallel) || sums[rank] != nil {
									t.Fatalf("owner-sum fault=%q owner=%d sum=%v cause=%v", fault, rank, sums[rank], causes[rank])
								}
							}
						}
						if fault == "goexit" {
							return
						}
						if fault != "healthy" {
							cause := causes[faultRank]
							switch fault {
							case "error":
								if !errors.Is(cause, primary) {
									t.Fatalf("primary lost: %v", cause)
								}
							case "selected", "returned":
								if !errors.Is(cause, selected) || !errors.Is(cause, primary) {
									t.Fatalf("selected cause lost: %v", cause)
								}
							case "value":
								if !strings.Contains(cause.Error(), "expert panic: owner value") {
									t.Fatalf("value conversion=%v", cause)
								}
							case "nil":
								if mode == "0" {
									var nilPanic *runtime.PanicNilError
									if !errors.As(cause, &nilPanic) {
										t.Fatalf("modern nil cause=%v", cause)
									}
								} else if !strings.Contains(cause.Error(), "expert panic: <nil>") {
									t.Fatalf("legacy nil cause=%v", cause)
								}
							}
						}
						epOwnerCompletionFrame(t, records[1], digest, fault != "healthy" && faultRank == 1, v41EPHeaderWords+2)
						replyWords := v41EPHeaderWords
						if fault == "healthy" {
							replyWords += 2
						}
						epOwnerCompletionFrame(t, records[0], digest, fault != "healthy", replyWords)
					})
				}
			}
		})
	}
}
