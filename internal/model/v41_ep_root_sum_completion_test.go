package model

import (
	"errors"
	"net"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"
)

// fak-test:runtime medium est=2s lane=default
func TestV41EPRootSumPreservesFailureVerdict(t *testing.T) {
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"nil", "error", "value", "selected", "returned", "goexit", "healthy"} {
				t.Run(fault, func(t *testing.T) {
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
					digest := [32]byte{2, 4, 6, 8}
					primary := errors.New("root sum primary")
					selected := v41ExpertOperationErr(0, "sum", &BackendForwardOperationError{Backend: "fixture", Path: "v41-routed-expert", Stage: "sum", Cause: primary})
					var calls [2]int
					var sums [2][]float32
					var causes, failures [2]error
					var payloads [2]any
					var returned, boundaryReturned [2]bool
					var wg sync.WaitGroup
					for rank, g := range groups {
						wg.Add(1)
						go func(rank int, g *DistComm) {
							defer wg.Done()
							// Only the fixture owns peer cancellation when the root exits its goroutine.
							defer g.Close()
							c := &v41ExpertParallelCollective{g: g, timeout: 2 * time.Second}
							defer func() { failures[rank] = c.failure }()
							func() {
								defer func() { payloads[rank] = recover() }()
								part := make([]float32, v41EPHeaderWords+2)
								v41EPHeader(part, digest, false)
								part[v41EPHeaderWords] = float32(rank + 1)
								part[v41EPHeaderWords+1] = float32(2 * (rank + 1))
								round := v41EPReduction{owners: owners, digest: digest, layer: 0, picks: picks, width: 2, sum: func(layer int, gotPicks []routePick, outputs [][]float32, width int) ([]float32, error) {
									calls[rank]++
									if rank != 0 || layer != 0 || width != 2 || len(gotPicks) != 2 || len(outputs) != 2 || len(outputs[0]) != 2 || len(outputs[1]) != 2 || outputs[0][0] != 1 || outputs[1][0] != 2 {
										panic("wrong root sum operands")
									}
									switch fault {
									case "nil":
										panic(nil)
									case "error":
										panic(primary)
									case "value":
										panic("root sum value")
									case "selected":
										panic(selected)
									case "returned":
										return nil, selected
									case "goexit":
										runtime.Goexit()
									}
									return []float32{3, 6}, nil
								}}
								sums[rank], causes[rank] = c.reduce(part, round)
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
						t.Fatal("root round timed out; sockets closed and workers joined")
					}
					if calls != [2]int{1, 0} {
						t.Fatalf("root sum calls=%v", calls)
					}
					ordinary := fault == "returned" || fault == "healthy"
					if returned[0] != ordinary || boundaryReturned[0] == (fault == "goexit") {
						t.Fatalf("root-sum fault=%q returned=%t boundaryReturned=%t sum=%v cause=%v", fault, returned[0], boundaryReturned[0], sums[0], causes[0])
					}
					if !returned[1] || !boundaryReturned[1] || payloads[1] != nil {
						t.Fatalf("peer return=%t/%t panic=%v", returned[1], boundaryReturned[1], payloads[1])
					}
					if fault == "healthy" {
						for rank := range groups {
							if payloads[rank] != nil || causes[rank] != nil || failures[rank] != nil || len(sums[rank]) != 2 || sums[rank][0] != 3 || sums[rank][1] != 6 {
								t.Fatalf("healthy rank%d sum=%v err=%v panic=%v", rank, sums[rank], causes[rank], payloads[rank])
							}
						}
					} else {
						if sums[0] != nil || sums[1] != nil || !errors.Is(causes[1], ErrV41ExpertParallel) {
							t.Fatalf("failed round sums=%v causes=%v", sums, causes)
						}
						if fault == "goexit" {
							if payloads[0] != nil || failures[0] != nil || len(records[0].written) != 0 {
								t.Fatal("Goexit manufactured a verdict or failure outcome")
							}
							return
						}
						if !errors.Is(failures[0], ErrV41ExpertParallel) {
							t.Fatalf("root failure not latched: %v", failures[0])
						}
						switch fault {
						case "nil":
							if mode == "0" {
								if _, ok := payloads[0].(*runtime.PanicNilError); !ok {
									t.Fatalf("modern nil=%T %v", payloads[0], payloads[0])
								}
							} else if payloads[0] != nil {
								t.Fatalf("legacy nil=%v", payloads[0])
							}
						case "error":
							if payloads[0] != primary {
								t.Fatalf("primary=%v", payloads[0])
							}
						case "value":
							if payloads[0] != "root sum value" {
								t.Fatalf("value=%v", payloads[0])
							}
						case "selected":
							if payloads[0] != selected {
								t.Fatalf("selected=%v", payloads[0])
							}
						case "returned":
							if payloads[0] != nil || !errors.Is(causes[0], selected) || !errors.Is(causes[0], primary) {
								t.Fatalf("returned cause=%v panic=%v", causes[0], payloads[0])
							}
						}
						if fault != "returned" {
							var carrier *v41EPRootPanic
							if !errors.As(failures[0], &carrier) || carrier.value != payloads[0] {
								t.Fatalf("root panic carrier=%v payload=%v", failures[0], payloads[0])
							}
						}
					}
					epOwnerCompletionFrame(t, records[1], digest, false, v41EPHeaderWords+2)
					words := v41EPHeaderWords
					if fault == "healthy" {
						words += 2
					}
					epOwnerCompletionFrame(t, records[0], digest, fault != "healthy", words)
				})
			}
		})
	}
}
