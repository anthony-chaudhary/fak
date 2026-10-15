package model

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"net"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41EPCountConn struct {
	net.Conn
	sent, received uint64
}

func (c *v41EPCountConn) Write(p []byte) (int, error) {
	n, err := c.Conn.Write(p)
	c.sent += uint64(n)
	return n, err
}

func (c *v41EPCountConn) Read(p []byte) (int, error) {
	n, err := c.Conn.Read(p)
	c.received += uint64(n)
	return n, err
}

func v41EPTestPart(round v41EPReduction, rank int, rows [][]float32) []float32 {
	plan, err := v41EPGatherPlan(round.picks, round.owners, round.width)
	if err != nil {
		panic(err)
	}
	part := make([]float32, plan.Shards[rank].Width())
	v41EPHeader(part, round.digest, false)
	offset, band := v41EPHeaderWords, round.owners.Shards[rank]
	for slot, pick := range round.picks {
		if pick.expert >= band.Lo && pick.expert < band.Hi {
			copy(part[offset:], rows[slot])
			offset += round.width
		}
	}
	return part
}

func v41EPTestCollective(g *DistComm) *v41ExpertParallelCollective {
	return &v41ExpertParallelCollective{g: g, timeout: time.Second}
}

// One real TCP round per case checks original slot bytes, the single root
// reduction, final bits on every rank, and actual socket bytes against counters.
// These small software fixtures make no link-speed or device-TPS claim.
// fak-test:runtime fast est=100ms lane=default
func TestV41ExpertParallelRootReduceTCP(t *testing.T) {
	owners, _ := ExpertParallelPlan(10, 3)
	for _, test := range []struct {
		name    string
		experts []int
		values  []float32
	}{
		{"balanced", []int{8, 0, 5, 1, 6, 9}, []float32{1, 2, 3, 4, 5, 6}},
		{"skewed", []int{8, 9, 8, 9, 8, 9}, []float32{1, 2, 3, 4, 5, 6}},
		{"duplicate-order", []int{8, 0, 8}, []float32{1 << 24, 1, -(1 << 24)}},
		{"final-f32", []int{0, 8}, []float32{1, 1.0 / 256}},
		{"empty", nil, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			const width = 256
			round := v41EPReduction{owners: owners, digest: sha256.Sum256([]byte(test.name)), width: width}
			rows := make([][]float32, len(test.experts))
			for slot, expert := range test.experts {
				round.picks = append(round.picks, routePick{expert: expert, weight: 1})
				rows[slot] = make([]float32, width)
				for i := range rows[slot] {
					rows[slot][i] = test.values[slot]
				}
			}
			want, err := v41FullRoutedSum(0, round.picks, rows, width)
			if err != nil {
				t.Fatal(err)
			}
			if test.name == "duplicate-order" && want[0] != 0 {
				t.Fatal("order witness must distinguish [1, 2^24, -2^24] from rank pre-sums")
			}
			calls := make([]int, 3)
			errs := v41EPTestGroup(t, func(g *DistComm) error {
				var wires []*v41EPCountConn
				if g.Rank() == 0 {
					for rank := 1; rank < g.Size(); rank++ {
						conn := &v41EPCountConn{Conn: g.workers[rank]}
						g.workers[rank] = conn
						wires = append(wires, conn)
					}
				} else {
					conn := &v41EPCountConn{Conn: g.coord}
					g.coord = conn
					wires = append(wires, conn)
				}
				local := round
				local.sum = func(layer int, picks []routePick, got [][]float32, h int) ([]float32, error) {
					calls[g.Rank()]++
					if g.Rank() != 0 || !reflect.DeepEqual(picks, round.picks) || len(got) != len(rows) {
						return nil, errors.New("sum ran on a follower or lost original slots")
					}
					for slot := range rows {
						if err := v41EPCheckBits(got[slot], rows[slot]); err != nil {
							return nil, err
						}
					}
					return v41FullRoutedSum(layer, picks, got, h)
				}
				c := v41EPTestCollective(g)
				got, err := c.reduce(v41EPTestPart(local, g.Rank(), rows), local)
				if err != nil {
					return err
				}
				if err := v41EPCheckBits(got, want); err != nil {
					return err
				}
				var sent, received uint64
				for _, wire := range wires {
					sent += wire.sent
					received += wire.received
				}
				if c.rounds.Load() != 1 || sent != c.sentBytes.Load() || received != c.receivedBytes.Load() {
					return fmt.Errorf("wire/counter mismatch: sent %d/%d received %d/%d", sent, c.sentBytes.Load(), received, c.receivedBytes.Load())
				}
				if g.Rank() == 0 {
					plan, _ := v41EPGatherPlan(local.picks, owners, width)
					wantBytes := uint64((g.Size() - 1) * (9 + 4*(v41EPHeaderWords+width)))
					oldBytes := uint64((g.Size() - 1) * (9 + 4*plan.Dim))
					if sent != wantBytes || (len(rows) >= 2 && sent >= oldBytes) {
						return fmt.Errorf("root egress %d, want %d, old allgather %d", sent, wantBytes, oldBytes)
					}
				}
				return nil
			})
			for rank, err := range errs {
				if err != nil {
					t.Fatalf("rank %d: %v", rank, err)
				}
			}
			if !reflect.DeepEqual(calls, []int{1, 0, 0}) {
				t.Fatalf("canonical sum calls=%v", calls)
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41ExpertParallelRootReduceErrors(t *testing.T) {
	owners, _ := ExpertParallelPlan(10, 3)
	selected := errors.New("private selected root failure")
	for _, mode := range []string{"error", "panic", "nonfinite", "short", "overflow"} {
		t.Run(mode, func(t *testing.T) {
			calls := make([]int, 3)
			errs := v41EPTestGroup(t, func(g *DistComm) error {
				round := v41EPReduction{owners: owners, width: 1, digest: sha256.Sum256([]byte(mode)), picks: []routePick{{expert: 0, weight: 1}, {expert: 8, weight: 1}}}
				rows := [][]float32{{1}, {2}}
				if mode == "overflow" {
					rows = [][]float32{{math.MaxFloat32}, {math.MaxFloat32}}
				}
				round.sum = func(layer int, picks []routePick, outputs [][]float32, width int) ([]float32, error) {
					calls[g.Rank()]++
					switch mode {
					case "error":
						return nil, selected
					case "panic":
						panic(selected)
					case "nonfinite":
						return []float32{float32(math.NaN())}, nil
					case "short":
						return nil, nil
					default:
						return v41FullRoutedSum(layer, picks, outputs, width)
					}
				}
				c := v41EPTestCollective(g)
				part := v41EPTestPart(round, g.Rank(), rows)
				var got []float32
				var err error
				panicked := panicAsError(func() { got, err = c.reduce(part, round) })
				if panicked != nil {
					if g.Rank() != 0 || mode != "panic" || panicked != selected {
						return fmt.Errorf("root panic identity changed: %v", panicked)
					}
					err = c.failure
				}
				if got != nil || !errors.Is(err, ErrV41ExpertParallel) {
					return fmt.Errorf("root failure accepted: %v / %v", got, err)
				}
				if g.Rank() == 0 && (mode == "error" || mode == "panic") && !errors.Is(err, selected) {
					return fmt.Errorf("selected root cause lost: %v", err)
				}
				if g.Rank() != 0 && (strings.Contains(err.Error(), "private selected") || !strings.Contains(err.Error(), "root refused")) {
					return fmt.Errorf("follower did not receive a generic failure verdict: %v", err)
				}
				if again, repeated := c.reduce(part, round); again != nil || repeated != err || c.rounds.Load() != 0 {
					return errors.New("failed round was replayed or counted")
				}
				return nil
			})
			for rank, err := range errs {
				if err != nil {
					t.Fatalf("rank %d: %v", rank, err)
				}
			}
			if !reflect.DeepEqual(calls, []int{1, 0, 0}) {
				t.Fatalf("failed canonical sum replayed: %v", calls)
			}
		})
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41ExpertParallelRootReduceMalformedTCP(t *testing.T) {
	owners, _ := ExpertParallelPlan(10, 3)
	for _, mode := range []string{"op", "version", "identity", "status", "nonfinite", "short-vector", "trailing", "oversize", "truncated", "disconnect", "stall"} {
		t.Run(mode, func(t *testing.T) {
			calls := make([]int, 3)
			errs := v41EPTestGroup(t, func(g *DistComm) error {
				round := v41EPReduction{owners: owners, width: 1, digest: sha256.Sum256([]byte(mode)), picks: []routePick{{expert: 8, weight: 1}}}
				part := v41EPTestPart(round, g.Rank(), [][]float32{{1}})
				if g.Rank() == 2 {
					op := opV41EPReduce
					switch mode {
					case "disconnect":
						_ = g.coord.Close()
						return errors.New("injected disconnect")
					case "stall":
						var one [1]byte
						_, err := g.coord.Read(one[:])
						return err
					case "op":
						op = opAllGather
					case "version":
						part[0] = math.Float32frombits(v41EPWireVersion - 1)
					case "identity":
						part[2] = math.Float32frombits(math.Float32bits(part[2]) ^ 1)
					case "status":
						part[1] = math.Float32frombits(1)
					case "nonfinite":
						part[len(part)-1] = float32(math.Inf(1))
					case "short-vector":
						part = part[:len(part)-1]
					}
					payload := append([]byte{byte(op)}, encodeF32(part)...)
					var err error
					switch mode {
					case "trailing":
						// Keep the frame size right but advertise a shorter vector.
						binary.LittleEndian.PutUint32(payload[1:5], uint32(len(part)-1))
						err = writeFrame(g.coord, payload)
					case "oversize":
						var header [4]byte
						binary.LittleEndian.PutUint32(header[:], ^uint32(0))
						_, err = g.coord.Write(header[:])
					case "truncated":
						var header [4]byte
						binary.LittleEndian.PutUint32(header[:], uint32(len(payload)))
						_, err = g.coord.Write(header[:])
						if err == nil {
							_, err = g.coord.Write(payload[:5])
						}
						_ = g.coord.Close()
						return errors.New("injected truncated frame")
					default:
						err = writeFrame(g.coord, payload)
					}
					if err != nil {
						return err
					}
					reply, err := readV41EPVector(g.coord, v41EPHeaderWords+1, false)
					if err != nil {
						return err
					}
					_, err = v41EPReducedRow(reply, round.digest, 1)
					return err
				}
				round.sum = func(int, []routePick, [][]float32, int) ([]float32, error) {
					calls[g.Rank()]++
					return []float32{1}, nil
				}
				c := v41EPTestCollective(g)
				c.timeout = 200 * time.Millisecond
				_, err := c.reduce(part, round)
				if err != nil && !errors.Is(err, ErrV41ExpertParallel) {
					return fmt.Errorf("failure lost EP classification: %v", err)
				}
				return err
			})
			for rank, err := range errs {
				if err == nil || (rank < 2 && !errors.Is(err, ErrV41ExpertParallel)) {
					t.Fatalf("rank %d admitted malformed %s or lost EP classification: %v", rank, mode, err)
				}
			}
			if !reflect.DeepEqual(calls, []int{0, 0, 0}) {
				t.Fatalf("malformed rows reached canonical sum: %v", calls)
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41ExpertParallelReducedResponseValidation(t *testing.T) {
	digest := sha256.Sum256([]byte("response"))
	for _, mode := range []string{"version", "identity", "status", "nonfinite", "short", "failure-result"} {
		reply := make([]float32, v41EPHeaderWords+1)
		v41EPHeader(reply, digest, false)
		switch mode {
		case "version":
			reply[0] = math.Float32frombits(v41EPWireVersion - 1)
		case "identity":
			reply[2] = math.Float32frombits(math.Float32bits(reply[2]) ^ 1)
		case "status":
			reply[1] = math.Float32frombits(2)
		case "nonfinite":
			reply[len(reply)-1] = float32(math.NaN())
		case "short":
			reply = reply[:v41EPHeaderWords]
		case "failure-result":
			reply[1] = math.Float32frombits(1)
		}
		if _, err := v41EPReducedRow(reply, digest, 1); err == nil {
			t.Fatalf("accepted %s response", mode)
		}
	}
	owners, _ := ExpertParallelPlan(10, 3)
	if _, err := v41EPGatherPlan(nil, owners, v41EPMaxWords); err == nil {
		t.Fatal("oversized result admitted before allocation")
	}
	if _, err := v41EPGatherPlan([]routePick{{expert: 0, weight: 1}, {expert: 8, weight: 1}}, owners, v41EPMaxWords/2); err == nil {
		t.Fatal("oversized aggregate admitted before allocation")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41ExpertParallelMalformedResponseTCP(t *testing.T) {
	owners, _ := ExpertParallelPlan(10, 3)
	for _, mode := range []string{"op", "version", "identity", "status", "nonfinite", "short", "trailing", "oversize", "truncated", "failure-result"} {
		t.Run(mode, func(t *testing.T) {
			round := v41EPReduction{owners: owners, width: 1, digest: sha256.Sum256([]byte(mode))}
			errs := v41EPTestGroup(t, func(g *DistComm) error {
				if g.Rank() == 0 {
					for rank := 1; rank < g.Size(); rank++ {
						if _, err := readV41EPVector(g.workers[rank], v41EPHeaderWords, true); err != nil {
							return err
						}
					}
					reply := make([]float32, v41EPHeaderWords+1)
					v41EPHeader(reply, round.digest, false)
					op := opV41EPReduce
					switch mode {
					case "op":
						op = opAllGather
					case "version":
						reply[0] = math.Float32frombits(v41EPWireVersion - 1)
					case "identity":
						reply[2] = math.Float32frombits(math.Float32bits(reply[2]) ^ 1)
					case "status":
						reply[1] = math.Float32frombits(2)
					case "nonfinite":
						reply[len(reply)-1] = float32(math.NaN())
					case "short":
						reply = reply[:v41EPHeaderWords]
					case "failure-result":
						reply[1] = math.Float32frombits(1)
					}
					payload := append([]byte{byte(op)}, encodeF32(reply)...)
					if mode == "trailing" {
						binary.LittleEndian.PutUint32(payload[1:5], uint32(len(reply)-1))
					}
					for rank := 1; rank < g.Size(); rank++ {
						conn := g.workers[rank]
						if mode == "oversize" || mode == "truncated" {
							var header [4]byte
							size := uint32(len(payload))
							if mode == "oversize" {
								size = ^uint32(0)
							}
							binary.LittleEndian.PutUint32(header[:], size)
							if _, err := conn.Write(header[:]); err != nil {
								return err
							}
						} else if err := writeFrame(conn, payload); err != nil {
							return err
						}
					}
					return nil
				}
				c := v41EPTestCollective(g)
				part := v41EPTestPart(round, g.Rank(), nil)
				got, err := c.reduce(part, round)
				if got != nil || !errors.Is(err, ErrV41ExpertParallel) || c.rounds.Load() != 0 {
					return fmt.Errorf("malformed %s response admitted: %v / %v", mode, got, err)
				}
				if _, repeated := c.reduce(part, round); repeated != err {
					return errors.New("malformed response did not make failure sticky")
				}
				return nil
			})
			for rank, err := range errs {
				if err != nil {
					t.Fatalf("rank %d: %v", rank, err)
				}
			}
		})
	}
}

type v41EPRootScopeBackend struct {
	compute.Backend
	owner sync.Mutex
}

func (b *v41EPRootScopeBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }

// The actual Session/coordinator path keeps the root's existing execution
// scope through the selected sum and its failure cleanup. This software owner
// witness does not advertise DeviceMemory or qualify a physical backend.
// fak-test:runtime medium est=2s lane=default
func TestV41ExpertParallelRootSelectedSumSession(t *testing.T) {
	t.Run("error", func(t *testing.T) { v41EPRootSelectedSumSession(t, nil) })
	t.Run("panic-identity", func(t *testing.T) { v41EPRootSelectedSumSession(t, &struct{ private int }{7}) })
}

func v41EPRootSelectedSumSession(t *testing.T, panicValue any) {
	t.Helper()
	full := v41FullStepModel(t)
	binding := v41EPFixtureBinding(full)
	owners, _ := ExpertParallelPlan(full.Cfg.NumExperts, 3)
	models := make([]*Model, 3)
	for rank, band := range owners.Shards {
		models[rank] = v41EPFixtureShard(full, band)
	}
	sessions, calls := make([]*Session, 3), make([]int, 3)
	selected := errors.New("selected root sum cleanup witness")
	errs := v41EPTestGroup(t, func(g *DistComm) error {
		rank, m := g.Rank(), models[g.Rank()]
		if err := v41EPConfigure(m, g, binding); err != nil {
			return err
		}
		backend := &v41EPRootScopeBackend{Backend: compute.Default()}
		newSession := func() *Session {
			s := m.NewBackendSession(backend)
			sessions[rank] = s
			st := s.v41State()
			st.setRoutedSum(func(layer int, picks []routePick, rows [][]float32, width int) ([]float32, error) {
				calls[rank]++
				if rank != 0 || s.backendExecutionScope == nil || !s.backendExecutionScope.live || s.backendExecutionScope.owner != &backend.owner {
					return nil, errors.New("root sum lost the calling Session's execution scope")
				}
				if backend.owner.TryLock() {
					backend.owner.Unlock()
					return nil, errors.New("root sum did not hold the backend owner")
				}
				if calls[rank] == 4 {
					return v41SelectedDeviceCall(s, layer, "v41-routed-sum", func(*string) ([]float32, error) {
						if panicValue != nil {
							panic(panicValue)
						}
						return nil, selected
					})
				}
				return v41FullRoutedSum(layer, picks, rows, width)
			})
			return s
		}
		if rank != 0 {
			return RunEPFollower(g, newSession)
		}
		coord, err := NewEPDecodeCoordinator(g)
		if err != nil {
			return err
		}
		m.SetEPDecodeCoordinator(coord)
		s := newSession()
		defer s.Close()
		if panicValue == nil {
			return panicAsError(func() { s.Prefill([]int{1, 2, 3}); s.Step(4) })
		}
		var caught any
		func() {
			defer func() { caught = recover() }()
			s.Prefill([]int{1, 2, 3})
			s.Step(4)
		}()
		if caught != panicValue {
			return fmt.Errorf("selected root panic identity changed: %v", caught)
		}
		return m.epColl.(*v41ExpertParallelCollective).failure
	})
	if (panicValue == nil && !errors.Is(errs[0], selected)) || !errors.Is(errs[0], ErrV41ExpertParallel) || !reflect.DeepEqual(calls, []int{4, 0, 0}) {
		t.Fatalf("selected root error/sum ownership changed: errors=%v calls=%v", errs, calls)
	}
	for rank, s := range sessions {
		if errs[rank] == nil || s == nil || s.epSeqLen() != 3 || s.modelWeightsHeld || s.backendExecutionScope != nil {
			t.Fatalf("rank %d failed round committed or retained ownership: session=%p err=%v", rank, s, errs[rank])
		}
	}
	if !sessions[0].BackendSessionClosed() {
		t.Fatal("selected root sum did not close its backend Session")
	}
}

type v41EPFailedReplyConn struct {
	net.Conn
	cause error
}

func (c v41EPFailedReplyConn) Write(p []byte) (int, error) {
	// Leave a genuine partial frame on the wire before reporting failure.
	n, _ := c.Conn.Write(p[:min(2, len(p))])
	return n, c.cause
}

// TCP cannot retract a row already received by an earlier peer. A later
// response failure must invalidate the group and every subsequent round;
// it does not promise an atomic all-rank rollback without a commit protocol.
// fak-test:runtime fast est=100ms lane=default
func TestV41ExpertParallelRootResponseWriteFailure(t *testing.T) {
	owners, _ := ExpertParallelPlan(10, 3)
	selected := errors.New("injected partial root response")
	calls := make([]int, 3)
	errs := v41EPTestGroup(t, func(g *DistComm) error {
		if g.Rank() == 0 {
			g.workers[2] = v41EPFailedReplyConn{Conn: g.workers[2], cause: selected}
		}
		round := v41EPReduction{owners: owners, width: 1, digest: sha256.Sum256([]byte("partial response"))}
		round.sum = func(int, []routePick, [][]float32, int) ([]float32, error) {
			calls[g.Rank()]++
			return []float32{0}, nil
		}
		c := v41EPTestCollective(g)
		part := v41EPTestPart(round, g.Rank(), nil)
		got, err := c.reduce(part, round)
		if g.Rank() == 1 {
			if err != nil || len(got) != 1 || got[0] != 0 || c.rounds.Load() != 1 {
				return fmt.Errorf("earlier peer did not consume its completed response: %v / %v", got, err)
			}
			_, err = c.reduce(part, round)
		} else if c.rounds.Load() != 0 || (g.Rank() == 0 && !errors.Is(err, selected)) {
			return fmt.Errorf("failed response counted or lost root cause: %v", err)
		}
		if !errors.Is(err, ErrV41ExpertParallel) {
			return fmt.Errorf("failed group reused on rank %d: %v", g.Rank(), err)
		}
		if got, repeated := c.reduce(part, round); got != nil || repeated != err {
			return errors.New("response failure did not remain terminal")
		}
		return nil
	})
	for rank, err := range errs {
		if err != nil {
			t.Fatalf("rank %d: %v", rank, err)
		}
	}
	if !reflect.DeepEqual(calls, []int{1, 0, 0}) {
		t.Fatalf("response failure replayed canonical sum: %v", calls)
	}
}
