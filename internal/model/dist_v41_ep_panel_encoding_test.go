package model

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"reflect"
	"testing"
)

type v41EPPanelEncodingConn struct {
	net.Conn
	fault   error
	writes  [][]byte
	payload []byte
}

func (c *v41EPPanelEncodingConn) Write(p []byte) (int, error) {
	c.writes = append(c.writes, append([]byte(nil), p...))
	if len(c.writes) == 2 {
		// Test-only retention proves shared identity and no later mutation.
		// Production writers need not retain bytes after synchronous Write.
		c.payload = p
		if c.fault != nil {
			n, _ := c.Conn.Write(p[:min(2, len(p))])
			return n, c.fault
		}
	}
	return c.Conn.Write(p)
}

// Exact wire bytes and selected-failure order, using real TCP. Allocation
// identity is a software mechanism witness, not a native memory/speed metric.
// fak-test:runtime fast est=100ms lane=default
func TestV41ExpertParallelPanelEncoding(t *testing.T) {
	t.Run("zero_peers", func(t *testing.T) {
		r, all := v41EPPanelTestFixture(3, 1, false)
		r.owners, _ = ExpertParallelPlan(10, 1)
		g := &DistComm{size: 1, workers: make([]net.Conn, 1)}
		c := v41EPTestCollective(g)
		got, err := c.reducePanel(v41EPPanelTestOwner(r, 0, all), r)
		if err != nil || len(got) != 3 || v41EPPanelTestStats(c).CompletedRows != 3 {
			t.Fatalf("zero-peer reduction changed: %v / %v", got, err)
		}
	})
	for _, mode := range []string{"healthy", "first_write_failure", "later_write_failure", "sum_and_write_failure"} {
		t.Run(mode, func(t *testing.T) {
			r, all := v41EPPanelTestFixture(3, 1, false)
			plan, err := v41EPPanelPlan(r.owners, r.picks, r.width)
			if err != nil {
				t.Fatal(err)
			}
			// Independent legacy wire construction, with no new encoder call.
			reply := make([]float32, plan.replyWords)
			v41EPPanelHeader(reply, r.identity(), len(r.picks), false)
			at := v41EPPanelHeaderWords
			for token, picks := range r.picks {
				row, err := v41FullRoutedSum(r.layer, picks, all[token], r.width)
				if err != nil {
					t.Fatal(err)
				}
				v41EPHeader(reply[at:], r.digests[token], false)
				at += v41EPHeaderWords
				copy(reply[at:], row)
				at += r.width
			}
			successWire := append([]byte{byte(opV41EPPanelReduce)}, encodeF32(reply)...)
			failure := make([]float32, v41EPPanelHeaderWords)
			v41EPPanelHeader(failure, r.identity(), len(r.picks), true)
			failureWire := append([]byte{byte(opV41EPPanelReduce)}, encodeF32(failure)...)
			writeFault, sumFault := errors.New("panel encoding write failure"), errors.New("panel encoding sum failure")
			calls := make([]int, 3)
			observed := make([]*v41EPPanelEncodingConn, 3)
			errs := v41EPTestGroup(t, func(g *DistComm) error {
				rank := g.Rank()
				if rank == 0 {
					for peer := 1; peer < g.Size(); peer++ {
						conn := &v41EPPanelEncodingConn{Conn: g.workers[peer]}
						if peer == 1 && (mode == "first_write_failure" || mode == "sum_and_write_failure") || peer == 2 && mode == "later_write_failure" {
							conn.fault = writeFault
						}
						observed[peer], g.workers[peer] = conn, conn
					}
				}
				local := r
				local.sum = func(layer int, picks []routePick, rows [][]float32, width int) ([]float32, error) {
					calls[rank]++
					if mode == "sum_and_write_failure" {
						return nil, sumFault
					}
					return v41FullRoutedSum(layer, picks, rows, width)
				}
				c := v41EPTestCollective(g)
				rows := v41EPPanelTestOwner(r, rank, all)
				got, err := c.reducePanel(rows, local)
				stats := v41EPPanelTestStats(c)
				if mode == "healthy" {
					if err != nil || len(got) != 3 || stats.CompletedGathers != 1 || stats.CompletedRows != 3 {
						return fmt.Errorf("healthy encoded response: %+v / %v", stats, err)
					}
					return nil
				}
				if mode == "later_write_failure" && rank == 1 && err == nil {
					// Earlier success is irrevocable; root abort can instead race
					// this peer's deadline-clear and prevent local completion.
					if len(got) != 3 || stats.CompletedGathers != 1 || stats.CompletedRows != 3 {
						return fmt.Errorf("earlier response lost completed rows: %+v", stats)
					}
					_, err = c.reducePanel(rows, local)
				} else if got != nil || stats.CompletedGathers != 0 || stats.CompletedRows != 0 {
					return fmt.Errorf("failed encoding credited rows: %+v / %v", stats, err)
				}
				if !errors.Is(err, ErrV41ExpertParallel) {
					return fmt.Errorf("encoding failure lost terminal type: %v", err)
				}
				if rank == 0 {
					want := writeFault
					if mode == "sum_and_write_failure" {
						want = sumFault
					}
					if !errors.Is(err, want) || mode == "sum_and_write_failure" && errors.Is(err, writeFault) {
						return fmt.Errorf("response changed primary cause: %v", err)
					}
				}
				before := calls[rank]
				if repeated, again := c.reducePanel(rows, local); repeated != nil || again != err || calls[rank] != before {
					return errors.New("response failure replayed or changed sticky identity")
				}
				return nil
			})
			for rank, err := range errs {
				if err != nil {
					t.Fatalf("rank %d: %v", rank, err)
				}
			}
			wantCalls := 3
			if mode == "sum_and_write_failure" {
				wantCalls = 1
			}
			if !reflect.DeepEqual(calls, []int{wantCalls, 0, 0}) {
				t.Fatalf("root callbacks changed: %v", calls)
			}
			for peer := 1; peer < 3; peer++ {
				conn := observed[peer]
				want := successWire
				if mode == "sum_and_write_failure" || mode == "first_write_failure" && peer == 2 {
					want = failureWire
				}
				if len(conn.writes) != 2 || len(conn.writes[0]) != 4 || int(binary.LittleEndian.Uint32(conn.writes[0])) != len(want) || !bytes.Equal(conn.writes[1], want) || !bytes.Equal(conn.payload, want) {
					t.Fatalf("peer %d wire bytes changed or cached payload mutated", peer)
				}
			}
			shared := &observed[1].payload[0] == &observed[2].payload[0]
			if shared != (mode != "first_write_failure") {
				t.Fatalf("response backing reuse/transition differs: shared=%v", shared)
			}
		})
	}
}
