package model

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"net"
	"reflect"
	"testing"
	"time"
)

type v41EPPanelRequestConn struct {
	net.Conn
	writes, deadlines int
}

func (c *v41EPPanelRequestConn) Write(p []byte) (int, error) {
	c.writes++
	return c.Conn.Write(p)
}

func (c *v41EPPanelRequestConn) SetDeadline(deadline time.Time) error {
	c.deadlines++
	return c.Conn.SetDeadline(deadline)
}

// The byte oracle compares the direct follower snapshot with the unchanged
// root pack plus checked encoder. TCP failure cases pin pre-I/O refusal order.
// fak-test:runtime fast est=100ms lane=default
func TestV41ExpertParallelPanelRequestEncoding(t *testing.T) {
	t.Run("exact_bytes_and_private_snapshot", func(t *testing.T) {
		for _, distribution := range []string{"balanced", "skew", "empty"} {
			for _, tokens := range []int{2, 3, 32} {
				r, all := v41EPPanelTestFixture(tokens, 7, distribution == "skew")
				for token := range all {
					if distribution == "empty" {
						r.picks[token], all[token] = nil, nil
						continue
					}
					// Original picks already contain duplicates and non-sorted
					// slots. Preserve signed zero and finite BF16 subnormal bits.
					all[token][0][0] = math.Float32frombits(0x80000000)
					all[token][1][1] = math.Float32frombits(0x00010000)
				}
				plan, err := v41EPPanelPlan(r.owners, r.picks, r.width)
				if err != nil {
					t.Fatal(err)
				}
				for rank := range r.owners.Shards {
					rows := v41EPPanelTestOwner(r, rank, all)
					part, err := r.pack(rows, plan, rank, r.identity())
					if err != nil {
						t.Fatal(err)
					}
					want, err := encodeV41EPPanelVector(part)
					if err != nil {
						t.Fatal(err)
					}
					got, err := r.encodeOwnerRequest(rows, plan, rank, r.identity())
					if err != nil || len(got) != cap(got) || !bytes.Equal(got, want) {
						t.Fatalf("%s tokens=%d rank=%d direct bytes differ: %v", distribution, tokens, rank, err)
					}
					for _, row := range rows {
						for i := range row {
							row[i] = float32(math.NaN())
						}
					}
					if !bytes.Equal(got, want) {
						t.Fatal("encoded snapshot aliases owner rows")
					}
				}
			}
		}
		// This is one eliminated follower-side intermediate, not total peak.
		r, _ := v41EPPanelTestFixture(32, 1, true)
		plan, err := v41EPPanelPlan(r.owners, r.picks, 5120)
		if err != nil || 4*plan.requests[2] != 3933484 {
			t.Fatalf("worst-skew intermediate accounting: %+v / %v", plan, err)
		}
	})
	t.Run("same_local_refusal_order", func(t *testing.T) {
		for _, mode := range []string{"late_width", "late_nonfinite", "early_width", "early_nonfinite"} {
			r, all := v41EPPanelTestFixture(3, 1, true)
			plan, err := v41EPPanelPlan(r.owners, r.picks, r.width)
			if err != nil {
				t.Fatal(err)
			}
			rows := v41EPPanelTestOwner(r, 2, all)
			switch mode {
			case "late_width":
				rows[2] = rows[2][:len(rows[2])-1]
			case "late_nonfinite":
				rows[2][len(rows[2])-1] = float32(math.Inf(1))
			case "early_width":
				rows[0] = rows[0][:len(rows[0])-1]
				rows[2][0] = float32(math.NaN())
			case "early_nonfinite":
				rows[0][0] = float32(math.NaN())
				rows[2] = rows[2][:len(rows[2])-1]
			}
			part, want := r.pack(rows, plan, 2, r.identity())
			got, err := r.encodeOwnerRequest(rows, plan, 2, r.identity())
			if part != nil || want == nil || got != nil || err == nil || err.Error() != want.Error() {
				t.Fatalf("%s refusal order changed: %v / %v", mode, err, want)
			}
		}
	})
	for _, mode := range []string{"late_width_before_write", "late_nonfinite_before_write"} {
		t.Run(mode, func(t *testing.T) {
			r, all := v41EPPanelTestFixture(3, 1, true)
			calls := make([]int, 3)
			var refused *v41EPPanelRequestConn
			errs := v41EPTestGroup(t, func(g *DistComm) error {
				rank := g.Rank()
				rows := v41EPPanelTestOwner(r, rank, all)
				if rank == 2 {
					refused = &v41EPPanelRequestConn{Conn: g.coord}
					g.coord = refused
					if mode == "late_width_before_write" {
						rows[2] = rows[2][:len(rows[2])-1]
					} else {
						rows[2][len(rows[2])-1] = float32(math.NaN())
					}
				}
				local := r
				local.sum = func(layer int, picks []routePick, outputs [][]float32, width int) ([]float32, error) {
					calls[rank]++
					return v41FullRoutedSum(layer, picks, outputs, width)
				}
				c := v41EPTestCollective(g)
				got, err := c.reducePanel(rows, local)
				stats := v41EPPanelTestStats(c)
				if got != nil || !errors.Is(err, ErrV41ExpertParallel) || stats.CompletedGathers != 0 || stats.CompletedRows != 0 {
					return fmt.Errorf("local request refusal published output: %+v / %v", stats, err)
				}
				if again, repeated := c.reducePanel(rows, local); again != nil || repeated != err {
					return errors.New("refused request replayed or changed sticky error")
				}
				return nil
			})
			for rank, err := range errs {
				if err != nil {
					t.Fatalf("rank %d: %v", rank, err)
				}
			}
			if refused == nil || refused.writes != 0 || refused.deadlines != 0 || !reflect.DeepEqual(calls, []int{0, 0, 0}) {
				t.Fatalf("late local refusal passed snapshot boundary: connection=%+v calls=%v", refused, calls)
			}
		})
	}
}
