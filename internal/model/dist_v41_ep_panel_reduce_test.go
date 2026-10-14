package model

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

func v41EPPanelTestFixture(tokens, width int, skew bool) (v41EPPanelReduction, [][][]float32) {
	owners, _ := ExpertParallelPlan(10, 3)
	r := v41EPPanelReduction{owners: owners, width: width, picks: make([][]routePick, tokens), digests: make([][32]byte, tokens)}
	rows := make([][][]float32, tokens)
	for token := range rows {
		r.digests[token] = sha256.Sum256([]byte(fmt.Sprint("original ordinal and token ", token)))
		experts := []int{8, 0, 8, 5, 1, 6}
		if skew {
			experts = []int{8, 8, 8, 9, 9, 9}
		}
		values := []float32{1 << 24, 1, -(1 << 24), 1, 1.0 / 256, 0}
		rows[token] = make([][]float32, len(experts))
		for slot, expert := range experts {
			r.picks[token] = append(r.picks[token], routePick{expert: expert, weight: float32(slot+1) / 8})
			rows[token][slot] = make([]float32, width)
			for i := range rows[token][slot] {
				rows[token][slot][i] = values[slot]
			}
		}
	}
	return r, rows
}

func v41EPPanelTestOwner(r v41EPPanelReduction, rank int, all [][][]float32) [][]float32 {
	rows := make([][]float32, len(r.picks))
	band := r.owners.Shards[rank]
	for token, picks := range r.picks {
		for slot, pick := range picks {
			if pick.expert >= band.Lo && pick.expert < band.Hi {
				rows[token] = append(rows[token], all[token][slot]...)
			}
		}
	}
	return rows
}

func v41EPPanelTestStats(c *v41ExpertParallelCollective) V41ExpertParallelTransportStats {
	return (&Model{epColl: c}).V41ExpertParallelTransportStats()
}

// The root covers the bounded protocol, with the existing Session witness
// providing packed owner-cohort math and Prefill/Step/history continuation.
// All traffic is real loopback TCP; this is no device or network speed claim.
// fak-test:runtime medium est=2s lane=default
func TestV41ExpertParallelPanelReduce(t *testing.T) {
	t.Run("cumulative_results", func(t *testing.T) {
		r, _ := v41EPPanelTestFixture(3, 1, true)
		for _, test := range []struct {
			tokens, width int
			backing       uint64
			upper         uint64
		}{
			{35, 256, 37328, 37988}, // actual 32+3-token CPU Session witness
			{35, 5120, 718288, 718948},
			{33, 256, 35156, 35816}, // one panel plus a scalar tail
			{1, 256, 1064, 1064},    // scalar follower retains its 10-word header
		} {
			picks := make([][]routePick, test.tokens)
			for token := range picks {
				picks[token] = r.picks[0]
			}
			storage, err := v41EPPrefillResultStorageBound(test.tokens, test.width)
			if err != nil || storage.backingBytes != test.upper || storage.totalBytes != storage.backingBytes+storage.rowHeaderBytes {
				t.Fatalf("cumulative bound %+v: %+v / %v", test, storage, err)
			}
			var backing uint64
			for first := 0; first < len(picks); {
				last := v41EPPrefillWindowEnd(r.owners, picks, test.width, first)
				if last <= first {
					t.Fatal("storage witness made no progress")
				}
				words := v41EPHeaderWords + test.width
				if last-first > 1 {
					plan, err := v41EPPanelPlan(r.owners, picks[first:last], test.width)
					if err != nil {
						t.Fatal(err)
					}
					words = plan.replyWords
				}
				backing += uint64(4 * words)
				first = last
			}
			if backing != test.backing || backing > storage.backingBytes {
				t.Fatalf("retained slabs %+v: %d, upper %d", test, backing, storage.backingBytes)
			}
			if test.tokens == 35 && test.width == 256 && storage.rowHeaderBytes == 35*24 && backing+storage.rowHeaderBytes != 38168 {
				t.Fatal("35-token retained response slabs plus row descriptors differ")
			}
		}
		maxInt := int(^uint(0) >> 1)
		one, _ := v41EPPrefillResultStorageBound(1, 1)
		descriptorProbe := maxInt / (66 + int(one.rowHeaderBytes)/2)
		for _, shape := range [][2]int{{0, 1}, {1, 0}, {maxInt, 1}, {1, maxInt}, {maxInt / 48, 1}, {descriptorProbe, 1}} {
			if storage, err := v41EPPrefillResultStorageBound(shape[0], shape[1]); err == nil || storage.totalBytes != 0 {
				t.Fatalf("invalid/overflow cumulative shape %v admitted: %+v / %v", shape, storage, err)
			}
		}
	})
	t.Run("admission", func(t *testing.T) {
		r, _ := v41EPPanelTestFixture(32, 5120, true)
		plan, err := v41EPPanelPlan(r.owners, r.picks, r.width)
		if err != nil || plan.liveBytes != 17872236 || 9+4*plan.requests[2] != 3933493 || 9+4*plan.replyWords != 656693 {
			t.Fatalf("Flash bound: %+v / %v", plan, err)
		}
		for _, mode := range []string{"singleton", "tokens", "picks", "width", "overflow", "nonfinite", "expert", "owner-bytes", "aggregate"} {
			bad := r
			bad.picks = append([][]routePick(nil), r.picks...)
			switch mode {
			case "singleton":
				bad.picks = bad.picks[:1]
			case "tokens":
				bad.picks = append(bad.picks, nil)
			case "picks":
				bad.picks[0] = append(append([]routePick(nil), bad.picks[0]...), routePick{expert: 8, weight: 1})
			case "width":
				bad.width = 0
			case "overflow":
				bad.width = int(^uint(0) >> 1)
			case "nonfinite", "expert":
				bad.picks[0] = append([]routePick(nil), bad.picks[0]...)
				if mode == "expert" {
					bad.picks[0][0].expert = bad.owners.Dim
				} else {
					bad.picks[0][0].weight = float32(math.NaN())
				}
			case "owner-bytes":
				bad.width = v41EPOwnerPanelMaxBytes / 4
			case "aggregate":
				bad.owners, _ = ExpertParallelPlan(20000, 20000)
			}
			if plan, err := v41EPPanelPlan(bad.owners, bad.picks, bad.width); err == nil || plan.requests != nil {
				t.Fatalf("%s admitted or allocated rejected plan: %+v / %v", mode, plan, err)
			}
		}
		// Common geometry can force a shorter window, independently of which
		// rank owns any of its picks. This is preselection, never a retry.
		owners, _ := ExpertParallelPlan(20000, 20000)
		last := len(r.picks)
		for last > 1 {
			if _, err := v41EPPanelPlan(owners, r.picks[:last], r.width); err == nil {
				break
			}
			last--
		}
		if last <= 1 || last >= 32 {
			t.Fatalf("expected a bounded shorter common window, got %d", last)
		}
	})

	for _, distribution := range []string{"balanced", "skew", "empty"} {
		for _, tokens := range []int{2, 3, 32} {
			t.Run(fmt.Sprintf("exact_%s_rows_%d", distribution, tokens), func(t *testing.T) {
				r, all := v41EPPanelTestFixture(tokens, 256, distribution == "skew")
				if distribution == "empty" {
					for token := range r.picks {
						r.picks[token], all[token] = nil, nil
					}
				}
				want := make([][]float32, tokens)
				for token := range want {
					var err error
					want[token], err = v41FullRoutedSum(0, r.picks[token], all[token], r.width)
					if err != nil {
						t.Fatal(err)
					}
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
					local := r
					local.sum = func(layer int, picks []routePick, got [][]float32, width int) ([]float32, error) {
						token := calls[g.Rank()]
						calls[g.Rank()]++
						if g.Rank() != 0 || token >= tokens || !reflect.DeepEqual(picks, r.picks[token]) {
							return nil, errors.New("wrong root, token or slot ordering")
						}
						for slot := range got {
							if err := v41EPCheckBits(got[slot], all[token][slot]); err != nil {
								return nil, err
							}
							if cap(got[slot]) != width {
								return nil, errors.New("uncapped private contribution")
							}
						}
						return v41FullRoutedSum(layer, picks, got, width)
					}
					c := v41EPTestCollective(g)
					got, err := c.reducePanel(v41EPPanelTestOwner(r, g.Rank(), all), local)
					if err != nil || len(got) != tokens {
						return fmt.Errorf("panel result: %d / %v", len(got), err)
					}
					for token := range got {
						if err := v41EPCheckBits(got[token], want[token]); err != nil {
							return err
						}
						if cap(got[token]) != r.width {
							return errors.New("uncapped result row")
						}
					}
					var sent, received uint64
					for _, wire := range wires {
						sent += wire.sent
						received += wire.received
					}
					stats := v41EPPanelTestStats(c)
					if stats.CompletedGathers != 1 || stats.CompletedRows != uint64(tokens) || stats.SentFrameBytes != sent || stats.ReceivedFrameBytes != received {
						return fmt.Errorf("completed panel stats %+v, wire %d/%d", stats, sent, received)
					}
					return nil
				})
				for rank, err := range errs {
					if err != nil {
						t.Fatalf("rank %d: %v", rank, err)
					}
				}
				if !reflect.DeepEqual(calls, []int{tokens, 0, 0}) {
					t.Fatalf("root exact sums: %v", calls)
				}
			})
		}
	}

	for _, mode := range []string{"op", "envelope", "rows", "status", "later-identity", "later-status", "later-nonfinite", "reorder", "short", "trailing", "oversize", "truncated", "disconnect", "stall"} {
		t.Run("refuse_"+mode, func(t *testing.T) {
			r, all := v41EPPanelTestFixture(3, 1, true)
			plan, _ := v41EPPanelPlan(r.owners, r.picks, r.width)
			calls := make([]int, 3)
			errs := v41EPTestGroup(t, func(g *DistComm) error {
				local := r
				local.sum = func(int, []routePick, [][]float32, int) ([]float32, error) {
					calls[g.Rank()]++
					return []float32{0}, nil
				}
				rows := v41EPPanelTestOwner(r, g.Rank(), all)
				if g.Rank() == 2 {
					part, err := r.pack(rows, plan, 2, r.identity())
					if err != nil {
						return err
					}
					at := v41EPPanelHeaderWords + 2*(v41EPHeaderWords+6)
					switch mode {
					case "disconnect":
						_ = g.coord.Close()
						return errors.New("injected disconnect")
					case "stall":
						var one [1]byte
						_, err := g.coord.Read(one[:])
						return err
					case "envelope":
						part[2] = math.Float32frombits(math.Float32bits(part[2]) ^ 1)
					case "rows":
						part[10] = math.Float32frombits(2)
					case "status":
						part[1] = math.Float32frombits(2)
					case "later-identity":
						part[at+2] = math.Float32frombits(math.Float32bits(part[at+2]) ^ 1)
					case "later-status":
						part[at+1] = math.Float32frombits(1)
					case "later-nonfinite":
						part[len(part)-1] = float32(math.Inf(1))
					case "reorder":
						for i := 0; i < v41EPHeaderWords+6; i++ {
							part[v41EPPanelHeaderWords+i], part[at+i] = part[at+i], part[v41EPPanelHeaderWords+i]
						}
					case "short":
						part = part[:len(part)-1]
					}
					payload := append([]byte{byte(opV41EPPanelReduce)}, encodeF32(part)...)
					if mode == "op" {
						payload[0] = byte(opV41EPReduce)
					}
					if mode == "trailing" {
						binary.LittleEndian.PutUint32(payload[1:5], uint32(len(part)-1))
					}
					if mode == "oversize" || mode == "truncated" {
						var header [4]byte
						n := ^uint32(0)
						if mode == "truncated" {
							n = uint32(len(payload))
						}
						binary.LittleEndian.PutUint32(header[:], n)
						_, err = g.coord.Write(header[:])
						if mode == "truncated" {
							_, _ = g.coord.Write(payload[:5])
							_ = g.coord.Close()
							return errors.New("injected truncation")
						}
					} else {
						err = writeFrame(g.coord, payload)
					}
					if err != nil {
						return err
					}
					reply, err := readV41EPPanelVector(g.coord, plan.replyWords, false)
					if err == nil {
						_, err = r.response(reply, r.identity())
					}
					return err
				}
				c := v41EPTestCollective(g)
				c.timeout = 200 * time.Millisecond
				out, err := c.reducePanel(rows, local)
				if out != nil || !errors.Is(err, ErrV41ExpertParallel) || v41EPPanelTestStats(c).CompletedRows != 0 {
					return fmt.Errorf("malformed panel published rows or lost terminal error: %v", err)
				}
				if retry, repeated := c.reducePanel(rows, local); retry != nil || repeated != err {
					return errors.New("malformed panel was reusable")
				}
				return err
			})
			for rank, err := range errs {
				if err == nil || rank < 2 && !errors.Is(err, ErrV41ExpertParallel) {
					t.Fatalf("rank %d admitted %s: %v", rank, mode, err)
				}
			}
			if !reflect.DeepEqual(calls, []int{0, 0, 0}) {
				t.Fatalf("later invalid row reached root math: %v", calls)
			}
		})
	}

	for _, mode := range []string{"version", "envelope", "rows", "status", "later-identity", "later-nonfinite", "short", "failure-result", "op", "oversize"} {
		t.Run("response_"+mode, func(t *testing.T) {
			r, all := v41EPPanelTestFixture(3, 1, true)
			plan, _ := v41EPPanelPlan(r.owners, r.picks, r.width)
			errs := v41EPTestGroup(t, func(g *DistComm) error {
				if g.Rank() == 0 {
					parts := make([][]float32, 3)
					parts[0], _ = r.pack(v41EPPanelTestOwner(r, 0, all), plan, 0, r.identity())
					for rank := 1; rank < 3; rank++ {
						var err error
						parts[rank], err = readV41EPPanelVector(g.workers[rank], plan.requests[rank], true)
						if err != nil {
							return err
						}
					}
					reply, err := r.reduce(parts, plan, r.identity())
					if err != nil {
						return err
					}
					at := v41EPPanelHeaderWords + 2*(v41EPHeaderWords+r.width)
					switch mode {
					case "version":
						reply[0] = math.Float32frombits(v41EPWireVersion)
					case "envelope":
						reply[2] = math.Float32frombits(math.Float32bits(reply[2]) ^ 1)
					case "rows":
						reply[10] = math.Float32frombits(2)
					case "status":
						reply[1] = math.Float32frombits(2)
					case "later-identity":
						reply[at+2] = math.Float32frombits(math.Float32bits(reply[at+2]) ^ 1)
					case "later-nonfinite":
						reply[len(reply)-1] = float32(math.NaN())
					case "short":
						reply = reply[:len(reply)-1]
					case "failure-result":
						reply[1] = math.Float32frombits(1)
					}
					for rank := 1; rank < 3; rank++ {
						if mode == "oversize" {
							var header [4]byte
							binary.LittleEndian.PutUint32(header[:], ^uint32(0))
							_, err = g.workers[rank].Write(header[:])
						} else if mode == "op" {
							err = writeRequest(g.workers[rank], opV41EPReduce, reply)
						} else {
							err = writeV41EPPanelVector(g.workers[rank], reply)
						}
						if err != nil {
							return err
						}
					}
					return nil
				}
				c := v41EPTestCollective(g)
				rows := v41EPPanelTestOwner(r, g.Rank(), all)
				out, err := c.reducePanel(rows, r)
				if out != nil || !errors.Is(err, ErrV41ExpertParallel) || v41EPPanelTestStats(c).CompletedRows != 0 {
					return fmt.Errorf("bad response returned rows or counted completion: %v", err)
				}
				if retry, repeated := c.reducePanel(rows, r); retry != nil || repeated != err {
					return errors.New("bad response was reusable")
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

	for _, mode := range []string{"error", "panic", "nonfinite", "short"} {
		t.Run("later_sum_"+mode, func(t *testing.T) {
			r, all := v41EPPanelTestFixture(3, 1, false)
			selected := errors.New("private selected panel root cause")
			calls := make([]int, 3)
			errs := v41EPTestGroup(t, func(g *DistComm) error {
				local := r
				local.sum = func(layer int, picks []routePick, outputs [][]float32, width int) ([]float32, error) {
					calls[g.Rank()]++
					if calls[g.Rank()] == 2 {
						switch mode {
						case "error":
							return nil, selected
						case "panic":
							panic(selected)
						case "nonfinite":
							return []float32{float32(math.NaN())}, nil
						case "short":
							return nil, nil
						}
					}
					return v41FullRoutedSum(layer, picks, outputs, width)
				}
				c := v41EPTestCollective(g)
				rows := v41EPPanelTestOwner(r, g.Rank(), all)
				var out [][]float32
				var err error
				p := panicAsError(func() { out, err = c.reducePanel(rows, local) })
				if g.Rank() == 0 && mode == "panic" {
					if p != selected {
						return fmt.Errorf("lost original root panic: %v", p)
					}
					err = c.failure
				} else if p != nil {
					return fmt.Errorf("unexpected peer panic: %v", p)
				}
				if out != nil || !errors.Is(err, ErrV41ExpertParallel) || v41EPPanelTestStats(c).CompletedRows != 0 {
					return fmt.Errorf("partial panel escaped: %v", err)
				}
				if g.Rank() == 0 && (mode == "error" || mode == "panic") && !errors.Is(err, selected) {
					return errors.New("root cause lost")
				}
				if g.Rank() != 0 && strings.Contains(err.Error(), selected.Error()) {
					return errors.New("private root cause leaked over wire")
				}
				if retry, repeated := c.reducePanel(rows, local); retry != nil || repeated != err {
					return errors.New("selected panel replayed")
				}
				return nil
			})
			for rank, err := range errs {
				if err != nil {
					t.Fatalf("rank %d: %v", rank, err)
				}
			}
			if !reflect.DeepEqual(calls, []int{2, 0, 0}) {
				t.Fatalf("sum failed to stop or replayed: %v", calls)
			}
		})
	}

	t.Run("partial_response_write", func(t *testing.T) {
		r, all := v41EPPanelTestFixture(3, 1, false)
		selected := errors.New("injected panel response failure")
		calls := make([]int, 3)
		errs := v41EPTestGroup(t, func(g *DistComm) error {
			if g.Rank() == 0 {
				g.workers[2] = v41EPFailedReplyConn{Conn: g.workers[2], cause: selected}
			}
			local := r
			local.sum = func(layer int, picks []routePick, outputs [][]float32, width int) ([]float32, error) {
				calls[g.Rank()]++
				return v41FullRoutedSum(layer, picks, outputs, width)
			}
			c := v41EPTestCollective(g)
			rows := v41EPPanelTestOwner(r, g.Rank(), all)
			out, err := c.reducePanel(rows, local)
			stats := v41EPPanelTestStats(c)
			if g.Rank() == 1 {
				// Root abort can race this peer's deadline-clear/completion.
				// Only a fully completed local exchange may credit its rows.
				if err == nil {
					if len(out) != 3 || stats.CompletedGathers != 1 || stats.CompletedRows != 3 {
						return fmt.Errorf("earlier peer lost completed response: %+v", stats)
					}
					_, err = c.reducePanel(rows, local)
				} else if out != nil || stats.CompletedGathers != 0 || stats.CompletedRows != 0 {
					return fmt.Errorf("incomplete earlier response was credited: %+v / %v", stats, err)
				}
			} else if out != nil || stats.CompletedGathers != 0 || stats.CompletedRows != 0 || g.Rank() == 0 && !errors.Is(err, selected) {
				return fmt.Errorf("failed response credited or root cause lost: %+v / %v", stats, err)
			}
			if !errors.Is(err, ErrV41ExpertParallel) {
				return fmt.Errorf("group was reusable: %v", err)
			}
			if retry, repeated := c.reducePanel(rows, local); retry != nil || repeated != err {
				return errors.New("response failure replayed")
			}
			return nil
		})
		for rank, err := range errs {
			if err != nil {
				t.Fatalf("rank %d: %v", rank, err)
			}
		}
		if !reflect.DeepEqual(calls, []int{3, 0, 0}) {
			t.Fatalf("partial response repeated root math: %v", calls)
		}
	})
}
