package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This is an opt-in CPU-recorder observation of the canonical scalar callback,
// not a native backend collector. Physical submissions/fences remain unknown.
type v41EPRootSumObservation struct {
	Scope                     string
	Callbacks, ValidatedRows  int
	Uploads, AddAttempts      int
	ReadAttempts, ReadReturns int
	FreedBuffers              int
	UploadBytes, ReturnBytes  uint64
	PeakOwnedTensorBytes      uint64
	PeakOwnedBuffers          int
	Physical                  *compute.BackendExecutionObservation `json:",omitempty"`
}

func v41EPRootSumObserve(b *v41SumTestBackend, calls, valid int) v41EPRootSumObservation {
	o := b.observation
	return v41EPRootSumObservation{
		Scope:     "CPU recorder HAL calls; physical submissions and fences unavailable",
		Callbacks: calls, ValidatedRows: valid, Uploads: b.uploads, AddAttempts: b.adds,
		ReadAttempts: o.readAttempts, ReadReturns: o.readReturned, FreedBuffers: len(o.freed),
		UploadBytes: o.uploadBytes, ReturnBytes: o.returnedBytes,
		PeakOwnedTensorBytes: o.peakBytes, PeakOwnedBuffers: o.peakBuffers,
	}
}

func v41EPRootSumFreed(b *v41SumTestBackend) error {
	o := b.observation
	if len(b.live) != 0 || len(o.sizes) != 0 || o.liveBytes != 0 || len(o.uploaded) != len(o.freed) {
		return fmt.Errorf("root sum retained tensors or omitted frees: live=%d bytes=%d uploaded=%d freed=%d", len(b.live), o.liveBytes, len(o.uploaded), len(o.freed))
	}
	identities := make(map[compute.Buffer]int, len(o.uploaded))
	for _, buf := range o.uploaded {
		if identities[buf] != 0 {
			return errors.New("recorder reused an owned upload identity")
		}
		identities[buf] = 1
	}
	for _, buf := range o.freed {
		if identities[buf] != 1 {
			return errors.New("root sum freed an unowned or repeated identity")
		}
		identities[buf] = 0
	}
	return nil
}

// Bare canonical-callback fixture over real TCP. This intentionally does not
// claim Session.Prefill entrypoint or native execution-owner coverage; the
// unchanged PrefillPanelTCP and RootSelectedSumSession controls cover those.
// No operation is batched, replaced or replayed by this observer.
// fak-test:runtime fast est=200ms lane=default
func TestV41ExpertParallelRootSumObservation(t *testing.T) {
	const width = 256
	for _, mode := range []string{"healthy_balanced", "healthy_skew", "early_overflow", "read_fault", "short_read", "nonfinite_read"} {
		t.Run(mode, func(t *testing.T) {
			healthy := mode == "healthy_balanced" || mode == "healthy_skew"
			full, all := v41EPPanelTestFixture(36, width, mode != "healthy_balanced")
			if mode == "early_overflow" {
				for _, row := range all[0] {
					for i := range row {
						row[i] = math.Float32frombits(0x7f7f0000) // finite maximum widened BF16
					}
				}
			}
			calls, validated := make([]int, 3), make([]int, 3)
			auditErrors := make([]error, 3)
			prefill, final := make([]v41EPRootSumObservation, 3), make([]v41EPRootSumObservation, 3)
			errs := v41EPTestGroup(t, func(g *DistComm) error {
				rank := g.Rank()
				b := newV41SumTestBackend()
				b.observation = &v41SumTestObservation{sizes: map[compute.Buffer]uint64{}}
				if _, available, err := compute.CaptureBackendExecutionSnapshot(b); available || err != nil {
					return errors.New("CPU recorder fabricated physical backend observations")
				}
				switch mode {
				case "early_overflow":
					b.faultAdd = 7 // first add of the later token; must never execute
				case "read_fault":
					b.faultRead = true
				case "short_read":
					b.shortRead = true
				case "nonfinite_read":
					b.nanRead = true
				}
				s := &Session{M: &Model{}, Backend: b}
				canonical := s.v41RoutedSumFunc()
				observed := func(layer int, picks []routePick, rows [][]float32, h int) ([]float32, error) {
					token := calls[rank]
					calls[rank]++
					if rank != 0 || token >= len(full.picks) || !reflect.DeepEqual(picks, full.picks[token]) || h != width {
						auditErrors[rank] = errors.New("root observer changed token/slot/callback identity")
					}
					result, err := canonical(layer, picks, rows, h) // exactly one original call
					if err == nil {
						validated[rank]++
					}
					if cleanup := v41EPRootSumFreed(b); cleanup != nil {
						auditErrors[rank] = cleanup // never replace a selected primary error
					}
					return result, err
				}
				c := v41EPTestCollective(g)
				r := full
				r.picks, r.digests, r.sum = full.picks[:32], full.digests[:32], observed
				rows := v41EPPanelTestOwner(r, rank, all[:32])
				wantRows := len(r.picks)
				got, err := c.reducePanel(rows, r)
				if !healthy {
					stats := v41EPPanelTestStats(c)
					if got != nil || !errors.Is(err, ErrV41ExpertParallel) || stats.CompletedGathers != 0 || stats.CompletedRows != 0 {
						return fmt.Errorf("failed observed panel credited output: %+v / %v", stats, err)
					}
					if rank == 0 {
						var operation *BackendForwardOperationError
						if !errors.As(err, &operation) || !s.BackendSessionClosed() {
							return fmt.Errorf("selected sum failure lost closing operation: %v", err)
						}
						var backend *compute.BackendError
						if mode == "early_overflow" && errors.As(err, &backend) {
							return fmt.Errorf("later Add fault overrode earlier nonfinite sum: %v", err)
						}
						if mode == "read_fault" && (!errors.As(err, &backend) || backend.Site != "Read") {
							return fmt.Errorf("Read fault identity changed: %v", err)
						}
					}
					before := v41EPRootSumObserve(b, calls[rank], validated[rank])
					if repeated, again := c.reducePanel(rows, r); repeated != nil || again != err || before != v41EPRootSumObserve(b, calls[rank], validated[rank]) {
						return errors.New("observed failed group replayed root work")
					}
					final[rank] = before
					return v41EPRootSumFreed(b)
				}
				if err != nil {
					return err
				}
				check := func(got [][]float32, first int) error {
					for i, row := range got {
						want, err := v41FullRoutedSum(0, full.picks[first+i], all[first+i], width)
						if err != nil {
							return err
						}
						if err := v41EPCheckBits(row, want); err != nil {
							return err
						}
					}
					return nil
				}
				if len(got) != wantRows {
					return errors.New("first observed panel lost rows")
				}
				if err := check(got, 0); err != nil {
					return err
				}
				r.picks, r.digests = full.picks[32:35], full.digests[32:35]
				got, err = c.reducePanel(v41EPPanelTestOwner(r, rank, all[32:35]), r)
				if err != nil || len(got) != 3 {
					return fmt.Errorf("observed short tail: %v", err)
				}
				if err := check(got, 32); err != nil {
					return err
				}
				prefill[rank] = v41EPRootSumObserve(b, calls[rank], validated[rank])
				stats := v41EPPanelTestStats(c)
				if stats.CompletedGathers != 2 || stats.CompletedRows != 35 {
					return fmt.Errorf("observed Prefill exchange/row counts: %+v", stats)
				}
				step := v41EPReduction{owners: full.owners, layer: 0, width: width, picks: full.picks[35], digest: full.digests[35], sum: observed}
				last, err := c.reduce(v41EPTestPart(step, rank, all[35]), step)
				if err != nil {
					return err
				}
				if err := check([][]float32{last}, 35); err != nil {
					return err
				}
				stats = v41EPPanelTestStats(c)
				if stats.CompletedGathers != 3 || stats.CompletedRows != 36 {
					return fmt.Errorf("observed scalar continuation counts: %+v", stats)
				}
				final[rank] = v41EPRootSumObserve(b, calls[rank], validated[rank])
				return v41EPRootSumFreed(b)
			})
			for rank, err := range errs {
				if err != nil || auditErrors[rank] != nil {
					t.Fatalf("rank %d: operation=%v observation=%v", rank, err, auditErrors[rank])
				}
				got := final[rank]
				if got.Physical != nil {
					t.Fatal("CPU recorder claimed physical submissions")
				}
				if rank != 0 {
					if got.Callbacks != 0 || got.Uploads != 0 || got.AddAttempts != 0 || got.ReadAttempts != 0 || got.ReadReturns != 0 || got.UploadBytes != 0 || got.ReturnBytes != 0 || got.FreedBuffers != 0 || got.PeakOwnedTensorBytes != 0 {
						t.Fatalf("follower performed root sum work: %+v", got)
					}
					continue
				}
				if healthy {
					for _, state := range []struct {
						got  v41EPRootSumObservation
						rows int
					}{{prefill[rank], 35}, {got, 36}} {
						x, rows := state.got, state.rows
						if x.Callbacks != rows || x.ValidatedRows != rows || x.Uploads != 7*rows || x.AddAttempts != 6*rows || x.ReadAttempts != rows || x.ReadReturns != rows || x.UploadBytes != uint64(4*width*7*rows) || x.ReturnBytes != uint64(4*width*rows) || x.FreedBuffers != 7*rows {
							t.Fatalf("root HAL counts for %d rows: %+v", rows, x)
						}
					}
				} else {
					wantReturns, wantBytes := 1, uint64(4*width)
					if mode == "read_fault" {
						wantReturns, wantBytes = 0, 0
					} else if mode == "short_read" {
						wantBytes -= 4
					}
					if got.Callbacks != 1 || got.ValidatedRows != 0 || got.Uploads != 7 || got.AddAttempts != 6 || got.ReadAttempts != 1 || got.ReadReturns != wantReturns || got.UploadBytes != 4*width*7 || got.ReturnBytes != wantBytes || got.FreedBuffers != 7 {
						t.Fatalf("failed root attempt/return/validation counts: %+v", got)
					}
				}
				if got.PeakOwnedBuffers != 7 || got.PeakOwnedTensorBytes != 4*width*7 {
					t.Fatalf("root retained more than one scalar sum's tensors: %+v", got)
				}
			}
			report := struct {
				Prefill *v41EPRootSumObservation `json:",omitempty"`
				Final   v41EPRootSumObservation
			}{Final: final[0]}
			if healthy {
				report.Prefill = &prefill[0]
			}
			encoded, _ := json.Marshal(report)
			t.Log(string(encoded))
		})
	}
}
