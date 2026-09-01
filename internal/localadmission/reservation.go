package localadmission

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/memgate"
)

const reservationSchema = "fak-local-memory-reservations/1"

var ErrReservationNotFound = errors.New("localadmission: reservation not found")

type MemoryPlan struct {
	StartupPeakBytes int64 `json:"startup_peak_bytes"`
	SteadyBytes      int64 `json:"steady_bytes"`
}

type ReservationRequest struct {
	OwnerPID int                     `json:"owner_pid"`
	Plan     MemoryPlan              `json:"plan"`
	Host     memgate.AdmissionSample `json:"host"`
}

type Reservation struct {
	ID               string `json:"id"`
	OwnerPID         int    `json:"owner_pid"`
	StartupPeakBytes int64  `json:"startup_peak_bytes"`
	SteadyBytes      int64  `json:"steady_bytes"`
	HeldBytes        int64  `json:"held_bytes"`
	Phase            string `json:"phase"`
}

type ReservationDecision struct {
	Admit              bool             `json:"admit"`
	Reason             string           `json:"reason"`
	CapacityBytes      int64            `json:"capacity_bytes"`
	ReservedBytes      int64            `json:"reserved_bytes"`
	RequestedPeakBytes int64            `json:"requested_peak_bytes"`
	Pressure           memgate.Pressure `json:"pressure"`
	Reservation        *Reservation     `json:"reservation,omitempty"`
	Reaped             int              `json:"reaped,omitempty"`
}

type reservationLedger struct {
	Schema       string        `json:"schema"`
	Reservations []Reservation `json:"reservations"`
}

type ReservationStore struct {
	dir       string
	now       func() time.Time
	alive     func(int) bool
	lockDelay time.Duration
}

func NewReservationStore(dir string) *ReservationStore {
	return &ReservationStore{dir: dir, now: time.Now, alive: processAlive, lockDelay: 5 * time.Millisecond}
}

func (s *ReservationStore) Reserve(ctx context.Context, req ReservationRequest) (ReservationDecision, error) {
	d := ReservationDecision{CapacityBytes: req.Host.AllocatableBytes, RequestedPeakBytes: req.Plan.StartupPeakBytes, Pressure: req.Host.Pressure}
	if req.Host.Pressure == memgate.PressureUnknown {
		d.Reason = "pressure_unknown"
		return d, nil
	}
	if req.Host.Pressure == memgate.PressureCritical {
		d.Reason = "pressure_critical"
		return d, nil
	}
	if req.Host.AllocatableBytes <= 0 {
		d.Reason = "capacity_unknown"
		return d, nil
	}
	if req.OwnerPID <= 0 || req.Plan.StartupPeakBytes <= 0 || req.Plan.SteadyBytes <= 0 || req.Plan.SteadyBytes > req.Plan.StartupPeakBytes {
		d.Reason = "invalid_request"
		return d, nil
	}
	unlock, err := s.lock(ctx)
	if err != nil {
		return d, err
	}
	defer unlock()
	ledger, err := s.readLedger()
	if err != nil {
		return d, err
	}
	ledger.Reservations, d.Reaped = s.reap(ledger.Reservations)
	for _, r := range ledger.Reservations {
		d.ReservedBytes += r.HeldBytes
	}
	if req.Plan.StartupPeakBytes > req.Host.AllocatableBytes-d.ReservedBytes {
		d.Reason = "aggregate_capacity"
		if d.Reaped > 0 {
			_ = s.writeLedger(ledger)
		}
		return d, nil
	}
	id, err := reservationID()
	if err != nil {
		return d, err
	}
	r := Reservation{ID: id, OwnerPID: req.OwnerPID, StartupPeakBytes: req.Plan.StartupPeakBytes, SteadyBytes: req.Plan.SteadyBytes, HeldBytes: req.Plan.StartupPeakBytes, Phase: "startup"}
	ledger.Reservations = append(ledger.Reservations, r)
	if err := s.writeLedger(ledger); err != nil {
		return d, err
	}
	d.Admit, d.Reason, d.Reservation = true, "reserved", &r
	return d, nil
}

func (s *ReservationStore) MarkSteady(ctx context.Context, id string) (Reservation, error) {
	return s.update(ctx, id, func(r *Reservation) { r.HeldBytes, r.Phase = r.SteadyBytes, "steady" })
}

func (s *ReservationStore) Release(ctx context.Context, id string) error {
	unlock, err := s.lock(ctx)
	if err != nil {
		return err
	}
	defer unlock()
	ledger, err := s.readLedger()
	if err != nil {
		return err
	}
	found := false
	kept := ledger.Reservations[:0]
	for _, r := range ledger.Reservations {
		if r.ID == id {
			found = true
			continue
		}
		kept = append(kept, r)
	}
	if !found {
		return ErrReservationNotFound
	}
	ledger.Reservations = kept
	return s.writeLedger(ledger)
}

func (s *ReservationStore) update(ctx context.Context, id string, fn func(*Reservation)) (Reservation, error) {
	unlock, err := s.lock(ctx)
	if err != nil {
		return Reservation{}, err
	}
	defer unlock()
	ledger, err := s.readLedger()
	if err != nil {
		return Reservation{}, err
	}
	for i := range ledger.Reservations {
		if ledger.Reservations[i].ID == id {
			fn(&ledger.Reservations[i])
			if err := s.writeLedger(ledger); err != nil {
				return Reservation{}, err
			}
			return ledger.Reservations[i], nil
		}
	}
	return Reservation{}, ErrReservationNotFound
}

func (s *ReservationStore) reap(in []Reservation) ([]Reservation, int) {
	out := in[:0]
	reaped := 0
	for _, r := range in {
		if !s.alive(r.OwnerPID) {
			reaped++
			continue
		}
		out = append(out, r)
	}
	return out, reaped
}

func (s *ReservationStore) lock(ctx context.Context) (func(), error) {
	if err := os.MkdirAll(s.dir, 0o700); err != nil {
		return nil, err
	}
	return lockReservationFile(ctx, filepath.Join(s.dir, "lock"), s.lockDelay)
}

func (s *ReservationStore) readLedger() (reservationLedger, error) {
	path := filepath.Join(s.dir, "reservations.json")
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return reservationLedger{Schema: reservationSchema}, nil
	}
	if err != nil {
		return reservationLedger{}, err
	}
	var l reservationLedger
	if err := json.Unmarshal(b, &l); err != nil {
		return reservationLedger{}, fmt.Errorf("decode reservation ledger: %w", err)
	}
	if l.Schema != reservationSchema {
		return reservationLedger{}, fmt.Errorf("unsupported reservation ledger schema %q", l.Schema)
	}
	return l, nil
}

func (s *ReservationStore) writeLedger(l reservationLedger) error {
	l.Schema = reservationSchema
	sort.Slice(l.Reservations, func(i, j int) bool { return l.Reservations[i].ID < l.Reservations[j].ID })
	b, err := json.MarshalIndent(l, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	tmp := filepath.Join(s.dir, fmt.Sprintf("reservations-%d.tmp", s.now().UnixNano()))
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, filepath.Join(s.dir, "reservations.json")); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func reservationID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(b[:]), nil
}
