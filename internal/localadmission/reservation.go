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
	"strconv"
	"strings"
	"time"
)

// reservationSchema is the schema stamped on every write. The class-breakdown
// additions are additive, so /2 ledgers are strict supersets of /1.
//
// Forward migration contract: the ledger is a shared host cache, so the newest
// writer must never poison an older reader. Readers therefore accept ANY
// generation within reservationSchemaFamilyPrefix, decoding only the stable
// reservation fields (encoding/json drops unknown fields from a newer
// generation). The next write self-heals by re-stamping reservationSchema (see
// writeLedger), so no explicit migration step is needed.
const reservationSchema = "fak-local-memory-reservations/2"

// reservationSchemaFamilyPrefix identifies the reservation-ledger schema family.
// Any schema with this prefix and a positive integer generation suffix is
// read-compatible regardless of generation.
const reservationSchemaFamilyPrefix = "fak-local-memory-reservations/"

type Pressure string

const (
	PressureUnknown  Pressure = "unknown"
	PressureNormal   Pressure = "normal"
	PressureWarning  Pressure = "warning"
	PressureCritical Pressure = "critical"
)

// AdmissionSample is the byte-precise host contract consumed by local admission.
// Producers may map their own host-memory sample onto this lower-layer value.
type AdmissionSample struct {
	TotalBytes       int64    `json:"total_bytes"`
	AllocatableBytes int64    `json:"allocatable_bytes"`
	CompressedBytes  int64    `json:"compressed_bytes"`
	WiredBytes       int64    `json:"wired_bytes"`
	Pressure         Pressure `json:"pressure"`
}

var ErrReservationNotFound = errors.New("localadmission: reservation not found")

type MemoryPlan struct {
	StartupPeakBytes int64 `json:"startup_peak_bytes"`
	SteadyBytes      int64 `json:"steady_bytes"`
}

type ReservationRequest struct {
	OwnerPID int             `json:"owner_pid"`
	Plan     MemoryPlan      `json:"plan"`
	Host     AdmissionSample `json:"host"`
	Policy   string          `json:"policy,omitempty"`
	// Classes is the optional class breakdown backing Plan, keyed by the
	// producer's MemoryClass name (weights, kv_cache, activation, ...). String
	// keys keep localadmission decoupled from compute while preserving the
	// breakdown through MarkSteady for class-aware reconciliation.
	Classes map[string]int64 `json:"classes,omitempty"`
}

type Reservation struct {
	ID               string           `json:"id"`
	OwnerPID         int              `json:"owner_pid"`
	StartupPeakBytes int64            `json:"startup_peak_bytes"`
	SteadyBytes      int64            `json:"steady_bytes"`
	HeldBytes        int64            `json:"held_bytes"`
	Phase            string           `json:"phase"`
	Classes          map[string]int64 `json:"classes,omitempty"`
}

type ReservationDecision struct {
	Admit              bool         `json:"admit"`
	Reason             string       `json:"reason"`
	RemedyHint         string       `json:"remedy_hint,omitempty"`
	AdmissionPolicy    string       `json:"admission_policy,omitempty"`
	CapacityBytes      int64        `json:"capacity_bytes"`
	ReservedBytes      int64        `json:"reserved_bytes"`
	RequestedPeakBytes int64        `json:"requested_peak_bytes"`
	Pressure           Pressure     `json:"pressure"`
	Reservation        *Reservation `json:"reservation,omitempty"`
	Reaped             int          `json:"reaped,omitempty"`
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
	if req.Host.Pressure == PressureUnknown {
		d.Reaped = s.reapPersisted(ctx)
		d.Reason = "pressure_unknown"
		return d, nil
	}
	policy := req.Policy
	if req.Host.Pressure == PressureCritical {
		if policy == "dev" {
			d.AdmissionPolicy = "dev-override"
		} else {
			d.Reaped = s.reapPersisted(ctx)
			d.Reason = "pressure_critical"
			if req.Host.TotalBytes > 0 {
				compPct := float64(req.Host.CompressedBytes) / float64(req.Host.TotalBytes) * 100.0
				d.RemedyHint = fmt.Sprintf("fleet-wide ambient pressure (compressed %.1f%%): free memory, reboot, or use the dev override (policy=dev)", compPct)
			} else {
				d.RemedyHint = "fleet-wide ambient pressure: free memory, reboot, or use the dev override (policy=dev)"
			}
			return d, nil
		}
	}
	if req.Host.AllocatableBytes <= 0 {
		d.Reaped = s.reapPersisted(ctx)
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
		avail := req.Host.AllocatableBytes - d.ReservedBytes
		if avail < 0 {
			avail = 0
		}
		if req.Host.TotalBytes > 0 {
			d.RemedyHint = fmt.Sprintf("requested startup peak %.2f GiB (steady %.2f GiB) exceeds available allocatable capacity %.2f GiB (host total %.2f GiB, active reservations %.2f GiB)",
				float64(req.Plan.StartupPeakBytes)/(1<<30),
				float64(req.Plan.SteadyBytes)/(1<<30),
				float64(avail)/(1<<30),
				float64(req.Host.TotalBytes)/(1<<30),
				float64(d.ReservedBytes)/(1<<30),
			)
		} else {
			d.RemedyHint = fmt.Sprintf("requested startup peak %.2f GiB (steady %.2f GiB) exceeds available allocatable capacity %.2f GiB (active reservations %.2f GiB)",
				float64(req.Plan.StartupPeakBytes)/(1<<30),
				float64(req.Plan.SteadyBytes)/(1<<30),
				float64(avail)/(1<<30),
				float64(d.ReservedBytes)/(1<<30),
			)
		}
		if d.ReservedBytes > 0 {
			d.RemedyHint += "; wait for active reservations to release, or select a smaller model or quantization"
		} else {
			d.RemedyHint += "; select a smaller model or quantization"
		}
		if d.Reaped > 0 {
			_ = s.writeLedger(ledger)
		}
		return d, nil
	}
	id, err := reservationID()
	if err != nil {
		return d, err
	}
	r := Reservation{ID: id, OwnerPID: req.OwnerPID, StartupPeakBytes: req.Plan.StartupPeakBytes, SteadyBytes: req.Plan.SteadyBytes, HeldBytes: req.Plan.StartupPeakBytes, Phase: "startup", Classes: cloneClasses(req.Classes)}
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

// ActiveReservations returns all currently active reservations after reaping dead processes.
func (s *ReservationStore) ActiveReservations(ctx context.Context) ([]Reservation, error) {
	unlock, err := s.lock(ctx)
	if err != nil {
		return nil, err
	}
	defer unlock()

	ledger, err := s.readLedger()
	if err != nil {
		return nil, err
	}

	active, reaped := s.reap(ledger.Reservations)
	// Persist the reaped list, not the stale pre-reap slice: reap aliases the
	// backing array, and writeLedger's in-place sort would otherwise resurrect
	// reaped entries and corrupt the returned active slice.
	ledger.Reservations = active
	if reaped > 0 {
		_ = s.writeLedger(ledger)
	}
	return active, nil
}

// TotalReservedBytes returns the sum of HeldBytes across all active reservations.
func (s *ReservationStore) TotalReservedBytes(ctx context.Context) (int64, error) {
	active, err := s.ActiveReservations(ctx)
	if err != nil {
		return 0, err
	}
	var total int64
	for _, r := range active {
		total += r.HeldBytes
	}
	return total, nil
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

// reapPersisted locks the ledger, reaps dead-owner reservations, and persists
// the result when anything was reaped. It is best-effort by design: a lock,
// read, or write failure degrades to "nothing reaped" rather than an error, so
// a failure to reclaim memory never converts an otherwise-clean refusal
// (capacity_unknown / pressure_critical) into an opaque error or strips its
// typed reason. It tolerates a missing ledger: readLedger returns an empty
// ledger for os.ErrNotExist without creating reservations.json, so the
// no-ledger-creation invariant on refusal paths is preserved even though s.lock
// may create the enclosing directory.
func (s *ReservationStore) reapPersisted(ctx context.Context) int {
	unlock, err := s.lock(ctx)
	if err != nil {
		return 0
	}
	defer unlock()
	ledger, err := s.readLedger()
	if err != nil {
		return 0
	}
	reaped := 0
	ledger.Reservations, reaped = s.reap(ledger.Reservations)
	if reaped > 0 {
		_ = s.writeLedger(ledger)
	}
	return reaped
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
	if !reservationSchemaKnown(l.Schema) {
		return reservationLedger{}, fmt.Errorf("unsupported reservation ledger schema %q", l.Schema)
	}
	return l, nil
}

// reservationSchemaKnown reports whether a ledger schema is read-compatible.
// Any generation of the reservation-ledger family is accepted: older (legacy)
// generations decode as today, and a NEWER generation written by a newer binary
// is decoded leniently over the stable reservation fields (unknown fields are
// ignored by encoding/json), so the newest writer never bricks an older reader.
// Only a schema outside the family, or a malformed generation suffix, is
// rejected. writeLedger re-stamps reservationSchema on the next mutation, which
// self-heals the on-disk generation back to the current one.
func reservationSchemaKnown(schema string) bool {
	return reservationSchemaFamily(schema)
}

// reservationSchemaFamily reports whether schema belongs to the reservation
// ledger family and carries a positive integer generation suffix, e.g.
// "fak-local-memory-reservations/3". A different family or a non-numeric or
// non-positive suffix is not read-compatible.
func reservationSchemaFamily(schema string) bool {
	gen, ok := strings.CutPrefix(schema, reservationSchemaFamilyPrefix)
	if !ok || gen == "" {
		return false
	}
	n, err := strconv.Atoi(gen)
	return err == nil && n > 0
}

// cloneClasses copies a class breakdown so a stored reservation cannot alias or
// mutate the caller's map.
func cloneClasses(in map[string]int64) map[string]int64 {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]int64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
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
