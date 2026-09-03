package localadmission

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const defaultReceiptSchema = "fak.metal-unified-memory-reservation/1"

// UnifiedMemoryReservation accounts for startup peak bytes, steady resident bytes,
// memory pressure level, and process owner on Apple Silicon unified memory.
type UnifiedMemoryReservation struct {
	ID               string                           `json:"id"`
	OwnerPID         int                              `json:"owner_pid"`
	StartupPeakBytes int64                            `json:"startup_peak_bytes"`
	SteadyBytes      int64                            `json:"steady_bytes"`
	HeldBytes        int64                            `json:"held_bytes"`
	Phase            string                           `json:"phase"` // "startup", "steady", or "released"
	Pressure         Pressure                         `json:"pressure"`
	HostUnified      bool                             `json:"host_unified"`
	CreatedAt        time.Time                        `json:"created_at"`
	UpdatedAt        time.Time                        `json:"updated_at"`
	manager          *UnifiedMemoryReservationManager `json:"-"`
}

// PromoteToSteady transitions the reservation from its initial startup peak
// down to its steady resident bytes once model loading succeeds.
func (r *UnifiedMemoryReservation) PromoteToSteady(ctx context.Context) error {
	if r.manager == nil {
		return errors.New("localadmission: unmanaged reservation")
	}
	updated, err := r.manager.PromoteToSteady(ctx, r.ID)
	if err == nil {
		*r = updated
	}
	return err
}

// Release releases the reservation and returns held memory back to the
// aggregate allocatable budget.
func (r *UnifiedMemoryReservation) Release(ctx context.Context) error {
	if r.manager == nil {
		return errors.New("localadmission: unmanaged reservation")
	}
	err := r.manager.Release(ctx, r.ID)
	if err == nil {
		r.Phase = "released"
		r.HeldBytes = 0
		r.UpdatedAt = r.manager.timeNow()
	}
	return err
}

// MetalReservationReceipt provides an evidentiary receipt for an admission decision
// or lifecycle state transition on Apple unified memory.
type MetalReservationReceipt struct {
	Schema           string     `json:"schema"`
	Engine           string     `json:"engine"`  // "fak-native"
	Verdict          string     `json:"verdict"` // "ADMIT", "REJECT", or "RELEASED"
	Reason           string     `json:"reason,omitempty"`
	Model            string     `json:"model,omitempty"`
	Device           string     `json:"device,omitempty"`
	Topology         string     `json:"topology"`         // "apple-unified-memory"
	HostUnified      bool       `json:"host_unified"`     // true: host and device draw from one physical pool
	HostAddressable  bool       `json:"host_addressable"` // false: device buffers are NOT host-dereferenceable
	PlannedBytes     MemoryPlan `json:"planned_bytes"`
	AllocatableBytes int64      `json:"allocatable_bytes"`
	AvailableBytes   int64      `json:"available_bytes"`
	ReservedBytes    int64      `json:"reserved_bytes"`
	TotalBytes       int64      `json:"total_bytes"`
	Pressure         Pressure   `json:"pressure"`
	OwnerPID         int        `json:"owner_pid"`
	ReservationID    string     `json:"reservation_id,omitempty"`
	Phase            string     `json:"phase,omitempty"` // "startup", "steady", or "released"
	PeakRSSBytes     int64      `json:"peak_rss_bytes,omitempty"`
	SwapBytes        int64      `json:"swap_bytes,omitempty"`
	Cleanup          string     `json:"cleanup,omitempty"` // "active", "released", "reaped", or "none"
	Timestamp        time.Time  `json:"timestamp"`
}

func (r *MetalReservationReceipt) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

func (r *MetalReservationReceipt) String() string {
	b, err := r.JSON()
	if err != nil {
		return fmt.Sprintf("MetalReservationReceipt{Engine:%s, Verdict:%s, Reason:%s}", r.Engine, r.Verdict, r.Reason)
	}
	return string(b)
}

type UnifiedMemoryReceipt = MetalReservationReceipt

// UnifiedMemoryReservationRequest requests a unified memory reservation for a model load.
type UnifiedMemoryReservationRequest struct {
	OwnerPID     int             `json:"owner_pid"`
	Plan         MemoryPlan      `json:"plan"`
	Host         AdmissionSample `json:"host"`
	Policy       string          `json:"policy,omitempty"`
	ModelName    string          `json:"model_name,omitempty"`
	DeviceName   string          `json:"device_name,omitempty"`
	AllowWarning bool            `json:"allow_warning,omitempty"`
}

type MetalReservationRequest = UnifiedMemoryReservationRequest

// UnifiedMemoryResult contains the admission outcome and evidentiary receipt.
type UnifiedMemoryResult struct {
	Admit              bool                      `json:"admit"`
	Verdict            string                    `json:"verdict"` // "ADMIT" or "REJECT"
	Reason             string                    `json:"reason"`
	RemedyHint         string                    `json:"remedy_hint,omitempty"`
	AllocatableBytes   int64                     `json:"allocatable_bytes"`
	AvailableBytes     int64                     `json:"available_bytes"`
	ReservedBytes      int64                     `json:"reserved_bytes"`
	RequestedPeakBytes int64                     `json:"requested_peak_bytes"`
	Pressure           Pressure                  `json:"pressure"`
	Reservation        *UnifiedMemoryReservation `json:"reservation,omitempty"`
	Reaped             int                       `json:"reaped,omitempty"`
	Receipt            *MetalReservationReceipt  `json:"receipt,omitempty"`
}

type UnifiedMemoryAdmission = UnifiedMemoryResult
type MetalReservationResult = UnifiedMemoryResult
type MetalAdmission = UnifiedMemoryResult

// UnifiedMemoryReservationManager manages atomic unified memory admission across
// concurrent processes on Apple Silicon.
type UnifiedMemoryReservationManager struct {
	store *ReservationStore
}

type MetalReservationManager = UnifiedMemoryReservationManager

// NewUnifiedMemoryReservationManager creates a manager rooted at the given directory.
// If dir is empty, a temporary directory is used.
func NewUnifiedMemoryReservationManager(dir string) *UnifiedMemoryReservationManager {
	if dir == "" {
		dir = filepath.Join(os.TempDir(), fmt.Sprintf("metal-res-%d", time.Now().UnixNano()))
	}
	return &UnifiedMemoryReservationManager{
		store: NewReservationStore(dir),
	}
}

// NewMetalReservationManager aliases NewUnifiedMemoryReservationManager.
func NewMetalReservationManager(dir string) *UnifiedMemoryReservationManager {
	return NewUnifiedMemoryReservationManager(dir)
}

func (m *UnifiedMemoryReservationManager) SetAlive(fn func(int) bool) {
	m.store.alive = fn
}

func (m *UnifiedMemoryReservationManager) SetNow(fn func() time.Time) {
	m.store.now = fn
}

func (m *UnifiedMemoryReservationManager) timeNow() time.Time {
	if m.store.now != nil {
		return m.store.now()
	}
	return time.Now()
}

// Reserve atomically checks host pressure, aggregate capacity, and active reservations
// to admit or reject a memory reservation.
func (m *UnifiedMemoryReservationManager) Reserve(ctx context.Context, req UnifiedMemoryReservationRequest) (UnifiedMemoryResult, error) {
	now := m.timeNow()

	// Fail closed on Warning pressure unless explicitly allowed or in dev mode
	if req.Host.Pressure == PressureWarning && req.Policy != "dev" && !req.AllowWarning {
		res := UnifiedMemoryResult{
			AllocatableBytes:   req.Host.AllocatableBytes,
			AvailableBytes:     req.Host.AllocatableBytes,
			RequestedPeakBytes: req.Plan.StartupPeakBytes,
			Pressure:           req.Host.Pressure,
			Verdict:            "REJECT",
			Reason:             "pressure_warning",
			RemedyHint:         "ambient memory pressure is warning: wait for memory pressure to clear or reduce concurrent model load",
		}
		res.Receipt = m.buildReceipt(req, res, nil, "none", now)
		return res, nil
	}

	rawReq := ReservationRequest{
		OwnerPID: req.OwnerPID,
		Plan:     req.Plan,
		Host:     req.Host,
		Policy:   req.Policy,
	}

	sDec, err := m.store.Reserve(ctx, rawReq)
	if err != nil {
		return UnifiedMemoryResult{}, err
	}

	verdict := "REJECT"
	if sDec.Admit {
		verdict = "ADMIT"
	}

	avail := sDec.CapacityBytes - sDec.ReservedBytes
	if avail < 0 {
		avail = 0
	}

	res := UnifiedMemoryResult{
		Admit:              sDec.Admit,
		Verdict:            verdict,
		Reason:             sDec.Reason,
		RemedyHint:         sDec.RemedyHint,
		AllocatableBytes:   sDec.CapacityBytes,
		AvailableBytes:     avail,
		ReservedBytes:      sDec.ReservedBytes,
		RequestedPeakBytes: sDec.RequestedPeakBytes,
		Pressure:           sDec.Pressure,
		Reaped:             sDec.Reaped,
	}

	var activeRes *UnifiedMemoryReservation
	cleanup := "none"

	if sDec.Admit && sDec.Reservation != nil {
		cleanup = "active"
		activeRes = &UnifiedMemoryReservation{
			ID:               sDec.Reservation.ID,
			OwnerPID:         sDec.Reservation.OwnerPID,
			StartupPeakBytes: sDec.Reservation.StartupPeakBytes,
			SteadyBytes:      sDec.Reservation.SteadyBytes,
			HeldBytes:        sDec.Reservation.HeldBytes,
			Phase:            sDec.Reservation.Phase,
			Pressure:         req.Host.Pressure,
			HostUnified:      true,
			CreatedAt:        now,
			UpdatedAt:        now,
			manager:          m,
		}
		res.Reservation = activeRes
		res.ReservedBytes += activeRes.HeldBytes
		res.AvailableBytes = req.Host.AllocatableBytes - res.ReservedBytes
		if res.AvailableBytes < 0 {
			res.AvailableBytes = 0
		}
	}

	res.Receipt = m.buildReceipt(req, res, activeRes, cleanup, now)
	return res, nil
}

// PromoteToSteady updates the held bytes of a reservation from startup peak to steady bytes.
func (m *UnifiedMemoryReservationManager) PromoteToSteady(ctx context.Context, id string) (UnifiedMemoryReservation, error) {
	r, err := m.store.MarkSteady(ctx, id)
	if err != nil {
		return UnifiedMemoryReservation{}, err
	}
	return UnifiedMemoryReservation{
		ID:               r.ID,
		OwnerPID:         r.OwnerPID,
		StartupPeakBytes: r.StartupPeakBytes,
		SteadyBytes:      r.SteadyBytes,
		HeldBytes:        r.HeldBytes,
		Phase:            r.Phase,
		Pressure:         PressureNormal,
		HostUnified:      true,
		UpdatedAt:        m.timeNow(),
		manager:          m,
	}, nil
}

// Release removes the specified reservation and frees held memory.
func (m *UnifiedMemoryReservationManager) Release(ctx context.Context, id string) error {
	return m.store.Release(ctx, id)
}

// ReapStale scans for dead owners and removes their reservations.
func (m *UnifiedMemoryReservationManager) ReapStale(ctx context.Context) (int, error) {
	unlock, err := m.store.lock(ctx)
	if err != nil {
		return 0, err
	}
	defer unlock()

	ledger, err := m.store.readLedger()
	if err != nil {
		return 0, err
	}

	var reaped int
	ledger.Reservations, reaped = m.store.reap(ledger.Reservations)
	if reaped > 0 {
		if err := m.store.writeLedger(ledger); err != nil {
			return 0, err
		}
	}
	return reaped, nil
}

// ActiveReservations returns all currently active reservations.
func (m *UnifiedMemoryReservationManager) ActiveReservations(ctx context.Context) ([]UnifiedMemoryReservation, error) {
	unlock, err := m.store.lock(ctx)
	if err != nil {
		return nil, err
	}
	defer unlock()

	ledger, err := m.store.readLedger()
	if err != nil {
		return nil, err
	}

	out := make([]UnifiedMemoryReservation, len(ledger.Reservations))
	for i, r := range ledger.Reservations {
		out[i] = UnifiedMemoryReservation{
			ID:               r.ID,
			OwnerPID:         r.OwnerPID,
			StartupPeakBytes: r.StartupPeakBytes,
			SteadyBytes:      r.SteadyBytes,
			HeldBytes:        r.HeldBytes,
			Phase:            r.Phase,
			Pressure:         PressureNormal,
			HostUnified:      true,
			manager:          m,
		}
	}
	return out, nil
}

// TotalReservedBytes returns the sum of HeldBytes across all active reservations.
func (m *UnifiedMemoryReservationManager) TotalReservedBytes(ctx context.Context) (int64, error) {
	res, err := m.ActiveReservations(ctx)
	if err != nil {
		return 0, err
	}
	var total int64
	for _, r := range res {
		total += r.HeldBytes
	}
	return total, nil
}

func (m *UnifiedMemoryReservationManager) buildReceipt(req UnifiedMemoryReservationRequest, res UnifiedMemoryResult, active *UnifiedMemoryReservation, cleanup string, timestamp time.Time) *MetalReservationReceipt {
	model := req.ModelName
	if model == "" {
		model = "fak-native-model"
	}
	device := req.DeviceName
	if device == "" {
		device = "Apple Silicon Unified Memory"
	}

	resID := ""
	phase := "none"
	if active != nil {
		resID = active.ID
		phase = active.Phase
	}

	avail := res.AvailableBytes
	if avail < 0 {
		avail = 0
	}

	return &MetalReservationReceipt{
		Schema:           defaultReceiptSchema,
		Engine:           "fak-native",
		Verdict:          res.Verdict,
		Reason:           res.Reason,
		Model:            model,
		Device:           device,
		Topology:         "apple-unified-memory",
		HostUnified:      true,
		HostAddressable:  false,
		PlannedBytes:     req.Plan,
		AllocatableBytes: req.Host.AllocatableBytes,
		AvailableBytes:   avail,
		ReservedBytes:    res.ReservedBytes,
		TotalBytes:       req.Host.TotalBytes,
		Pressure:         req.Host.Pressure,
		OwnerPID:         req.OwnerPID,
		ReservationID:    resID,
		Phase:            phase,
		PeakRSSBytes:     req.Plan.StartupPeakBytes,
		SwapBytes:        req.Host.CompressedBytes,
		Cleanup:          cleanup,
		Timestamp:        timestamp,
	}
}

// GenerateReleaseReceipt generates an evidentiary receipt for a released reservation.
func (m *UnifiedMemoryReservationManager) GenerateReleaseReceipt(r UnifiedMemoryReservation, host AdmissionSample) *MetalReservationReceipt {
	now := m.timeNow()
	return &MetalReservationReceipt{
		Schema:          defaultReceiptSchema,
		Engine:          "fak-native",
		Verdict:         "RELEASED",
		Reason:          "released",
		Model:           "fak-native-model",
		Device:          "Apple Silicon Unified Memory",
		Topology:        "apple-unified-memory",
		HostUnified:     true,
		HostAddressable: false,
		PlannedBytes: MemoryPlan{
			StartupPeakBytes: r.StartupPeakBytes,
			SteadyBytes:      r.SteadyBytes,
		},
		AllocatableBytes: host.AllocatableBytes,
		AvailableBytes:   host.AllocatableBytes,
		ReservedBytes:    0,
		TotalBytes:       host.TotalBytes,
		Pressure:         host.Pressure,
		OwnerPID:         r.OwnerPID,
		ReservationID:    r.ID,
		Phase:            "released",
		PeakRSSBytes:     0,
		SwapBytes:        host.CompressedBytes,
		Cleanup:          "released",
		Timestamp:        now,
	}
}

// SampleM3ProQwen38Receipt returns an exact serialized M3 Pro Qwen3.8 Metal admission receipt
// demonstrating the physical unified topology without treating device buffers as host-addressable.
func SampleM3ProQwen38Receipt() MetalReservationReceipt {
	return MetalReservationReceipt{
		Schema:          defaultReceiptSchema,
		Engine:          "fak-native",
		Verdict:         "ADMIT",
		Reason:          "reserved",
		Model:           "qwen38:27b",
		Device:          "Apple M3 Pro",
		Topology:        "apple-unified-memory",
		HostUnified:     true,
		HostAddressable: false,
		PlannedBytes: MemoryPlan{
			StartupPeakBytes: 20 * (1 << 30), // 20 GiB startup peak
			SteadyBytes:      16 * (1 << 30), // 16 GiB steady residency
		},
		AllocatableBytes: 30 * (1 << 30), // 30 GiB allocatable
		AvailableBytes:   30 * (1 << 30),
		ReservedBytes:    20 * (1 << 30),
		TotalBytes:       36 * (1 << 30), // 36 GiB unified memory
		Pressure:         PressureNormal,
		OwnerPID:         os.Getpid(),
		ReservationID:    "res-m3pro-qwen38-001",
		Phase:            "startup",
		PeakRSSBytes:     17 * (1 << 30),
		SwapBytes:        0,
		Cleanup:          "active",
		Timestamp:        time.Date(2026, time.August, 29, 12, 0, 0, 0, time.UTC),
	}
}
