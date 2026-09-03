package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
	"github.com/anthony-chaudhary/fak/internal/memgate"
)

func defaultLocalReservationDir() string {
	if dir := os.Getenv("FAK_RESERVATION_DIR"); dir != "" {
		return dir
	}
	root, err := os.UserCacheDir()
	if err != nil {
		root = os.TempDir()
	}
	return filepath.Join(root, "fak", "reservations")
}

// loadLocalLauncherModelWithMetalLease serializes the retained residency created by a
// local fak-native Metal GGUF load. Acquisition is deliberately no-wait and
// precedes load: a second heavy process must fail before allocating, not queue
// after it has already consumed unified memory. The returned release function
// belongs to the whole serve lifecycle, not merely to model loading.
//
// opts.Path is an injection seam for the concurrency witness. Production leaves
// it empty, selecting the same FAK_GPU_LEASE/default path as modelbench -metal.
func loadLocalLauncherModelWithMetalLease(useMetal bool, ggufPath string, opts gpulease.Options, load func()) (release func(), err error) {
	if !useMetal || strings.TrimSpace(ggufPath) == "" {
		load()
		return func() {}, nil
	}

	opts.NoWait = true
	opts.Timeout = 0
	lease, err := gpulease.Acquire(opts)
	if err != nil {
		path := opts.Path
		if path == "" {
			path = gpulease.DefaultPath()
		}
		if errors.Is(err, gpulease.ErrBusy) {
			return func() {}, fmt.Errorf("fak local launcher: Metal residency admission refused before model load: %w; stop the holder process and retry, or run a CPU/non-Metal serve", err)
		}
		return func() {}, fmt.Errorf("fak local launcher: acquire Metal residency lease %s before model load: %w", path, err)
	}

	resDir := defaultLocalReservationDir()
	store := localadmission.NewReservationStore(resDir)
	mem, memErr := memgate.ReadMemory()
	var resID string
	if memErr == nil && mem.TotalBytes > 0 && os.Getenv("FAK_NATIVE_ADMISSION") != "exclusive" {
		sample := memgate.AdmissionSampleFor(mem)
		req := localadmission.ReservationRequest{
			OwnerPID: os.Getpid(),
			Plan: localadmission.MemoryPlan{
				StartupPeakBytes: 3 << 30,
				SteadyBytes:      2 << 30,
			},
			Host: localadmission.AdmissionSample{
				TotalBytes:       sample.TotalBytes,
				AllocatableBytes: sample.AllocatableBytes,
				CompressedBytes:  sample.CompressedBytes,
				WiredBytes:       sample.WiredBytes,
				Pressure:         localadmission.Pressure(sample.Pressure),
			},
			Policy: os.Getenv("FAK_ADMISSION_POLICY"),
		}
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		dec, rerr := store.Reserve(ctx, req)
		cancel()
		if rerr == nil {
			if !dec.Admit {
				lease.Release()
				hint := dec.RemedyHint
				if hint != "" {
					return func() {}, fmt.Errorf("fak local launcher: local memory reservation refused: %s (%s)", dec.Reason, hint)
				}
				return func() {}, fmt.Errorf("fak local launcher: local memory reservation refused: %s", dec.Reason)
			}
			if dec.Reservation != nil {
				resID = dec.Reservation.ID
			}
		}
	}

	loaded := false
	defer func() {
		if !loaded {
			if resID != "" {
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
				_ = store.Release(ctx, resID)
				cancel()
			}
			lease.Release()
		}
	}()
	load()
	loaded = true
	if resID != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, _ = store.MarkSteady(ctx, resID)
		cancel()
	}
	return func() {
		if resID != "" {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			_ = store.Release(ctx, resID)
			cancel()
		}
		lease.Release()
	}, nil
}
