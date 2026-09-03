package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
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

// estimateMetalModelMemoryBounds estimates the startup peak and steady resident bytes
// for a model path on Apple Silicon unified memory. Test environments may inject
// explicit bounds via FAK_TEST_STARTUP_PEAK_BYTES / FAK_TEST_STEADY_BYTES.
func estimateMetalModelMemoryBounds(ggufPath string) localadmission.MemoryPlan {
	if peakStr := os.Getenv("FAK_TEST_STARTUP_PEAK_BYTES"); peakStr != "" {
		if peak, err := strconv.ParseInt(peakStr, 10, 64); err == nil && peak > 0 {
			steady := peak * 2 / 3
			if steadyStr := os.Getenv("FAK_TEST_STEADY_BYTES"); steadyStr != "" {
				if s, err := strconv.ParseInt(steadyStr, 10, 64); err == nil && s > 0 {
					steady = s
				}
			}
			return localadmission.MemoryPlan{
				StartupPeakBytes: peak,
				SteadyBytes:      steady,
			}
		}
	}

	trimmed := strings.TrimSpace(ggufPath)
	if trimmed != "" {
		if _, err := os.Stat(trimmed); err == nil {
			ws, err := ggufload.OpenWeights(trimmed)
			if err == nil {
				defer ws.Close()
				plan, err := ws.EstimateLoadMemoryPlan()
				if err == nil && plan.Total() > 0 {
					steady := plan.Total()
					total, _, known := compute.HostSystemMemoryInfo()
					if os.Getenv("FAK_STREAM_Q4K") == "1" || os.Getenv("FAK_METAL_STREAM_Q4K") == "1" {
						reqPeak, _, _ := streamedQ4KMetalCapacity(total, known, os.Getenv("FAK_Q4K_FREE_CPU") == "1")
						if reqPeak < steady {
							reqPeak = steady
						}
						return localadmission.MemoryPlan{
							StartupPeakBytes: reqPeak,
							SteadyBytes:      steady,
						}
					}
					peak, _ := metalGGUFPeakCapacity(true, steady, total, known)
					if peak <= steady {
						peak = int64(float64(steady) * metalGGUFObservedPeakMultiplier)
					}
					if peak < steady {
						peak = steady
					}
					return localadmission.MemoryPlan{
						StartupPeakBytes: peak,
						SteadyBytes:      steady,
					}
				}
			}
		}
	}

	return localadmission.MemoryPlan{
		StartupPeakBytes: 3 << 30,
		SteadyBytes:      2 << 30,
	}
}

// loadLocalLauncherModelWithMetalLease coordinates local native Metal memory admission,
// combining the model's startup/steady memory plan, current Darwin allocatable memory and pressure,
// active FAK reservations, and GPU leases.
//
// In exclusive mode (FAK_NATIVE_ADMISSION=exclusive), it retains the conservative exclusive
// GPU lease across the entire serve lifecycle as a verified rollback mechanism.
//
// In aggregate mode (default), it acquires an exclusive load lease during initial model allocation
// to prevent transient allocation races, reserves startup peak memory in the persistent reservation store,
// downshifts to steady residency once loaded, and releases the transient load lease so that proven-small
// models can safely coexist when aggregate capacity permits.
func loadLocalLauncherModelWithMetalLease(useMetal bool, ggufPath string, opts gpulease.Options, load func()) (release func(), err error) {
	if !useMetal || strings.TrimSpace(ggufPath) == "" {
		load()
		return func() {}, nil
	}

	exclusiveMode := os.Getenv("FAK_NATIVE_ADMISSION") == "exclusive"

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
	plan := estimateMetalModelMemoryBounds(ggufPath)

	var resID string
	retainLease := os.Getenv("FAK_NATIVE_ADMISSION") != "aggregate"

	if !exclusiveMode {
		mem, memErr := memgate.ReadMemory()
		if memErr == nil && mem.TotalBytes > 0 {
			sample := memgate.AdmissionSampleFor(mem)
			req := localadmission.ReservationRequest{
				OwnerPID: os.Getpid(),
				Plan:     plan,
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
			if rerr != nil {
				lease.Release()
				return func() {}, fmt.Errorf("fak local launcher: local memory reservation error: %w", rerr)
			}
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

			if opts.Path != "" || plan.StartupPeakBytes > sample.AllocatableBytes/2 || plan.SteadyBytes > sample.AllocatableBytes/2 {
				retainLease = true
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
	if !retainLease {
		lease.Release()
	}

	var once sync.Once
	return func() {
		once.Do(func() {
			if resID != "" {
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
				_ = store.Release(ctx, resID)
				cancel()
			}
			if retainLease {
				lease.Release()
			}
		})
	}, nil
}
