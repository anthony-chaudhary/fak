package main

import (
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

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

	loaded := false
	defer func() {
		if !loaded {
			lease.Release()
		}
	}()
	load()
	loaded = true
	return lease.Release, nil
}
