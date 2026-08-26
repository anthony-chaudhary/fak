package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const guardAllowWatchInterval = time.Second

type guardAllowWatchEvent struct {
	Reloaded bool
	Rejected bool
	Detail   string
}

type guardAllowWatcher struct {
	interval time.Duration
	reload   gateway.PolicyReloadFunc
	onEvent  func(guardAllowWatchEvent)

	mu       sync.Mutex
	lastGood string
}

func newGuardAllowWatcher(interval time.Duration, reload gateway.PolicyReloadFunc, onEvent func(guardAllowWatchEvent)) *guardAllowWatcher {
	if interval <= 0 {
		interval = guardAllowWatchInterval
	}
	w := &guardAllowWatcher{interval: interval, reload: reload, onEvent: onEvent}
	if sig, err := guardAllowLayersSignature(); err == nil {
		w.lastGood = sig
	}
	return w
}

func guardAllowLayersSignature() (string, error) {
	var b strings.Builder
	// Watch the same effective layer set the reload consumes. Omitting the
	// per-launch session layer made `guard allow --session` write a grant that
	// could never affect its live guard: the file was only read at launch, before
	// the child could create it, then deleted at teardown.
	for _, layer := range guardAllowEffectiveReadLayers() {
		b.WriteString(layer.Name)
		b.WriteByte(0)
		b.WriteString(layer.Path)
		b.WriteByte(0)
		raw, err := os.ReadFile(layer.Path)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				b.WriteString("<missing>")
				b.WriteByte(0)
				continue
			}
			return "", err
		}
		// Include the time-filtered form as well as the bytes. A TTL crossing changes
		// the effective allow set even when the operator has not rewritten the file.
		effective, err := loadGuardAllowOverlay(layer.Path)
		if err != nil {
			return "", err
		}
		effectiveRaw, err := json.Marshal(effective)
		if err != nil {
			return "", err
		}
		b.Write(raw)
		b.WriteByte(0)
		b.Write(effectiveRaw)
		b.WriteByte(0)
	}
	return b.String(), nil
}

func (w *guardAllowWatcher) Reload(ctx context.Context) guardAllowWatchEvent {
	w.mu.Lock()
	defer w.mu.Unlock()
	sig, err := guardAllowLayersSignature()
	if err != nil {
		e := guardAllowWatchEvent{Rejected: true, Detail: err.Error()}
		w.emit(e)
		return e
	}
	if sig == w.lastGood {
		return guardAllowWatchEvent{}
	}
	if w.reload == nil {
		e := guardAllowWatchEvent{Rejected: true, Detail: "policy reload callback unavailable"}
		w.emit(e)
		return e
	}
	if _, err := w.reload(ctx); err != nil {
		e := guardAllowWatchEvent{Rejected: true, Detail: err.Error()}
		w.emit(e)
		return e
	}
	w.lastGood = sig
	e := guardAllowWatchEvent{Reloaded: true, Detail: "operator allow overlay reloaded"}
	w.emit(e)
	return e
}

func (w *guardAllowWatcher) emit(e guardAllowWatchEvent) {
	if w.onEvent != nil && (e.Reloaded || e.Rejected) {
		w.onEvent(e)
	}
}

func (w *guardAllowWatcher) Run(ctx context.Context) error {
	t := time.NewTicker(w.interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
			w.Reload(ctx)
		}
	}
}

func startGuardAllowWatcher(ctx context.Context, reload gateway.PolicyReloadFunc, _ bool) {
	w := newGuardAllowWatcher(guardAllowWatchInterval, reload, func(e guardAllowWatchEvent) {
		if !e.Rejected {
			return
		}
		fmt.Fprintf(os.Stderr, "fak guard allow watcher: rejected; keeping last-good floor: %s\n", e.Detail)
	})
	go func() { _ = w.Run(ctx) }()
}
