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
	interval         time.Duration
	reload           gateway.PolicyReloadFunc
	onEvent          func(guardAllowWatchEvent)
	signature        func() (string, error)
	onSignatureError func(context.Context) error

	mu              sync.Mutex
	lastGood        string
	signatureFailed bool
}

func newGuardAllowWatcher(interval time.Duration, reload gateway.PolicyReloadFunc, onEvent func(guardAllowWatchEvent)) *guardAllowWatcher {
	if interval <= 0 {
		interval = guardAllowWatchInterval
	}
	w := &guardAllowWatcher{interval: interval, reload: reload, onEvent: onEvent, signature: guardAllowLayersSignature}
	if sig, err := w.signature(); err == nil {
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

// guardOperatorApprovalLayersSignature tracks only effective, unexpired exact-call
// approvals. Unrelated allow/deny edits remain owned by their existing policy paths.
func guardOperatorApprovalLayersSignature() (string, error) {
	var b strings.Builder
	for _, layer := range guardAllowEffectiveReadLayers() {
		overlay, err := loadGuardAllowOverlay(layer.Path)
		if err != nil {
			return "", err
		}
		raw, err := json.Marshal(overlay.Approve)
		if err != nil {
			return "", err
		}
		b.WriteString(layer.Name)
		b.WriteByte(0)
		b.WriteString(layer.Path)
		b.WriteByte(0)
		b.Write(raw)
		b.WriteByte(0)
	}
	return b.String(), nil
}

func (w *guardAllowWatcher) Reload(ctx context.Context) guardAllowWatchEvent {
	return w.reloadIfChanged(ctx, false)
}

// Refresh re-applies the effective overlay even when its signature has not changed.
// Long-lived servers use this once before accepting requests: policy installation and
// watcher construction are separate steps, so an approval written between them must not
// be mistaken for an already-applied last-good signature.
func (w *guardAllowWatcher) Refresh(ctx context.Context) guardAllowWatchEvent {
	return w.reloadIfChanged(ctx, true)
}

func (w *guardAllowWatcher) reloadIfChanged(ctx context.Context, force bool) guardAllowWatchEvent {
	w.mu.Lock()
	defer w.mu.Unlock()
	sig, err := w.signature()
	if err != nil {
		detail := err.Error()
		if w.onSignatureError != nil {
			if clearErr := w.onSignatureError(ctx); clearErr != nil {
				detail += "; fail-closed approval clear rejected: " + clearErr.Error()
			}
		}
		w.signatureFailed = true
		e := guardAllowWatchEvent{Rejected: true, Detail: detail}
		w.emit(e)
		return e
	}
	force = force || w.signatureFailed
	if !force && sig == w.lastGood {
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
	w.signatureFailed = false
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

// startGuardAllowWatcher owns one Run goroutine. Its stop function cancels and joins that run,
// including any in-flight Reload, before session-scope state may be torn down.
// Repeated or concurrent stops are safe, even after the parent was canceled.
func startGuardAllowWatcher(ctx context.Context, reload gateway.PolicyReloadFunc, _ bool) context.CancelFunc {
	w := newLoggedGuardAllowWatcher(reload)
	return startGuardAllowWatcherRun(ctx, w)
}

// startOperatorApprovalWatcherWithInitialRefresh closes the startup gap between
// installing the base floor and beginning transport service. It never reloads
// the base/profile or general allow/deny overlays.
func startOperatorApprovalWatcherWithInitialRefresh(ctx context.Context, reload gateway.PolicyReloadFunc) context.CancelFunc {
	w := newLoggedGuardAllowWatcher(reload)
	w.signature = guardOperatorApprovalLayersSignature
	w.onSignatureError = clearServeOperatorApprovals
	w.lastGood = ""
	w.Refresh(ctx)
	return startGuardAllowWatcherRun(ctx, w)
}

func newLoggedGuardAllowWatcher(reload gateway.PolicyReloadFunc) *guardAllowWatcher {
	return newGuardAllowWatcher(guardAllowWatchInterval, reload, func(e guardAllowWatchEvent) {
		if !e.Rejected {
			return
		}
		fmt.Fprintf(os.Stderr, "fak guard allow watcher: rejected; keeping last-good floor: %s\n", e.Detail)
	})
}

func startGuardAllowWatcherRun(ctx context.Context, w *guardAllowWatcher) context.CancelFunc {
	ctx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = w.Run(ctx)
	}()
	return func() {
		cancel()
		<-done
	}
}
