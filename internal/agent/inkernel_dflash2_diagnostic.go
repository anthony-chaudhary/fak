package agent

import (
	"context"
	"errors"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

type dflash2DiagnosticOwner interface {
	Bootstrap(context.Context, []int) error
	PrepareTimed(context.Context, int) (*model.Qwen38DFlash2Round, model.DFlash2PrepareTiming, error)
	CommittedPosition(context.Context) (int, error)
	Close() error
}

// DFlash2DiagnosticRequest owns the same domain gate and active target lease as
// serving. The returned rounds belong to this request and become unusable after
// Close. Callers must close it before closing its authenticated target.
type DFlash2DiagnosticRequest struct {
	mu       sync.Mutex
	target   *DFlash2Target
	lease    *dflash2Lease
	request  dflash2DiagnosticOwner
	release  func()
	closed   bool
	closeErr error
}

// OpenDiagnosticRequest routes through the target's immutable typed options.
// It never exposes a borrowed Session or bypasses serving-domain quarantine.
func (t *DFlash2Target) OpenDiagnosticRequest(ctx context.Context) (out *DFlash2DiagnosticRequest, err error) {
	if t == nil {
		return nil, errors.New("dflash2: target required")
	}
	if err = t.ValidateBinding(t.m, t.tok, t.backend, true, false); err != nil {
		return nil, err
	}
	release, err := t.domain.acquire(ctx)
	if err != nil {
		return nil, err
	}
	d := &DFlash2DiagnosticRequest{target: t, release: release}
	completed, published := false, false
	defer func() {
		if !completed {
			t.domain.quarantine(errors.New("dflash2: diagnostic acquisition interrupted"), d)
		}
		if completed && !published && d.release != nil {
			t.domain.mu.Lock()
			healthy := t.domain.failure == nil
			t.domain.mu.Unlock()
			if healthy {
				d.release()
				d.release = nil
			} else {
				t.domain.quarantine(err, d)
			}
		}
	}()
	d.lease, err = t.newRequest(ctx)
	if err != nil {
		completed = true
		return nil, err
	}
	var ok bool
	d.request, ok = d.lease.request.(dflash2DiagnosticOwner)
	if !ok {
		err = errors.Join(errors.New("dflash2: native diagnostic request unavailable"), dflash2CleanupCall("diagnostic refusal close", d.Close))
		completed = true
		return nil, err
	}
	completed, published = true, true
	return d, nil
}

func (d *DFlash2DiagnosticRequest) live() error {
	if d == nil || d.closed || d.request == nil {
		return errors.New("dflash2: diagnostic request retired")
	}
	return nil
}
func (d *DFlash2DiagnosticRequest) Bootstrap(ctx context.Context, prompt []int) error {
	if d == nil {
		return errors.New("dflash2: diagnostic request required")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if err := d.live(); err != nil {
		return err
	}
	return d.request.Bootstrap(ctx, prompt)
}
func (d *DFlash2DiagnosticRequest) PrepareTimed(ctx context.Context, maximum int) (*model.Qwen38DFlash2Round, model.DFlash2PrepareTiming, error) {
	if d == nil {
		return nil, model.DFlash2PrepareTiming{}, errors.New("dflash2: diagnostic request required")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if err := d.live(); err != nil {
		return nil, model.DFlash2PrepareTiming{}, err
	}
	return d.request.PrepareTimed(ctx, maximum)
}
func (d *DFlash2DiagnosticRequest) CommittedPosition(ctx context.Context) (int, error) {
	if d == nil {
		return 0, errors.New("dflash2: diagnostic request required")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if err := d.live(); err != nil {
		return 0, err
	}
	return d.request.CommittedPosition(ctx)
}

// Close publishes incomplete retirement before callbacks. A panic/Goexit keeps
// the lease, target and domain gate in quarantine. A later
// Close retains the failure rather than claiming successful retirement.
func (d *DFlash2DiagnosticRequest) Close() (err error) {
	if d == nil {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return d.closeErr
	}
	d.closed = true
	d.closeErr = errors.New("dflash2: diagnostic retirement incomplete")
	completed := false
	defer func() {
		if completed {
			d.closeErr = err
		}
		if d.closeErr != nil {
			d.target.domain.quarantine(d.closeErr, d)
		}
		if d.closeErr == nil && d.release != nil {
			d.release()
			d.release = nil
		}
	}()
	err = d.lease.close()
	completed = true
	return err
}
