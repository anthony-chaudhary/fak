package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const sessionRegistryEnv = "FAK_SESSION_REGISTRY"

type sessionLeasePublisher interface {
	PublishSession(context.Context, leaseref.SessionDescriptor) (string, error)
	RemoveSession(context.Context, string) error
}

type sessionDurability struct {
	registry *session.Registry
	leases   sessionLeasePublisher
	host     string
	meta     session.DescriptorMeta
	ttl      time.Duration
	now      func() time.Time
	warnf    func(string, ...any)
}

var serveSessionDurability *sessionDurability

type unknownSessionError struct {
	id string
}

func (e unknownSessionError) Error() string {
	return fmt.Sprintf("unknown session id %q", e.id)
}

func newSessionDurability(registry *session.Registry, leases sessionLeasePublisher, host string, ttl time.Duration, now func() time.Time, warnf func(string, ...any), meta ...session.DescriptorMeta) *sessionDurability {
	if ttl <= 0 {
		ttl = session.DefaultDescriptorTTL
	}
	if now == nil {
		now = time.Now
	}
	var m session.DescriptorMeta
	if len(meta) > 0 {
		m = meta[0]
	}
	return &sessionDurability{
		registry: registry,
		leases:   leases,
		host:     strings.TrimSpace(host),
		meta:     m,
		ttl:      ttl,
		now:      now,
		warnf:    warnf,
	}
}

func configureServeSessionDurability(tbl *session.Table, path string, stderr io.Writer, meta ...session.DescriptorMeta) error {
	path = strings.TrimSpace(path)
	if path == "" {
		path = defaultSessionRegistryPath()
	}
	if strings.EqualFold(path, "off") {
		serveSessionDurability = nil
		return nil
	}
	path = pathutil.ExpandTilde(path)
	warnf := func(format string, args ...any) {
		if stderr != nil {
			fmt.Fprintf(stderr, format+"\n", args...)
		}
	}
	mirror := newSessionDurability(
		session.NewRegistry(session.NewFileStore(path)),
		leaseref.NewInDir(""),
		sessionDurabilityHost(),
		session.DefaultDescriptorTTL,
		time.Now,
		warnf,
		sessionDurabilityMeta(meta...),
	)
	if err := mirror.restore(tbl); err != nil {
		return err
	}
	serveSessionDurability = mirror
	return nil
}

func defaultSessionRegistryPath() string {
	if p := strings.TrimSpace(os.Getenv(sessionRegistryEnv)); p != "" {
		return p
	}
	if dir, err := os.UserConfigDir(); err == nil && strings.TrimSpace(dir) != "" {
		return filepath.Join(dir, "fak", "session-registry.json")
	}
	return filepath.Join(".fak", "session-registry.json")
}

func sessionDurabilityHost() string {
	host, _ := os.Hostname()
	host = strings.TrimSpace(host)
	if host == "" {
		host = "localhost"
	}
	return host
}

func sessionDurabilityMeta(meta ...session.DescriptorMeta) session.DescriptorMeta {
	if len(meta) > 0 {
		return meta[0]
	}
	return sessionDescriptorMeta(os.Args[1:])
}

func sessionDescriptorMeta(argv []string) session.DescriptorMeta {
	copied := append([]string(nil), argv...)
	startSHA := sessionStartSHA()
	cacheKey := sessionCacheKey(sessionDurabilityHost(), sessionWorkingDir(), startSHA, copied)
	return session.DescriptorMeta{
		PID:      os.Getpid(),
		Argv:     copied,
		StartSHA: startSHA,
		CacheKey: cacheKey,
	}
}

func defaultSessionIDFromMeta(meta session.DescriptorMeta) string {
	if meta.CacheKey == "" {
		return "guard"
	}
	key := meta.CacheKey
	if len(key) > 16 {
		key = key[:16]
	}
	return "guard-" + key
}

func sessionWorkingDir() string {
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}
	return wd
}

func sessionStartSHA() string {
	cmd := exec.Command("git", "rev-parse", "HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func sessionCacheKey(host, cwd, startSHA string, argv []string) string {
	h := sha256.New()
	for _, part := range []string{host, cwd, startSHA} {
		h.Write([]byte(part))
		h.Write([]byte{0})
	}
	for _, arg := range argv {
		h.Write([]byte(arg))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func registerServeSessionDurability(ctx context.Context, id string) error {
	if serveSessionDurability == nil {
		return nil
	}
	id = strings.TrimSpace(id)
	if id == "" {
		return nil
	}
	return serveSessionDurability.register(ctx, id, serveSessions.Get(id))
}

func applySessionControlDurable(ctx context.Context, tbl *session.Table, mirror *sessionDurability, traceID, verb string, req gateway.SessionControlRequest) (session.State, bool, error) {
	if mirror != nil && verb == "run" {
		if err := mirror.requireKnown(tbl, traceID); err != nil {
			return session.State{}, false, err
		}
	}
	st, ok, err := applySessionControl(tbl, traceID, verb, req)
	if err != nil || !ok || mirror == nil {
		return st, ok, err
	}
	switch verb {
	case "run", "budget", "pace", "priority":
		if err := mirror.writeThrough(ctx, traceID, st); err != nil {
			return st, false, err
		}
	}
	return st, ok, nil
}

func (d *sessionDurability) restore(tbl *session.Table) error {
	if d == nil || d.registry == nil || tbl == nil {
		return nil
	}
	descs, err := d.registry.List(d.now())
	if err != nil {
		return fmt.Errorf("restore session registry: %w", err)
	}
	for _, desc := range descs {
		trace := strings.TrimSpace(desc.Trace)
		if trace == "" {
			trace = desc.ID
		}
		tbl.Restore(trace, desc.RestoredState())
	}
	return nil
}

func (d *sessionDurability) register(ctx context.Context, id string, st session.State) error {
	if d == nil || d.registry == nil {
		return nil
	}
	id = strings.TrimSpace(id)
	if id == "" {
		return errors.New("session id is required")
	}
	now := d.now()
	if _, err := d.registry.RegisterWithMeta(id, d.host, st, d.ttl, now, d.meta); err != nil {
		return fmt.Errorf("persist session descriptor: %w", err)
	}
	d.publishBestEffort(ctx, id, st, now)
	return nil
}

func (d *sessionDurability) writeThrough(ctx context.Context, id string, st session.State) error {
	if d == nil || d.registry == nil {
		return nil
	}
	id = strings.TrimSpace(id)
	if id == "" {
		return errors.New("session id is required")
	}
	now := d.now()
	if _, err := d.registry.UpdateWithMeta(id, st, now, d.meta); err != nil {
		return fmt.Errorf("persist session descriptor: %w", err)
	}
	d.publishBestEffort(ctx, id, st, now)
	return nil
}

func persistServeSessionRevision(ctx context.Context, id string, st session.State) {
	if serveSessionDurability == nil {
		return
	}
	if strings.TrimSpace(id) == "" {
		id = st.TraceID
	}
	if err := serveSessionDurability.writeThrough(ctx, id, st); err != nil && serveSessionDurability.warnf != nil {
		serveSessionDurability.warnf("fak: session descriptor update failed for %s: %v", id, err)
	}
}

func (d *sessionDurability) requireKnown(tbl *session.Table, id string) error {
	id = strings.TrimSpace(id)
	if id == "" {
		return unknownSessionError{id: id}
	}
	if sessionTableHas(tbl, id) {
		return nil
	}
	st, ok, err := d.lookupState(id)
	if err != nil {
		return err
	}
	if ok {
		tbl.Restore(st.TraceID, st)
		return nil
	}
	return unknownSessionError{id: id}
}

func (d *sessionDurability) lookupState(id string) (session.State, bool, error) {
	if d == nil || d.registry == nil {
		return session.State{}, false, nil
	}
	id = strings.TrimSpace(id)
	descs, err := d.registry.List(d.now())
	if err != nil {
		return session.State{}, false, fmt.Errorf("read session registry: %w", err)
	}
	for _, desc := range descs {
		if desc.ID != id && desc.Trace != id {
			continue
		}
		trace := strings.TrimSpace(desc.Trace)
		if trace == "" {
			trace = desc.ID
		}
		st := desc.RestoredState()
		st.TraceID = trace
		return st, true, nil
	}
	return session.State{}, false, nil
}

func (d *sessionDurability) snapshotStates() ([]session.State, error) {
	if d == nil || d.registry == nil {
		return nil, nil
	}
	descs, err := d.registry.List(d.now())
	if err != nil {
		return nil, fmt.Errorf("read session registry: %w", err)
	}
	out := make([]session.State, 0, len(descs))
	for _, desc := range descs {
		trace := strings.TrimSpace(desc.Trace)
		if trace == "" {
			trace = desc.ID
		}
		st := desc.RestoredState()
		st.TraceID = trace
		out = append(out, st)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Priority != out[j].Priority {
			return out[i].Priority < out[j].Priority
		}
		if out[i].Rev != out[j].Rev {
			return out[i].Rev > out[j].Rev
		}
		return out[i].TraceID < out[j].TraceID
	})
	return out, nil
}

func (d *sessionDurability) publishBestEffort(ctx context.Context, id string, st session.State, now time.Time) {
	if d == nil || d.leases == nil {
		return
	}
	if st.Run == session.Stopped {
		if err := d.leases.RemoveSession(ctx, id); err != nil && d.warnf != nil {
			d.warnf("fak: session side-ref remove failed for %s: %v", id, err)
		}
		return
	}
	ttlSecs := int64(d.ttl / time.Second)
	if ttlSecs < 0 {
		ttlSecs = 0
	}
	_, err := d.leases.PublishSession(ctx, leaseref.SessionDescriptor{
		ID:        id,
		Host:      d.host,
		PCBState:  strings.ToUpper(st.Run.String()),
		UpdatedAt: now.Unix(),
		TTLSecs:   ttlSecs,
	})
	if err != nil && d.warnf != nil {
		d.warnf("fak: session side-ref publish failed for %s: %v", id, err)
	}
}

func sessionTableHas(tbl *session.Table, traceID string) bool {
	if tbl == nil {
		return false
	}
	for _, st := range tbl.Snapshot() {
		if st.TraceID == traceID {
			return true
		}
	}
	return false
}
