package servechain

import (
	"errors"
	"fmt"
	"net/url"
	"sync"
	"sync/atomic"
	"time"

	billing "github.com/anthony-chaudhary/fak/pkg/billing"
	billingkeystore "github.com/anthony-chaudhary/fak/pkg/billing/keystore"
	billingledger "github.com/anthony-chaudhary/fak/pkg/billing/ledger"
	billingmanifest "github.com/anthony-chaudhary/fak/pkg/billing/manifest"
	billingpricebook "github.com/anthony-chaudhary/fak/pkg/billing/pricebook"
)

// Defaults for the zero-valued Config fields. The unit scale keeps the
// ledger's integer units whole for the price book's per-million-token float
// amounts: one ledger unit is one micro-currency-unit by default.
const (
	DefaultUnitsPerCurrencyUnit = 1_000_000.0
	DefaultReserveUnits         = 1_000_000
	DefaultMaxBodyBytes         = 1 << 20
	DefaultSettleQueueDepth     = 256
)

// ErrInvalidConfig is the root sentinel for a Config that names no manifest,
// ledger, key store, or price book, or that carries a non-positive scale or
// reserve.
var ErrInvalidConfig = errors.New("servechain: invalid config")

// Config is everything the chain needs to run. Manifest, Prices, Ledger, and
// Keys are required; Policy is optional (nil means no external policy gate —
// the manifest's own limits still apply).
type Config struct {
	Manifest *billingmanifest.Manifest
	Prices   *billingpricebook.PriceBook
	Ledger   *billingledger.Ledger
	Keys     *billingkeystore.Store

	// Policy is the commercial authority consulted before reservation. A
	// nil Policy admits; the chain then enforces only the manifest's own
	// limits and balances.
	Policy billing.Policy

	// UnitsPerCurrencyUnit is how many ledger units one currency unit is
	// worth. Zero applies DefaultUnitsPerCurrencyUnit.
	UnitsPerCurrencyUnit float64

	// ReserveUnits is the hold placed on every admitted request before the
	// upstream answers. Zero applies DefaultReserveUnits.
	ReserveUnits int64

	// MaxBodyBytes bounds the inbound request body. Zero applies
	// DefaultMaxBodyBytes.
	MaxBodyBytes int64

	// SettleQueueDepth bounds the asynchronous settle queue; overflow falls
	// back to synchronous settlement inline. Zero applies
	// DefaultSettleQueueDepth.
	SettleQueueDepth int

	// Now overrides the clock, for tests. Nil means time.Now.
	Now func() time.Time
}

func (c *Config) fillDefaults() error {
	switch {
	case c == nil:
		return fmt.Errorf("%w: config is nil", ErrInvalidConfig)
	case c.Manifest == nil:
		return fmt.Errorf("%w: manifest is required", ErrInvalidConfig)
	case c.Prices == nil:
		return fmt.Errorf("%w: price book is required", ErrInvalidConfig)
	case c.Ledger == nil:
		return fmt.Errorf("%w: ledger is required", ErrInvalidConfig)
	case c.Keys == nil:
		return fmt.Errorf("%w: key store is required", ErrInvalidConfig)
	}
	if c.UnitsPerCurrencyUnit <= 0 {
		c.UnitsPerCurrencyUnit = DefaultUnitsPerCurrencyUnit
	}
	if c.ReserveUnits <= 0 {
		c.ReserveUnits = DefaultReserveUnits
	}
	if c.MaxBodyBytes <= 0 {
		c.MaxBodyBytes = DefaultMaxBodyBytes
	}
	if c.SettleQueueDepth <= 0 {
		c.SettleQueueDepth = DefaultSettleQueueDepth
	}
	if c.Now == nil {
		c.Now = time.Now
	}
	return nil
}

// route is one admitted served model: the manifest ref plus its upstream's
// parsed base URL.
type route struct {
	ref  billingmanifest.ModelRef
	base *url.URL
}

// Stats is the chain's counters. Every field is an atomic count of one
// observable outcome; Snapshot reads them consistently enough for operator
// reporting (not for billing — the ledger is the billing record).
type Stats struct {
	Requests    atomic.Uint64 // requests that entered the chain
	Admitted    atomic.Uint64 // requests that reached the proxy
	Rejected    atomic.Uint64 // chain-refused (4xx, not upstream errors)
	Released    atomic.Uint64 // reservations released after upstream failure
	Settled     atomic.Uint64 // reservations settled with known units
	Pending     atomic.Uint64 // reservations left PENDING (unknown usage)
	ProxyErrors atomic.Uint64 // upstream failures surfaced as 502
}

// snapshot copies the counters.
func (s *Stats) snapshot() map[string]uint64 {
	return map[string]uint64{
		"requests": s.Requests.Load(), "admitted": s.Admitted.Load(),
		"rejected": s.Rejected.Load(), "released": s.Released.Load(),
		"settled": s.Settled.Load(), "pending": s.Pending.Load(),
		"proxy_errors": s.ProxyErrors.Load(),
	}
}

// Chain is the billing gateway handler. Construct with New; serve with
// ServeHTTP; stop with Close. Safe for concurrent use.
type Chain struct {
	cfg       Config
	routes    map[string]route    // served model name -> route
	upstreams map[string]*url.URL // upstream name -> parsed base URL
	caps      *billing.SpendCaps
	window    *rateWindow
	settle    *settleQueue
	closeOnce sync.Once
	stats     Stats
}

// New validates cfg, resolves the manifest's model→upstream bindings, and
// starts the settle worker. It performs no I/O beyond that resolution.
func New(cfg Config) (*Chain, error) {
	if err := cfg.fillDefaults(); err != nil {
		return nil, err
	}
	if err := cfg.Manifest.Validate(); err != nil {
		return nil, fmt.Errorf("%w: manifest: %w", ErrInvalidConfig, err)
	}
	if err := cfg.Prices.Validate(); err != nil {
		return nil, fmt.Errorf("%w: price book: %w", ErrInvalidConfig, err)
	}

	upstreams := make(map[string]*url.URL, len(cfg.Manifest.Upstreams))
	for i := range cfg.Manifest.Upstreams {
		u := &cfg.Manifest.Upstreams[i]
		parsed, err := url.Parse(u.BaseURL)
		if err != nil || parsed.Host == "" {
			return nil, fmt.Errorf("%w: upstream %q base %q", ErrInvalidConfig, u.Name, u.BaseURL)
		}
		upstreams[u.Name] = parsed
	}

	routes := make(map[string]route, len(cfg.Manifest.Models))
	for i := range cfg.Manifest.Models {
		ref := cfg.Manifest.Models[i]
		base, ok := upstreams[ref.Upstream]
		if !ok {
			return nil, fmt.Errorf("%w: model %q references undeclared upstream %q", ErrInvalidConfig, ref.Name, ref.Upstream)
		}
		routes[ref.Name] = route{ref: ref, base: base}
	}
	if len(routes) == 0 {
		return nil, fmt.Errorf("%w: manifest declares no models", ErrInvalidConfig)
	}

	c := &Chain{
		cfg:       cfg,
		routes:    routes,
		upstreams: upstreams,
		caps: billing.NewSpendCaps(billing.CapConfig{
			DailyTokenCap:   int64(cfg.Manifest.Limits.DailyTokenCap),
			SessionTokenCap: int64(cfg.Manifest.Limits.SessionTokenCap),
		}),
		window: newRateWindow(cfg.Manifest.Limits),
	}
	c.settle = newSettleQueue(c, cfg.SettleQueueDepth)
	return c, nil
}

// Close stops the settle worker after it drains its queue. Idempotent.
func (c *Chain) Close() {
	if c == nil {
		return
	}
	c.closeOnce.Do(func() { c.settle.stop() })
}

// StatsSnapshot returns the current counter values.
func (c *Chain) StatsSnapshot() map[string]uint64 {
	return c.stats.snapshot()
}
