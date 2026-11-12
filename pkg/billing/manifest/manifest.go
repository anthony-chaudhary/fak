package manifest

import (
	"errors"
	"fmt"
	"net/url"
)

// ManifestVersionV1 is the only manifest format version this package
// produces and validates. Files evolve additively: later revisions may add
// optional fields, and consumers that decode strictly must expect them;
// breaking changes get a new version string.
const ManifestVersionV1 = "fak-billing-manifest/v1"

// Sentinel errors. Every validation failure wraps ErrInvalidManifest, so
// errors.Is(err, ErrInvalidManifest) identifies any malformed manifest
// while the specific sentinel names the fault. Load-level faults wrap
// ErrMalformedPayload: the bytes were not one decodable v1 file at all.
var (
	// ErrInvalidManifest is the root sentinel for every malformed Manifest.
	ErrInvalidManifest = errors.New("manifest: invalid billing manifest")

	// ErrUnknownVersion marks a manifest whose version this package does
	// not validate.
	ErrUnknownVersion = fmt.Errorf("%w: unknown version", ErrInvalidManifest)

	// ErrMalformedPayload marks serialized input that is not exactly one
	// decodable v1 manifest: bad JSON, a field outside the v1 schema, or
	// trailing data.
	ErrMalformedPayload = fmt.Errorf("%w: malformed serialized payload", ErrInvalidManifest)

	// ErrMissingStateDir marks a manifest without its state directory.
	ErrMissingStateDir = fmt.Errorf("%w: state_dir is required", ErrInvalidManifest)

	// ErrMissingPricesPath marks a manifest without its price-book path.
	ErrMissingPricesPath = fmt.Errorf("%w: prices_path is required", ErrInvalidManifest)

	// ErrMissingModelName marks a model ref without its served name.
	ErrMissingModelName = fmt.Errorf("%w: model name is required", ErrInvalidManifest)

	// ErrDuplicateModelName marks two model refs serving the same name.
	ErrDuplicateModelName = fmt.Errorf("%w: duplicate model name", ErrInvalidManifest)

	// ErrMissingUpstreamModel marks a model ref without its upstream model id.
	ErrMissingUpstreamModel = fmt.Errorf("%w: upstream_model is required", ErrInvalidManifest)

	// ErrUnknownUpstreamRef marks a model ref naming an upstream the
	// manifest does not declare.
	ErrUnknownUpstreamRef = fmt.Errorf("%w: model references undeclared upstream", ErrInvalidManifest)

	// ErrMissingUpstreamName marks an upstream without its name.
	ErrMissingUpstreamName = fmt.Errorf("%w: upstream name is required", ErrInvalidManifest)

	// ErrDuplicateUpstreamName marks two upstreams carrying the same name.
	ErrDuplicateUpstreamName = fmt.Errorf("%w: duplicate upstream name", ErrInvalidManifest)

	// ErrInvalidBaseURL marks an upstream whose base_url is not an
	// absolute http or https URL.
	ErrInvalidBaseURL = fmt.Errorf("%w: base_url must be an absolute http or https URL", ErrInvalidManifest)

	// ErrNegativeLimit marks a limit value below zero.
	ErrNegativeLimit = fmt.Errorf("%w: limit values must be non-negative", ErrInvalidManifest)
)

// ModelRef binds one served model name to the upstream that serves it and
// the model id that upstream knows it by. The binding is declared, not
// inferred: nothing in this package routes or resolves.
type ModelRef struct {
	// Name is the served model name clients request. Required; unique
	// across the manifest's Models.
	Name string `json:"name"`

	// Upstream names the Upstream that serves this model. Required; must
	// match a name declared in the manifest's Upstreams.
	Upstream string `json:"upstream"`

	// UpstreamModel is the model id as the named upstream knows it.
	// Required.
	UpstreamModel string `json:"upstream_model"`
}

// Validate checks the closed model-ref contract: a non-empty served name and
// a non-empty upstream model id. Set-level rules (name uniqueness and the
// upstream reference) are enforced by Manifest.Validate.
func (r *ModelRef) Validate() error {
	if r == nil {
		return ErrInvalidManifest
	}
	if r.Name == "" {
		return ErrMissingModelName
	}
	if r.UpstreamModel == "" {
		return fmt.Errorf("%w: model %q", ErrMissingUpstreamModel, r.Name)
	}
	return nil
}

// Upstream is one OpenAI-compatible engine endpoint the manifest may serve
// from: stock SGLang/vLLM or anything else speaking the compatible API.
// This package neither calls nor probes it; BaseURL is carried opaquely past
// absoluteness checking.
type Upstream struct {
	// Name identifies the upstream within the manifest. Required; unique.
	Name string `json:"name"`

	// BaseURL is the absolute http/https base of the upstream's
	// OpenAI-compatible API (for example "http://host:8000/v1"). Required.
	BaseURL string `json:"base_url"`
}

// Validate checks the closed upstream contract: a non-empty name and an
// absolute http/https base URL. Set-level rules (name uniqueness) are
// enforced by Manifest.Validate.
func (u *Upstream) Validate() error {
	if u == nil {
		return ErrInvalidManifest
	}
	if u.Name == "" {
		return ErrMissingUpstreamName
	}
	return u.validateBaseURL()
}

// validateBaseURL enforces the closed base-URL contract: an absolute http or
// https URL with a host. Path, port, and credential material are carried
// opaquely: this package checks scheme and host only.
func (u *Upstream) validateBaseURL() error {
	parsed, err := url.Parse(u.BaseURL)
	if err != nil {
		return fmt.Errorf("%w: %q: %v", ErrInvalidBaseURL, u.BaseURL, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return fmt.Errorf("%w: %q has scheme %q", ErrInvalidBaseURL, u.BaseURL, parsed.Scheme)
	}
	if parsed.Host == "" {
		return fmt.Errorf("%w: %q has no host", ErrInvalidBaseURL, u.BaseURL)
	}
	return nil
}

// Limits is the set of neutral caps an operator may declare on a serving
// deployment. Every field is a plain integer count; zero means uncapped,
// and zero is the default: no cap, threshold, or formula lives in this type.
// Whether a consumer enforces a declared cap, how, and what happens at the
// boundary are decisions of the consuming command.
type Limits struct {
	// RequestsPerMinute is the per-minute request cap; 0 means uncapped.
	RequestsPerMinute int `json:"requests_per_minute"`

	// TokensPerMinute is the per-minute token cap; 0 means uncapped.
	TokensPerMinute int `json:"tokens_per_minute"`

	// DailyTokenCap is the per-day token cap; 0 means uncapped.
	DailyTokenCap int `json:"daily_token_cap"`

	// SessionTokenCap is the per-session token cap; 0 means uncapped.
	SessionTokenCap int `json:"session_token_cap"`
}

// Validate checks the closed limits contract: no negative value. Zero is
// the uncapped default and is always valid.
func (l Limits) Validate() error {
	values := []int{l.RequestsPerMinute, l.TokensPerMinute, l.DailyTokenCap, l.SessionTokenCap}
	for _, n := range values {
		if n < 0 {
			return fmt.Errorf("%w: %d", ErrNegativeLimit, n)
		}
	}
	return nil
}

// Manifest is the operator-facing billing configuration for one serving
// deployment: which models are served, which OpenAI-compatible upstreams
// serve them, where the price book lives, which caps apply, and where the
// billing state (ledger, key store, reports) lives.
//
// The operator writes the file; this package only declares its closed v1
// shape and checks it. Nothing here is a policy decision: the manifest
// carries declarations, and the consuming command decides what they mean.
type Manifest struct {
	// Version tags the file format. Always ManifestVersionV1 for
	// manifests this package produces or validates.
	Version string `json:"version"`

	// StateDir is where the billing state lives: the ledger, the key
	// store, and the reports. A relative or absolute path. Required.
	StateDir string `json:"state_dir"`

	// PricesPath is the path to the price-book file, loaded separately by
	// the caller (see pkg/billing/pricebook). This package carries the
	// reference only; it never reads the price book. Required.
	PricesPath string `json:"prices_path"`

	// Models is the set of served model names and their upstream
	// bindings. Each name is unique.
	Models []ModelRef `json:"models"`

	// Upstreams is the set of OpenAI-compatible engine endpoints the
	// model refs may name. Each name is unique.
	Upstreams []Upstream `json:"upstreams"`

	// Limits is the neutral cap set; zero fields are uncapped.
	Limits Limits `json:"limits"`
}

// Validate checks the closed v1 contract: the exact v1 version tag, a
// non-empty state dir and price-book path, uniquely named upstreams with
// absolute http/https base URLs, uniquely named model refs that each bind a
// non-empty upstream model id to a declared upstream, and non-negative
// limits. On the first violated rule it returns a wrapped sentinel error,
// matchable with errors.Is against the specific sentinel or against
// ErrInvalidManifest.
func (m *Manifest) Validate() error {
	if m == nil {
		return ErrInvalidManifest
	}
	if m.Version != ManifestVersionV1 {
		return fmt.Errorf("%w: have %q", ErrUnknownVersion, m.Version)
	}
	if m.StateDir == "" {
		return ErrMissingStateDir
	}
	if m.PricesPath == "" {
		return ErrMissingPricesPath
	}
	upstreamNames := make(map[string]struct{}, len(m.Upstreams))
	for i := range m.Upstreams {
		u := &m.Upstreams[i]
		if err := u.Validate(); err != nil {
			return err
		}
		if _, dup := upstreamNames[u.Name]; dup {
			return fmt.Errorf("%w: %q", ErrDuplicateUpstreamName, u.Name)
		}
		upstreamNames[u.Name] = struct{}{}
	}
	seen := make(map[string]struct{}, len(m.Models))
	for i := range m.Models {
		r := &m.Models[i]
		if err := r.Validate(); err != nil {
			return err
		}
		if _, dup := seen[r.Name]; dup {
			return fmt.Errorf("%w: %q", ErrDuplicateModelName, r.Name)
		}
		seen[r.Name] = struct{}{}
		if _, ok := upstreamNames[r.Upstream]; !ok {
			return fmt.Errorf("%w: model %q references %q", ErrUnknownUpstreamRef, r.Name, r.Upstream)
		}
	}
	return m.Limits.Validate()
}
