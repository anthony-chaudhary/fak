// Package harnesskit defines fak's supported public vocabulary for agent products.
//
// The package is deliberately contract-only: importing it does not expose fak's
// internal engine, policy, or gateway packages. Extensions receive only the
// capability-checked Services supplied by a host.
package harnesskit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
)

// ContractVersion is the compatibility line implemented by this package.
const ContractVersion = "v1alpha1"

// Capability names authority that a product may request. A declaration is not
// a grant; the host intersects it with the effective policy at every operation.
type Capability string

// ExtensionPlane is a supported place where a builder may add behavior.
type ExtensionPlane string

const (
	PlaneTools        ExtensionPlane = "tools"
	PlaneModels       ExtensionPlane = "models"
	PlaneContext      ExtensionPlane = "context"
	PlaneInstructions ExtensionPlane = "instructions"
	PlaneTransports   ExtensionPlane = "transports"
	PlaneEvents       ExtensionPlane = "events"
	PlaneHardware     ExtensionPlane = "hardware"
)

// LifecycleState is the host-observable state of an extension instance.
type LifecycleState string

const (
	StateDeclared LifecycleState = "declared"
	StateStarting LifecycleState = "starting"
	StateRunning  LifecycleState = "running"
	StateDraining LifecycleState = "draining"
	StateClosed   LifecycleState = "closed"
	StateFailed   LifecycleState = "failed"
)

// Ownership states who must release a resource.
type Ownership string

const (
	OwnershipCaller Ownership = "caller"
	OwnershipHost   Ownership = "host"
)

// Provenance identifies the reviewed implementation selected by a builder.
type Provenance struct {
	Source  string `json:"source"`
	Version string `json:"version"`
	Digest  string `json:"digest,omitempty"`
}

// Extension describes one implementation attached to an extension plane.
type Extension struct {
	ID            string         `json:"id"`
	Plane         ExtensionPlane `json:"plane"`
	Compatibility string         `json:"compatibility"`
	Requires      []Capability   `json:"requires,omitempty"`
	Provenance    Provenance     `json:"provenance"`
}

// Profile is a named, reusable set of extensions and requested capabilities.
type Profile struct {
	ID           string       `json:"id"`
	Capabilities []Capability `json:"capabilities,omitempty"`
	Extensions   []Extension  `json:"extensions,omitempty"`
}

// ProductSpec is the portable description accepted by Builder.
type ProductSpec struct {
	ID         string      `json:"id"`
	Version    string      `json:"version"`
	Profile    Profile     `json:"profile"`
	Transports []Transport `json:"transports,omitempty"`
}

// Transport is a public transport declaration. Configuration remains with its
// adapter so secrets do not enter the portable product description.
type Transport struct {
	ID         string     `json:"id"`
	Provenance Provenance `json:"provenance"`
}

// Product is an immutable, validated builder result.
type Product struct{ spec ProductSpec }

// Spec returns a copy of the portable product description.
func (p Product) Spec() ProductSpec { return cloneSpec(p.spec) }

// Builder validates the smallest public product vocabulary. It performs no I/O.
type Builder struct{ spec ProductSpec }

// New creates a builder for a product identity and compatibility version.
func New(id, version string) *Builder { return &Builder{spec: ProductSpec{ID: id, Version: version}} }

// WithProfile replaces the product profile.
func (b *Builder) WithProfile(p Profile) *Builder { b.spec.Profile = p; return b }

// WithTransport adds a transport declaration.
func (b *Builder) WithTransport(t Transport) *Builder {
	b.spec.Transports = append(b.spec.Transports, t)
	return b
}

// Build validates and freezes the product description.
func (b *Builder) Build() (Product, error) {
	if b == nil {
		return Product{}, &Error{Code: CodeInvalid, Op: "build", Err: errors.New("nil builder")}
	}
	s := cloneSpec(b.spec)
	if s.ID == "" || s.Version == "" || s.Profile.ID == "" {
		return Product{}, &Error{Code: CodeInvalid, Op: "build", Err: errors.New("product id, version, and profile id are required")}
	}
	seen := map[string]bool{}
	allowed := SupportedPlanes()
	for _, e := range s.Profile.Extensions {
		if e.ID == "" || e.Compatibility == "" || e.Provenance.Source == "" || e.Provenance.Version == "" {
			return Product{}, &Error{Code: CodeInvalid, Op: "extension", Err: fmt.Errorf("incomplete extension %q", e.ID)}
		}
		if !slices.Contains(allowed, e.Plane) {
			return Product{}, &Error{Code: CodeUnsupported, Op: "extension", Err: fmt.Errorf("plane %q", e.Plane)}
		}
		key := string(e.Plane) + "/" + e.ID
		if seen[key] {
			return Product{}, &Error{Code: CodeConflict, Op: "extension", Err: fmt.Errorf("duplicate %q", key)}
		}
		seen[key] = true
	}
	return Product{spec: s}, nil
}

func cloneSpec(s ProductSpec) ProductSpec {
	out := s
	out.Profile.Capabilities = slices.Clone(s.Profile.Capabilities)
	out.Profile.Extensions = slices.Clone(s.Profile.Extensions)
	for i := range out.Profile.Extensions {
		out.Profile.Extensions[i].Requires = slices.Clone(out.Profile.Extensions[i].Requires)
	}
	out.Transports = slices.Clone(s.Transports)
	return out
}

// Code is a stable, machine-actionable error category.
type Code string

const (
	CodeInvalid      Code = "invalid"
	CodeUnsupported  Code = "unsupported"
	CodeConflict     Code = "conflict"
	CodeDenied       Code = "denied"
	CodeCanceled     Code = "canceled"
	CodeBackpressure Code = "backpressure"
	CodeInternal     Code = "internal"
)

// Error preserves a stable code while retaining the underlying cause.
type Error struct {
	Code Code
	Op   string
	Err  error
}

func (e *Error) Error() string {
	if e == nil {
		return "<nil>"
	}
	return fmt.Sprintf("harnesskit %s: %s: %v", e.Code, e.Op, e.Err)
}
func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// Stream is a cancellation-aware, backpressured event stream. Send and Recv
// block until accepted, completed, or ctx is done; io.EOF means clean close.
type Stream[T any] interface {
	Send(context.Context, T) error
	Recv(context.Context) (T, error)
	Close() error
}

// Invocation is the only tool reachability exposed to public extensions.
type Invocation struct {
	Tool      string
	Arguments json.RawMessage
}

// Result is an adjudicated invocation result.
type Result struct{ Content json.RawMessage }

// Services is the capability-filtered host surface. Invoke still adjudicates
// every call; extension registration never grants authority.
type Services interface {
	Invoke(context.Context, Invocation) (Result, error)
}

// Runtime is owned and closed by the host after Start succeeds. Drain stops new
// work and waits for accepted work, bounded by ctx; Close must be idempotent.
type Runtime interface {
	Drain(context.Context) error
	io.Closer
}

// Factory starts an extension. The host owns a non-nil returned Runtime; the
// caller owns inputs it passes through Services unless their contract says otherwise.
type Factory interface {
	Manifest() Extension
	Start(context.Context, Services) (Runtime, error)
}
