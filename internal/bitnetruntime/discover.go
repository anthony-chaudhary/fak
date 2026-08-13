package bitnetruntime

import (
	"context"
	"strconv"
	"strings"
)

// Prober collects the delegate's own probe output. It is injected rather than
// implemented here so this leaf stays pure: cmd/fak supplies one that runs the
// real bitnet.cpp binary under a deadline, and the tests supply a fake runtime.
// A non-nil error means the runtime could not be run at all, which is a
// different answer from a runtime that ran and reported nothing.
type Prober func(ctx context.Context) ([]byte, error)

// Discover runs the probe and folds its report into a Runtime. An empty Reason
// means the runtime is usable; any other Reason is the obstacle, and the
// returned Runtime is then only as complete as the report was.
func Discover(ctx context.Context, probe Prober) (Runtime, Reason) {
	if probe == nil {
		return Runtime{Name: RuntimeName}, ReasonProbeFailed
	}
	raw, err := probe(ctx)
	if err != nil {
		return Runtime{Name: RuntimeName}, ReasonProbeFailed
	}
	return ParseReport(raw)
}

// ParseReport reads a probe report: `key: value` lines, one per line, in the
// shape bitnet.cpp's own banner prints. Unrecognized keys are ignored so a
// future build's extra lines are not an error, but a recognized key declared
// twice with two different values is a contradiction and refuses — a report
// that disagrees with itself cannot be read for any of its other fields either.
//
// The report is a TRANSCRIPTION of what the delegate printed, not a fak-owned
// format: this package defines no artifact schema and requires no conversion.
func ParseReport(raw []byte) (Runtime, Reason) {
	rt := Runtime{Name: RuntimeName}
	seen := map[string]string{}
	var kernelsDeclared bool

	for _, line := range strings.Split(string(raw), "\n") {
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.ToLower(strings.TrimSpace(key))
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		switch key {
		case "version", RuntimeName, "kernels":
		default:
			continue
		}
		if prior, dup := seen[key]; dup && prior != value {
			return rt, ReasonProbeConflict
		}
		seen[key] = value

		switch key {
		case "version":
			rt.Build = value
		case RuntimeName:
			rt.Version = value
		case "kernels":
			kernelsDeclared = true
			rt.Kernels = parseKernels(value)
		}
	}

	if len(seen) == 0 {
		return rt, ReasonProbeEmpty
	}
	if rt.Version == "" {
		return rt, ReasonVersionUndeclared
	}
	cmp, ok := compareVersions(rt.Version, MinRuntimeVersion)
	if !ok {
		return rt, ReasonVersionMalformed
	}
	if cmp < 0 {
		return rt, ReasonVersionTooOld
	}
	if !kernelsDeclared {
		// A version banner is evidence of a version and of nothing else:
		// bitnet.cpp selects its kernels at compile time, so a build that did
		// not say which it carries has not been probed for them.
		return rt, ReasonKernelsUndeclared
	}
	return rt, ""
}

// parseKernels reads the build's comma-separated kernel list. A token outside
// this contract's vocabulary is kept out of the list rather than rejected: a
// future kernel is not an error, it is simply not something this contract can
// admit a model onto.
func parseKernels(value string) []Kernel {
	var out []Kernel
	for _, token := range strings.Split(value, ",") {
		k := Kernel(strings.ToLower(strings.TrimSpace(token)))
		if _, known := kernelSpecs[k]; known {
			out = append(out, k)
		}
	}
	return out
}

// compareVersions orders two dotted numeric versions. ok is false when either
// side is not dotted numerics, which is a malformed report rather than an old
// one.
func compareVersions(got, want string) (int, bool) {
	a, ok := parseVersion(got)
	if !ok {
		return 0, false
	}
	b, ok := parseVersion(want)
	if !ok {
		return 0, false
	}
	for i := range a {
		if a[i] != b[i] {
			if a[i] < b[i] {
				return -1, true
			}
			return 1, true
		}
	}
	return 0, true
}

// parseVersion reads major.minor.patch, tolerating a missing minor or patch and
// ignoring any pre-release or build suffix after the patch number.
func parseVersion(v string) ([3]int, bool) {
	v = strings.TrimSpace(v)
	if base, _, ok := strings.Cut(v, "-"); ok {
		v = strings.TrimSpace(base)
	}
	var out [3]int
	parts := strings.Split(v, ".")
	if len(parts) > 3 {
		return out, false
	}
	for i, part := range parts {
		n, err := strconv.Atoi(part)
		if err != nil || n < 0 {
			return out, false
		}
		out[i] = n
	}
	return out, true
}

// DiscoverAndAdmit is the whole public path: probe a runtime, then decide
// whether one artifact may be delegated to it on this host. A discovery
// obstacle short-circuits, because there is no runtime to admit anything
// against — reporting a host or model verdict beside it would imply the request
// was evaluated when it was not.
func DiscoverAndAdmit(ctx context.Context, probe Prober, host Host, model Model) Result {
	rt, reason := Discover(ctx, probe)
	if reason != "" {
		return finish([]Reason{reason}, Delegation{Kernel: KernelUnknown})
	}
	return Admit(rt, host, model)
}
