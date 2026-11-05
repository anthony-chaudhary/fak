package main

import (
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

// serve_load_manifest_admission.go — the layout-guard (lg-preflight-wire) admission: the
// header-only manifest preflight (ggufload.BuildManifestPreflight on the GGUF leg, the
// safetensors twin on the directory leg) runs on every serve load path, fail-closed, BEFORE
// model dispatch. A refusal aborts the load with ServeManifestPreflightRefusal — a typed error
// carrying the full fak.new-model-refusal/1 report — never a warning. The census reads no
// tensor payload: it is pure metadata plus the (merged-shard) tensor directory.

// ErrServeManifestPreflight is the sentinel every serve-side manifest-preflight refusal wraps,
// so a caller can classify the abort with errors.Is without parsing prose.
var ErrServeManifestPreflight = errors.New("serve: manifest preflight refused the checkpoint")

// servePreflightErrorMax bounds the refusal lines folded into the error text.
const servePreflightErrorMax = 8

// ServeManifestPreflightRefusal is the typed, fail-closed admission refusal: the checkpoint
// header failed the manifest preflight (unread keys, HF mismatch, missing/orphan/mis-shaped
// forward-requested tensors, unsupported dtypes, declared-but-absent MTP head) before any
// tensor byte was read. Report carries the complete aggregate so one run names every seam.
type ServeManifestPreflightRefusal struct {
	Path   string
	Report ggufload.ManifestPreflight
}

// Error renders the verdict plus a bounded refusal list.
func (e *ServeManifestPreflightRefusal) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "manifest preflight REFUSED %s:", e.Path)
	shown := e.Report.Refusals
	if len(shown) > servePreflightErrorMax {
		for _, r := range shown[:servePreflightErrorMax] {
			fmt.Fprintf(&b, "\n  %s on %s: %s", r.Reason, r.Axis, r.Detail)
		}
		fmt.Fprintf(&b, "\n  (+%d more)", len(shown)-servePreflightErrorMax)
		return b.String()
	}
	for _, r := range shown {
		fmt.Fprintf(&b, "\n  %s on %s: %s", r.Reason, r.Axis, r.Detail)
	}
	return b.String()
}

// Unwrap exposes the sentinel for errors.Is classification.
func (e *ServeManifestPreflightRefusal) Unwrap() error { return ErrServeManifestPreflight }

// Refusals returns the full refusal report (stable reason/axis order, set by finish()).
func (e *ServeManifestPreflightRefusal) Refusals() []newmodel.Refusal { return e.Report.Refusals }

// Has reports whether the report carries a refusal with the given reason token.
func (e *ServeManifestPreflightRefusal) Has(reason newmodel.RefusalReason) bool {
	for _, r := range e.Report.Refusals {
		if r.Reason == reason {
			return true
		}
	}
	return false
}

// serveAdmitManifestPreflight runs the header-only GGUF manifest preflight over an already
// opened (merged-shard) header and returns the typed refusal when it refuses.
func serveAdmitManifestPreflight(path string, f *ggufload.File) error {
	rep := ggufload.BuildManifestPreflight(ggufload.ManifestPreflightInput{Path: path, File: f})
	if !rep.Refused() {
		return nil
	}
	return &ServeManifestPreflightRefusal{Path: path, Report: rep}
}

// serveAdmitSafetensorsPreflight is the safetensors leg: the header-only plan over the
// checkpoint directory, mapped onto the shared manifest report (the CLI verb's own mapping),
// refused with the same typed error.
func serveAdmitSafetensorsPreflight(dir string) error {
	plan, err := fakmodel.BuildSafetensorsPreflight(fakmodel.SafetensorsPreflightInput{Dir: dir})
	if err != nil {
		return &ServeManifestPreflightRefusal{Path: dir, Report: safetensorsPlanErrorReport(dir, err)}
	}
	rep := manifestFromSafetensorsPlan(dir, plan)
	if !rep.Refused() {
		return nil
	}
	return &ServeManifestPreflightRefusal{Path: dir, Report: rep}
}

// safetensorsPlanErrorReport folds a plan-level I/O or config-parse failure into the shared
// report shape so the typed refusal covers both legs uniformly.
func safetensorsPlanErrorReport(dir string, err error) ggufload.ManifestPreflight {
	return ggufload.ManifestPreflight{
		Schema:   ggufload.ManifestPreflightSchema,
		Path:     dir,
		Verdict:  ggufload.ManifestPreflightRefused,
		Refusals: []newmodel.Refusal{newmodel.HeaderRefusal(newmodel.RefusalHeaderConfigInvalid, "safetensors", err.Error())},
	}
}
