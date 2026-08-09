package accounts

// Per-account settings projection — the durable, in-tree replacement for the external
// `sync_account_settings.py` ("csync") chore.
//
// Each rotated Claude config home reads ONLY its own `<config_dir>/settings.json` — it never
// sees `~/.claude/settings.json`. So a default that must apply to every launched account,
// most importantly `permissions.defaultMode: bypassPermissions` (what lets a guarded launch
// skip Claude's per-tool prompts, since under `fak guard` the kernel IS the permission system)
// plus `skipDangerousModePermissionPrompt: true`, has to be WRITTEN INTO each account's file.
//
// The single source of truth for those defaults already lives in the registry, under the dos
// view's `defaults` block: registry.Views["dos"].Blocks["defaults"]["settings"]. Before this,
// that block was projected into each account's settings.json only by a hand-run Python script
// living in another repo — so a brand-new account added with `fak accounts add` launched
// WITHOUT the bypass default until someone remembered to run csync, and the setting appeared
// to "get lost". This file ports csync's deep-merge into Go so `fak accounts add` seeds the
// new seat and `fak accounts sync` re-projects the whole roster, keeping the defaults in ONE
// place with no external step.
//
// The merge is a faithful port of csync's `deep_merge`: the defaults win for every leaf key
// they name, dict values recurse, and every other key the account already has (theme,
// effortLevel, …) is left untouched. The exceptions are the SEAT-OWNED keys handled in
// seatSettingsDefaults — `model` (#3091) and `askUserQuestionTimeout` — where an already-set
// per-seat value is preserved and a fresh seat is seeded with the launch default rather than
// whatever the registry pinned. It is idempotent — a second projection reports no change
// — because the serialized form is byte-stable (sorted keys, 2-space indent, trailing newline),
// exactly matching csync's `json.dumps(..., indent=2, sort_keys=True) + "\n"` so the two
// implementations cannot drift.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
)

// settingsFilename is the per-config-dir settings file every rotated Claude account reads.
const settingsFilename = "settings.json"

// SettingsResult records what the projection did to one home's settings.json: the seat name,
// the file path, whether it changed, and (when skipped) the reason. The caller renders these
// as the human report ("updated" / "ok (no change)" / "skipped: tombstoned").
type SettingsResult struct {
	Name    string
	Path    string
	Changed bool
	Skipped string // "" when acted on; else why it was skipped (e.g. "tombstoned", "no dir")
}

// DefaultsSettings returns the registry's per-account settings defaults — the map at
// Views["dos"].Blocks["defaults"]["settings"] — and ok=false when the block is absent or not
// the expected shape. Every rung is a guarded type assertion (mirroring rotationBlock), so a
// pre-defaults registry, or one whose `defaults` block carries no `settings` sub-map, is a
// clean no-op signal rather than a panic or an error.
func (r Registry) DefaultsSettings() (map[string]any, bool) {
	vc, ok := r.Views["dos"]
	if !ok {
		return nil, false
	}
	defaults, ok := vc.Blocks["defaults"].(map[string]any)
	if !ok {
		return nil, false
	}
	settings, ok := defaults["settings"].(map[string]any)
	if !ok || len(settings) == 0 {
		return nil, false
	}
	return settings, true
}

// ProjectedDefaultModel is the model the projection seeds into a seat that carries no model of
// its own: Opus 5, matching cmd/fak's compiled-in defaultLaunchModel so a bare `claude
// --resume`, a direct `claude` under a seat's CLAUDE_CONFIG_DIR, or a guard budget-restart
// relaunch — none of which pass --model — inherits the SAME primary the account-switched launcher
// pins. It MUST stay equal to that launch default; cmd/fak binds them with a guard test.
//
// The registry's defaults.settings block historically pinned model=claude-fable-5 — the fleet
// launch FALLBACK, not the primary — so the deep-merge force-overwrote every seat's model back to
// fable-5 on each sync/add, and only the launch path (which passes --model explicitly) honored
// opus (#3091). The projection now overrides that pinned value with the launch default and treats
// model as seat-owned. Fable 5 stays the launch fallback CHAIN and is not touched here.
const ProjectedDefaultModel = "claude-opus-5"

// questionTimeoutKey is the harness setting behind the "Question auto-continue timeout" config
// row: the idle time before a pending AskUserQuestion auto-continues with whatever answers are
// selected so far. The harness accepts a closed enum — "60s", "5m", "10m", "never" — and
// defaults to "never", i.e. a question a seat cannot answer blocks that session forever.
const questionTimeoutKey = "askUserQuestionTimeout"

// ProjectedQuestionTimeout is the value the projection seeds into a seat that carries none:
// the shortest rung the harness offers, so an unattended launch never parks on a question. A
// guarded fleet seat has no operator at the keyboard, so the harness default ("never") turns
// every ask into an indefinite stall; auto-continuing after 60s of idle keeps the session moving
// while still leaving a real operator a full minute to answer an interactive launch.
//
// It MUST stay one of the harness's accepted enum values — the setting is parsed with a
// `.catch(undefined)`, so a typo degrades silently back to "never" rather than erroring.
const ProjectedQuestionTimeout = "60s"

// seatSettingsDefaults specializes the registry defaults overlay for one seat's current settings,
// applying the SEAT-OWNED carve-outs. For `model` (#3091), when the defaults pin one:
//   - if the seat already carries its own model, model is DROPPED from the overlay so the merge
//     preserves the seat's choice (a hand-set "model":"sonnet" survives the next sync/add);
//   - otherwise model is SEEDED as ProjectedDefaultModel (the launch default), not whatever the
//     registry pinned (historically the fable-5 fallback).
//
// When the defaults name no model at all there is nothing to reconcile for that key.
// `askUserQuestionTimeout` follows the same shape but needs no registry pin to fire: a seat that
// has never chosen one is seeded with ProjectedQuestionTimeout so every launched account carries
// the auto-continue default, and a seat that HAS chosen one keeps it.
//
// The shared registry defaults map is never mutated — the overlay is a shallow copy, which is
// sufficient because both carve-out keys are top-level scalars in the settings block (the
// recursion in deepMergeJSON never reaches inside them).
func seatSettingsDefaults(defaults, seat map[string]any) map[string]any {
	overlay := make(map[string]any, len(defaults)+1)
	for k, v := range defaults {
		overlay[k] = v
	}
	if _, pins := defaults["model"]; pins {
		if _, seatHasModel := seat["model"]; seatHasModel {
			delete(overlay, "model")
		} else {
			overlay["model"] = ProjectedDefaultModel
		}
	}
	if _, seatHasTimeout := seat[questionTimeoutKey]; seatHasTimeout {
		delete(overlay, questionTimeoutKey)
	} else {
		overlay[questionTimeoutKey] = ProjectedQuestionTimeout
	}
	return overlay
}

// ProjectSettings deep-merges the registry's defaults.settings block into each home's
// settings.json, writing through the injected writeFn (kept injectable so the projection is
// unit-tested without touching disk, the same seam pattern as accountsLaunchRun). It returns a
// per-home result list, ok=false when the registry carries no defaults.settings block (a clean
// whole-roster no-op), and an error only on a genuine write failure.
//
// A home is skipped — recorded, never written — when it is tombstoned (an inactive seat must
// not be reseeded) or carries no dir. For an active seat the existing settings.json is read
// tolerantly (a missing or unparseable file reads as {} and is (re)created), the defaults are
// merged in — with the seat-owned carve-outs (seatSettingsDefaults) applied first, which is also
// what seeds `askUserQuestionTimeout` onto a seat the registry says nothing about — and
// the file is rewritten ONLY when the merge changed something, so a re-run over an already-synced
// roster writes nothing.
func (r Registry) ProjectSettings(homes []Home, writeFn func(path string, b []byte) error) ([]SettingsResult, bool, error) {
	defaults, ok := r.DefaultsSettings()
	if !ok {
		return nil, false, nil
	}
	results := make([]SettingsResult, 0, len(homes))
	for _, h := range homes {
		res := SettingsResult{Name: h.Name}
		switch {
		case !h.Active():
			res.Skipped = "tombstoned"
			results = append(results, res)
			continue
		case h.Dir == "":
			res.Skipped = "no dir"
			results = append(results, res)
			continue
		}
		path := filepath.Join(h.Dir, settingsFilename)
		res.Path = path
		current := readSettings(path)
		merged, changed := deepMergeJSON(current, seatSettingsDefaults(defaults, current))
		if changed {
			b, err := marshalSettings(merged)
			if err != nil {
				return results, true, err
			}
			if err := writeFn(path, b); err != nil {
				return results, true, err
			}
		}
		res.Changed = changed
		results = append(results, res)
	}
	return results, true, nil
}

// readSettings reads and JSON-decodes an account's settings.json into a map. Any read error
// (a missing file is the common case for a brand-new seat) OR a decode error (a corrupt file,
// or a top-level array/scalar) yields an empty map, so the merge treats it as "no existing
// settings" and repairs it — matching csync's try/except-to-{} behavior.
func readSettings(path string) map[string]any {
	b, err := os.ReadFile(path)
	if err != nil {
		return map[string]any{}
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil || m == nil {
		return map[string]any{}
	}
	return m
}

// deepMergeJSON merges overlay into base and reports whether anything changed. It is a faithful
// port of csync's deep_merge: overlay wins for every leaf key it names; when BOTH base[k] and
// overlay[k] are maps it recurses (OR-ing the changed bit); any other overlay value replaces
// the base value wholesale (including a map replacing a scalar, or the reverse — the two
// map type-asserts fail, so it falls through to the replace path, exactly as the Python
// isinstance guard does). Keys absent from overlay are preserved. base is not mutated (out is
// a shallow copy; recursion copies each nested level it touches).
//
// Equality uses reflect.DeepEqual, not ==, because JSON-decoded values include []any and
// map[string]any which are not comparable with == (it would panic); DeepEqual also reproduces
// Python's `out.get(k) != v` for the missing-key case (a nil base value differs from any
// non-nil overlay value, so the key is added and change is reported).
func deepMergeJSON(base, overlay map[string]any) (map[string]any, bool) {
	out := make(map[string]any, len(base))
	for k, v := range base {
		out[k] = v
	}
	changed := false
	for k, ov := range overlay {
		if ovMap, ok := ov.(map[string]any); ok {
			if baseMap, ok := out[k].(map[string]any); ok {
				merged, subChanged := deepMergeJSON(baseMap, ovMap)
				out[k] = merged
				changed = changed || subChanged
				continue
			}
		}
		if cur, present := out[k]; !present || !reflect.DeepEqual(cur, ov) {
			out[k] = ov
			changed = true
		}
	}
	return out, changed
}

// marshalSettings serializes a merged settings map to the byte-stable form that makes the
// projection idempotent: sorted keys (encoding/json sorts map[string]any keys), 2-space
// indent, and a trailing newline — matching csync's `json.dumps(m, indent=2, sort_keys=True)
// + "\n"` so a value written by either implementation round-trips as "no change" through the
// other. Whole numbers emit without a ".0" (encoding/json renders the float64 2 as "2"),
// matching Python.
func marshalSettings(m map[string]any) ([]byte, error) {
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}
