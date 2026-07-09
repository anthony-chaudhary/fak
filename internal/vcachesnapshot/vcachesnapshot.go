// Package vcachesnapshot persists the gateway's observed per-turn provider-cache window
// to a small JSONL file at a well-known per-user path, so a SEPARATE `fak vcache score`
// process can read the REALIZED cache window a finished `fak guard`/`fak serve` session
// observed — instead of falling back to the synthetic-Zipf planned forecast.
//
// The score CLI and the gateway are different processes; the gateway holds the live
// window in memory (internal/gateway, observeVCacheTurn -> m.vcacheTurns) and exposes a
// copy via Server.VCacheTurnsSnapshot(). This package is the durable hop between them: the
// host Writes the snapshot at session exit (mirroring cachevalueledger.Append), and the
// score CLI Reads it when no explicit --telemetry file is given. Each row is one
// vcacheobserve.Turn in its JSON shape, so the reader's output folds directly through
// vcacheobserve.Observe with no schema translation.
package vcachesnapshot

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

// EnvPath is the optional override used by guard/serve writers and vcache readers.
// Set it to a file path to keep a probe/replay artifact separate from the well-known
// live snapshot, or to "off" to suppress the automatic writer.
const EnvPath = "FAK_VCACHE_SNAPSHOT"

// EnvContextPath is the optional read-side override for a separate context-plane
// witness. It lets `fak vcache score` compose the ordinary provider-cache window with
// a no-key guard replay artifact without overwriting the provider snapshot.
const EnvContextPath = "FAK_VCACHE_CONTEXT_SNAPSHOT"

// EnvWindow optionally overrides the bounded live-window size (in turns) the automatic
// guard/serve writer retains. A missing, non-numeric, or non-positive value keeps
// DefaultWindowTurns; set it larger to retain a wider window, or to a small value to
// tighten it. It does NOT re-enable persistence — EnvPath=off still suppresses the writer.
const EnvWindow = "FAK_VCACHE_SNAPSHOT_WINDOW"

// DefaultWindowTurns bounds the automatic guard/serve snapshot to the most recent N
// observed turns, so default persistence leaves a bounded live window rather than an
// unbounded per-session log (#1524). The score reads the realized recent window; turns
// older than the bound fall out of it. Chosen to comfortably hold an ordinary session's
// window while capping a pathologically long one — override with EnvWindow if an operator
// needs a wider or tighter live window.
const DefaultWindowTurns = 512

// DefaultRel is the per-user default snapshot path's basename under the config dir.
const DefaultRel = "vcache-turns.jsonl"

// DefaultContextRel is the per-user context witness snapshot basename under the
// config dir. It is intentionally separate from DefaultRel so a context replay never
// clobbers the live provider telemetry window.
const DefaultContextRel = "vcache-context-turns.jsonl"

// DefaultPath resolves the well-known snapshot path: <UserConfigDir>/fak/vcache-turns.jsonl,
// falling back to .fak/vcache-turns.jsonl when no user config dir is available — the same
// "config dir, else .fak" convention guardDefaultAuditPath uses, so the snapshot lives
// beside the decision journal and the cache-value ledger.
func DefaultPath() string {
	return defaultSnapshotPath(DefaultRel)
}

// DefaultContextPath resolves the well-known context witness snapshot path. It follows
// the same config-dir convention as DefaultPath, but uses a separate basename so the
// provider-cache and context planes can be witnessed independently and composed later.
func DefaultContextPath() string {
	return defaultSnapshotPath(DefaultContextRel)
}

func defaultSnapshotPath(rel string) string {
	dir := strings.TrimSpace(userConfigDir())
	if dir != "" {
		return filepath.Join(dir, "fak", rel)
	}
	return filepath.Join(".fak", rel)
}

func userConfigDir() string {
	dir, err := os.UserConfigDir()
	if err != nil {
		return ""
	}
	return dir
}

// ConfiguredPath resolves the automatic guard/serve snapshot target. It mirrors the
// reader-side FAK_VCACHE_SNAPSHOT override used by `fak vcache score/status`, while
// keeping DefaultPath as the stable well-known fallback for callers that explicitly ask
// for "default".
func ConfiguredPath() (string, bool) {
	path := strings.TrimSpace(os.Getenv(EnvPath))
	if path == "" {
		return DefaultPath(), true
	}
	if strings.EqualFold(path, "off") {
		return "", false
	}
	return path, true
}

// WindowTurns resolves the configured bounded live-window size, honoring EnvWindow and
// falling back to DefaultWindowTurns. A non-numeric or non-positive override is ignored so
// a fat-fingered env value can never turn the bound off (that path stays unbounded-unsafe).
func WindowTurns() int {
	raw := strings.TrimSpace(os.Getenv(EnvWindow))
	if raw == "" {
		return DefaultWindowTurns
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return DefaultWindowTurns
	}
	return n
}

// boundWindow returns the most recent n turns — the bounded live window. A non-positive n
// or a slice already within the bound is returned unchanged; otherwise it keeps the tail,
// because the score wants the REALIZED recent window, not the session's cold prefix.
func boundWindow(turns []vcacheobserve.Turn, n int) []vcacheobserve.Turn {
	if n <= 0 || len(turns) <= n {
		return turns
	}
	return turns[len(turns)-n:]
}

// WriteConfigured writes the automatic guard/serve snapshot to ConfiguredPath, bounded to
// the most recent WindowTurns turns so a finished session leaves a BOUNDED replayable live
// window with no operator flag (#1524). The returned bool is false only when
// FAK_VCACHE_SNAPSHOT=off disabled the writer.
func WriteConfigured(turns []vcacheobserve.Turn) (string, bool, error) {
	path, ok := ConfiguredPath()
	if !ok {
		return "", false, nil
	}
	return path, true, Write(path, boundWindow(turns, WindowTurns()))
}

// Write replaces the snapshot at path with one JSONL row per turn (truncating any prior
// snapshot — the score reads the MOST RECENT session's window, not an ever-growing log).
// A nil/empty turns slice writes an empty file, which Read treats as "no observed window"
// so the score correctly falls open to the planned forecast. Creates parent dirs as needed.
func Write(path string, turns []vcacheobserve.Turn) error {
	if strings.TrimSpace(path) == "" {
		path = DefaultPath()
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	for i := range turns {
		if err := enc.Encode(turns[i]); err != nil {
			return err
		}
	}
	if err := w.Flush(); err != nil {
		return err
	}
	return f.Sync()
}

// Read loads the turns from the snapshot at path. A missing file is NOT an error — it
// returns (nil, false, nil), the "no observed window" signal the score uses to fall open
// to the planned forecast. A malformed line is skipped rather than failing the whole read,
// so a partially-written snapshot still yields the turns it can. ok is true only when at
// least one turn parsed.
func Read(path string) (turns []vcacheobserve.Turn, ok bool, err error) {
	if strings.TrimSpace(path) == "" {
		path = DefaultPath()
	}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var t vcacheobserve.Turn
		if json.Unmarshal([]byte(line), &t) != nil {
			continue // skip a malformed row rather than fail the read
		}
		turns = append(turns, t)
	}
	if err := sc.Err(); err != nil {
		return turns, len(turns) > 0, err
	}
	return turns, len(turns) > 0, nil
}
