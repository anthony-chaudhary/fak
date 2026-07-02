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
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

// EnvPath is the optional override used by guard/serve writers and vcache readers.
// Set it to a file path to keep a probe/replay artifact separate from the well-known
// live snapshot, or to "off" to suppress the automatic writer.
const EnvPath = "FAK_VCACHE_SNAPSHOT"

// DefaultRel is the per-user default snapshot path's basename under the config dir.
const DefaultRel = "vcache-turns.jsonl"

// DefaultPath resolves the well-known snapshot path: <UserConfigDir>/fak/vcache-turns.jsonl,
// falling back to .fak/vcache-turns.jsonl when no user config dir is available — the same
// "config dir, else .fak" convention guardDefaultAuditPath uses, so the snapshot lives
// beside the decision journal and the cache-value ledger.
func DefaultPath() string {
	if dir, err := os.UserConfigDir(); err == nil && strings.TrimSpace(dir) != "" {
		return filepath.Join(dir, "fak", DefaultRel)
	}
	return filepath.Join(".fak", DefaultRel)
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

// WriteConfigured writes the automatic guard/serve snapshot to ConfiguredPath. The
// returned bool is false only when FAK_VCACHE_SNAPSHOT=off disabled the writer.
func WriteConfigured(turns []vcacheobserve.Turn) (string, bool, error) {
	path, ok := ConfiguredPath()
	if !ok {
		return "", false, nil
	}
	return path, true, Write(path, turns)
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
