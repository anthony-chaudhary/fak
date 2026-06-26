package secretload

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// DotEnv is an OPT-IN SecretSource backed by a .env-style file (KEY=value lines). It is
// off by default — a loader gets a DotEnv only when a caller explicitly wires one, since
// silently reading a .env surprises operators and can shadow a real env value. It parses
// the common .env dialect: blank lines and # comments are ignored, an optional leading
// `export ` is stripped, and single/double-quoted values are unquoted.
//
// expiresAt preflight: a reserved key (ExpiresAtKey) whose RFC3339 value, when present,
// declares when the file's secrets go stale. Preflight(now) returns an error once that
// instant has passed, so a rotated-but-not-refreshed .env fails LOUDLY at startup rather
// than authenticating with a dead credential. The expiry key is consumed by Preflight and
// is never returned from Lookup.
type DotEnv struct {
	name      string
	vals      map[string]string
	expiresAt time.Time // zero = no declared expiry
}

// ExpiresAtKey is the reserved .env key carrying an RFC3339 expiry for the file's secrets.
const ExpiresAtKey = "FAK_SECRETS_EXPIRES_AT"

// LoadDotEnv parses path as a .env file. A missing file is NOT an error — it returns a
// DotEnv that supplies nothing (so wiring an optional .env is safe whether or not the file
// is present); the bool reports whether the file existed.
func LoadDotEnv(path string) (d *DotEnv, existed bool, err error) {
	raw, rerr := os.ReadFile(path)
	if os.IsNotExist(rerr) {
		return &DotEnv{name: "dotenv:" + filepath.Base(path), vals: map[string]string{}}, false, nil
	}
	if rerr != nil {
		return nil, false, fmt.Errorf("secretload: read %s: %w", path, rerr)
	}
	d, err = parseDotEnv(filepath.Base(path), raw)
	if err != nil {
		return nil, true, err
	}
	return d, true, nil
}

func parseDotEnv(name string, raw []byte) (*DotEnv, error) {
	d := &DotEnv{name: "dotenv:" + name, vals: map[string]string{}}
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		t := strings.TrimSpace(sc.Text())
		if t == "" || strings.HasPrefix(t, "#") {
			continue
		}
		t = strings.TrimPrefix(t, "export ")
		eq := strings.IndexByte(t, '=')
		if eq < 0 {
			return nil, fmt.Errorf("secretload: %s:%d: not KEY=value", name, line)
		}
		key := strings.TrimSpace(t[:eq])
		if key == "" {
			return nil, fmt.Errorf("secretload: %s:%d: empty key", name, line)
		}
		val := unquote(strings.TrimSpace(t[eq+1:]))
		if key == ExpiresAtKey {
			ts, err := time.Parse(time.RFC3339, val)
			if err != nil {
				return nil, fmt.Errorf("secretload: %s:%d: %s must be RFC3339: %w", name, line, ExpiresAtKey, err)
			}
			d.expiresAt = ts
			continue
		}
		d.vals[key] = val
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("secretload: scan %s: %w", name, err)
	}
	return d, nil
}

// unquote strips one matching pair of surrounding single or double quotes.
func unquote(v string) string {
	if len(v) >= 2 {
		if (v[0] == '"' && v[len(v)-1] == '"') || (v[0] == '\'' && v[len(v)-1] == '\'') {
			return v[1 : len(v)-1]
		}
	}
	return v
}

// Name identifies the source for diagnostics; it includes the file's base name.
func (d *DotEnv) Name() string { return d.name }

// Lookup returns the parsed value for key, reporting a hit only for a non-empty value. The
// reserved ExpiresAtKey is never returned (it is preflight metadata, not a secret).
func (d *DotEnv) Lookup(key string) (string, bool) {
	if key == ExpiresAtKey {
		return "", false
	}
	v, ok := d.vals[key]
	if !ok || v == "" {
		return "", false
	}
	return v, true
}

// ExpiresAt reports the file's declared expiry and whether one was declared.
func (d *DotEnv) ExpiresAt() (time.Time, bool) {
	return d.expiresAt, !d.expiresAt.IsZero()
}

// Preflight returns an error if the file declared an expiry that now has passed. A file
// with no declared expiry is always ok.
func (d *DotEnv) Preflight(now time.Time) error {
	if d.expiresAt.IsZero() {
		return nil
	}
	if now.After(d.expiresAt) {
		return fmt.Errorf("secretload: %s expired at %s (now %s)",
			d.name, d.expiresAt.Format(time.RFC3339), now.Format(time.RFC3339))
	}
	return nil
}

// DiscoverDotEnv loads the conventional .env stack under dir in precedence order (highest
// first): .env.<profile>.local, .env.local, .env.<profile>, .env. It returns one DotEnv
// per file that EXISTS, in that order, so the caller can AddSource them as priority drops.
// Missing files are skipped. profile may be "" (no environment profile), which drops the
// profile-scoped names.
func DiscoverDotEnv(dir, profile string) ([]*DotEnv, error) {
	var names []string
	if profile != "" {
		names = append(names, ".env."+profile+".local")
	}
	names = append(names, ".env.local")
	if profile != "" {
		names = append(names, ".env."+profile)
	}
	names = append(names, ".env")

	var out []*DotEnv
	for _, n := range names {
		d, existed, err := LoadDotEnv(filepath.Join(dir, n))
		if err != nil {
			return nil, err
		}
		if existed {
			out = append(out, d)
		}
	}
	return out, nil
}
