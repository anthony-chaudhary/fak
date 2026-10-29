package treedoctor

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
)

// CheckGoCachePlan consumes the exact data-only plan, not reconstructed shell
// variables. It checks GOCACHE against all named filesystem budget domains.
// The caller must still bind the eventual child argv/environment to this plan.
func CheckGoCachePlan(path string) (GoCachePlacement, error) {
	var zero GoCachePlacement
	f, err := openPlacementPlan(path)
	if err != nil {
		return zero, err
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, 4<<20+1))
	if err != nil {
		return zero, err
	}
	if len(raw) > 4<<20 {
		return zero, fmt.Errorf("plan exceeds 4 MiB")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	if err := placementUniqueKeys(d, 0); err != nil {
		return zero, fmt.Errorf("plan JSON: %w", err)
	}
	if _, err := d.Token(); err != io.EOF {
		return zero, fmt.Errorf("plan JSON has trailing data")
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil || top == nil {
		return zero, fmt.Errorf("plan must be an object")
	}
	var env map[string]json.RawMessage
	if err := json.Unmarshal(top["environment"], &env); err != nil || env == nil {
		return zero, fmt.Errorf("plan requires environment object")
	}
	var cache string
	if err := json.Unmarshal(env["GOCACHE"], &cache); err != nil || cache == "" {
		return zero, fmt.Errorf("plan requires explicit GOCACHE")
	}
	if prog, ok := env["GOCACHEPROG"]; ok {
		var v string
		if bytes.Equal(bytes.TrimSpace(prog), []byte("null")) {
			return zero, fmt.Errorf("GOCACHEPROG must be absent or empty")
		}
		if err := json.Unmarshal(prog, &v); err != nil || v != "" {
			return zero, fmt.Errorf("GOCACHEPROG must be absent or empty")
		}
	}
	var specs []map[string]json.RawMessage
	if err := json.Unmarshal(top["filesystems"], &specs); err != nil || len(specs) == 0 || len(specs) > 8 {
		return zero, fmt.Errorf("plan requires 1..8 filesystems")
	}
	names := map[string]bool{}
	matched := false
	var report GoCachePlacement
	for _, spec := range specs {
		var name, root string
		var budget *uint64
		if json.Unmarshal(spec["name"], &name) != nil || name == "" || names[name] || json.Unmarshal(spec["path"], &root) != nil || root == "" || json.Unmarshal(spec["sampled_budget_bytes"], &budget) != nil || budget == nil {
			return zero, fmt.Errorf("each filesystem requires a unique name, path, and explicit non-null sampled budget")
		}
		names[name] = true
		next, err := CheckGoCachePlacement(cache, root, *budget)
		if err != nil {
			return zero, fmt.Errorf("filesystem %s: %w", name, err)
		}
		if next.SameFilesystem {
			if matched {
				return zero, fmt.Errorf("multiple filesystem records match GOCACHE")
			}
			matched, report = true, next
			report.FilesystemName = name
		}
	}
	if !matched {
		return zero, fmt.Errorf("no planned filesystem matches GOCACHE")
	}
	sum := sha256.Sum256(raw)
	report.PlanSHA256 = hex.EncodeToString(sum[:])
	return report, nil
}

// Refuse duplicate keys anywhere, including escaped aliases, before decoding.
func placementUniqueKeys(d *json.Decoder, depth int) error {
	if depth > 64 {
		return fmt.Errorf("nesting limit exceeded")
	}
	t, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	if delim != '{' && delim != '[' {
		return fmt.Errorf("unexpected delimiter")
	}
	keys := map[string]bool{}
	for d.More() {
		if delim == '{' {
			k, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := k.(string)
			if !ok || keys[key] {
				return fmt.Errorf("duplicate or invalid key %v", k)
			}
			keys[key] = true
		}
		if err := placementUniqueKeys(d, depth+1); err != nil {
			return err
		}
	}
	_, err = d.Token()
	return err
}
