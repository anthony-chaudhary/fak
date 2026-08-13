package portability

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

const Schema = "fak.portability/v1"

var known = map[string]bool{"skill": true, "workflow": true, "policy": true}

type Object struct {
	ID      string          `json:"id"`
	Kind    string          `json:"kind"`
	Name    string          `json:"name"`
	Payload json.RawMessage `json:"payload"`
	Digest  string          `json:"digest"`
	Active  bool            `json:"active"`
	Reason  string          `json:"reason,omitempty"`
}
type Package struct {
	Schema  string   `json:"schema"`
	ID      string   `json:"id"`
	Objects []Object `json:"objects"`
	Digest  string   `json:"digest"`
}
type Receipt struct {
	Schema    string `json:"schema"`
	ID        string `json:"id"`
	Operation string `json:"operation"`
	PackageID string `json:"package_id,omitempty"`
	From      string `json:"from,omitempty"`
	To        string `json:"to,omitempty"`
	Status    string `json:"status"`
	At        string `json:"at"`
	Detail    string `json:"detail,omitempty"`
}
type Preview struct {
	Objects  []Object `json:"objects"`
	Rejected []string `json:"rejected,omitempty"`
}

type Store struct {
	Home string
	Now  func() time.Time
}

func New(home string) Store { return Store{Home: home, Now: time.Now} }
func (s Store) now() time.Time {
	if s.Now != nil {
		return s.Now().UTC()
	}
	return time.Now().UTC()
}

func (s Store) Discover(selectors []string) (Preview, error) {
	out := Preview{}
	root := filepath.Join(s.Home, "managed")
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(d.Name(), ".json") {
			return nil
		}
		rel, e := filepath.Rel(root, path)
		if e != nil {
			return e
		}
		bits := strings.Split(filepath.ToSlash(rel), "/")
		if len(bits) != 2 {
			return nil
		}
		kind := singular(bits[0])
		name := strings.TrimSuffix(bits[1], ".json")
		if !selected(kind, name, selectors) {
			return nil
		}
		b, e := os.ReadFile(path)
		if e != nil {
			return e
		}
		var v any
		if json.Unmarshal(b, &v) != nil {
			out.Rejected = append(out.Rejected, fmt.Sprintf("%s: invalid JSON", rel))
			return nil
		}
		if why := unsafe(v); why != "" {
			out.Rejected = append(out.Rejected, fmt.Sprintf("%s: %s", rel, why))
			return nil
		}
		canon, _ := json.Marshal(v)
		sum := sha256.Sum256(canon)
		o := Object{ID: kind + ":" + name, Kind: kind, Name: name, Payload: canon, Digest: "sha256:" + hex.EncodeToString(sum[:]), Active: known[kind]}
		if !known[kind] {
			o.Reason = "unknown object type"
		}
		out.Objects = append(out.Objects, o)
		return nil
	})
	if errors.Is(err, os.ErrNotExist) {
		return out, nil
	}
	sort.Slice(out.Objects, func(i, j int) bool { return out.Objects[i].ID < out.Objects[j].ID })
	sort.Strings(out.Rejected)
	return out, err
}
func selected(k, n string, ss []string) bool {
	if len(ss) == 0 {
		return true
	}
	for _, s := range ss {
		if s == k || s == k+":"+n {
			return true
		}
	}
	return false
}

var secretKey = regexp.MustCompile(`(?i)(token|secret|password|credential|api[_-]?key|private[_-]?key|conversation[_-]?history|chat[_-]?history|trajectory|transcript)`)
var tokenValue = regexp.MustCompile(`(?i)(bearer\s+\S+|gh[pousr]_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9]{8,})`)
var privateHost = regexp.MustCompile(`(?i)(^|[^a-z0-9])([a-z0-9-]+\.(internal|local|lan)|localhost)([^a-z0-9]|$)`)
var winAbs = regexp.MustCompile(`^[A-Za-z]:[\\/]`)

func unsafe(v any) string {
	switch x := v.(type) {
	case map[string]any:
		for k, v := range x {
			if secretKey.MatchString(k) {
				return "credential-bearing field " + k
			}
			if r := unsafe(v); r != "" {
				return r
			}
		}
	case []any:
		for _, v := range x {
			if r := unsafe(v); r != "" {
				return r
			}
		}
	case string:
		if tokenValue.MatchString(x) {
			return "token-like value"
		}
		if privateHost.MatchString(x) {
			return "private hostname"
		}
		if filepath.IsAbs(x) || winAbs.MatchString(x) {
			return "absolute host path"
		}
	}
	return ""
}
func packageDigest(p Package) string {
	q := struct {
		Schema  string   `json:"schema"`
		Objects []Object `json:"objects"`
	}{p.Schema, p.Objects}
	b, _ := json.Marshal(q)
	h := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(h[:])
}
func (s Store) Export(out string, selectors []string, commit bool) (Package, Receipt, error) {
	pview, e := s.Discover(selectors)
	if e != nil {
		return Package{}, Receipt{}, e
	}
	if len(pview.Rejected) > 0 {
		return Package{}, Receipt{}, fmt.Errorf("export refused: %s", strings.Join(pview.Rejected, "; "))
	}
	if len(pview.Objects) < 1 {
		return Package{}, Receipt{}, errors.New("export refused: no selected managed objects")
	}
	p := Package{Schema: Schema, Objects: pview.Objects}
	p.Digest = packageDigest(p)
	p.ID = "pkg-" + strings.TrimPrefix(p.Digest, "sha256:")[:16]
	r := s.receipt("export", p.ID, "", "", map[bool]string{true: "committed", false: "preview"}[commit], "")
	if commit {
		b, _ := json.MarshalIndent(p, "", "  ")
		if e = atomicWrite(out, b); e != nil {
			return Package{}, Receipt{}, e
		}
		if e := s.writeReceipt(r); e != nil {
			return Package{}, Receipt{}, e
		}
	}
	return p, r, nil
}
func ReadPackage(path string) (Package, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return Package{}, e
	}
	var p Package
	if e = json.Unmarshal(b, &p); e != nil {
		return p, fmt.Errorf("invalid package: %w", e)
	}
	if p.Schema != Schema {
		return p, fmt.Errorf("incompatible package schema %q", p.Schema)
	}
	if p.Digest != packageDigest(p) {
		return p, errors.New("invalid package digest")
	}
	for i := range p.Objects {
		var v any
		if json.Unmarshal(p.Objects[i].Payload, &v) != nil {
			return p, fmt.Errorf("invalid object %s", p.Objects[i].ID)
		}
		if why := unsafe(v); why != "" {
			return p, fmt.Errorf("unsafe object %s: %s", p.Objects[i].ID, why)
		}
		canon, _ := json.Marshal(v)
		sum := sha256.Sum256(canon)
		if p.Objects[i].Digest != "sha256:"+hex.EncodeToString(sum[:]) {
			return p, fmt.Errorf("invalid object digest %s", p.Objects[i].ID)
		}
	}
	return p, nil
}
func (s Store) Apply(path string, commit bool, interruptAfter int) (Receipt, error) {
	p, e := ReadPackage(path)
	if e != nil {
		return Receipt{}, e
	}
	target := filepath.Join(s.Home, "contexts", p.ID)
	if b, e := os.ReadFile(filepath.Join(target, "package.json")); e == nil {
		var old Package
		if json.Unmarshal(b, &old) == nil && old.Digest == p.Digest {
			return s.receiptDone("apply", p.ID, "", "", commit, "already applied")
		}
	}
	r := s.receipt("apply", p.ID, "", p.ID, map[bool]string{true: "committed", false: "preview"}[commit], "")
	if !commit {
		return r, nil
	}
	stage := target + ".staging"
	os.RemoveAll(stage)
	if e = os.MkdirAll(filepath.Join(stage, "objects"), 0700); e != nil {
		return r, e
	}
	for i, o := range p.Objects {
		if interruptAfter > 0 && i >= interruptAfter {
			os.RemoveAll(stage)
			return r, errors.New("apply interrupted; prior context remains active; retry apply")
		}
		o.Active = known[o.Kind]
		if !o.Active {
			o.Reason = "unknown or incompatible object type"
		}
		b, _ := json.MarshalIndent(o, "", "  ")
		if e = atomicWrite(filepath.Join(stage, "objects", safeName(o.ID)+".json"), b); e != nil {
			os.RemoveAll(stage)
			return r, e
		}
	}
	b, _ := json.MarshalIndent(p, "", "  ")
	if e = atomicWrite(filepath.Join(stage, "package.json"), b); e != nil {
		os.RemoveAll(stage)
		return r, e
	}
	os.RemoveAll(target)
	if e = os.Rename(stage, target); e != nil {
		return r, e
	}
	if e := s.writeReceipt(r); e != nil {
		return r, e
	}
	return r, nil
}
func (s Store) Switch(id string, commit bool) (Receipt, error) {
	if _, e := os.Stat(filepath.Join(s.Home, "contexts", id, "package.json")); e != nil {
		return Receipt{}, fmt.Errorf("context %s is not applied", id)
	}
	from, _ := s.Active()
	r := s.receipt("switch", "", from, id, map[bool]string{true: "committed", false: "preview"}[commit], "")
	if commit {
		if e := atomicWrite(filepath.Join(s.Home, "active"), []byte(id+"\n")); e != nil {
			return r, e
		}
		if e := s.writeReceipt(r); e != nil {
			return r, e
		}
	}
	return r, nil
}
func (s Store) Active() (string, error) {
	b, e := os.ReadFile(filepath.Join(s.Home, "active"))
	if errors.Is(e, os.ErrNotExist) {
		return "", nil
	}
	return strings.TrimSpace(string(b)), e
}
func (s Store) Readback() (map[string]string, error) {
	id, e := s.Active()
	if e != nil || id == "" {
		return nil, errors.New("no active context")
	}
	files, e := os.ReadDir(filepath.Join(s.Home, "contexts", id, "objects"))
	if e != nil {
		return nil, e
	}
	out := map[string]string{}
	for _, f := range files {
		b, _ := os.ReadFile(filepath.Join(s.Home, "contexts", id, "objects", f.Name()))
		var o Object
		if json.Unmarshal(b, &o) == nil && o.Active {
			var v map[string]any
			if json.Unmarshal(o.Payload, &v) == nil {
				if x, ok := v["behavior"].(string); ok {
					out[o.Kind+":"+o.Name] = x
				}
			}
		}
	}
	return out, nil
}
func (s Store) Rollback(receiptID string, commit bool) (Receipt, error) {
	b, e := os.ReadFile(filepath.Join(s.Home, "receipts", receiptID+".json"))
	if e != nil {
		return Receipt{}, e
	}
	var prior Receipt
	if e = json.Unmarshal(b, &prior); e != nil {
		return Receipt{}, e
	}
	if prior.Operation != "switch" || prior.Status != "committed" {
		return Receipt{}, errors.New("receipt is not a committed switch")
	}
	cur, _ := s.Active()
	r := s.receipt("rollback", "", cur, prior.From, map[bool]string{true: "committed", false: "preview"}[commit], "reverses "+receiptID)
	if commit {
		if prior.From == "" {
			os.Remove(filepath.Join(s.Home, "active"))
		} else if e = atomicWrite(filepath.Join(s.Home, "active"), []byte(prior.From+"\n")); e != nil {
			return r, e
		}
		if e := s.writeReceipt(r); e != nil {
			return r, e
		}
	}
	return r, nil
}
func (s Store) receipt(op, pkg, from, to, status, detail string) Receipt {
	seed := fmt.Sprintf("%s|%s|%s|%s|%s", op, pkg, from, to, s.now().Format(time.RFC3339Nano))
	h := sha256.Sum256([]byte(seed))
	return Receipt{Schema: "fak.portability.receipt/v1", ID: "rcpt-" + hex.EncodeToString(h[:8]), Operation: op, PackageID: pkg, From: from, To: to, Status: status, At: s.now().Format(time.RFC3339Nano), Detail: detail}
}
func (s Store) receiptDone(op, pkg, from, to string, commit bool, detail string) (Receipt, error) {
	r := s.receipt(op, pkg, from, to, map[bool]string{true: "committed", false: "preview"}[commit], detail)
	if commit {
		return r, s.writeReceipt(r)
	}
	return r, nil
}
func (s Store) writeReceipt(r Receipt) error {
	b, _ := json.MarshalIndent(r, "", "  ")
	return atomicWrite(filepath.Join(s.Home, "receipts", r.ID+".json"), b)
}
func atomicWrite(path string, b []byte) error {
	if e := os.MkdirAll(filepath.Dir(path), 0700); e != nil {
		return e
	}
	tmp := path + ".tmp"
	if e := os.WriteFile(tmp, b, 0600); e != nil {
		return e
	}
	return os.Rename(tmp, path)
}
func safeName(s string) string { return strings.NewReplacer(":", "--", "/", "-", "\\", "-").Replace(s) }

func singular(s string) string {
	if s == "policies" {
		return "policy"
	}
	return strings.TrimSuffix(s, "s")
}
