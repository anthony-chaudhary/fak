package portability

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const MergeSchema = "fak.portability.merge/v1"

type ConflictKind string

const (
	ConflictDivergent   ConflictKind = "divergent-edit"
	ConflictDeleteEdit  ConflictKind = "delete-vs-edit"
	ConflictType        ConflictKind = "type-change"
	ConflictVersion     ConflictKind = "version-change"
	ConflictPrecedence  ConflictKind = "precedence-change"
	ConflictDependency  ConflictKind = "dependency-change"
	ConflictBaseMissing ConflictKind = "common-base-missing"
	ConflictSchema      ConflictKind = "schema-skew"
)

type MergeConflict struct {
	Kind        ConflictKind `json:"kind"`
	ObjectID    string       `json:"object_id,omitempty"`
	Path        string       `json:"path,omitempty"`
	Explanation string       `json:"explanation"`
}
type MergeStep struct {
	ObjectID string   `json:"object_id"`
	Action   string   `json:"action"`
	Source   string   `json:"source"`
	Paths    []string `json:"paths,omitempty"`
}
type mergeSnapshot struct {
	Path    string `json:"path"`
	Existed bool   `json:"existed"`
	Bytes   []byte `json:"bytes,omitempty"`
}
type mergeTransaction struct {
	Schema    string          `json:"schema"`
	PlanID    string          `json:"plan_id"`
	Receipt   string          `json:"receipt"`
	Snapshots []mergeSnapshot `json:"snapshots"`
}
type MergePlan struct {
	Schema       string          `json:"schema"`
	ID           string          `json:"id"`
	BaseDigest   string          `json:"base_digest,omitempty"`
	LocalDigest  string          `json:"local_digest"`
	RemoteDigest string          `json:"remote_digest"`
	Channel      Channel         `json:"channel"`
	Steps        []MergeStep     `json:"steps"`
	Conflicts    []MergeConflict `json:"conflicts,omitempty"`
	Result       Package         `json:"result"`
	Egress       []EgressPreview `json:"egress"`
	Digest       string          `json:"digest"`
}

func PreviewMerge(base *Package, local, remote Package, channel Channel) (MergePlan, error) {
	p := MergePlan{Schema: MergeSchema, LocalDigest: local.Digest, RemoteDigest: remote.Digest, Channel: channel}
	if base == nil {
		p.Conflicts = append(p.Conflicts, MergeConflict{Kind: ConflictBaseMissing, Explanation: "a common-base package is required; no ancestry guess or last-writer-wins was attempted"})
	} else {
		p.BaseDigest = base.Digest
	}
	if local.Schema != Schema || remote.Schema != Schema || (base != nil && base.Schema != Schema) {
		p.Conflicts = append(p.Conflicts, MergeConflict{Kind: ConflictSchema, Explanation: "package schema differs from fak.portability/v1; incompatible objects remain inactive and byte-preserved"})
	}
	bm, lm, rm := objectMap(base), objectMap(&local), objectMap(&remote)
	ids := map[string]bool{}
	for id := range bm {
		ids[id] = true
	}
	for id := range lm {
		ids[id] = true
	}
	for id := range rm {
		ids[id] = true
	}
	ordered := make([]string, 0, len(ids))
	for id := range ids {
		ordered = append(ordered, id)
	}
	sort.Strings(ordered)
	out := make([]Object, 0, len(ordered))
	for _, id := range ordered {
		b, bok := bm[id]
		l, lok := lm[id]
		r, rok := rm[id]
		obj, keep, step, cs := mergeObject(id, b, bok, l, lok, r, rok)
		p.Steps = append(p.Steps, step)
		p.Conflicts = append(p.Conflicts, cs...)
		if keep {
			out = append(out, obj)
		}
	}
	sort.Slice(p.Conflicts, func(i, j int) bool {
		a, b := p.Conflicts[i], p.Conflicts[j]
		if a.ObjectID != b.ObjectID {
			return a.ObjectID < b.ObjectID
		}
		if a.Path != b.Path {
			return a.Path < b.Path
		}
		return a.Kind < b.Kind
	})
	p.Result = Package{Schema: Schema, Objects: out}
	// The egress policy is run over every candidate before the merged export receives identity.
	for i := range p.Result.Objects {
		e, err := PreviewEgress(channel, p.Result.Objects[i].Payload)
		if err != nil {
			if !p.Result.Objects[i].Active && channel == ChannelMachineLocal {
				e = EgressPreview{Channel: channel, Allowed: true, Payload: append(json.RawMessage(nil), p.Result.Objects[i].Payload...)}
			} else {
				return p, err
			}
		}
		p.Egress = append(p.Egress, e)
		if !e.Allowed {
			p.Conflicts = append(p.Conflicts, MergeConflict{Kind: ConflictDivergent, ObjectID: p.Result.Objects[i].ID, Path: "$", Explanation: "egress policy denied merged object before export write"})
			p.Result.Objects[i].Active = false
		}
		p.Result.Objects[i].Payload = append(json.RawMessage(nil), e.Payload...)
		p.Result.Objects[i].Digest = payloadDigest(e.Payload)
	}
	if len(p.Conflicts) == 0 {
		p.Result.Digest = packageDigest(p.Result)
		p.Result.ID = "pkg-" + strings.TrimPrefix(p.Result.Digest, "sha256:")[:16]
	}
	p.Digest = mergePlanDigest(p)
	p.ID = "merge-" + strings.TrimPrefix(p.Digest, "sha256:")[:16]
	return p, nil
}
func objectMap(p *Package) map[string]Object {
	m := map[string]Object{}
	if p != nil {
		for _, o := range p.Objects {
			m[o.ID] = o
		}
	}
	return m
}
func sameObject(a, b Object) bool {
	return a.ID == b.ID && a.Kind == b.Kind && a.Name == b.Name && a.Digest == b.Digest && a.Active == b.Active && a.Reason == b.Reason && bytes.Equal(a.Payload, b.Payload)
}
func mergeObject(id string, b Object, bok bool, l Object, lok bool, r Object, rok bool) (Object, bool, MergeStep, []MergeConflict) {
	step := MergeStep{ObjectID: id}
	if lok && rok && sameObject(l, r) {
		step.Action = "keep"
		step.Source = "both"
		return l, true, step, nil
	}
	if bok {
		lc := !lok || !sameObject(l, b)
		rc := !rok || !sameObject(r, b)
		if !lc {
			if !rok {
				step.Action = "delete"
				step.Source = "remote"
				return Object{}, false, step, nil
			}
			step.Action = "take"
			step.Source = "remote"
			return r, true, step, nil
		}
		if !rc {
			if !lok {
				step.Action = "delete"
				step.Source = "local"
				return Object{}, false, step, nil
			}
			step.Action = "take"
			step.Source = "local"
			return l, true, step, nil
		}
		if !lok || !rok {
			step.Action = "conflict"
			return b, true, step, []MergeConflict{{Kind: ConflictDeleteEdit, ObjectID: id, Path: "$", Explanation: "one home deleted the object while the other edited it; base is retained inactive pending decision"}}
		}
	} else {
		if lok && !rok {
			step.Action = "add"
			step.Source = "local"
			return l, true, step, nil
		}
		if rok && !lok {
			step.Action = "add"
			step.Source = "remote"
			return r, true, step, nil
		}
	}
	if l.Kind != r.Kind {
		step.Action = "conflict"
		return inactiveRaw(l), true, step, []MergeConflict{{Kind: ConflictType, ObjectID: id, Path: "$/kind", Explanation: "homes changed object type differently; local bytes are retained inactive pending decision"}}
	}
	var bv, lv, rv any
	if bok {
		if json.Unmarshal(b.Payload, &bv) != nil {
			return binaryConflict(id, l, ConflictDivergent, "base object is opaque; divergent bytes cannot be semantically merged")
		}
	}
	if json.Unmarshal(l.Payload, &lv) != nil || json.Unmarshal(r.Payload, &rv) != nil {
		return binaryConflict(id, l, ConflictDivergent, "opaque or binary object changed on both homes; bytes remain inactive and no last-writer-wins was attempted")
	}
	merged, paths, cs := mergeValue(id, "$", bv, lv, rv, bok)
	if len(cs) > 0 {
		step.Action = "conflict"
		return inactiveRaw(l), true, step, cs
	}
	raw, _ := json.Marshal(merged)
	l.Payload = raw
	l.Digest = payloadDigest(raw)
	step.Action = "merge"
	step.Source = "three-way"
	step.Paths = paths
	return l, true, step, nil
}
func binaryConflict(id string, l Object, k ConflictKind, msg string) (Object, bool, MergeStep, []MergeConflict) {
	return inactiveRaw(l), true, MergeStep{ObjectID: id, Action: "conflict"}, []MergeConflict{{Kind: k, ObjectID: id, Path: "$", Explanation: msg}}
}
func inactiveRaw(o Object) Object { o.Active = false; return o }
func mergeValue(id, path string, b, l, r any, hasBase bool) (any, []string, []MergeConflict) {
	if equal(l, r) {
		return l, nil, nil
	}
	if hasBase && equal(l, b) {
		return r, []string{path}, nil
	}
	if hasBase && equal(r, b) {
		return l, []string{path}, nil
	}
	lm, lok := l.(map[string]any)
	rm, rok := r.(map[string]any)
	bm, _ := b.(map[string]any)
	if lok && rok {
		out := map[string]any{}
		keys := map[string]bool{}
		for k := range bm {
			keys[k] = true
		}
		for k := range lm {
			keys[k] = true
		}
		for k := range rm {
			keys[k] = true
		}
		ks := make([]string, 0, len(keys))
		for k := range keys {
			ks = append(ks, k)
		}
		sort.Strings(ks)
		var paths []string
		var cs []MergeConflict
		for _, k := range ks {
			bv, bok := bm[k]
			lv, lok := lm[k]
			rv, rok := rm[k]
			kp := path + "/" + escapePtr(k)
			if lok && rok {
				v, ps, x := mergeValue(id, kp, bv, lv, rv, bok)
				if len(x) == 0 {
					out[k] = v
				}
				paths = append(paths, ps...)
				cs = append(cs, x...)
				continue
			}
			if !lok && !rok {
				continue
			}
			if bok {
				if !lok && equal(rv, bv) {
					continue
				}
				if !rok && equal(lv, bv) {
					continue
				}
				cs = append(cs, MergeConflict{Kind: ConflictDeleteEdit, ObjectID: id, Path: kp, Explanation: "one home deleted this field while the other edited it"})
				continue
			}
			if lok {
				out[k] = lv
			} else {
				out[k] = rv
			}
			paths = append(paths, kp)
		}
		return out, paths, cs
	}
	kind := ConflictDivergent
	low := strings.ToLower(path)
	if strings.Contains(low, "version") {
		kind = ConflictVersion
	} else if strings.Contains(low, "precedence") || strings.Contains(low, "priority") {
		kind = ConflictPrecedence
	} else if strings.Contains(low, "depend") || strings.Contains(low, "requires") {
		kind = ConflictDependency
	} else if fmt.Sprintf("%T", l) != fmt.Sprintf("%T", r) {
		kind = ConflictType
	}
	return nil, nil, []MergeConflict{{Kind: kind, ObjectID: id, Path: path, Explanation: "both homes changed the same semantic field differently; operator choice is required"}}
}
func equal(a, b any) bool { x, _ := json.Marshal(a); y, _ := json.Marshal(b); return bytes.Equal(x, y) }
func escapePtr(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "~", "~0"), "/", "~1")
}
func payloadDigest(b []byte) string {
	var v any
	if json.Unmarshal(b, &v) == nil {
		b, _ = json.Marshal(v)
	}
	h := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(h[:])
}
func mergePlanDigest(p MergePlan) string {
	p.ID = ""
	p.Digest = ""
	b, _ := json.Marshal(p)
	return payloadDigest(b)
}

func WriteMergePlan(path string, p MergePlan) error {
	b, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	return atomicWrite(path, append(b, '\n'))
}
func ReadMergePlan(path string) (MergePlan, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return MergePlan{}, err
	}
	var p MergePlan
	if json.Unmarshal(b, &p) != nil {
		return p, errors.New("invalid merge plan")
	}
	want := p.Digest
	if mergePlanDigest(p) != want {
		return p, errors.New("merge plan digest mismatch")
	}
	return p, nil
}
func (s Store) CommitMerge(plan MergePlan, out string, commit bool, interruptAfter int) (Receipt, error) {
	if err := s.RecoverMerge(); err != nil {
		return Receipt{}, err
	}
	if mergePlanDigest(plan) != plan.Digest {
		return Receipt{}, errors.New("merge plan digest mismatch")
	}
	if len(plan.Conflicts) > 0 {
		return Receipt{}, fmt.Errorf("merge blocked by %d typed conflict(s)", len(plan.Conflicts))
	}
	if plan.Result.ID == "" {
		return Receipt{}, errors.New("merge result has no identity")
	}
	from, _ := s.Active()
	r := s.receipt("merge", plan.Result.ID, from, plan.Result.ID, "preview", "atomic merged export and context activation")
	if !commit {
		return r, nil
	}
	if len(plan.Egress) != len(plan.Result.Objects) {
		return r, errors.New("egress policy evidence missing at commit boundary")
	}
	for i := range plan.Result.Objects {
		e := plan.Egress[i]
		if !e.Allowed || e.Channel != plan.Channel {
			return r, errors.New("egress policy denied merge at commit boundary")
		}
	}
	b, err := json.MarshalIndent(plan.Result, "", "  ")
	if err != nil {
		return r, err
	}
	b = append(b, '\n')
	r.Status = "committed"
	rb, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return r, err
	}
	rb = append(rb, '\n')
	paths := []string{out, filepath.Join(s.Home, "packages", safeName(plan.Result.ID)+".json"), filepath.Join(s.Home, "active"), filepath.Join(s.Home, "receipts", safeName(r.ID)+".json")}
	payloads := [][]byte{b, b, []byte(plan.Result.ID + "\n"), rb}
	tx := mergeTransaction{Schema: MergeSchema, PlanID: plan.ID, Receipt: r.ID}
	for _, path := range paths {
		old, e := os.ReadFile(path)
		if e != nil && !os.IsNotExist(e) {
			return r, e
		}
		tx.Snapshots = append(tx.Snapshots, mergeSnapshot{Path: path, Existed: e == nil, Bytes: old})
	}
	txb, err := json.MarshalIndent(tx, "", "  ")
	if err != nil {
		return r, err
	}
	if err := atomicWrite(s.mergeJournalPath(), append(txb, '\n')); err != nil {
		return r, err
	}
	if interruptAfter == 1 {
		return r, errors.New("simulated interruption after merge journal")
	}
	for i, path := range paths {
		if err := atomicWrite(path, payloads[i]); err != nil {
			return r, err
		}
		if interruptAfter == i+2 {
			return r, fmt.Errorf("simulated interruption after merge write %d", i+1)
		}
	}
	if err := os.Remove(s.mergeJournalPath()); err != nil && !os.IsNotExist(err) {
		return r, err
	}
	return r, nil
}
func (s Store) mergeJournalPath() string { return filepath.Join(s.Home, "merge-transaction.json") }

// RecoverMerge restores exact pre-commit bytes after interruption. A durable
// committed receipt proves completion, so recovery only removes a stale journal.
func (s Store) RecoverMerge() error {
	b, err := os.ReadFile(s.mergeJournalPath())
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	var tx mergeTransaction
	if json.Unmarshal(b, &tx) != nil || tx.Schema != MergeSchema || tx.PlanID == "" || tx.Receipt == "" {
		return errors.New("invalid merge recovery journal")
	}
	if rb, e := os.ReadFile(filepath.Join(s.Home, "receipts", safeName(tx.Receipt)+".json")); e == nil {
		var r Receipt
		if json.Unmarshal(rb, &r) == nil && r.Status == "committed" && r.Operation == "merge" {
			return os.Remove(s.mergeJournalPath())
		}
	}
	for i := len(tx.Snapshots) - 1; i >= 0; i-- {
		snap := tx.Snapshots[i]
		if snap.Existed {
			if err := atomicWrite(snap.Path, snap.Bytes); err != nil {
				return err
			}
		} else if err := os.Remove(snap.Path); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return os.Remove(s.mergeJournalPath())
}
