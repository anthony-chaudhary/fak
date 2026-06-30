package commitintent

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	QueueSchema  = "fak.commit-intent.queue.v1"
	RecordSchema = "fak.commit-intent.submit.v1"
)

type StampKind string

const (
	StampNone    StampKind = "none"
	StampTrailer StampKind = "trailer"
	StampDirect  StampKind = "direct"
	StampRelease StampKind = "release"
)

type State string

const (
	StatePending  State = "pending"
	StateDraining State = "draining"
	StateDone     State = "done"
	StateFailed   State = "failed"
)

var (
	ErrMissingField       = errors.New("commitintent: missing field")
	ErrInvalidField       = errors.New("commitintent: invalid field")
	ErrPathDigestMismatch = errors.New("commitintent: path digest mismatch")
	ErrStaleBase          = errors.New("commitintent: stale base")
)

var (
	idRE          = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)
	shaRE         = regexp.MustCompile(`^(?i:[0-9a-f]{40}|[0-9a-f]{64})$`)
	leafRE        = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)
	trailerLeafRE = regexp.MustCompile(`(?i)\((?:refs\s+)?fak[ :]+([A-Za-z0-9][\w.\-]*)\)\s*$`)
	directLeafRE  = regexp.MustCompile(`(?i)^fak/([A-Za-z0-9][\w.\-]*):`)
	releaseRE     = regexp.MustCompile(`^v\d+\.\d+\.\d+:`)
)

type FieldError struct {
	Field  string
	Reason string
	Err    error
}

func (e FieldError) Error() string {
	if e.Reason == "" {
		return e.Field + ": " + e.Err.Error()
	}
	return e.Field + ": " + e.Err.Error() + ": " + e.Reason
}

func (e FieldError) Unwrap() error { return e.Err }

type Queue struct {
	Schema       string         `json:"schema"`
	NextSequence int64          `json:"next_sequence"`
	Records      []SubmitRecord `json:"records"`
}

type SubmitRecord struct {
	Schema      string    `json:"schema"`
	Sequence    int64     `json:"sequence"`
	SubmittedAt time.Time `json:"submitted_at"`
	State       State     `json:"state"`
	Intent      Intent    `json:"intent"`
}

type Intent struct {
	ID         string        `json:"id"`
	BaseSHA    string        `json:"base_sha"`
	Paths      []string      `json:"paths"`
	PathDigest string        `json:"path_digest"`
	Subject    string        `json:"subject"`
	Stamp      Stamp         `json:"stamp"`
	Metadata   StampMetadata `json:"metadata,omitempty"`
}

type Stamp struct {
	Kind StampKind `json:"kind"`
	Leaf string    `json:"leaf,omitempty"`
	Text string    `json:"text,omitempty"`
}

type StampMetadata struct {
	Issue      int      `json:"issue,omitempty"`
	Generation string   `json:"generation,omitempty"`
	Source     string   `json:"source,omitempty"`
	Requester  string   `json:"requester,omitempty"`
	Labels     []string `json:"labels,omitempty"`
}

type InvalidRecord struct {
	Record SubmitRecord `json:"record"`
	Error  string       `json:"error"`
}

type DrainPlan struct {
	Ready   []SubmitRecord  `json:"ready"`
	Stale   []SubmitRecord  `json:"stale"`
	Invalid []InvalidRecord `json:"invalid"`
}

func NewQueue() Queue {
	return Queue{Schema: QueueSchema, NextSequence: 1}
}

func Submit(q Queue, now time.Time, intent Intent) (Queue, SubmitRecord, error) {
	if strings.TrimSpace(q.Schema) == "" {
		q.Schema = QueueSchema
	}
	if q.Schema != QueueSchema {
		return q, SubmitRecord{}, fieldError("schema", ErrInvalidField, q.Schema)
	}
	if q.NextSequence <= 0 {
		q.NextSequence = nextSequence(q.Records)
	}
	intent, err := NormalizeIntent(intent)
	if err != nil {
		return q, SubmitRecord{}, err
	}
	rec := SubmitRecord{
		Schema:      RecordSchema,
		Sequence:    q.NextSequence,
		SubmittedAt: now.UTC(),
		State:       StatePending,
		Intent:      intent,
	}
	if err := ValidateRecord(rec); err != nil {
		return q, SubmitRecord{}, err
	}
	next := q
	next.Records = append(append([]SubmitRecord(nil), q.Records...), rec)
	next.NextSequence++
	return next, rec, nil
}

func Drain(records []SubmitRecord, currentBaseSHA string, limit int) DrainPlan {
	base, err := NormalizeSHA(currentBaseSHA)
	if err != nil {
		return DrainPlan{Invalid: invalidAll(records, "current_base_sha: "+err.Error())}
	}
	ordered := Ordered(records)
	var plan DrainPlan
	for _, rec := range ordered {
		if rec.State != "" && rec.State != StatePending {
			continue
		}
		if err := ValidateRecord(rec); err != nil {
			plan.Invalid = append(plan.Invalid, InvalidRecord{Record: rec, Error: err.Error()})
			continue
		}
		if rec.Intent.BaseSHA != base {
			plan.Stale = append(plan.Stale, rec)
			continue
		}
		if limit > 0 && len(plan.Ready) >= limit {
			continue
		}
		plan.Ready = append(plan.Ready, rec)
	}
	return plan
}

func Ordered(records []SubmitRecord) []SubmitRecord {
	out := append([]SubmitRecord(nil), records...)
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Sequence != b.Sequence {
			return a.Sequence < b.Sequence
		}
		at, bt := a.SubmittedAt.UTC(), b.SubmittedAt.UTC()
		if !at.Equal(bt) {
			if at.IsZero() {
				return false
			}
			if bt.IsZero() {
				return true
			}
			return at.Before(bt)
		}
		if a.Intent.ID != b.Intent.ID {
			return a.Intent.ID < b.Intent.ID
		}
		return a.Intent.PathDigest < b.Intent.PathDigest
	})
	return out
}

func NormalizeIntent(in Intent) (Intent, error) {
	var err error
	out := in
	if out.ID, err = NormalizeID(out.ID); err != nil {
		return Intent{}, err
	}
	if out.BaseSHA, err = NormalizeSHA(out.BaseSHA); err != nil {
		return Intent{}, err
	}
	if out.Paths, err = NormalizePaths(out.Paths); err != nil {
		return Intent{}, err
	}
	digest := PathDigest(out.Paths)
	if strings.TrimSpace(out.PathDigest) == "" {
		out.PathDigest = digest
	} else if out.PathDigest != digest {
		return Intent{}, fieldError("path_digest", ErrPathDigestMismatch, fmt.Sprintf("got %s want %s", out.PathDigest, digest))
	}
	out.Subject = strings.TrimSpace(out.Subject)
	if err := ValidateSubject(out.Subject); err != nil {
		return Intent{}, err
	}
	out.Stamp, err = NormalizeStamp(out.Stamp, out.Subject)
	if err != nil {
		return Intent{}, err
	}
	if err := ValidateMetadata(out.Metadata); err != nil {
		return Intent{}, err
	}
	return out, nil
}

func ValidateIntent(in Intent) error {
	_, err := NormalizeIntent(in)
	return err
}

func ValidateIntentForBase(in Intent, currentBaseSHA string) error {
	intent, err := NormalizeIntent(in)
	if err != nil {
		return err
	}
	base, err := NormalizeSHA(currentBaseSHA)
	if err != nil {
		return err
	}
	if intent.BaseSHA != base {
		return fieldError("base_sha", ErrStaleBase, "intent base "+intent.BaseSHA+" != current base "+base)
	}
	return nil
}

func ValidateRecord(rec SubmitRecord) error {
	if rec.Schema == "" {
		return fieldError("schema", ErrMissingField, "record schema is required")
	}
	if rec.Schema != RecordSchema {
		return fieldError("schema", ErrInvalidField, rec.Schema)
	}
	if rec.Sequence <= 0 {
		return fieldError("sequence", ErrInvalidField, "must be positive")
	}
	if rec.SubmittedAt.IsZero() {
		return fieldError("submitted_at", ErrMissingField, "submitted time is required")
	}
	switch rec.State {
	case StatePending, StateDraining, StateDone, StateFailed:
	default:
		return fieldError("state", ErrInvalidField, string(rec.State))
	}
	return ValidateIntent(rec.Intent)
}

func MarshalRecord(rec SubmitRecord) ([]byte, error) {
	if err := ValidateRecord(rec); err != nil {
		return nil, err
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

func ParseRecord(data []byte) (SubmitRecord, error) {
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return SubmitRecord{}, fieldError("record", ErrMissingField, "empty record")
	}
	var rec SubmitRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return SubmitRecord{}, err
	}
	if err := ValidateRecord(rec); err != nil {
		return SubmitRecord{}, err
	}
	rec.Intent, _ = NormalizeIntent(rec.Intent)
	return rec, nil
}

func NormalizeID(id string) (string, error) {
	id = strings.TrimSpace(id)
	if id == "" {
		return "", fieldError("id", ErrMissingField, "id is required")
	}
	if !idRE.MatchString(id) {
		return "", fieldError("id", ErrInvalidField, id)
	}
	return id, nil
}

func NormalizeSHA(sha string) (string, error) {
	sha = strings.ToLower(strings.TrimSpace(sha))
	if sha == "" {
		return "", fieldError("base_sha", ErrMissingField, "base sha is required")
	}
	if !shaRE.MatchString(sha) {
		return "", fieldError("base_sha", ErrInvalidField, sha)
	}
	return sha, nil
}

func NormalizePaths(paths []string) ([]string, error) {
	if len(paths) == 0 {
		return nil, fieldError("paths", ErrMissingField, "at least one path is required")
	}
	seen := map[string]bool{}
	out := make([]string, 0, len(paths))
	for _, raw := range paths {
		p, err := NormalizePath(raw)
		if err != nil {
			return nil, err
		}
		if !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	sort.Strings(out)
	return out, nil
}

func NormalizePath(raw string) (string, error) {
	p := strings.TrimSpace(strings.ReplaceAll(raw, "\\", "/"))
	if p == "" {
		return "", fieldError("paths", ErrMissingField, "empty path")
	}
	if strings.ContainsRune(p, '\x00') {
		return "", fieldError("paths", ErrInvalidField, "path contains NUL")
	}
	if strings.HasPrefix(p, "/") || strings.Contains(p, ":") {
		return "", fieldError("paths", ErrInvalidField, raw)
	}
	p = path.Clean(p)
	p = strings.TrimPrefix(p, "./")
	if p == "." || p == "" || p == ".." || strings.HasPrefix(p, "../") {
		return "", fieldError("paths", ErrInvalidField, raw)
	}
	if p == ".git" || strings.HasPrefix(p, ".git/") {
		return "", fieldError("paths", ErrInvalidField, raw)
	}
	return p, nil
}

func PathDigest(paths []string) string {
	normalized, err := NormalizePaths(paths)
	if err != nil {
		return ""
	}
	h := sha256.New()
	for _, p := range normalized {
		_, _ = h.Write([]byte(strconv.Itoa(len(p))))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(p))
		_, _ = h.Write([]byte{0})
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

func ValidateSubject(subject string) error {
	subject = strings.TrimSpace(subject)
	if subject == "" {
		return fieldError("subject", ErrMissingField, "subject is required")
	}
	if strings.ContainsAny(subject, "\r\n") {
		return fieldError("subject", ErrInvalidField, "subject must be one line")
	}
	if len(subject) > 240 {
		return fieldError("subject", ErrInvalidField, "subject is longer than 240 bytes")
	}
	return nil
}

func NormalizeStamp(stamp Stamp, subject string) (Stamp, error) {
	subjectStamp := StampFromSubject(subject)
	out := Stamp{
		Kind: StampKind(strings.ToLower(strings.TrimSpace(string(stamp.Kind)))),
		Leaf: strings.ToLower(strings.TrimSpace(stamp.Leaf)),
		Text: strings.TrimSpace(stamp.Text),
	}
	if out.Kind == "" || out.Kind == StampNone {
		out = subjectStamp
	}
	if err := ValidateStamp(out); err != nil {
		return Stamp{}, err
	}
	if subjectStamp.Kind != StampNone && (subjectStamp.Kind != out.Kind || subjectStamp.Leaf != out.Leaf) {
		return Stamp{}, fieldError("stamp", ErrInvalidField, "stamp metadata does not match subject")
	}
	if subjectStamp.Kind == StampNone {
		return Stamp{}, fieldError("stamp", ErrMissingField, "subject carries no fak stamp")
	}
	if out.Text == "" {
		out.Text = subjectStamp.Text
	}
	return out, nil
}

func ValidateStamp(stamp Stamp) error {
	switch stamp.Kind {
	case StampTrailer, StampDirect:
		if stamp.Leaf == "" {
			return fieldError("stamp.leaf", ErrMissingField, "leaf is required")
		}
		if !leafRE.MatchString(stamp.Leaf) {
			return fieldError("stamp.leaf", ErrInvalidField, stamp.Leaf)
		}
	case StampRelease:
		return fieldError("stamp.kind", ErrInvalidField, "release stamps are not drainable commit intents")
	default:
		return fieldError("stamp.kind", ErrMissingField, "bindable fak stamp is required")
	}
	return nil
}

func StampFromSubject(subject string) Stamp {
	subject = strings.TrimSpace(subject)
	if m := trailerLeafRE.FindStringSubmatch(subject); m != nil {
		leaf := strings.ToLower(m[1])
		return Stamp{Kind: StampTrailer, Leaf: leaf, Text: "(fak " + leaf + ")"}
	}
	if m := directLeafRE.FindStringSubmatch(subject); m != nil {
		leaf := strings.ToLower(m[1])
		return Stamp{Kind: StampDirect, Leaf: leaf, Text: "fak/" + leaf + ":"}
	}
	if releaseRE.MatchString(subject) {
		return Stamp{Kind: StampRelease}
	}
	return Stamp{Kind: StampNone}
}

func ValidateMetadata(meta StampMetadata) error {
	if meta.Issue < 0 {
		return fieldError("metadata.issue", ErrInvalidField, "issue must not be negative")
	}
	if meta.Generation != "" {
		switch meta.Generation {
		case "gen/now", "gen/next", "gen/second-next", "gen/future":
		default:
			return fieldError("metadata.generation", ErrInvalidField, meta.Generation)
		}
	}
	for _, s := range []struct {
		field string
		value string
	}{
		{"metadata.source", meta.Source},
		{"metadata.requester", meta.Requester},
	} {
		if strings.ContainsAny(s.value, "\r\n") {
			return fieldError(s.field, ErrInvalidField, "must be one line")
		}
	}
	for _, label := range meta.Labels {
		if strings.TrimSpace(label) == "" || strings.ContainsAny(label, "\r\n") {
			return fieldError("metadata.labels", ErrInvalidField, label)
		}
	}
	return nil
}

func nextSequence(records []SubmitRecord) int64 {
	var maxSeq int64
	for _, rec := range records {
		if rec.Sequence > maxSeq {
			maxSeq = rec.Sequence
		}
	}
	return maxSeq + 1
}

func invalidAll(records []SubmitRecord, msg string) []InvalidRecord {
	out := make([]InvalidRecord, 0, len(records))
	for _, rec := range records {
		out = append(out, InvalidRecord{Record: rec, Error: msg})
	}
	return out
}

func fieldError(field string, err error, reason string) error {
	return FieldError{Field: field, Err: err, Reason: reason}
}
