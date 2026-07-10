package modelroute

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

const AuditReceiptLedgerRowSchema = "fak-crossaudit-ledger-row/v1"

// AuditFindingSeverity is a closed vocabulary. UNKNOWN means a REFUTE finding
// was real enough to preserve but its impact was not calibrated; NONE is used
// only when the audit did not produce a finding.
type AuditFindingSeverity string

const (
	AuditSeverityNone     AuditFindingSeverity = "NONE"
	AuditSeverityUnknown  AuditFindingSeverity = "UNKNOWN"
	AuditSeverityLow      AuditFindingSeverity = "LOW"
	AuditSeverityMedium   AuditFindingSeverity = "MEDIUM"
	AuditSeverityHigh     AuditFindingSeverity = "HIGH"
	AuditSeverityCritical AuditFindingSeverity = "CRITICAL"
)

func (s AuditFindingSeverity) Valid() bool {
	switch s {
	case AuditSeverityNone, AuditSeverityUnknown, AuditSeverityLow, AuditSeverityMedium, AuditSeverityHigh, AuditSeverityCritical:
		return true
	default:
		return false
	}
}

// NormalizeAuditFindingSeverity prevents an untrusted reviewer from attaching
// finding severity to a non-finding verdict. A REFUTE with no calibrated
// severity remains UNKNOWN rather than being silently upgraded or dropped.
func NormalizeAuditFindingSeverity(verdict CrossAuditVerdict, severity AuditFindingSeverity) AuditFindingSeverity {
	severity = AuditFindingSeverity(strings.ToUpper(strings.TrimSpace(string(severity))))
	if verdict != CrossAuditRefute {
		return AuditSeverityNone
	}
	if !severity.Valid() || severity == AuditSeverityNone {
		return AuditSeverityUnknown
	}
	return severity
}

func (s AuditFindingSeverity) ValidateForVerdict(verdict CrossAuditVerdict) error {
	if !s.Valid() {
		return fmt.Errorf("modelroute: cross-audit finding severity %q is invalid", s)
	}
	if verdict == CrossAuditRefute && s == AuditSeverityNone {
		return fmt.Errorf("modelroute: cross-audit REFUTE requires a finding severity")
	}
	if verdict != CrossAuditRefute && s != AuditSeverityNone {
		return fmt.Errorf("modelroute: cross-audit verdict %s cannot carry finding severity %s", verdict, s)
	}
	return nil
}

// AuditTiming uses exact integer nanoseconds so receipt hashing never depends
// on locale, float formatting, or lossy duration conversion.
type AuditTiming struct {
	StartedAtUnixNano   int64 `json:"started_at_unix_nano"`
	CompletedAtUnixNano int64 `json:"completed_at_unix_nano"`
	DurationNanos       int64 `json:"duration_nanos"`
}

func (t AuditTiming) Validate() error {
	if t.StartedAtUnixNano <= 0 || t.CompletedAtUnixNano < t.StartedAtUnixNano || t.DurationNanos < 0 {
		return fmt.Errorf("modelroute: cross-audit timing is invalid")
	}
	if t.CompletedAtUnixNano-t.StartedAtUnixNano != t.DurationNanos {
		return fmt.Errorf("modelroute: cross-audit duration does not match timing endpoints")
	}
	return nil
}

// AuditTokenCost records exact token counts and USD microunits. It deliberately
// avoids floats in the durable receipt and ledger hash preimages.
type AuditTokenCost struct {
	Measured      bool   `json:"measured"`
	InputTokens   int64  `json:"input_tokens"`
	OutputTokens  int64  `json:"output_tokens"`
	TotalTokens   int64  `json:"total_tokens"`
	CostMicrosUSD int64  `json:"cost_micros_usd"`
	Currency      string `json:"currency"`
	Basis         string `json:"basis"`
}

func (u AuditTokenCost) Normalize() AuditTokenCost {
	if u.InputTokens != 0 || u.OutputTokens != 0 || u.TotalTokens != 0 || u.CostMicrosUSD != 0 {
		u.Measured = true
	}
	if u.TotalTokens == 0 && (u.InputTokens != 0 || u.OutputTokens != 0) {
		u.TotalTokens = u.InputTokens + u.OutputTokens
	}
	if strings.TrimSpace(u.Currency) == "" {
		u.Currency = "USD"
	}
	if strings.TrimSpace(u.Basis) == "" {
		u.Basis = "unreported"
	}
	u.Currency = strings.ToUpper(strings.TrimSpace(u.Currency))
	u.Basis = strings.ToLower(strings.TrimSpace(u.Basis))
	return u
}

func (u AuditTokenCost) Validate() error {
	if u.InputTokens < 0 || u.OutputTokens < 0 || u.TotalTokens < 0 || u.CostMicrosUSD < 0 {
		return fmt.Errorf("modelroute: cross-audit token/cost usage cannot be negative")
	}
	if u.TotalTokens != u.InputTokens+u.OutputTokens {
		return fmt.Errorf("modelroute: cross-audit total tokens %d, want input+output %d", u.TotalTokens, u.InputTokens+u.OutputTokens)
	}
	if !u.Measured && (u.InputTokens != 0 || u.OutputTokens != 0 || u.TotalTokens != 0 || u.CostMicrosUSD != 0) {
		return fmt.Errorf("modelroute: unmeasured cross-audit usage cannot carry token/cost values")
	}
	if u.Currency != "USD" {
		return fmt.Errorf("modelroute: cross-audit cost currency %q, want USD", u.Currency)
	}
	switch u.Basis {
	case "unreported", "provider-reported", "host-calculated":
	default:
		return fmt.Errorf("modelroute: cross-audit cost basis %q is invalid", u.Basis)
	}
	return nil
}

// AuditReceiptKey is stable across retries and auditor selection: the subject
// digest and exact policy version+digest are its complete preimage.
func AuditReceiptKey(subjectDigest, policyVersion, policyDigest string) string {
	return "audit:v2:" + strings.TrimPrefix(hashJSON(struct {
		SubjectDigest string `json:"subject_digest"`
		PolicyVersion string `json:"policy_version"`
		PolicyDigest  string `json:"policy_digest"`
	}{subjectDigest, policyVersion, policyDigest}), "sha256:")
}

func auditReceiptIdentityDigest(receipt IssueAuditReceipt) string {
	return hashJSON(struct {
		Author          AuditIdentity  `json:"author"`
		Auditor         AuditIdentity  `json:"auditor"`
		ObservedAuditor *AuditIdentity `json:"observed_auditor,omitempty"`
	}{receipt.Author, receipt.Auditor, receipt.ObservedAuditor})
}

func validSHA256Digest(digest string) bool {
	if !strings.HasPrefix(digest, "sha256:") || len(digest) != len("sha256:")+64 {
		return false
	}
	for _, r := range strings.TrimPrefix(digest, "sha256:") {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f')) {
			return false
		}
	}
	return true
}

// LegacyAuditReceiptLedgerKey imports a verified v1 receipt without rewriting
// or inventing facts it never carried. V1 had no policy digest, so its typed
// legacy key binds exactly the available subject digest + policy version. The
// ledger row hash then binds that unchanged historical receipt and key.
func LegacyAuditReceiptLedgerKey(subjectDigest, policyVersion string) string {
	return "audit:v1:" + strings.TrimPrefix(hashJSON(struct {
		SubjectDigest string `json:"subject_digest"`
		PolicyVersion string `json:"policy_version"`
	}{subjectDigest, policyVersion}), "sha256:")
}

func auditReceiptLedgerKey(receipt IssueAuditReceipt) (string, error) {
	switch receipt.Schema {
	case CrossAuditReceiptSchema:
		return receipt.AuditKey, nil
	case CrossAuditReceiptSchemaV1:
		return LegacyAuditReceiptLedgerKey(receipt.Subject.Digest, receipt.PolicyVersion), nil
	default:
		return "", fmt.Errorf("modelroute: audit ledger receipt schema %q is unsupported", receipt.Schema)
	}
}

// AuditReceiptLedgerRow embeds the canonical IssueAuditReceipt rather than
// defining a second receipt schema. Hash covers the entire verified receipt,
// its cursor, append time, and previous row hash.
type AuditReceiptLedgerRow struct {
	Schema             string            `json:"schema"`
	Seq                uint64            `json:"seq"`
	AppendedAtUnixNano int64             `json:"appended_at_unix_nano"`
	AuditKey           string            `json:"audit_key"`
	Receipt            IssueAuditReceipt `json:"receipt"`
	PrevHash           string            `json:"prev_hash"`
	Hash               string            `json:"hash"`
}

func (row AuditReceiptLedgerRow) recomputeHash() string {
	row.Hash = ""
	return hashJSON(row)
}

func validateAuditReceiptLedgerRow(row AuditReceiptLedgerRow, wantSeq uint64, wantPrev string) error {
	if row.Schema != AuditReceiptLedgerRowSchema {
		return fmt.Errorf("modelroute: audit ledger row schema %q, want %q", row.Schema, AuditReceiptLedgerRowSchema)
	}
	if row.Seq != wantSeq {
		return fmt.Errorf("modelroute: audit ledger sequence %d, want %d", row.Seq, wantSeq)
	}
	if row.AppendedAtUnixNano <= 0 {
		return fmt.Errorf("modelroute: audit ledger row %d has no append timestamp", row.Seq)
	}
	if row.PrevHash != wantPrev {
		return fmt.Errorf("modelroute: audit ledger row %d prev_hash %q, want %q", row.Seq, row.PrevHash, wantPrev)
	}
	if row.PrevHash != "" && !validSHA256Digest(row.PrevHash) {
		return fmt.Errorf("modelroute: audit ledger row %d prev_hash is not a sha256 digest", row.Seq)
	}
	if err := row.Receipt.Verify(); err != nil {
		return fmt.Errorf("modelroute: audit ledger row %d receipt: %w", row.Seq, err)
	}
	wantKey, err := auditReceiptLedgerKey(row.Receipt)
	if err != nil {
		return err
	}
	if row.AuditKey != wantKey {
		return fmt.Errorf("modelroute: audit ledger row %d audit_key %q, want %q", row.Seq, row.AuditKey, wantKey)
	}
	if !validSHA256Digest(row.Hash) || row.Hash != row.recomputeHash() {
		return fmt.Errorf("modelroute: audit ledger row %d hash mismatch", row.Seq)
	}
	return nil
}

type AuditReceiptLedgerIntegrity struct {
	Broken    bool   `json:"broken"`
	TornTail  bool   `json:"torn_tail,omitempty"`
	AtLine    int    `json:"at_line,omitempty"`
	AtSeq     uint64 `json:"at_seq,omitempty"`
	Reason    string `json:"reason,omitempty"`
	Recovered int    `json:"recovered_rows"`
}

type AuditReceiptLedgerIntegrityError struct {
	Integrity AuditReceiptLedgerIntegrity
}

func (e *AuditReceiptLedgerIntegrityError) Error() string {
	kind := "corrupt"
	if e.Integrity.TornTail {
		kind = "torn tail"
	}
	return fmt.Sprintf("modelroute: audit receipt ledger %s at line %d: %s", kind, e.Integrity.AtLine, e.Integrity.Reason)
}

// ParseAuditReceiptLedgerPrefix returns the longest verified prefix. A torn
// final write and a semantic/hash corruption are both surfaced in Integrity;
// neither is silently treated as a complete ledger.
func ParseAuditReceiptLedgerPrefix(r io.Reader) ([]AuditReceiptLedgerRow, AuditReceiptLedgerIntegrity, error) {
	reader := bufio.NewReader(r)
	var rows []AuditReceiptLedgerRow
	prev := ""
	seenKeys := make(map[string]string)
	lineNo := 0
	for {
		raw, readErr := reader.ReadBytes('\n')
		if len(raw) == 0 && errors.Is(readErr, io.EOF) {
			return rows, AuditReceiptLedgerIntegrity{Recovered: len(rows)}, nil
		}
		if readErr != nil && !errors.Is(readErr, io.EOF) {
			return rows, AuditReceiptLedgerIntegrity{}, fmt.Errorf("modelroute: read audit receipt ledger: %w", readErr)
		}
		lineNo++
		if errors.Is(readErr, io.EOF) {
			integ := AuditReceiptLedgerIntegrity{
				Broken: true, TornTail: true, AtLine: lineNo, Recovered: len(rows),
				Reason: "unterminated final ledger row",
			}
			return rows, integ, nil
		}
		line := bytes.TrimSpace(raw)
		if len(line) == 0 {
			integ := AuditReceiptLedgerIntegrity{Broken: true, AtLine: lineNo, Recovered: len(rows), Reason: "blank ledger row"}
			return rows, integ, nil
		}
		var row AuditReceiptLedgerRow
		if err := decodeAuditLedgerLine(line, &row); err != nil {
			integ := AuditReceiptLedgerIntegrity{
				Broken: true, AtLine: lineNo,
				Recovered: len(rows), Reason: "decode: " + err.Error(),
			}
			return rows, integ, nil
		}
		if err := validateAuditReceiptLedgerRow(row, uint64(len(rows)+1), prev); err != nil {
			integ := AuditReceiptLedgerIntegrity{Broken: true, AtLine: lineNo, AtSeq: row.Seq, Recovered: len(rows), Reason: err.Error()}
			return rows, integ, nil
		}
		if priorDigest, ok := seenKeys[row.AuditKey]; ok {
			reason := fmt.Sprintf("%v: audit_key %s repeats receipt %s", ErrAuditReceiptDuplicateRow, row.AuditKey, priorDigest)
			if priorDigest != row.Receipt.ReceiptDigest {
				reason = fmt.Sprintf("%v: audit_key %s has receipt digests %s and %s", ErrAuditReceiptKeyConflict, row.AuditKey, priorDigest, row.Receipt.ReceiptDigest)
			}
			integ := AuditReceiptLedgerIntegrity{Broken: true, AtLine: lineNo, AtSeq: row.Seq, Recovered: len(rows), Reason: reason}
			return rows, integ, nil
		}
		seenKeys[row.AuditKey] = row.Receipt.ReceiptDigest
		rows = append(rows, row)
		prev = row.Hash
	}
}

func decodeAuditLedgerLine(line []byte, out *AuditReceiptLedgerRow) error {
	dec := json.NewDecoder(bytes.NewReader(line))
	dec.DisallowUnknownFields()
	if err := dec.Decode(out); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("multiple JSON values")
		}
		return err
	}
	return nil
}

// ParseAuditReceiptLedger is strict: any torn tail, malformed row, receipt
// mutation, sequence gap, or hash discontinuity is an error.
func ParseAuditReceiptLedger(r io.Reader) ([]AuditReceiptLedgerRow, error) {
	rows, integ, err := ParseAuditReceiptLedgerPrefix(r)
	if err != nil {
		return rows, err
	}
	if integ.Broken {
		return rows, &AuditReceiptLedgerIntegrityError{Integrity: integ}
	}
	return rows, nil
}

func VerifyAuditReceiptRows(rows []AuditReceiptLedgerRow) (string, error) {
	prev := ""
	for i, row := range rows {
		if err := validateAuditReceiptLedgerRow(row, uint64(i+1), prev); err != nil {
			return prev, err
		}
		prev = row.Hash
	}
	return prev, nil
}

type AuditReceiptFoldEntry struct {
	AuditKey      string               `json:"audit_key"`
	Seq           uint64               `json:"seq"`
	Verdict       CrossAuditVerdict    `json:"verdict"`
	Severity      AuditFindingSeverity `json:"severity"`
	Passed        bool                 `json:"passed"`
	ReceiptDigest string               `json:"receipt_digest"`
	RowHash       string               `json:"row_hash"`
}

type AuditReceiptFold struct {
	Rows          int                              `json:"rows"`
	UniqueAudits  int                              `json:"unique_audits"`
	DuplicateRows int                              `json:"duplicate_rows"`
	Verdict       CrossAuditVerdict                `json:"verdict"`
	CleanPass     bool                             `json:"clean_pass"`
	VerdictCounts map[CrossAuditVerdict]int        `json:"verdict_counts"`
	ByAuditKey    map[string]AuditReceiptFoldEntry `json:"by_audit_key"`
	OrderedKeys   []string                         `json:"ordered_keys"`
}

// FoldAuditReceiptLedger verifies the complete chain before reducing it. Only
// the literal PASS verdict sets Passed; REFUTE, INCONCLUSIVE, and UNAVAILABLE
// remain false regardless of reason or severity text.
func FoldAuditReceiptLedger(rows []AuditReceiptLedgerRow) (AuditReceiptFold, error) {
	if _, err := VerifyAuditReceiptRows(rows); err != nil {
		return AuditReceiptFold{}, err
	}
	fold := AuditReceiptFold{
		Rows: len(rows),
		VerdictCounts: map[CrossAuditVerdict]int{
			CrossAuditPass: 0, CrossAuditRefute: 0, CrossAuditInconclusive: 0, CrossAuditUnavailable: 0,
		},
		ByAuditKey: make(map[string]AuditReceiptFoldEntry),
	}
	for _, row := range rows {
		key := row.AuditKey
		if prior, ok := fold.ByAuditKey[key]; ok {
			if prior.ReceiptDigest != row.Receipt.ReceiptDigest {
				return AuditReceiptFold{}, fmt.Errorf("%w: audit_key %s has receipt digests %s and %s", ErrAuditReceiptKeyConflict, key, prior.ReceiptDigest, row.Receipt.ReceiptDigest)
			}
			return AuditReceiptFold{}, fmt.Errorf("%w: audit_key %s repeats receipt %s", ErrAuditReceiptDuplicateRow, key, prior.ReceiptDigest)
		}
		entry := AuditReceiptFoldEntry{
			AuditKey: key, Seq: row.Seq, Verdict: row.Receipt.Verdict, Severity: row.Receipt.Severity,
			Passed: row.Receipt.Verdict == CrossAuditPass, ReceiptDigest: row.Receipt.ReceiptDigest, RowHash: row.Hash,
		}
		fold.ByAuditKey[key] = entry
		fold.VerdictCounts[row.Receipt.Verdict]++
		fold.OrderedKeys = append(fold.OrderedKeys, key)
	}
	fold.UniqueAudits = len(fold.ByAuditKey)
	sort.Strings(fold.OrderedKeys)
	fold.Verdict = foldAuditReceiptVerdict(fold.VerdictCounts, fold.UniqueAudits)
	fold.CleanPass = fold.Verdict == CrossAuditPass && fold.UniqueAudits > 0
	return fold, nil
}

func foldAuditReceiptVerdict(counts map[CrossAuditVerdict]int, unique int) CrossAuditVerdict {
	if unique == 0 {
		return CrossAuditUnavailable
	}
	if counts[CrossAuditRefute] > 0 {
		return CrossAuditRefute
	}
	if counts[CrossAuditUnavailable] > 0 || counts[CrossAuditInconclusive] > 0 {
		return CrossAuditInconclusive
	}
	return CrossAuditPass
}

type AuditReceiptLedgerVerification struct {
	Rows          int                       `json:"rows"`
	UniqueAudits  int                       `json:"unique_audits"`
	DuplicateRows int                       `json:"duplicate_rows"`
	HeadHash      string                    `json:"head_hash"`
	VerdictCounts map[CrossAuditVerdict]int `json:"verdict_counts"`
	Cursor        AuditReceiptLedgerCursor  `json:"cursor"`
}

const AuditReceiptLedgerCursorSchema = "fak-crossaudit-ledger-cursor/v1"

// AuditReceiptLedgerCursor is the external truncation witness a consumer saves
// after append/verify. A bare hash-chain proves its present prefix but cannot
// detect deletion of the final valid rows without this expected head.
type AuditReceiptLedgerCursor struct {
	Schema   string `json:"schema"`
	Rows     uint64 `json:"rows"`
	HeadHash string `json:"head_hash"`
}

func (cursor AuditReceiptLedgerCursor) Validate() error {
	if cursor.Schema != AuditReceiptLedgerCursorSchema {
		return fmt.Errorf("modelroute: audit receipt cursor schema %q, want %q", cursor.Schema, AuditReceiptLedgerCursorSchema)
	}
	if cursor.Rows == 0 {
		if cursor.HeadHash != "" {
			return fmt.Errorf("modelroute: empty audit receipt cursor cannot carry a head hash")
		}
		return nil
	}
	if !validSHA256Digest(cursor.HeadHash) {
		return fmt.Errorf("modelroute: audit receipt cursor head is not sha256")
	}
	return nil
}

func auditReceiptLedgerCursor(rows []AuditReceiptLedgerRow) AuditReceiptLedgerCursor {
	cursor := AuditReceiptLedgerCursor{Schema: AuditReceiptLedgerCursorSchema, Rows: uint64(len(rows))}
	if len(rows) > 0 {
		cursor.HeadHash = rows[len(rows)-1].Hash
	}
	return cursor
}

func VerifyAuditReceiptLedgerReader(r io.Reader) (AuditReceiptLedgerVerification, error) {
	rows, err := ParseAuditReceiptLedger(r)
	if err != nil {
		return AuditReceiptLedgerVerification{}, err
	}
	return verifyAuditReceiptLedgerRows(rows)
}

func verifyAuditReceiptLedgerRows(rows []AuditReceiptLedgerRow) (AuditReceiptLedgerVerification, error) {
	fold, err := FoldAuditReceiptLedger(rows)
	if err != nil {
		return AuditReceiptLedgerVerification{}, err
	}
	head, err := VerifyAuditReceiptRows(rows)
	if err != nil {
		return AuditReceiptLedgerVerification{}, err
	}
	return AuditReceiptLedgerVerification{
		Rows: fold.Rows, UniqueAudits: fold.UniqueAudits, DuplicateRows: fold.DuplicateRows,
		HeadHash: head, VerdictCounts: fold.VerdictCounts, Cursor: auditReceiptLedgerCursor(rows),
	}, nil
}

func VerifyAuditReceiptLedger(path string) (AuditReceiptLedgerVerification, error) {
	f, err := os.Open(path)
	if err != nil {
		return AuditReceiptLedgerVerification{}, fmt.Errorf("modelroute: open audit receipt ledger: %w", err)
	}
	defer f.Close()
	return VerifyAuditReceiptLedgerReader(f)
}

func VerifyAuditReceiptLedgerAtCursor(path string, expected AuditReceiptLedgerCursor) (AuditReceiptLedgerVerification, error) {
	if err := expected.Validate(); err != nil {
		return AuditReceiptLedgerVerification{}, err
	}
	f, err := os.Open(path)
	if err != nil {
		return AuditReceiptLedgerVerification{}, fmt.Errorf("modelroute: open audit receipt ledger: %w", err)
	}
	defer f.Close()
	rows, err := ParseAuditReceiptLedger(f)
	if err != nil {
		return AuditReceiptLedgerVerification{}, err
	}
	verification, err := verifyAuditReceiptLedgerRows(rows)
	if err != nil {
		return AuditReceiptLedgerVerification{}, err
	}
	if expected.Rows == 0 {
		return verification, nil
	}
	if uint64(len(rows)) < expected.Rows {
		return AuditReceiptLedgerVerification{}, fmt.Errorf(
			"modelroute: audit receipt ledger cursor shrank: rows %d, expected at least %d", len(rows), expected.Rows,
		)
	}
	if got := rows[expected.Rows-1].Hash; got != expected.HeadHash {
		return AuditReceiptLedgerVerification{}, fmt.Errorf(
			"modelroute: audit receipt ledger cursor prefix mismatch at row %d: head %s, want %s", expected.Rows, got, expected.HeadHash,
		)
	}
	return verification, nil
}

var (
	ErrAuditReceiptKeyConflict  = errors.New("modelroute: conflicting duplicate audit receipt key")
	ErrAuditReceiptDuplicateRow = errors.New("modelroute: duplicate audit receipt ledger row")
	ErrAuditReceiptLedgerBusy   = errors.New("modelroute: audit receipt ledger append lock is busy")
	auditReceiptLedgerLockWait  = 2 * time.Second
)

type AuditReceiptAppendResult struct {
	Row       AuditReceiptLedgerRow    `json:"row"`
	Appended  bool                     `json:"appended"`
	Duplicate bool                     `json:"duplicate"`
	Cursor    AuditReceiptLedgerCursor `json:"cursor"`
}

// AppendAuditReceiptLedger verifies the existing ledger before appending. An
// exact duplicate is an idempotent no-op; the same subject+policy key carrying
// a different receipt digest is a typed conflict and is never appended.
func AppendAuditReceiptLedger(path string, receipt IssueAuditReceipt) (AuditReceiptAppendResult, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return AuditReceiptAppendResult{}, fmt.Errorf("modelroute: audit receipt ledger path is required")
	}
	if err := receipt.Verify(); err != nil {
		return AuditReceiptAppendResult{}, err
	}
	ledgerKey, err := auditReceiptLedgerKey(receipt)
	if err != nil {
		return AuditReceiptAppendResult{}, err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return AuditReceiptAppendResult{}, fmt.Errorf("modelroute: create audit receipt ledger directory: %w", err)
	}
	var result AuditReceiptAppendResult
	err = withAuditReceiptLedgerLock(path, func() error {
		rows, err := readAuditReceiptLedger(path)
		if err != nil {
			return err
		}
		for _, row := range rows {
			if row.AuditKey != ledgerKey {
				continue
			}
			if row.Receipt.ReceiptDigest == receipt.ReceiptDigest {
				result = AuditReceiptAppendResult{Row: row, Duplicate: true, Cursor: auditReceiptLedgerCursor(rows)}
				return nil
			}
			return fmt.Errorf("%w: %s", ErrAuditReceiptKeyConflict, ledgerKey)
		}

		row := AuditReceiptLedgerRow{
			Schema: AuditReceiptLedgerRowSchema, Seq: uint64(len(rows) + 1), AppendedAtUnixNano: time.Now().UTC().UnixNano(),
			AuditKey: ledgerKey, Receipt: receipt,
		}
		if len(rows) > 0 {
			row.PrevHash = rows[len(rows)-1].Hash
		}
		row.Hash = row.recomputeHash()
		if err := validateAuditReceiptLedgerRow(row, row.Seq, row.PrevHash); err != nil {
			return err
		}
		b, err := json.Marshal(row)
		if err != nil {
			return fmt.Errorf("modelroute: encode audit receipt ledger row: %w", err)
		}
		b = append(b, '\n')
		f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
		if err != nil {
			return fmt.Errorf("modelroute: open audit receipt ledger for append: %w", err)
		}
		defer f.Close()
		if _, err := f.Write(b); err != nil {
			return fmt.Errorf("modelroute: append audit receipt ledger: %w", err)
		}
		if err := f.Sync(); err != nil {
			return fmt.Errorf("modelroute: sync audit receipt ledger: %w", err)
		}
		result = AuditReceiptAppendResult{Row: row, Appended: true, Cursor: auditReceiptLedgerCursor(append(rows, row))}
		return nil
	})
	return result, err
}

func readAuditReceiptLedger(path string) ([]AuditReceiptLedgerRow, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("modelroute: open audit receipt ledger: %w", err)
	}
	defer f.Close()
	return ParseAuditReceiptLedger(f)
}

func withAuditReceiptLedgerLock(path string, fn func() error) error {
	lockFile, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return fmt.Errorf("modelroute: open audit receipt ledger lock: %w", err)
	}
	defer lockFile.Close()
	deadline := time.Now().Add(auditReceiptLedgerLockWait)
	for {
		err = flock.TryLock(lockFile)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			return fmt.Errorf("modelroute: lock audit receipt ledger: %w", err)
		}
		if time.Now().After(deadline) {
			return ErrAuditReceiptLedgerBusy
		}
		time.Sleep(10 * time.Millisecond)
	}
	defer flock.Unlock(lockFile) // best effort; close also releases the advisory lock
	return fn()
}
