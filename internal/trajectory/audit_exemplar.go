package trajectory

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	// AuditUnknownExemplarMaxCount and AuditUnknownExemplarMaxBytes bound each
	// persisted reservoir. The byte bound covers the exact JSON encoding of the
	// exemplar array, including delimiters, rather than an approximate payload sum.
	AuditUnknownExemplarMaxCount = 32
	AuditUnknownExemplarMaxBytes = 16 * 1024

	auditUnknownExemplarStructureMaxBytes = 1024
)

// AuditUnknownExemplar is a content-free shape retained for an event whose
// subtype or visibility could not be classified. Structure contains JSON field
// names and JSON types only. Scalar values are discarded before ShapeHash, ID,
// or any persisted field is produced.
type AuditUnknownExemplar struct {
	ID            string `json:"id"`
	Source        string `json:"source"`
	Subtype       string `json:"subtype"`
	Visibility    string `json:"visibility"`
	Aggregate     string `json:"aggregate"`
	ShapeHash     string `json:"shape_hash"`
	Structure     string `json:"structure"`
	Observations  int    `json:"observations"`
	ObservedBytes int64  `json:"observed_bytes"`
}

// AuditUnknownExemplarReservoir describes both the fixed production limits and
// the retained set. DroppedObservations includes observations whose shapes were
// evicted or could not fit; it keeps the cap honest without an unbounded side map.
type AuditUnknownExemplarReservoir struct {
	CardinalityLimit    int                    `json:"cardinality_limit"`
	ByteLimit           int                    `json:"byte_limit"`
	Retained            int                    `json:"retained"`
	StoredBytes         int                    `json:"stored_bytes"`
	DroppedObservations int                    `json:"dropped_observations"`
	Exemplars           []AuditUnknownExemplar `json:"exemplars,omitempty"`
}

type auditUnknownExemplarReservoir struct {
	maxCount          int
	maxBytes          int
	totalObservations int
	byID              map[string]AuditUnknownExemplar
}

func newDefaultAuditUnknownExemplarReservoir() *auditUnknownExemplarReservoir {
	return newAuditUnknownExemplarReservoir(AuditUnknownExemplarMaxCount, AuditUnknownExemplarMaxBytes)
}

func newAuditUnknownExemplarReservoir(maxCount, maxBytes int) *auditUnknownExemplarReservoir {
	if maxCount < 0 {
		maxCount = 0
	}
	if maxBytes < 2 { // [] is the smallest exact JSON array encoding.
		maxBytes = 2
	}
	return &auditUnknownExemplarReservoir{maxCount: maxCount, maxBytes: maxBytes, byID: map[string]AuditUnknownExemplar{}}
}

func (r *auditUnknownExemplarReservoir) observe(source, subtype, visibility, aggregate string, line []byte, observedBytes int64) {
	structure, shapeHash := auditStructuralShape(line)
	scrubbedSource, scrubbedSubtype := auditScrubExemplarLabel(source), auditScrubExemplarLabel(subtype)
	auditExemplarShape := auditScrubExemplarLabel(aggregate)
	identity := sha256.Sum256([]byte(strings.Join([]string{scrubbedSource, scrubbedSubtype, visibility, auditExemplarShape, shapeHash}, "\x00")))
	id := "ex-" + hex.EncodeToString(identity[:8])
	r.totalObservations++
	if previous, ok := r.byID[id]; ok {
		previous.Observations++
		previous.ObservedBytes += observedBytes
		r.byID[id] = previous
		r.rebalance()
		return
	}
	r.byID[id] = AuditUnknownExemplar{
		ID: id, Source: scrubbedSource, Subtype: scrubbedSubtype,
		Visibility: visibility, Aggregate: auditExemplarShape, ShapeHash: shapeHash, Structure: auditBoundStructure(structure, shapeHash),
		Observations: 1, ObservedBytes: observedBytes,
	}
	r.rebalance()
}

func (r *auditUnknownExemplarReservoir) merge(snapshot AuditUnknownExemplarReservoir) {
	r.totalObservations += snapshot.DroppedObservations
	for _, exemplar := range snapshot.Exemplars {
		if exemplar.Observations < 1 {
			exemplar.Observations = 1
		}
		r.totalObservations += exemplar.Observations
		if previous, ok := r.byID[exemplar.ID]; ok {
			previous.Observations += exemplar.Observations
			previous.ObservedBytes += exemplar.ObservedBytes
			r.byID[exemplar.ID] = previous
		} else {
			r.byID[exemplar.ID] = exemplar
		}
		r.rebalance()
	}
}

func (r *auditUnknownExemplarReservoir) rebalance() {
	rows := r.sorted()
	kept := make(map[string]AuditUnknownExemplar, min(len(rows), r.maxCount))
	selected := make([]AuditUnknownExemplar, 0, min(len(rows), r.maxCount))
	for _, row := range rows {
		if len(selected) >= r.maxCount {
			break
		}
		candidate := append(append([]AuditUnknownExemplar(nil), selected...), row)
		if auditSerializedExemplarBytes(candidate) > r.maxBytes {
			continue
		}
		selected = candidate
		kept[row.ID] = row
	}
	r.byID = kept
}

func (r *auditUnknownExemplarReservoir) sorted() []AuditUnknownExemplar {
	rows := make([]AuditUnknownExemplar, 0, len(r.byID))
	for _, row := range r.byID {
		rows = append(rows, row)
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].ID < rows[j].ID })
	return rows
}

func (r *auditUnknownExemplarReservoir) snapshot() AuditUnknownExemplarReservoir {
	rows := r.sorted()
	retainedObservations := 0
	for _, row := range rows {
		retainedObservations += row.Observations
	}
	return AuditUnknownExemplarReservoir{
		CardinalityLimit: r.maxCount, ByteLimit: r.maxBytes, Retained: len(rows),
		StoredBytes:         auditSerializedExemplarBytes(rows),
		DroppedObservations: max(0, r.totalObservations-retainedObservations), Exemplars: rows,
	}
}

func auditSerializedExemplarBytes(rows []AuditUnknownExemplar) int {
	encoded, _ := json.Marshal(rows)
	return len(encoded)
}

func auditStructuralShape(line []byte) (string, string) {
	var value any
	decoder := json.NewDecoder(strings.NewReader(string(line)))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		value = map[string]any{"malformed": nil}
	}
	var out strings.Builder
	auditWriteStructuralShape(&out, value)
	structure := out.String()
	digest := sha256.Sum256([]byte(structure))
	return structure, hex.EncodeToString(digest[:])
}

func auditWriteStructuralShape(out *strings.Builder, value any) {
	switch typed := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		out.WriteString("object{")
		for i, key := range keys {
			if i > 0 {
				out.WriteByte(',')
			}
			out.WriteString(strconv.Quote(auditScrubStructuralField(key)))
			out.WriteByte(':')
			auditWriteStructuralShape(out, typed[key])
		}
		out.WriteByte('}')
	case []any:
		shapes := make(map[string]struct{})
		for _, item := range typed {
			var itemShape strings.Builder
			auditWriteStructuralShape(&itemShape, item)
			shapes[itemShape.String()] = struct{}{}
		}
		items := make([]string, 0, len(shapes))
		for shape := range shapes {
			items = append(items, shape)
		}
		sort.Strings(items)
		out.WriteString("array<")
		out.WriteString(strings.Join(items, "|"))
		out.WriteByte('>')
	case string:
		out.WriteString("string")
	case json.Number, float64:
		out.WriteString("number")
	case bool:
		out.WriteString("boolean")
	case nil:
		out.WriteString("null")
	default:
		out.WriteString("unknown")
	}
}

func auditScrubStructuralField(field string) string {
	field = strings.ToValidUTF8(field, "�")
	lower := strings.ToLower(field)
	secretShaped := strings.HasPrefix(lower, "sk-") || strings.HasPrefix(lower, "ghp_") ||
		strings.HasPrefix(lower, "github_pat_") || strings.HasPrefix(field, "AKIA") ||
		strings.HasPrefix(lower, "bearer ") || strings.Contains(field, "\\") || strings.Contains(field, "/") ||
		strings.Contains(field, "@") || len(field) > 64
	if !secretShaped {
		return field
	}
	digest := sha256.Sum256([]byte(field))
	return "field#" + hex.EncodeToString(digest[:6])
}

func auditScrubExemplarLabel(label string) string {
	label = strings.ToValidUTF8(label, "�")
	if strings.HasPrefix(label, "/") {
		return auditScrubStructuralField(label)
	}
	parts := strings.Split(label, "/")
	for i, part := range parts {
		parts[i] = auditScrubStructuralField(part)
	}
	return strings.Join(parts, "/")
}

func auditUnknownDiscriminatorSuffix(values []string) string {
	hashes := make([]string, 0, len(values))
	seen := map[string]bool{}
	for _, value := range values {
		digest := sha256.Sum256([]byte(value))
		hash := hex.EncodeToString(digest[:6])
		if !seen[hash] {
			hashes = append(hashes, hash)
			seen[hash] = true
		}
	}
	if len(hashes) == 0 {
		return ""
	}
	sort.Strings(hashes)
	return "discriminator#" + strings.Join(hashes, ".")
}

func auditBoundStructure(structure, shapeHash string) string {
	if len(structure) <= auditUnknownExemplarStructureMaxBytes {
		return structure
	}
	suffix := "…#" + shapeHash[:12]
	limit := auditUnknownExemplarStructureMaxBytes - len(suffix)
	if limit <= 0 {
		return suffix
	}
	cut := limit
	for cut > 0 && !utf8.ValidString(structure[:cut]) {
		cut--
	}
	return structure[:cut] + suffix
}

func hasExemplarID(rows []AuditUnknownExemplar, id string) bool {
	for _, row := range rows {
		if row.ID == id {
			return true
		}
	}
	return false
}

func auditExemplarIDs(rows []AuditUnknownExemplar, match func(AuditUnknownExemplar) bool) []string {
	ids := make([]string, 0)
	for _, row := range rows {
		if match(row) {
			ids = append(ids, row.ID)
		}
	}
	sort.Strings(ids)
	return ids
}

func linkAuditUnknownExemplars(distribution []AuditDistributionRow, storage []AuditStorageRow, exemplars []AuditUnknownExemplar) {
	for i := range distribution {
		name := distribution[i].Name
		distribution[i].ExemplarIDs = auditExemplarIDs(exemplars, func(exemplar AuditUnknownExemplar) bool {
			return exemplar.Visibility == "visible_unknown" && exemplar.Aggregate == name
		})
	}
	for i := range storage {
		source, subtype := storage[i].Source, storage[i].Subtype
		storage[i].ExemplarIDs = auditExemplarIDs(exemplars, func(exemplar AuditUnknownExemplar) bool {
			return exemplar.Visibility == "storage_unknown" && exemplar.Source == auditScrubExemplarLabel(source) && exemplar.Subtype == auditScrubExemplarLabel(subtype)
		})
	}
}

func mergeAuditUnknownExemplarReservoirs(left, right AuditUnknownExemplarReservoir) AuditUnknownExemplarReservoir {
	reservoir := newDefaultAuditUnknownExemplarReservoir()
	reservoir.merge(left)
	reservoir.merge(right)
	return reservoir.snapshot()
}

func mergeAuditDistributionRows(groups ...[]AuditDistributionRow) []AuditDistributionRow {
	byteTotals := map[string]int64{}
	callTotals := map[string]int{}
	for _, group := range groups {
		for _, row := range group {
			byteTotals[row.Name] += row.Bytes
			callTotals[row.Name] += row.Calls
		}
	}
	rows := distributionRows(byteTotals)
	for i := range rows {
		rows[i].Calls = callTotals[rows[i].Name]
	}
	return rows
}

func mergeAuditStorageRows(groups ...[]AuditStorageRow) []AuditStorageRow {
	totals := map[string]*AuditStorageRow{}
	for _, group := range groups {
		for _, row := range group {
			key := row.Source + "\x00" + row.Subtype
			total := totals[key]
			if total == nil {
				total = &AuditStorageRow{Source: row.Source, Subtype: row.Subtype}
				totals[key] = total
			}
			total.Bytes += row.Bytes
			total.Records += row.Records
		}
	}
	return storageDistributionRows(totals)
}
