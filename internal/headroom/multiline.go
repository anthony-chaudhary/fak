package headroom

import "bytes"

// recordLineKind is the compiled-in filter's classification of one physical
// line. Continuations attach to the current record; every other kind starts a
// new record. Unknown lines are always preserved by consumers.
type recordLineKind uint8

const (
	recordUnknown recordLineKind = iota
	recordRoutine
	recordError
	recordContinuation
)

type recordLineClassifier func(line []byte) recordLineKind

// distillRecord is a byte-exact group of physical lines. ForceKeep marks an
// overflow chunk whose classification is intentionally unavailable: filters
// must preserve it rather than risk hiding an unterminated diagnostic.
type distillRecord struct {
	Bytes     []byte
	Kind      recordLineKind
	ForceKeep bool
}

// groupDistillRecords groups raw bytes without normalizing line endings or the
// final newline. maxRecordBytes bounds a classified record; once exceeded, all
// continuation bytes are emitted as force-keep chunks until a new record starts.
func groupDistillRecords(raw []byte, maxRecordBytes int, classify recordLineClassifier) []distillRecord {
	if len(raw) == 0 {
		return nil
	}
	if maxRecordBytes <= 0 {
		maxRecordBytes = 64 * 1024
	}
	lines := splitLinesExact(raw)
	records := make([]distillRecord, 0, len(lines))
	for _, line := range lines {
		kind := classify(bytes.TrimSuffix(bytes.TrimSuffix(line, []byte("\n")), []byte("\r")))
		if kind == recordContinuation && len(records) > 0 {
			last := &records[len(records)-1]
			if !last.ForceKeep && len(last.Bytes)+len(line) <= maxRecordBytes {
				last.Bytes = append(last.Bytes, line...)
				continue
			}
			records = append(records, distillRecord{Bytes: append([]byte(nil), line...), Kind: recordUnknown, ForceKeep: true})
			continue
		}
		if kind == recordContinuation {
			kind = recordUnknown
		}
		records = append(records, distillRecord{Bytes: append([]byte(nil), line...), Kind: kind})
	}
	return records
}

func splitLinesExact(raw []byte) [][]byte {
	lines := make([][]byte, 0, bytes.Count(raw, []byte("\n"))+1)
	for len(raw) > 0 {
		i := bytes.IndexByte(raw, '\n')
		if i < 0 {
			lines = append(lines, raw)
			break
		}
		lines = append(lines, raw[:i+1])
		raw = raw[i+1:]
	}
	return lines
}

func joinDistillRecords(records []distillRecord) []byte {
	var size int
	for _, record := range records {
		size += len(record.Bytes)
	}
	out := make([]byte, 0, size)
	for _, record := range records {
		out = append(out, record.Bytes...)
	}
	return out
}
