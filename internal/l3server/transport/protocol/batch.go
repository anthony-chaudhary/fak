package protocol

import (
	"encoding/binary"
	"fmt"
)

// MaxBatchCount is the absolute cap on batch element count to prevent OOM
// from malformed messages claiming huge counts with tiny bodies.
const MaxBatchCount = 100_000

// EncodeMGetBody encodes multiple keys for MGET/MTEST.
// Format: count(4) + [keyLen(2) + key]*
func EncodeMGetBody(keys [][]byte) []byte {
	size := 4
	for _, k := range keys {
		size += 2 + len(k)
	}
	body := make([]byte, size)
	binary.LittleEndian.PutUint32(body[0:4], uint32(len(keys)))
	off := 4
	for _, k := range keys {
		binary.LittleEndian.PutUint16(body[off:], uint16(len(k)))
		off += 2
		copy(body[off:], k)
		off += len(k)
	}
	return body
}

// DecodeMGetBody decodes multiple keys.
func DecodeMGetBody(body []byte) ([][]byte, error) {
	if len(body) < 4 {
		return nil, fmt.Errorf("mget body too short")
	}
	count := binary.LittleEndian.Uint32(body[0:4])
	if count > MaxBatchCount {
		return nil, fmt.Errorf("mget count %d exceeds max %d", count, MaxBatchCount)
	}
	// Physical bound: each key needs at least 2 bytes (keyLen field)
	if count > uint32((len(body)-4)/2) {
		return nil, fmt.Errorf("mget count %d impossible for body size %d", count, len(body))
	}
	off := 4
	keys := make([][]byte, count)
	for i := uint32(0); i < count; i++ {
		if off+2 > len(body) {
			return nil, fmt.Errorf("mget body truncated at key %d", i)
		}
		keyLen := binary.LittleEndian.Uint16(body[off:])
		off += 2
		if off+int(keyLen) > len(body) {
			return nil, fmt.Errorf("mget body key %d truncated", i)
		}
		keys[i] = body[off : off+int(keyLen)]
		off += int(keyLen)
	}
	return keys, nil
}

// EncodeMSetBody encodes multiple key-value pairs for MSET.
// Format: count(4) + [keyLen(2) + key + valueLen(4) + value]*
func EncodeMSetBody(keys, values [][]byte) []byte {
	size := 4
	for i := range keys {
		size += 2 + len(keys[i]) + 4 + len(values[i])
	}
	body := make([]byte, size)
	binary.LittleEndian.PutUint32(body[0:4], uint32(len(keys)))
	off := 4
	for i := range keys {
		binary.LittleEndian.PutUint16(body[off:], uint16(len(keys[i])))
		off += 2
		copy(body[off:], keys[i])
		off += len(keys[i])
		binary.LittleEndian.PutUint32(body[off:], uint32(len(values[i])))
		off += 4
		copy(body[off:], values[i])
		off += len(values[i])
	}
	return body
}

// DecodeMSetBody decodes multiple key-value pairs.
func DecodeMSetBody(body []byte) (keys, values [][]byte, err error) {
	if len(body) < 4 {
		return nil, nil, fmt.Errorf("mset body too short")
	}
	count := binary.LittleEndian.Uint32(body[0:4])
	if count > MaxBatchCount {
		return nil, nil, fmt.Errorf("mset count %d exceeds max %d", count, MaxBatchCount)
	}
	// Physical bound: each entry needs at least 6 bytes (keyLen(2)+valLen(4))
	if count > uint32((len(body)-4)/6) {
		return nil, nil, fmt.Errorf("mset count %d impossible for body size %d", count, len(body))
	}
	off := 4
	keys = make([][]byte, count)
	values = make([][]byte, count)
	for i := uint32(0); i < count; i++ {
		if off+2 > len(body) {
			return nil, nil, fmt.Errorf("mset body truncated at entry %d key length", i)
		}
		keyLen := binary.LittleEndian.Uint16(body[off:])
		off += 2
		if off+int(keyLen) > len(body) {
			return nil, nil, fmt.Errorf("mset body key %d truncated", i)
		}
		keys[i] = body[off : off+int(keyLen)]
		off += int(keyLen)
		if off+4 > len(body) {
			return nil, nil, fmt.Errorf("mset body truncated at entry %d value length", i)
		}
		valLen := binary.LittleEndian.Uint32(body[off:])
		off += 4
		if off+int(valLen) > len(body) {
			return nil, nil, fmt.Errorf("mset body value %d truncated", i)
		}
		values[i] = body[off : off+int(valLen)]
		off += int(valLen)
	}
	return keys, values, nil
}

// EncodeMultiValueResponse encodes multiple values for RespMultiValue.
// Format: count(4) + [found(1) + valueLen(4) + value]*
func EncodeMultiValueResponse(values [][]byte, founds []bool) []byte {
	size := 4
	for i := range values {
		size += 1
		if founds[i] {
			size += 4 + len(values[i])
		}
	}
	body := make([]byte, size)
	binary.LittleEndian.PutUint32(body[0:4], uint32(len(values)))
	off := 4
	for i := range values {
		if founds[i] {
			body[off] = 1
			off++
			binary.LittleEndian.PutUint32(body[off:], uint32(len(values[i])))
			off += 4
			copy(body[off:], values[i])
			off += len(values[i])
		} else {
			body[off] = 0
			off++
		}
	}
	return body
}

// EncodeMDelBody encodes multiple keys for MDEL.
// Format: count(4) + [keyLen(2) + key]*
// (Same wire format as MGet keys.)
func EncodeMDelBody(keys [][]byte) []byte {
	return EncodeMGetBody(keys)
}

// DecodeMDelBody decodes multiple keys for MDEL.
func DecodeMDelBody(body []byte) ([][]byte, error) {
	return DecodeMGetBody(body)
}

// EncodeKeysBody encodes a key pattern for KEYS scan.
// Format: patternLen(2) + pattern
func EncodeKeysBody(pattern []byte) []byte {
	body := make([]byte, 2+len(pattern))
	binary.LittleEndian.PutUint16(body[0:2], uint16(len(pattern)))
	copy(body[2:], pattern)
	return body
}

// DecodeKeysBody decodes a key pattern for KEYS scan.
func DecodeKeysBody(body []byte) ([]byte, error) {
	if len(body) < 2 {
		return nil, fmt.Errorf("keys body too short")
	}
	patLen := binary.LittleEndian.Uint16(body[0:2])
	if int(patLen)+2 > len(body) {
		return nil, fmt.Errorf("keys body truncated")
	}
	return body[2 : 2+patLen], nil
}

// EncodeKeysResponse encodes a list of keys for KEYS response.
// Format: count(4) + [keyLen(2) + key]*
func EncodeKeysResponse(keys [][]byte) []byte {
	return EncodeMGetBody(keys)
}

// DecodeKeysResponse decodes a list of keys from KEYS response.
func DecodeKeysResponse(body []byte) ([][]byte, error) {
	return DecodeMGetBody(body)
}

// EncodeMTestResponse encodes batch existence results.
// Format: count(4) + [found(1)]*
func EncodeMTestResponse(founds []bool) []byte {
	body := make([]byte, 4+len(founds))
	binary.LittleEndian.PutUint32(body[0:4], uint32(len(founds)))
	for i, f := range founds {
		if f {
			body[4+i] = 1
		}
	}
	return body
}

// DecodeMTestResponse decodes batch existence results.
func DecodeMTestResponse(body []byte) ([]bool, error) {
	if len(body) < 4 {
		return nil, fmt.Errorf("mtest response too short")
	}
	count := binary.LittleEndian.Uint32(body[0:4])
	if uint32(len(body)-4) < count {
		return nil, fmt.Errorf("mtest response truncated")
	}
	founds := make([]bool, count)
	for i := uint32(0); i < count; i++ {
		founds[i] = body[4+i] == 1
	}
	return founds, nil
}

// DecodeMultiValueResponse decodes multiple values.
func DecodeMultiValueResponse(body []byte) (values [][]byte, founds []bool, err error) {
	if len(body) < 4 {
		return nil, nil, fmt.Errorf("multi value response too short")
	}
	count := binary.LittleEndian.Uint32(body[0:4])
	if count > MaxBatchCount {
		return nil, nil, fmt.Errorf("multi value count %d exceeds max %d", count, MaxBatchCount)
	}
	// Physical bound: each entry needs at least 1 byte (found flag)
	if count > uint32(len(body)-4) {
		return nil, nil, fmt.Errorf("multi value count %d impossible for body size %d", count, len(body))
	}
	off := 4
	values = make([][]byte, count)
	founds = make([]bool, count)
	for i := uint32(0); i < count; i++ {
		if off >= len(body) {
			return nil, nil, fmt.Errorf("multi value response truncated at entry %d", i)
		}
		if body[off] == 0 {
			off++
			continue
		}
		off++ // skip found byte
		if off+4 > len(body) {
			return nil, nil, fmt.Errorf("multi value response value %d length truncated", i)
		}
		valLen := binary.LittleEndian.Uint32(body[off:])
		off += 4
		if off+int(valLen) > len(body) {
			return nil, nil, fmt.Errorf("multi value response value %d truncated", i)
		}
		values[i] = body[off : off+int(valLen)]
		founds[i] = true
		off += int(valLen)
	}
	return values, founds, nil
}

// EncodeMSetResultResponse encodes per-key MSET status bytes.
// Format: count(4) + [status(1)]*
func EncodeMSetResultResponse(statuses []byte) []byte {
	body := make([]byte, 4+len(statuses))
	binary.LittleEndian.PutUint32(body[0:4], uint32(len(statuses)))
	copy(body[4:], statuses)
	return body
}

// DecodeMSetResultResponse decodes per-key MSET status bytes.
func DecodeMSetResultResponse(body []byte) ([]byte, error) {
	if len(body) < 4 {
		return nil, fmt.Errorf("mset result body too short: %d", len(body))
	}
	count := int(binary.LittleEndian.Uint32(body[0:4]))
	if len(body) < 4+count {
		return nil, fmt.Errorf("mset result truncated: have %d, need %d", len(body), 4+count)
	}
	return body[4 : 4+count], nil
}
