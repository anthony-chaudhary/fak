package quantdetect

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// Family identifies the manifest parser selected by the caller.
type Family string

const (
	FamilySafetensors Family = "safetensors-config"
	FamilyGGUF        Family = "gguf"
	FamilyRuntime     Family = "runtime-manifest"
)

// Status is an explicit detection outcome; unknown is not an error or fallback.
type Status string

const (
	StatusDetected      Status = "detected"
	StatusUnknown       Status = "unknown"
	StatusMalformed     Status = "malformed"
	StatusLimitExceeded Status = "limit-exceeded"
	StatusUnsupported   Status = "unsupported"
)

// ReasonCode is stable machine-readable detail.
type ReasonCode string

const (
	ReasonDetected          ReasonCode = "QUANT_DETECTED"
	ReasonNoMetadata        ReasonCode = "QUANT_METADATA_UNKNOWN"
	ReasonMalformed         ReasonCode = "QUANT_MANIFEST_MALFORMED"
	ReasonReadLimit         ReasonCode = "QUANT_READ_LIMIT_EXCEEDED"
	ReasonFamilyUnsupported ReasonCode = "QUANT_FAMILY_UNSUPPORTED"
)

// Metadata is a neutral detection record suitable for adaptation into a richer
// quantization descriptor. Detector never opens or reads referenced weight data.
type Metadata struct {
	Family    Family            `json:"family"`
	Scheme    string            `json:"scheme,omitempty"`
	Format    string            `json:"format,omitempty"`
	Version   string            `json:"version,omitempty"`
	Bits      int               `json:"bits,omitempty"`
	GroupSize int               `json:"group_size,omitempty"`
	Runtime   string            `json:"runtime,omitempty"`
	Raw       map[string]string `json:"raw,omitempty"`
}

// Result includes exactly how many bytes were consumed from the bounded input.
type Result struct {
	Status    Status     `json:"status"`
	Reason    ReasonCode `json:"reason"`
	BytesRead int        `json:"bytes_read"`
	Detail    string     `json:"detail,omitempty"`
	Metadata  Metadata   `json:"metadata"`
}

// Detect reads at most maxBytes+1 bytes to distinguish an exact-boundary input
// from overflow, then dispatches without reading referenced model weights.
func Detect(r io.Reader, family Family, maxBytes int64) Result {
	if maxBytes <= 0 {
		return Result{Status: StatusLimitExceeded, Reason: ReasonReadLimit, Detail: "max_bytes must be positive"}
	}
	limited := io.LimitReader(r, maxBytes+1)
	data, err := io.ReadAll(limited)
	if err != nil {
		return Result{Status: StatusMalformed, Reason: ReasonMalformed, Detail: err.Error(), BytesRead: len(data)}
	}
	result := Result{BytesRead: len(data), Metadata: Metadata{Family: family}}
	if int64(len(data)) > maxBytes {
		result.Status, result.Reason, result.Detail = StatusLimitExceeded, ReasonReadLimit, fmt.Sprintf("input exceeds %d bytes", maxBytes)
		return result
	}
	switch family {
	case FamilySafetensors:
		return detectJSON(data, family, result, []string{"quantization_config", "quant_method", "quantization", "bits", "group_size"})
	case FamilyRuntime:
		return detectJSON(data, family, result, []string{"quantization", "quant_method", "format", "bits", "group_size", "runtime"})
	case FamilyGGUF:
		return detectGGUF(data, result)
	default:
		result.Status, result.Reason, result.Detail = StatusUnsupported, ReasonFamilyUnsupported, string(family)
		return result
	}
}

func detectJSON(data []byte, family Family, result Result, keys []string) Result {
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		result.Status, result.Reason, result.Detail = StatusMalformed, ReasonMalformed, err.Error()
		return result
	}
	flat := make(map[string]any)
	flatten("", value, flat)
	pick := func(names ...string) any {
		for path, v := range flat {
			leaf := path[strings.LastIndex(path, ".")+1:]
			for _, n := range names {
				if leaf == n {
					return v
				}
			}
		}
		return nil
	}
	method := text(pick("quant_method", "quantization", "type", "scheme"))
	format := text(pick("format", "quant_format"))
	if method == "" && format == "" && pick(keys...) == nil {
		result.Status, result.Reason = StatusUnknown, ReasonNoMetadata
		return result
	}
	result.Metadata.Scheme = method
	result.Metadata.Format = format
	result.Metadata.Version = text(pick("version", "quantizer_version"))
	result.Metadata.Runtime = text(pick("runtime", "engine"))
	result.Metadata.Bits = integer(pick("bits", "wbits"))
	result.Metadata.GroupSize = integer(pick("group_size", "groupsize"))
	result.Metadata.Raw = map[string]string{}
	for p, v := range flat {
		leaf := p[strings.LastIndex(p, ".")+1:]
		for _, k := range keys {
			if leaf == k {
				result.Metadata.Raw[p] = text(v)
			}
		}
	}
	result.Status, result.Reason = StatusDetected, ReasonDetected
	return result
}

func flatten(prefix string, value any, out map[string]any) {
	switch typed := value.(type) {
	case map[string]any:
		for k, v := range typed {
			p := k
			if prefix != "" {
				p = prefix + "." + k
			}
			flatten(p, v, out)
		}
	default:
		out[prefix] = value
	}
}

func detectGGUF(data []byte, result Result) Result {
	r := bytes.NewReader(data)
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil || string(magic[:]) != "GGUF" {
		result.Status, result.Reason = StatusMalformed, ReasonMalformed
		result.Detail = "missing GGUF magic"
		return result
	}
	var version uint32
	var tensors, count uint64
	if binary.Read(r, binary.LittleEndian, &version) != nil || binary.Read(r, binary.LittleEndian, &tensors) != nil || binary.Read(r, binary.LittleEndian, &count) != nil {
		result.Status, result.Reason = StatusMalformed, ReasonMalformed
		result.Detail = "truncated GGUF header"
		return result
	}
	result.Metadata.Version = fmt.Sprint(version)
	result.Metadata.Format = "gguf"
	result.Metadata.Raw = map[string]string{}
	for i := uint64(0); i < count; i++ {
		key, err := readString(r)
		if err != nil {
			return malformedGGUF(result, err)
		}
		var typ uint32
		if binary.Read(r, binary.LittleEndian, &typ) != nil {
			return malformedGGUF(result, io.ErrUnexpectedEOF)
		}
		val, err := readGGUFValue(r, typ)
		if err != nil {
			return malformedGGUF(result, err)
		}
		if strings.Contains(key, "quant") || strings.Contains(key, "file_type") {
			result.Metadata.Raw[key] = val
			if result.Metadata.Scheme == "" {
				result.Metadata.Scheme = val
			}
		}
	}
	if len(result.Metadata.Raw) == 0 {
		result.Status, result.Reason = StatusUnknown, ReasonNoMetadata
		return result
	}
	result.Status, result.Reason = StatusDetected, ReasonDetected
	return result
}
func malformedGGUF(r Result, err error) Result {
	r.Status, r.Reason, r.Detail = StatusMalformed, ReasonMalformed, err.Error()
	return r
}
func readString(r *bytes.Reader) (string, error) {
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	if n > uint64(r.Len()) {
		return "", io.ErrUnexpectedEOF
	}
	b := make([]byte, n)
	_, err := io.ReadFull(r, b)
	return string(b), err
}
func readGGUFValue(r *bytes.Reader, typ uint32) (string, error) {
	switch typ {
	case 4:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return fmt.Sprint(v), err
	case 8:
		return readString(r)
	default:
		return "", errors.New("unsupported GGUF metadata value type")
	}
}
func text(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	default:
		b, _ := json.Marshal(x)
		return string(b)
	}
}
func integer(v any) int {
	switch x := v.(type) {
	case float64:
		return int(x)
	case json.Number:
		n, _ := x.Int64()
		return int(n)
	default:
		return 0
	}
}
