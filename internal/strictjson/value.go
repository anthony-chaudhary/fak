package strictjson

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// NumberValue decodes exactly one JSON value while preserving numbers.
func NumberValue(data []byte) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("trailing data after JSON value")
		}
		return nil, err
	}
	return value, nil
}
