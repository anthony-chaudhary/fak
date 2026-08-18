// Package strictjson decodes exactly one JSON value while rejecting unknown fields.
package strictjson

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// Decode rejects unknown fields and any second JSON value. trailingMessage describes a nil-error second decode.
func Decode(raw []byte, destination any, trailingMessage string) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return fmt.Errorf("%s", trailingMessage)
		}
		return err
	}
	return nil
}
