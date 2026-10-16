package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

func openAIResponsesSSEFinalResponse(raw []byte) ([]byte, error) {
	var (
		event       string
		data        []string
		final       []byte
		firstError  []byte
		outputItems []responsesSSEOutputItem
	)
	flush := func() error {
		if len(data) == 0 {
			event = ""
			return nil
		}
		payload := strings.TrimSpace(strings.Join(data, "\n"))
		ev := strings.TrimSpace(event)
		event, data = "", nil
		if payload == "" || payload == "[DONE]" {
			return nil
		}
		payloadRaw := []byte(payload)
		item, hasItem, err := openAIResponsesSSEOutputItem(payloadRaw, ev, len(outputItems))
		if err != nil {
			return err
		}
		if hasItem {
			outputItems = append(outputItems, item)
		}
		resp, ok, err := openAIResponsesSSEPayloadResponse(payloadRaw, ev)
		if err != nil {
			return err
		}
		if ok {
			var envelope struct {
				Error json.RawMessage `json:"error"`
			}
			if json.Unmarshal(resp, &envelope) == nil && len(envelope.Error) > 0 &&
				!bytes.Equal(bytes.TrimSpace(envelope.Error), []byte("null")) && firstError == nil {
				// Keep the first provider failure even if a later event claims success.
				// Returning its envelope preserves the adapter's existing API error path.
				firstError = resp
			}
			final = resp
		}
		return nil
	}
	for _, lineb := range bytes.Split(raw, []byte{'\n'}) {
		line := strings.TrimRight(string(lineb), "\r")
		switch {
		case line == "":
			if err := flush(); err != nil {
				return nil, err
			}
		case strings.HasPrefix(line, ":"):
			continue
		case strings.HasPrefix(line, "event:"):
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			data = append(data, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := flush(); err != nil {
		return nil, err
	}
	if firstError != nil {
		return firstError, nil
	}
	if len(final) > 0 {
		return openAIResponsesSSEFillOutput(final, outputItems)
	}
	return nil, fmt.Errorf("no response.completed payload (body: %s)", truncate(raw, 200))
}

func openAIResponsesSSEPayloadResponse(raw []byte, event string) ([]byte, bool, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, false, fmt.Errorf("decode SSE data: %w (data: %s)", err, truncate(raw, 200))
	}
	var typ string
	_ = json.Unmarshal(top["type"], &typ)
	for _, terminal := range []string{"response.failed", "response.incomplete"} {
		if event != terminal && typ != terminal {
			continue
		}
		// Reject before output reconstruction: even item-done tool arguments
		// cannot be released from an unsuccessful enclosing response.
		var response struct {
			Error             json.RawMessage `json:"error"`
			IncompleteDetails json.RawMessage `json:"incomplete_details"`
		}
		if err := json.Unmarshal(top["response"], &response); err != nil {
			return nil, false, fmt.Errorf("%s: decode response: %w", terminal, err)
		}
		return nil, false, fmt.Errorf("%s: error=%s incomplete_details=%s", terminal, response.Error, response.IncompleteDetails)
	}
	if apiError := top["error"]; len(apiError) > 0 && !bytes.Equal(bytes.TrimSpace(apiError), []byte("null")) {
		return append([]byte(nil), raw...), true, nil
	}
	if event == "error" || typ == "error" {
		// The Responses error event puts code/message/param at the top level.
		// Wrap it in the existing adapter envelope without releasing any output.
		wrapped := append([]byte(`{"error":`), raw...)
		wrapped = append(wrapped, '}')
		return wrapped, true, nil
	}
	if resp := top["response"]; len(resp) > 0 && (event == "response.completed" || typ == "response.completed") {
		return append([]byte(nil), resp...), true, nil
	}
	if _, ok := top["output"]; ok {
		return append([]byte(nil), raw...), true, nil
	}
	if _, ok := top["error"]; ok {
		return append([]byte(nil), raw...), true, nil
	}
	if event == "response.completed" || typ == "response.completed" {
		return nil, false, fmt.Errorf("response.completed event missing response (data: %s)", truncate(raw, 200))
	}
	return nil, false, nil
}

type responsesSSEOutputItem struct {
	index int
	raw   json.RawMessage
}

func openAIResponsesSSEOutputItem(raw []byte, event string, fallbackIndex int) (responsesSSEOutputItem, bool, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil {
		return responsesSSEOutputItem{}, false, fmt.Errorf("decode SSE data: %w (data: %s)", err, truncate(raw, 200))
	}
	typ := strings.Trim(string(top["type"]), `"`)
	if event != "response.output_item.done" && typ != "response.output_item.done" {
		return responsesSSEOutputItem{}, false, nil
	}
	item := top["item"]
	if len(item) == 0 {
		return responsesSSEOutputItem{}, false, fmt.Errorf("response.output_item.done missing item (data: %s)", truncate(raw, 200))
	}
	index := fallbackIndex
	if rawIndex := top["output_index"]; len(rawIndex) > 0 {
		var n int
		index = -1 // Defer malformed-index refusal until fragments are needed.
		if err := json.Unmarshal(rawIndex, &n); err == nil && !bytes.Equal(bytes.TrimSpace(rawIndex), []byte("null")) {
			index = n
		}
	}
	return responsesSSEOutputItem{index: index, raw: append([]byte(nil), item...)}, true, nil
}

func openAIResponsesSSEFillOutput(final []byte, outputItems []responsesSSEOutputItem) ([]byte, error) {
	if bytes.Equal(bytes.TrimSpace(final), []byte("null")) {
		return nil, fmt.Errorf("response.completed payload is null")
	}
	if len(outputItems) == 0 {
		return final, nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(final, &obj); err != nil {
		return nil, fmt.Errorf("decode final response: %w (body: %s)", err, truncate(final, 200))
	}
	if existing := obj["output"]; len(existing) > 0 {
		var out []json.RawMessage
		if err := json.Unmarshal(existing, &out); err == nil && len(out) > 0 {
			return final, nil
		}
	}
	// A populated final output is authoritative above. Only reconstructed
	// fragments need an unambiguous position; do not turn one repeated item into
	// separate tool actions when later normalization generates distinct call IDs.
	seen := make(map[int]bool, len(outputItems))
	for _, item := range outputItems {
		if item.index < 0 {
			return nil, fmt.Errorf("response.output_item.done invalid output_index")
		}
		if seen[item.index] {
			return nil, fmt.Errorf("response.output_item.done duplicate output_index %d", item.index)
		}
		seen[item.index] = true
	}
	sort.SliceStable(outputItems, func(i, j int) bool {
		return outputItems[i].index < outputItems[j].index
	})
	out := make([]json.RawMessage, 0, len(outputItems))
	for _, item := range outputItems {
		out = append(out, item.raw)
	}
	rawOut, err := json.Marshal(out)
	if err != nil {
		return nil, err
	}
	obj["output"] = rawOut
	filled, err := json.Marshal(obj)
	if err != nil {
		return nil, err
	}
	return filled, nil
}
