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
	typ := strings.Trim(string(top["type"]), `"`)
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
		if err := json.Unmarshal(rawIndex, &n); err == nil {
			index = n
		}
	}
	return responsesSSEOutputItem{index: index, raw: append([]byte(nil), item...)}, true, nil
}

func openAIResponsesSSEFillOutput(final []byte, outputItems []responsesSSEOutputItem) ([]byte, error) {
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
