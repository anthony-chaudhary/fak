// JSON-LD parsing and shape helpers.
// Split verbatim from seoaeoscore.go along concern seams to hold the god-file ceiling (#3022).
package seoaeoscore

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

// ---------------------------------------------------------------------------
// JSON-LD helpers.
// ---------------------------------------------------------------------------

func sortedMapKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func collectJSONLDTypes(data any) map[string]bool {
	out := map[string]bool{}
	var walk func(any)
	walk = func(d any) {
		switch v := d.(type) {
		case map[string]any:
			switch t := v["@type"].(type) {
			case string:
				out[t] = true
			case []any:
				for _, x := range t {
					if s, ok := x.(string); ok {
						out[s] = true
					}
				}
			}
			for _, k := range sortedMapKeys(v) {
				walk(v[k])
			}
		case []any:
			for _, x := range v {
				walk(x)
			}
		}
	}
	walk(data)
	return out
}

func jsonldHasType(obj map[string]any, typ string) bool {
	switch t := obj["@type"].(type) {
	case string:
		return t == typ
	case []any:
		for _, x := range t {
			if s, ok := x.(string); ok && s == typ {
				return true
			}
		}
	}
	return false
}

func iterJSONLDObjects(data any) []map[string]any {
	var out []map[string]any
	var walk func(any)
	walk = func(d any) {
		switch v := d.(type) {
		case map[string]any:
			out = append(out, v)
			for _, k := range sortedMapKeys(v) {
				walk(v[k])
			}
		case []any:
			for _, x := range v {
				walk(x)
			}
		}
	}
	walk(data)
	return out
}

func jsonldObjectsWithType(values []any, typ string) []map[string]any {
	var out []map[string]any
	for _, data := range values {
		for _, obj := range iterJSONLDObjects(data) {
			if jsonldHasType(obj, typ) {
				out = append(out, obj)
			}
		}
	}
	return out
}

func jsonldURL(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case map[string]any:
		for _, key := range []string{"@id", "url", "item"} {
			if s, ok := v[key].(string); ok {
				return s
			}
		}
	}
	return ""
}

func numEquals(v any, want int) bool {
	switch n := v.(type) {
	case float64:
		return n == float64(want)
	case int:
		return n == want
	default:
		return false
	}
}

func breadcrumbShapeOK(values []any) (bool, string) {
	for _, bc := range jsonldObjectsWithType(values, "BreadcrumbList") {
		items, ok := bc["itemListElement"].([]any)
		if !ok || len(items) < 2 {
			continue
		}
		good := true
		for i, itAny := range items {
			item, ok := itAny.(map[string]any)
			if !ok || !jsonldHasType(item, "ListItem") {
				good = false
				break
			}
			if !numEquals(item["position"], i+1) {
				good = false
				break
			}
			name, _ := item["name"].(string)
			url := jsonldURL(item["item"])
			if strings.TrimSpace(name) == "" || !(strings.HasPrefix(url, "https://") || strings.HasPrefix(url, "http://")) {
				good = false
				break
			}
		}
		if good {
			return true, fmt.Sprintf("BreadcrumbList has %d ordered absolute ListItem entries", len(items))
		}
	}
	return false, "BreadcrumbList JSON-LD missing or structurally invalid"
}

func faqJSONLDSyncOK(values []any, faqText string) (bool, string) {
	var visible []string
	for _, h := range reH2.FindAllStringSubmatch(faqText, -1) {
		if isQuestion(h[1]) {
			visible = append(visible, strings.TrimSpace(h[1]))
		}
	}
	var questions []string
	answersOK := true
	for _, faq := range jsonldObjectsWithType(values, "FAQPage") {
		entities, ok := faq["mainEntity"].([]any)
		if !ok {
			continue
		}
		for _, entAny := range entities {
			ent, ok := entAny.(map[string]any)
			if !ok || !jsonldHasType(ent, "Question") {
				answersOK = false
				continue
			}
			if q, ok := ent["name"].(string); ok {
				questions = append(questions, strings.TrimSpace(q))
			}
			ans, ok := ent["acceptedAnswer"].(map[string]any)
			if ok {
				text, ok := ans["text"].(string)
				if !ok || utf8.RuneCountInString(strings.TrimSpace(text)) < 20 {
					answersOK = false
				}
			} else {
				answersOK = false
			}
		}
	}
	if len(visible) > 0 && strSliceEqual(questions, visible) && answersOK {
		return true, fmt.Sprintf("FAQPage JSON-LD mirrors %d visible FAQ questions", len(visible))
	}
	return false, fmt.Sprintf("FAQPage JSON-LD stale or incomplete (%d schema questions vs %d visible questions)", len(questions), len(visible))
}

func strSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
