package aeogen

import (
	"fmt"
	"strconv"
	"strings"
)

// jsonObj is an insertion-ordered JSON object, so output key order matches a Python dict.
type jsonObj []jsonKV

type jsonKV struct {
	K string
	V any
}

// pyJSONDumps is json.dumps(v, ensure_ascii=False, indent=2). encoding/json cannot be
// used: it reorders map keys and escapes <, >, &, U+2028 and U+2029, which Python does not.
func pyJSONDumps(v any) string {
	var sb strings.Builder
	writePyJSON(&sb, v, 0)
	return sb.String()
}

func writePyJSON(sb *strings.Builder, v any, depth int) {
	pad := func(d int) { sb.WriteString("\n" + strings.Repeat("  ", d)) }
	switch x := v.(type) {
	case string:
		writePyJSONString(sb, x)
	case int:
		sb.WriteString(strconv.Itoa(x))
	case []any:
		if len(x) == 0 {
			sb.WriteString("[]")
			return
		}
		sb.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				sb.WriteByte(',')
			}
			pad(depth + 1)
			writePyJSON(sb, e, depth+1)
		}
		pad(depth)
		sb.WriteByte(']')
	case jsonObj:
		if len(x) == 0 {
			sb.WriteString("{}")
			return
		}
		sb.WriteByte('{')
		for i, kv := range x {
			if i > 0 {
				sb.WriteByte(',')
			}
			pad(depth + 1)
			writePyJSONString(sb, kv.K)
			sb.WriteString(": ")
			writePyJSON(sb, kv.V, depth+1)
		}
		pad(depth)
		sb.WriteByte('}')
	default:
		panic(fmt.Sprintf("aeogen: unsupported JSON value %T", v))
	}
}

func writePyJSONString(sb *strings.Builder, s string) {
	sb.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			sb.WriteString(`\"`)
		case '\\':
			sb.WriteString(`\\`)
		case '\n':
			sb.WriteString(`\n`)
		case '\r':
			sb.WriteString(`\r`)
		case '\t':
			sb.WriteString(`\t`)
		case '\b':
			sb.WriteString(`\b`)
		case '\f':
			sb.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(sb, `\u%04x`, r)
			} else {
				sb.WriteRune(r)
			}
		}
	}
	sb.WriteByte('"')
}
