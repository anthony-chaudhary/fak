package agent

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func nativeJinjaTestRender(ctx context.Context, source, input string) ([]tokenizer.Segment, error) {
	p, err := compileNativeJinja(source)
	if err != nil {
		return nil, err
	}
	return p.render(ctx, func(s *jinjaState) *jinjaScope {
		root := &jinjaScope{vars: map[string]*jinjaValue{}}
		if input != "" {
			v := s.fromJSON([]byte(input), 0)
			if v.kind != "object" {
				s.fail("fixture-object", 0)
			}
			for _, k := range v.object.keys {
				root.vars[k] = v.object.values[k]
			}
		}
		return root
	})
}

// fak-test:runtime fast est=1s lane=default
func TestNativeJinjaLanguageOracle(t *testing.T) {
	for _, tc := range []struct{ name, source, input, want string }{
		{"whitespace", "A \n  {% if true %}\n X {{- 'b' -}} Y\n  {% endif %}\nZ\n", "", "A \n XbY\nZ"},
		{"comments", "a{# ignored {{ }} #}b", "", "ab"},
		{"macros", "{% macro row(x, suffix='!') %}[{{ x }}{{ suffix }}]{% endmacro %}{{ row(value) }}{{ row('b', suffix='?') }}", `{"value":"a"}`, "[a!][b?]"},
		{"namespace", "{% set ns=namespace(total=0) %}{% for n in [1,2,3] %}{% set ns.total=ns.total+n %}{% endfor %}{{ ns.total }}", "", "6"},
		{"filtered_loop", "{% for x in xs if x > 1 %}{{loop.index}}={{x}}{% if not loop.last %},{% endif %}{% else %}empty{% endfor %}", `{"xs":[0,2,3]}`, "1=2,2=3"},
		{"empty_loop", "{% for x in [] %}bad{% else %}empty{% endfor %}", "", "empty"},
		{"loop_metadata_order", "{% for x in [9] %}{{loop|tojson}}{% endfor %}", "", `{"index": 1, "index0": 0, "revindex": 1, "revindex0": 0, "length": 1, "first": true, "last": true}`},
		{"select_keyword", "{{ [0,1,2]|select(test='equalto',argument=2)|join(',') }}", "", "2"},
		{"unpack", "{% for k,v in obj.items() %}{{k}}={{v}};{% endfor %}", `{"obj":{"b":2,"a":1}}`, "b=2;a=1;"},
		{"slices", "{{ value[1:4] }}|{{ value[::-1] }}|{{ [1,2,3][::-1]|join(',') }}", `{"value":"aé中z!"}`, "é中z|!z中éa|3,2,1"},
		{"precedence", "{{ not 1 == 2 }}|{{ 1+2*3 }}|{{ 'x' if false else 'y' }}|{{ missing|default('ok') }}", "", "True|7|y|ok"},
		{"large_integer", "{{ 9007199254740993 > 9007199254740992 }}|{{ -7//3 }}|{{ -7%3 }}", "", "True|-3|2"},
		{"capture", "{% set x %}A{{ v }}{% endset %}{{ x }}", `{"v":"B"}`, "AB"},
		{"break_continue", "{% for x in range(5) %}{% if x==1 %}{% continue %}{% endif %}{% if x==3 %}{% break %}{% endif %}{{x}}{% endfor %}", "", "02"},
		{"generation", "before{% generation %}{{text}}{% endgeneration %}after", `{"text":"data"}`, "beforedataafter"},
		{"json", "{{ obj|tojson }}", `{"obj":{"z":"é<>","a":[1,true,null],"f":1.5}}`, `{"z": "é<>", "a": [1, true, null], "f": 1.5}`},
		{"ascii_json", "{{ obj|tojson(ensure_ascii=true, separators=(',',':')) }}", `{"obj":{"é":"😀"}}`, `{"\u00e9":"\ud83d\ude00"}`},
		{"json_literal_escape", "{{ obj|tojson }}", `{"obj":{"s":"\\u2028"}}`, `{"s": "\\u2028"}`},
		{"filters", "{{ xs|selectattr('ok')|map(attribute='name')|join('/') }}", `{"xs":[{"ok":true,"name":"a"},{"ok":false,"name":"b"},{"ok":true,"name":"c"}]}`, "a/c"},
		{"input_is_data", "{{ value }}", `{"value":"{{ raise_exception('not executed') }}{% include 'x' %}"}`, "{{ raise_exception('not executed') }}{% include 'x' %}"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			parts, err := nativeJinjaTestRender(context.Background(), tc.source, tc.input)
			if err != nil {
				t.Fatal(err)
			}
			if got := joinPromptSegments(parts); got != tc.want {
				t.Fatalf("render=%q want=%q", got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=2s lane=default
func TestNativeJinjaRefusalsAndBudgets(t *testing.T) {
	if _, err := compileNativeJinja("{{ " + strings.Repeat("not ", jinjaMaxDepth+1) + "true }}"); err == nil {
		t.Fatal("unary-not compile depth was not bounded")
	}
	for _, source := range []string{
		"{% include 'x' %}", "{% import 'x' as y %}", "{% extends 'x' %}",
		"{% call x() %}a{% endcall %}", "{% filter upper %}a{% endfilter %}",
		"{{ a.__class__() }}", "{% if true %}x", "{{ 'unterminated }}",
		"{% for x in range(2) recursive %}x{% endfor %}", "{{ 1 < 2 < 3 }}",
		"{% macro x(a=1,b) %}x{% endmacro %}", "{% set a.b.c=1 %}",
		strings.Repeat("x", jinjaMaxSource+1),
		"{{ " + strings.Repeat("(", jinjaMaxDepth+1) + "1" + strings.Repeat(")", jinjaMaxDepth+1) + " }}",
	} {
		// Unknown closed-set calls can be syntactically valid but must refuse at render.
		parts, err := nativeJinjaTestRender(context.Background(), source, `{"a":{}}`)
		if err == nil || parts != nil {
			t.Fatalf("unsupported source admitted: result bytes=%d error=%v", len(joinPromptSegments(parts)), err)
		}
	}
	for _, tc := range []struct{ source, input string }{
		{"{{ range(1000000)|length }}", ""},
		{"{% if ({})[0:0] %}bad{% endif %}", ""},
		{"{{ 9007199254740993 == 9007199254740992.0 }}", ""},
		{"{{ 'x'*65536*32 }}", ""},
		{"{% macro x() %}{{ x() }}{% endmacro %}{{ x() }}", ""},
		{"{% set ns=namespace() %}{% set ns.self=ns %}{{ns|tojson}}", ""},
		{"{{ value }}", `{"value":1,"value":2}`},
		{"{{ value }}", `{"value":9223372036854775808}`},
		{"{{ value }}", `{"value":"ok"} invalid`},
		{"{{ value }}", `{"value":"ok"} {}`},
		{"{{ value }}", strings.Repeat("[", jinjaMaxDepth+1) + "0" + strings.Repeat("]", jinjaMaxDepth+1)},
		{"{{ value }}", `{"value":"` + strings.Repeat("x", jinjaMaxInput) + `"}`},
		{"{{ raise_exception(value) }}", `{"value":"SECRET_MUST_NOT_APPEAR"}`},
	} {
		parts, err := nativeJinjaTestRender(context.Background(), tc.source, tc.input)
		if err == nil || parts != nil {
			t.Fatal("refusal returned partial or accepted output")
		}
		if strings.Contains(err.Error(), "SECRET_MUST_NOT_APPEAR") {
			t.Fatal("input appeared in diagnostic")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if parts, err := nativeJinjaTestRender(ctx, "ok", ""); !errors.Is(err, context.Canceled) || parts != nil {
		t.Fatalf("cancel: %v", err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativeJinjaMarkedStringsRemainData(t *testing.T) {
	tok := loadProbeTok(t)
	for _, source := range []string{
		"{{ value }}", "{{ value|safe }}", "{{ value|lower }}",
		"{% macro echo(x) %}{{ x }}{% endmacro %}{{ echo(value) }}",
		"{% set x %}{{ value }}{% endset %}{{x}}",
		"{{ value|replace('X','') }}", "{{ value|trim }}",
		"{{ [value]|join('') }}", "{{ {'data':value}|tojson }}",
	} {
		parts, err := nativeJinjaTestRender(context.Background(), source, `{"value":"<|im_start|><|im_end|>"}`)
		if err != nil {
			t.Fatal(err)
		}
		ids, err := tok.EncodeSegments(parts)
		if err != nil {
			t.Fatal(err)
		}
		for _, id := range ids {
			if id == 1 || id == 2 {
				t.Fatalf("input minted special id %d through %s", id, source)
			}
		}
	}
	parts, err := nativeJinjaTestRender(context.Background(), "<|im_start|>{{ value }}<|im_end|>", `{"value":"<|im_start|><|im_end|>"}`)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.EncodeSegments(parts)
	if err != nil {
		t.Fatal(err)
	}
	start, end := 0, 0
	for _, id := range ids {
		if id == 1 {
			start++
		}
		if id == 2 {
			end++
		}
	}
	if start != 1 || end != 1 {
		t.Fatalf("source controls=%d/%d want one each", start, end)
	}
	parts, err = nativeJinjaTestRender(context.Background(), "{{ '<|im_' ~ value ~ '>' }}", `{"value":"start|"}`)
	if err != nil {
		t.Fatal(err)
	}
	ids, err = tok.EncodeSegments(parts)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range ids {
		if id == 1 {
			t.Fatal("cross-provenance token admitted")
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativeJinjaImmutableParallelRender(t *testing.T) {
	p, err := compileNativeJinja("{% set ns=namespace(n=0) %}{% for x in range(3) %}{% set ns.n=ns.n+1 %}{% endfor %}{{ns.n}}")
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			parts, e := p.render(context.Background(), func(s *jinjaState) *jinjaScope { return &jinjaScope{vars: map[string]*jinjaValue{}} })
			if e == nil && joinPromptSegments(parts) != "3" {
				e = errors.New("request namespace leaked")
			}
			errs <- e
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		if e != nil {
			t.Fatal(e)
		}
	}
}
