package sessionreg

import (
	"bytes"
	"encoding/json"
	"os/exec"
	"path/filepath"
	"testing"
)

type regexCase struct {
	name, text string
	want       bool
}

// The non-ASCII runes probe Python's Unicode \b and \s: letters and digits beside a token
// (é, ü, ٣), the Kelvin sign and long s that (?i) folds to ASCII, and NBSP, ideographic
// space, zero-width space and line separator around \s.
var regexCases = []regexCase{
	{"apierr", "Fehler é529 beim Laden", false},
	{"apierr", "Fehler 529é beim Laden", false},
	{"apierr", "Fehler 529 beim Laden", true},
	{"apierr", "status 529: overloaded", true},
	{"apierr", "über·overloaded·now", true},
	{"apierr", "löverloaded", false},
	{"apierr", "x_529", false},
	{"apierr", "٣529", false},
	{"done", "Task complete.", true},
	{"done", "Tâche complèteé", false},
	{"done", "incomplete", false},
	{"done", "completeK", false},
	{"done", "Kompleted", false},
	{"done", "all checKs", true},
	{"done", "all ſet", true},
	{"auton", "<command-name>/loopé</command-name>", false},
	{"auton", "<command-name>/loop</command-name>", true},
	{"superv", "/dispatch-loopö", false},
	{"taskcmd", "<command-name>\v/goal</command-name>", true},
	{"taskcmd", "<command-name> 　/goal x", true},
	{"taskcmd", "<command-name>​/goal x", false},
	{"wrapper", " Caveat: harness text", true},
	{"wrapper", "​Caveat: harness text", false},
	{"userclose", "<command-name>/exitß", false},
	{"userclose", "<command-name>/exit</command-name>", true},
	{"userclose", "<command-name>/ſo", false},
	{"userclose", "<command-name>/logout", true},
	{"httpstatus", "Fehler é529 beim Laden", false},
	{"httpstatus", "Code ü503ö", false},
	{"httpstatus", "Fehler 529 beim Laden", true},
	{"httpstatus", "x_429", false},
	{"operatorstop", "HTTP 409 operator control", true},
	{"operatorstop", "HTTP 409é operator control", false},
	{"operatorstop", "409 refused: operator　stop", true},
	{"operatorstop", "budget_context_exhausted", true},
}

var namedRegexps = map[string]pyRegexp{
	"apierr": apiErrRE, "done": doneRE, "auton": autonRE, "superv": supervRE,
	"taskcmd": taskCmdRE, "wrapper": wrapperRE, "userclose": userCloseRE,
	"httpstatus": httpStatusRE, "operatorstop": operatorStopRE,
}

func TestPyRegexpSemantics(t *testing.T) {
	for _, c := range regexCases {
		if got := namedRegexps[c.name].MatchString(c.text); got != c.want {
			t.Errorf("%s.MatchString(%q) = %v, want %v", c.name, c.text, got, c.want)
		}
	}
}

// Groups come from the original text, not from the folded copy the match ran on, even after
// multi-byte runes and invalid UTF-8.
func TestPyRegexpSubmatchSlicesOriginal(t *testing.T) {
	if got := httpStatus("Fehlerü \xff— Code 503"); got != "503" {
		t.Errorf("httpStatus = %q, want 503", got)
	}
	if got := httpStatus("Fehler é529 beim Laden"); got != "" {
		t.Errorf("httpStatus = %q, want none: é is a word rune beside 529", got)
	}
	re := mustCompilePy(`(?i)\b(code) (\d+)\b`)
	if got := re.FindStringSubmatch("Prüfung — CODE 401 ö"); len(got) != 3 || got[0] != "CODE 401" || got[1] != "CODE" || got[2] != "401" {
		t.Errorf("FindStringSubmatch = %q, want [CODE 401, CODE, 401]", got)
	}
	if got := re.FindStringSubmatch("Prüfung codeé 401"); got != nil {
		t.Errorf("FindStringSubmatch = %q, want no match across a word rune", got)
	}
}

// TestPyRegexpParityWithPython asks fleet_sessions' own compiled patterns about the same
// texts, so the table above is pinned to the oracle rather than to a reading of re's docs.
func TestPyRegexpParityWithPython(t *testing.T) {
	py, args := pythonExe()
	if py == "" {
		t.Skip("python not on PATH; the Python oracle cannot run")
	}
	pairs := make([][2]string, len(regexCases))
	for i, c := range regexCases {
		pairs[i] = [2]string{c.name, c.text}
	}
	in, err := json.Marshal(pairs)
	if err != nil {
		t.Fatal(err)
	}
	tools, _ := filepath.Abs(filepath.Join("..", "..", "..", "tools"))
	cmd := exec.Command(py, append(args, filepath.Join("testdata", "pyregex_oracle.py"), tools)...)
	cmd.Stdin = bytes.NewReader(in)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("oracle: %v\n%s", err, stderr.String())
	}
	var got []bool
	if err := json.Unmarshal(out, &got); err != nil || len(got) != len(regexCases) {
		t.Fatalf("oracle output %q: %v", out, err)
	}
	for i, c := range regexCases {
		if got[i] != c.want {
			t.Errorf("python %s.search(%q) = %v, the Go table says %v", c.name, c.text, got[i], c.want)
		}
	}
}

func TestParseRecordAcceptsPythonJSONConstants(t *testing.T) {
	o := parseRecord(`{"type":"assistant","cost":NaN,"m":[Infinity,-Infinity],"s":"NaN \"Infinity\""}`)
	if o == nil || o["type"] != "assistant" || o["s"] != `NaN "Infinity"` || o["cost"] != nil {
		t.Fatalf("parseRecord = %#v, want the record with the constants read as null", o)
	}
	if parseRecord(`{"type":"assistant",}`) != nil || parseRecord(`[1,2,3]`) != nil {
		t.Fatal("parseRecord must still reject malformed JSON and non-object lines")
	}
}
