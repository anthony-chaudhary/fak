package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/codexmcpdiag"
)

var readCodexMCPEvents = readCodexMCPEventsSQLite

func runDoctorCodexMCPWarning(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("doctor codex-mcp-warning", flag.ContinueOnError)
	fs.SetOutput(stderr)
	servers := fs.String("servers", "", "comma-separated server names from the Codex warning")
	db := fs.String("db", defaultCodexLogDB(), "Codex structured-log SQLite path")
	jsonOut := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	names := splitServerNames(*servers)
	if len(names) == 0 {
		fmt.Fprintln(stderr, "fak doctor codex-mcp-warning: --servers is required")
		return 2
	}
	events, err := readCodexMCPEvents(*db, names)
	rep := codexmcpdiag.Report{Verdict: codexmcpdiag.VerdictInsufficient, Servers: make([]codexmcpdiag.Server, 0, len(names))}
	if err == nil {
		rep = codexmcpdiag.Classify(names, events)
	} else {
		for _, n := range names {
			rep.Servers = append(rep.Servers, codexmcpdiag.Server{Name: n, Status: "missing"})
		}
	}
	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(rep)
		return 0
	}
	fmt.Fprintln(stdout, rep.Verdict)
	for _, s := range rep.Servers {
		fmt.Fprintf(stdout, "- %s: %s\n", s.Name, s.Status)
	}
	return 0
}
func splitServerNames(s string) []string {
	var out []string
	seen := map[string]bool{}
	for _, n := range strings.Split(s, ",") {
		n = strings.TrimSpace(n)
		if n != "" && !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	return out
}
func readCodexMCPEventsSQLite(path string, names []string) ([]codexmcpdiag.Event, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	script := `import json,sqlite3,sys
p=sys.argv[1]; names=json.loads(sys.argv[2]); uri='file:'+p.replace('\\','/')+'?mode=ro'
c=sqlite3.connect(uri,uri=True,timeout=5); c.execute('pragma query_only=on')
rows=c.execute("select level,target,coalesce(feedback_log_body,'') from logs order by id desc limit 50000").fetchall()
out=[]
for level,target,body in rows:
 text=(str(target)+' '+str(body)).lower()
 if any(n.lower() in text for n in names): out.append({'level':level,'target':target,'body':body})
print(json.dumps(out))`
	nb, _ := json.Marshal(names)
	cmd := exec.Command("python", "-c", script, filepath.Clean(path), string(nb))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("read log: %w", err)
	}
	var events []codexmcpdiag.Event
	if err := json.Unmarshal(stdout.Bytes(), &events); err != nil {
		return nil, errors.New("read log: invalid helper output")
	}
	return events, nil
}
