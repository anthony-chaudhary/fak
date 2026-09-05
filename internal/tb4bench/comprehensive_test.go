package tb4bench

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

func makeTask(id, cat, prompt, setup, oracle string, timeout, budget int) TaskManifest {
	h := sha256.Sum256([]byte(oracle))
	return TaskManifest{
		TaskID:                 id,
		Category:               cat,
		Prompt:                 prompt,
		EnvironmentImageDigest: "ghcr.io/fak/tb4-sandbox@sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
		SetupCommand:           setup,
		VerificationOracle:     oracle,
		VerificationOracleHash: "sha256:" + hex.EncodeToString(h[:]),
		TimeoutSeconds:         timeout,
		BudgetTurns:            budget,
	}
}

func buildComprehensiveTasks() []TaskManifest {
	return []TaskManifest{
		// -------------------------------------------------------------
		// Category: build (8 tasks >= 7)
		// -------------------------------------------------------------
		makeTask(
			"tb4-comp-01-build-cgo-flag",
			CategoryBuild,
			"Fix the compilation error in Makefile by adding the required -pthread flag to CFLAGS.",
			"echo -e 'all:\n\tgcc -o app main.c' > Makefile && echo -e '#include <pthread.h>\nint main(){return 0;}' > main.c",
			"#!/bin/bash\ngrep -q -- '-pthread' Makefile && make app\n",
			45, 5,
		),
		makeTask(
			"tb4-comp-02-build-cmake-target",
			CategoryBuild,
			"Add missing add_executable target 'engine' linking main.cpp in CMakeLists.txt.",
			"echo 'cmake_minimum_required(VERSION 3.10)\nproject(Engine)' > CMakeLists.txt && echo 'int main(){return 0;}' > main.cpp",
			"#!/bin/bash\ngrep -q 'add_executable(engine main.cpp)' CMakeLists.txt\n",
			40, 5,
		),
		makeTask(
			"tb4-comp-03-build-cargo-dep",
			CategoryBuild,
			"Add serde dependency with derive feature to Cargo.toml.",
			"echo -e '[package]\nname = \"demo\"\nversion = \"0.1.0\"\n[dependencies]' > Cargo.toml",
			"#!/bin/bash\ngrep -E -q 'serde.*features.*derive' Cargo.toml\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-04-build-go-ldflags",
			CategoryBuild,
			"Configure build.sh to pass -ldflags '-X main.version=1.0.0' during go build.",
			"echo 'go build -o server ./cmd/server' > build.sh",
			"#!/bin/bash\ngrep -q -- \"-X main.version=1.0.0\" build.sh\n",
			30, 5,
		),
		makeTask(
			"tb4-comp-05-build-dockerfile-multistage",
			CategoryBuild,
			"Update Dockerfile to copy the compiled binary 'app' from builder stage into /usr/local/bin/.",
			"echo -e 'FROM golang:1.22 AS builder\nRUN go build -o app .\nFROM alpine:latest' > Dockerfile",
			"#!/bin/bash\ngrep -q 'COPY --from=builder /app /usr/local/bin/app' Dockerfile || grep -q 'COPY --from=builder.*app /usr/local/bin/' Dockerfile\n",
			45, 6,
		),
		makeTask(
			"tb4-comp-06-build-npm-bundle",
			CategoryBuild,
			"Add build script 'esbuild src/index.ts --bundle --outfile=dist/bundle.js' to package.json.",
			"echo '{\"name\":\"app\",\"scripts\":{}}' > package.json && mkdir -p src && touch src/index.ts",
			"#!/bin/bash\ngrep -q '\"build\": \"esbuild src/index.ts --bundle --outfile=dist/bundle.js\"' package.json\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-07-build-rust-profile",
			CategoryBuild,
			"Configure Cargo.toml release profile with opt-level = 3 and lto = true.",
			"echo -e '[package]\nname=\"compute\"\nversion=\"0.1.0\"' > Cargo.toml",
			"#!/bin/bash\ngrep -q 'opt-level = 3' Cargo.toml && grep -q 'lto = true' Cargo.toml\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-08-build-mvn-java-version",
			CategoryBuild,
			"Set maven.compiler.source and maven.compiler.target properties to 17 in pom.xml.",
			"echo '<project><properties></properties></project>' > pom.xml",
			"#!/bin/bash\ngrep -q '<maven.compiler.source>17</maven.compiler.source>' pom.xml && grep -q '<maven.compiler.target>17</maven.compiler.target>' pom.xml\n",
			40, 5,
		),

		// -------------------------------------------------------------
		// Category: sysadmin (8 tasks >= 7)
		// -------------------------------------------------------------
		makeTask(
			"tb4-comp-09-sysadmin-nginx-conf",
			CategorySysadmin,
			"Update nginx.conf to proxy requests from /api/ to http://127.0.0.1:8080/.",
			"echo -e 'server {\n  listen 80;\n}' > nginx.conf",
			"#!/bin/bash\ngrep -q 'proxy_pass http://127.0.0.1:8080/;' nginx.conf\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-10-sysadmin-logrotate",
			CategorySysadmin,
			"Create a logrotate config in etc/logrotate.d/app for /var/log/app.log with daily rotation and rotate count 7.",
			"mkdir -p etc/logrotate.d",
			"#!/bin/bash\ntest -f etc/logrotate.d/app && grep -q 'daily' etc/logrotate.d/app && grep -q 'rotate 7' etc/logrotate.d/app\n",
			40, 6,
		),
		makeTask(
			"tb4-comp-11-sysadmin-systemd-service",
			CategorySysadmin,
			"Create a systemd unit file etc/systemd/system/agent.service with ExecStart=/usr/bin/agent and Restart=always.",
			"mkdir -p etc/systemd/system",
			"#!/bin/bash\ntest -f etc/systemd/system/agent.service && grep -q 'ExecStart=/usr/bin/agent' etc/systemd/system/agent.service && grep -q 'Restart=always' etc/systemd/system/agent.service\n",
			40, 5,
		),
		makeTask(
			"tb4-comp-12-sysadmin-cron-schedule",
			CategorySysadmin,
			"Write a crontab entry in cron.d/backup that runs /usr/local/bin/backup.sh at 02:30 AM every day.",
			"mkdir -p cron.d",
			"#!/bin/bash\ntest -f cron.d/backup && grep -E -q '30 2 \\* \\* \\* .*backup.sh' cron.d/backup\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-13-sysadmin-iptables-rule",
			CategorySysadmin,
			"Add iptables command to rules.sh to drop incoming TCP traffic on port 23 (telnet).",
			"echo '#!/bin/bash' > rules.sh && chmod +x rules.sh",
			"#!/bin/bash\ngrep -E -q 'iptables -A INPUT -p tcp --dport 23 -j DROP' rules.sh\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-14-sysadmin-sysctl-tuning",
			CategorySysadmin,
			"Configure sysctl.conf to set fs.file-max = 2097152 and net.core.somaxconn = 65535.",
			"touch sysctl.conf",
			"#!/bin/bash\ngrep -q 'fs.file-max = 2097152' sysctl.conf && grep -q 'net.core.somaxconn = 65535' sysctl.conf\n",
			30, 5,
		),
		makeTask(
			"tb4-comp-15-sysadmin-env-exporter",
			CategorySysadmin,
			"Write export_env.sh to parse .env.json and export each key-value pair as an environment variable.",
			"echo '{\"DATABASE_URL\":\"postgres://localhost/db\",\"PORT\":\"3000\"}' > .env.json",
			"#!/bin/bash\ntest -x export_env.sh && source ./export_env.sh && [ \"$DATABASE_URL\" = \"postgres://localhost/db\" ] && [ \"$PORT\" = \"3000\" ]\n",
			45, 6,
		),
		makeTask(
			"tb4-comp-16-sysadmin-disk-cleanup",
			CategorySysadmin,
			"Create cleanup.sh script that deletes all files older than 30 days ending in .tmp under /var/tmp.",
			"touch cleanup.sh && chmod +x cleanup.sh",
			"#!/bin/bash\ngrep -E -q 'find /var/tmp -name [\"'\"'\"*]\\.tmp.*-mtime \\+30.*-delete' cleanup.sh || grep -E -q 'find /var/tmp.*-mtime \\+30.*-name [\"'\"'\"*]\\.tmp.*-delete' cleanup.sh\n",
			40, 6,
		),

		// -------------------------------------------------------------
		// Category: git (8 tasks >= 7)
		// -------------------------------------------------------------
		makeTask(
			"tb4-comp-17-git-submodule-init",
			CategoryGit,
			"Add git submodule https://example.com/libs/common.git at path vendor/common in .gitmodules.",
			"touch .gitmodules",
			"#!/bin/bash\ngrep -q 'path = vendor/common' .gitmodules && grep -q 'url = https://example.com/libs/common.git' .gitmodules\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-18-git-commit-amend-signoff",
			CategoryGit,
			"Amend the last commit in the repository to include Signed-off-by: Developer <dev@example.com>.",
			"git init && git config user.name 'Dev' && git config user.email 'dev@example.com' && echo initial > README.md && git add README.md && git commit -m 'initial'",
			"#!/bin/bash\ngit log -1 --pretty=%B | grep -q 'Signed-off-by: Developer <dev@example.com>'\n",
			45, 6,
		),
		makeTask(
			"tb4-comp-19-git-ignore-pattern",
			CategoryGit,
			"Update .gitignore to ignore all .log files, the dist/ directory, and .env files while keeping .env.example.",
			"touch .gitignore",
			"#!/bin/bash\ngrep -q '^\\*\\.log$' .gitignore && grep -q '^dist/$' .gitignore && grep -q '^\\.env$' .gitignore && grep -q '^!\\.env\\.example$' .gitignore\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-20-git-tag-annotated",
			CategoryGit,
			"Create an annotated git tag 'v1.0.0' on HEAD with message 'Release 1.0.0'.",
			"git init && git config user.name 'Dev' && git config user.email 'dev@example.com' && echo 'code' > main.c && git add main.c && git commit -m 'feat: ready'",
			"#!/bin/bash\ngit tag -l -n1 v1.0.0 | grep -q 'Release 1.0.0'\n",
			40, 5,
		),
		makeTask(
			"tb4-comp-21-git-cherry-pick-conflict",
			CategoryGit,
			"Cherry-pick commit on branch 'patch-1' onto main and resolve conflict keeping both changes.",
			"git init && git config user.name 'Dev' && git config user.email 'dev@example.com' && echo 'base' > f.txt && git add f.txt && git commit -m 'base' && git checkout -b patch-1 && echo -e 'base\npatch' > f.txt && git commit -am 'patch' && git checkout main && echo -e 'base\nmain' > f.txt && git commit -am 'main change'",
			"#!/bin/bash\ngrep -q 'main' f.txt && grep -q 'patch' f.txt && ! grep -q '<<<<<<' f.txt\n",
			50, 7,
		),
		makeTask(
			"tb4-comp-22-git-hooks-pre-commit",
			CategoryGit,
			"Create an executable .git/hooks/pre-commit script that checks that gofmt returns no differences.",
			"git init && mkdir -p .git/hooks",
			"#!/bin/bash\ntest -x .git/hooks/pre-commit && grep -q 'gofmt' .git/hooks/pre-commit\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-23-git-revert-merge",
			CategoryGit,
			"Revert the merge commit HEAD with parent mainline 1.",
			"git init && git config user.name 'Dev' && git config user.email 'dev@example.com' && echo init > f && git add f && git commit -m init && git checkout -b b1 && echo b1 > f && git commit -am b1 && git checkout main && echo main > g && git add g && git commit -m main && git merge --no-ff b1 -m 'merge b1'",
			"#!/bin/bash\ngit log -1 --pretty=%B | grep -q 'Revert \"merge b1\"' || git log -1 --pretty=%B | grep -q 'Revert'\n",
			45, 6,
		),
		makeTask(
			"tb4-comp-24-git-branch-rename",
			CategoryGit,
			"Rename local branch 'master' to 'main' and verify the current branch is 'main'.",
			"git init -b master && git config user.name 'Dev' && git config user.email 'dev@example.com' && echo a > a && git add a && git commit -m a",
			"#!/bin/bash\n[ \"$(git branch --show-current)\" = \"main\" ]\n",
			30, 5,
		),

		// -------------------------------------------------------------
		// Category: refactor (8 tasks >= 7)
		// -------------------------------------------------------------
		makeTask(
			"tb4-comp-25-refactor-extract-function",
			CategoryRefactor,
			"Refactor process_data in process.py by extracting calculate_tax(amount, rate) helper function.",
			"echo -e 'def process_data(amount, rate):\n    tax = amount * rate\n    return amount + tax' > process.py",
			"#!/bin/bash\ngrep -q 'def calculate_tax(amount, rate):' process.py && python3 -c 'import process; assert process.calculate_tax(100, 0.05) == 5.0'\n",
			40, 5,
		),
		makeTask(
			"tb4-comp-26-refactor-rename-symbols",
			CategoryRefactor,
			"Rename variable 'old_val' to 'normalized_value' across stats.py.",
			"echo -e 'def normalize(xs):\n    old_val = sum(xs)\n    return [x / old_val for x in xs]' > stats.py",
			"#!/bin/bash\n! grep -q 'old_val' stats.py && grep -q 'normalized_value' stats.py\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-27-refactor-decompose-conditional",
			CategoryRefactor,
			"Refactor discount calculation in discount.py by replacing nested if with a guard clause returning early.",
			"echo -e 'def get_discount(user):\n    if user is not None:\n        if user.is_vip:\n            return 0.2\n    return 0.0' > discount.py",
			"#!/bin/bash\ngrep -q 'if not user' discount.py || grep -q 'if user is None' discount.py\n",
			40, 5,
		),
		makeTask(
			"tb4-comp-28-refactor-go-interface",
			CategoryRefactor,
			"Define a Reader interface in reader.go with Read(p []byte) (n int, err error) method.",
			"touch reader.go",
			"#!/bin/bash\ngrep -q 'type Reader interface' reader.go && grep -q 'Read(p \\[\\]byte) (n int, err error)' reader.go\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-29-refactor-inline-temp",
			CategoryRefactor,
			"Refactor order.py to inline temporary variable 'base_price' directly in the return statement.",
			"echo -e 'def total(qty, price):\n    base_price = qty * price\n    return base_price' > order.py",
			"#!/bin/bash\n! grep -q 'base_price' order.py && grep -q 'return qty \\* price' order.py\n",
			30, 5,
		),
		makeTask(
			"tb4-comp-30-refactor-singleton-to-di",
			CategoryRefactor,
			"Refactor Service class in service.py to accept db instance via __init__(self, db) instead of calling DB.get_instance().",
			"echo -e 'class Service:\n    def __init__(self):\n        self.db = DB.get_instance()' > service.py",
			"#!/bin/bash\ngrep -q 'def __init__(self, db):' service.py && grep -q 'self.db = db' service.py\n",
			40, 5,
		),
		makeTask(
			"tb4-comp-31-refactor-replace-loop-comprehension",
			CategoryRefactor,
			"Replace the imperative for loop in filter.py with a list comprehension: [x for x in data if x > 0].",
			"echo -e 'def positive_only(data):\n    res = []\n    for x in data:\n        if x > 0:\n            res.append(x)\n    return res' > filter.py",
			"#!/bin/bash\ngrep -q '\\[x for x in data if x > 0\\]' filter.py\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-32-refactor-move-package",
			CategoryRefactor,
			"Move math utilities from utils/math.go to pkg/math/math.go and update package declaration to 'package math'.",
			"mkdir -p utils && echo -e 'package utils\nfunc Add(a, b int) int { return a + b }' > utils/math.go",
			"#!/bin/bash\ntest -f pkg/math/math.go && ! test -f utils/math.go && grep -q '^package math$' pkg/math/math.go\n",
			40, 5,
		),

		// -------------------------------------------------------------
		// Category: debug (8 tasks >= 7)
		// -------------------------------------------------------------
		makeTask(
			"tb4-comp-33-debug-off-by-one",
			CategoryDebug,
			"Fix the off-by-one error in binary_search.py so that high is initialized to len(arr) - 1.",
			"echo -e 'def binary_search(arr, target):\n    low = 0\n    high = len(arr)\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1' > binary_search.py",
			"#!/bin/bash\ngrep -q 'high = len(arr) - 1' binary_search.py && python3 -c 'from binary_search import binary_search; assert binary_search([1,3,5], 5) == 2'\n",
			40, 5,
		),
		makeTask(
			"tb4-comp-34-debug-nil-pointer",
			CategoryDebug,
			"Fix nil pointer dereference in GetName method in user.go by checking if u == nil.",
			"echo -e 'package user\ntype User struct{ Name string }\nfunc (u *User) GetName() string {\n    return u.Name\n}' > user.go",
			"#!/bin/bash\ngrep -q 'if u == nil' user.go\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-35-debug-race-condition",
			CategoryDebug,
			"Fix data race in counter.go by protecting Count increments with sync.Mutex.",
			"echo -e 'package counter\ntype Counter struct{ n int }\nfunc (c *Counter) Inc() { c.n++ }' > counter.go",
			"#!/bin/bash\ngrep -q 'sync.Mutex' counter.go && grep -q 'Lock()' counter.go && grep -q 'Unlock()' counter.go\n",
			45, 6,
		),
		makeTask(
			"tb4-comp-36-debug-memory-leak-goroutine",
			CategoryDebug,
			"Fix goroutine leak in worker.go by passing a context.Context to the worker loop and exiting when ctx.Done() receives.",
			"echo -e 'package worker\nfunc Start() {\n    go func() {\n        for {\n            doWork()\n        }\n    }()\n}' > worker.go",
			"#!/bin/bash\ngrep -q 'ctx.Done()' worker.go\n",
			45, 6,
		),
		makeTask(
			"tb4-comp-37-debug-unclosed-file",
			CategoryDebug,
			"Fix unclosed file descriptor resource leak in read_config.py using with open(...) context manager.",
			"echo -e 'def load(path):\n    f = open(path, \"r\")\n    return f.read()' > read_config.py",
			"#!/bin/bash\ngrep -q 'with open(path' read_config.py\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-38-debug-division-by-zero",
			CategoryDebug,
			"Handle division by zero in average.py by returning 0.0 when input list is empty.",
			"echo -e 'def compute_average(items):\n    return sum(items) / len(items)' > average.py",
			"#!/bin/bash\npython3 -c 'import average; assert average.compute_average([]) == 0.0'\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-39-debug-json-unmarshal-type",
			CategoryDebug,
			"Fix unmarshal failure in payload.go: Port field should be int instead of string in JSON tag `json:\"port\"`.",
			"echo -e 'package payload\ntype Config struct {\n    Port string `json:\"port\"`\n}' > payload.go",
			"#!/bin/bash\ngrep -q 'Port *int *`json:\"port\"`' payload.go\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-40-debug-deadlock-lock-order",
			CategoryDebug,
			"Fix potential deadlock in transfer.go by ensuring mu1 and mu2 are always locked in consistent order based on account ID.",
			"echo -e 'package transfer\nfunc Transfer(a, b *Account) {\n    a.mu.Lock()\n    b.mu.Lock()\n}' > transfer.go",
			"#!/bin/bash\ngrep -q 'a.ID < b.ID' transfer.go || grep -q 'a.id < b.id' transfer.go\n",
			45, 6,
		),

		// -------------------------------------------------------------
		// Category: test (8 tasks >= 7)
		// -------------------------------------------------------------
		makeTask(
			"tb4-comp-41-test-table-driven-go",
			CategoryTest,
			"Implement table-driven test TestSum in sum_test.go covering positive numbers, negative numbers, and zero.",
			"echo -e 'package math\nfunc Sum(a, b int) int { return a + b }' > sum.go && touch sum_test.go",
			"#!/bin/bash\ngrep -q 'tests := \\[\\]struct' sum_test.go && grep -q 't.Run(' sum_test.go\n",
			40, 6,
		),
		makeTask(
			"tb4-comp-42-test-pytest-mock",
			CategoryTest,
			"Write test_fetch_user in test_service.py mocking requests.get using pytest-mock or unittest.mock.",
			"echo -e 'import requests\ndef fetch_user(url):\n    return requests.get(url).json()' > service.py && touch test_service.py",
			"#!/bin/bash\ngrep -q 'mock' test_service.py && grep -q 'def test_fetch_user' test_service.py\n",
			45, 6,
		),
		makeTask(
			"tb4-comp-43-test-benchmark-go",
			CategoryTest,
			"Add BenchmarkFib(b *testing.B) to fib_test.go with b.N loop.",
			"echo -e 'package fib\nfunc Fib(n int) int { if n < 2 { return n }; return Fib(n-1)+Fib(n-2) }' > fib.go && touch fib_test.go",
			"#!/bin/bash\ngrep -q 'func BenchmarkFib(b \\*testing.B)' fib_test.go && grep -q 'for i := 0; i < b.N; i++' fib_test.go\n",
			40, 5,
		),
		makeTask(
			"tb4-comp-44-test-fuzz-target-go",
			CategoryTest,
			"Add FuzzReverse(f *testing.F) fuzz test in reverse_test.go.",
			"echo -e 'package str\nfunc Reverse(s string) string { r := []rune(s); for i,j:=0,len(r)-1; i<j; i,j=i+1,j-1 { r[i],r[j]=r[j],r[i] }; return string(r) }' > reverse.go && touch reverse_test.go",
			"#!/bin/bash\ngrep -q 'func FuzzReverse(f \\*testing.F)' reverse_test.go && grep -q 'f.Fuzz(func(t \\*testing.T,' reverse_test.go\n",
			45, 6,
		),
		makeTask(
			"tb4-comp-45-test-coverage-threshold",
			CategoryTest,
			"Add coverage threshold check in test.sh asserting go test -cover output is at least 80%.",
			"echo 'go test ./...' > test.sh && chmod +x test.sh",
			"#!/bin/bash\ngrep -q 'cover' test.sh && grep -q '80' test.sh\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-46-test-jest-snapshot",
			CategoryTest,
			"Write component snapshot test in __tests__/button.test.js asserting toMatchSnapshot().",
			"mkdir -p __tests__ && touch __tests__/button.test.js",
			"#!/bin/bash\ngrep -q 'toMatchSnapshot()' __tests__/button.test.js\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-47-test-property-based-hypothesis",
			CategoryTest,
			"Write property-based test in test_sort.py using @given(st.lists(st.integers())) to verify sort idempotence.",
			"touch test_sort.py",
			"#!/bin/bash\ngrep -q '@given' test_sort.py && grep -q 'st.lists' test_sort.py\n",
			45, 6,
		),
		makeTask(
			"tb4-comp-48-test-integration-docker-compose",
			CategoryTest,
			"Configure docker-compose.test.yml with postgres service healthcheck and exit_code_from runner.",
			"touch docker-compose.test.yml",
			"#!/bin/bash\ngrep -q 'healthcheck:' docker-compose.test.yml && grep -q 'postgres' docker-compose.test.yml\n",
			40, 5,
		),

		// -------------------------------------------------------------
		// Category: security (8 tasks >= 8)
		// -------------------------------------------------------------
		makeTask(
			"tb4-comp-49-security-sql-injection",
			CategorySecurity,
			"Fix SQL injection vulnerability in user_repo.py by replacing string concatenation with parameterized query.",
			"echo -e 'def get_user(db, name):\n    query = f\"SELECT * FROM users WHERE name = {name}\"\n    return db.execute(query)' > user_repo.py",
			"#!/bin/bash\n! grep -q 'f\"SELECT' user_repo.py && (grep -q 'execute(.*?, *(name,)\\|.*?, *\\[name\\])' user_repo.py || grep -q 'execute(\"SELECT \\* FROM users WHERE name = ?\"' user_repo.py)\n",
			45, 6,
		),
		makeTask(
			"tb4-comp-50-security-path-traversal",
			CategorySecurity,
			"Sanitize filename parameter in file_server.py to prevent directory traversal outside /safe/dir.",
			"echo -e 'import os\ndef read_file(filename):\n    return open(os.path.join(\"/safe/dir\", filename)).read()' > file_server.py",
			"#!/bin/bash\ngrep -q 'os.path.abspath' file_server.py || grep -q 'os.path.realpath' file_server.py || grep -q 'os.path.basename' file_server.py\n",
			40, 5,
		),
		makeTask(
			"tb4-comp-51-security-command-injection",
			CategorySecurity,
			"Fix command injection vulnerability in ping_tool.py by passing args as a list with shell=False in subprocess.run.",
			"echo -e 'import subprocess\ndef ping(host):\n    subprocess.run(f\"ping -c 1 {host}\", shell=True)' > ping_tool.py",
			"#!/bin/bash\ngrep -q 'shell=False' ping_tool.py && grep -q '\\[\"ping\"' ping_tool.py\n",
			40, 5,
		),
		makeTask(
			"tb4-comp-52-security-xss-escaping",
			CategorySecurity,
			"Fix reflected XSS in template_view.py by escaping user_input with html.escape.",
			"echo -e 'def render_greeting(user_input):\n    return f\"<h1>Hello {user_input}</h1>\"' > template_view.py",
			"#!/bin/bash\ngrep -q 'html.escape' template_view.py && python3 -c 'import template_view; assert \"&lt;script&gt;\" in template_view.render_greeting(\"<script>\")'\n",
			40, 5,
		),
		makeTask(
			"tb4-comp-53-security-hardcoded-secret",
			CategorySecurity,
			"Remove hardcoded SECRET_KEY from settings.py and read it from os.environ.get('SECRET_KEY').",
			"echo 'SECRET_KEY = \"supersecret123\"' > settings.py",
			"#!/bin/bash\n! grep -q '\"supersecret123\"' settings.py && grep -q \"os.environ.get(['\\\"]SECRET_KEY['\\\"])\" settings.py\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-54-security-insecure-deserialization",
			CategorySecurity,
			"Replace unsafe pickle.loads in session.py with json.loads to prevent insecure deserialization.",
			"echo -e 'import pickle\ndef load_session(data):\n    return pickle.loads(data)' > session.py",
			"#!/bin/bash\n! grep -q 'pickle' session.py && grep -q 'json.loads' session.py\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-55-security-cors-misconfig",
			CategorySecurity,
			"Fix CORS header in server.py: replace wildcard Access-Control-Allow-Origin: * with https://app.example.com.",
			"echo 'headers = {\"Access-Control-Allow-Origin\": \"*\"}' > server.py",
			"#!/bin/bash\n! grep -q '\"\\*\"' server.py && grep -q '\"https://app.example.com\"' server.py\n",
			35, 5,
		),
		makeTask(
			"tb4-comp-56-security-weak-hashing",
			CategorySecurity,
			"Replace weak MD5 password hashing in auth.py with bcrypt.hashpw.",
			"echo -e 'import hashlib\ndef hash_password(password):\n    return hashlib.md5(password.encode()).hexdigest()' > auth.py",
			"#!/bin/bash\n! grep -q 'hashlib.md5' auth.py && grep -q 'bcrypt' auth.py\n",
			40, 5,
		),
	}
}

// TestGenerateComprehensiveSuite writes testdata/tb4bench/comprehensive_suite.json.
func TestGenerateComprehensiveSuite(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "testdata", "tb4bench", "comprehensive_suite.json")
	suite := ManifestSuite{
		Benchmark: BenchmarkName,
		Version:   "1.0",
		Tasks:     buildComprehensiveTasks(),
	}

	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(suite); err != nil {
		t.Fatalf("failed to marshal comprehensive suite: %v", err)
	}

	if err := os.WriteFile(fixturePath, buf.Bytes(), 0644); err != nil {
		t.Fatalf("failed to write fixture file: %v", err)
	}
}

// TestComprehensiveSuite validates that the comprehensive suite JSON exists, satisfies
// all requirements (50+ tasks, all 7 categories with required minima, correct sha256 oracle hashes,
// unique IDs, and strict formatting).
func TestComprehensiveSuite(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "testdata", "tb4bench", "comprehensive_suite.json")
	suite, err := LoadManifestFile(fixturePath)
	if err != nil {
		t.Fatalf("failed to load comprehensive suite manifest: %v", err)
	}

	// 1. Assert suite is valid and has expected benchmark metadata.
	if err := suite.Validate(); err != nil {
		t.Fatalf("suite.Validate() failed: %v", err)
	}
	if suite.Benchmark != BenchmarkName {
		t.Errorf("expected benchmark %q, got %q", BenchmarkName, suite.Benchmark)
	}

	// 2. Assert >= 50 tasks.
	if len(suite.Tasks) < 50 {
		t.Fatalf("expected at least 50 tasks in comprehensive suite, got %d", len(suite.Tasks))
	}

	// 3. Category count assertions.
	categoryMinima := map[string]int{
		CategoryBuild:    7,
		CategorySysadmin: 7,
		CategoryGit:      7,
		CategoryRefactor: 7,
		CategoryDebug:    7,
		CategoryTest:     7,
		CategorySecurity: 8,
	}

	categoryCounts := make(map[string]int)
	seenIDs := make(map[string]bool)
	taskIDRegex := regexp.MustCompile(`^tb4-comp-\d{2}-[a-z0-9-]+$`)

	expectedImage := "ghcr.io/fak/tb4-sandbox@sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

	for i, task := range suite.Tasks {
		// Unique and strictly formatted task ID
		if seenIDs[task.TaskID] {
			t.Errorf("task %d: duplicate task ID %q", i, task.TaskID)
		}
		seenIDs[task.TaskID] = true

		if !taskIDRegex.MatchString(task.TaskID) {
			t.Errorf("task %d: task ID %q does not match strict naming format ^tb4-comp-\\d{2}-[a-z0-9-]+$", i, task.TaskID)
		}

		// Category tracking
		categoryCounts[task.Category]++

		// Environment image digest
		if task.EnvironmentImageDigest != expectedImage {
			t.Errorf("task %s: expected image %q, got %q", task.TaskID, expectedImage, task.EnvironmentImageDigest)
		}

		// Cryptographic oracle hash verification
		h := sha256.Sum256([]byte(task.VerificationOracle))
		computedHash := "sha256:" + hex.EncodeToString(h[:])
		if computedHash != task.VerificationOracleHash {
			t.Errorf("task %s: verification oracle hash mismatch: computed %s, manifest has %s",
				task.TaskID, computedHash, task.VerificationOracleHash)
		}
		if err := task.VerifyOracleScript([]byte(task.VerificationOracle)); err != nil {
			t.Errorf("task %s: task.VerifyOracleScript failed: %v", task.TaskID, err)
		}

		// Bounds check: timeout and budget
		if task.TimeoutSeconds < 30 || task.TimeoutSeconds > 60 {
			t.Errorf("task %s: timeout_seconds %d out of expected range [30, 60]", task.TaskID, task.TimeoutSeconds)
		}
		if task.BudgetTurns < 5 || task.BudgetTurns > 10 {
			t.Errorf("task %s: budget_turns %d out of expected range [5, 10]", task.TaskID, task.BudgetTurns)
		}

		// Oracle script sanity
		if len(task.VerificationOracle) == 0 {
			t.Errorf("task %s: verification_oracle is empty", task.TaskID)
		}
		if len(task.Prompt) == 0 {
			t.Errorf("task %s: prompt is empty", task.TaskID)
		}
	}

	// Validate category thresholds
	for cat, minCount := range categoryMinima {
		actual := categoryCounts[cat]
		if actual < minCount {
			t.Errorf("category %q has %d tasks, minimum required is %d", cat, actual, minCount)
		}
	}

	t.Logf("Comprehensive suite validated successfully: %d tasks across %d categories", len(suite.Tasks), len(categoryCounts))
	for cat, count := range categoryCounts {
		t.Logf("  Category %-10s: %d tasks", cat, count)
	}
}
