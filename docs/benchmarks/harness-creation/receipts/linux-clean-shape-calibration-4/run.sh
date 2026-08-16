#!/usr/bin/env bash
set -euo pipefail

commit=${FAK_COMMIT:-83c7641b909e}
base=$(mktemp -d /tmp/fak-harness-calibration.XXXXXX)
mkdir -p "$base"/{home,gopath,gocache,gomodcache,product}
export HOME="$base/home" GOPATH="$base/gopath" GOCACHE="$base/gocache" GOMODCACHE="$base/gomodcache" GOTOOLCHAIN=auto
transcript="$base/transcript.txt"
exec > >(tee "$transcript") 2>&1
stamp() { date -u +%Y-%m-%dT%H:%M:%S.%NZ; }
ns() { date +%s%N; }
elapsed() { python3 - "$1" "$2" <<'PY'
import sys
print(f"{(int(sys.argv[2])-int(sys.argv[1]))/1e9:.3f}")
PY
}

start=$(ns)
echo "run_id=linux-clean-shape-calibration-4"
echo "participant_id=maintainer-calibration"
echo "participant_class=maintainer-calibration independent=false"
echo "start=$(stamp)"
echo "base=$base"
echo "os=$(uname -srmo)"
echo "go_host=$(go version)"
echo "cache_state=empty HOME/GOPATH/GOCACHE/GOMODCACHE"
echo "artifact=github.com/anthony-chaudhary/fak/cmd/fak@$commit"
t=$(ns); go install "github.com/anthony-chaudhary/fak/cmd/fak@$commit"; u=$(ns)
echo "install_seconds=$(elapsed "$t" "$u")"
FAK="$GOPATH/bin/fak"
"$FAK" harness init --dir "$base/product" --module example.test/calibration4
cat > "$base/product/product/config.go" <<'EOF'
package product

type Config struct {
 ID string
 Version string
 Profile string
 SystemPrompt string
 Task string
}

func DefaultConfig() Config {
 return Config{
  ID: "local-support-harness",
  Version: "0.1.0",
  Profile: "support-summary",
  SystemPrompt: "Answer from admitted context.",
  Task: "Summarize the support request.",
 }
}

func OfflineReply(prompt string) string { return "admitted support summary: " + prompt }
EOF
user_before=$(sha256sum "$base/product/product/config.go" | cut -d' ' -f1)
cd "$base/product"
t=$(ns); go build -o "$base/product-bin" ./cmd/product; u=$(ns)
echo "build_seconds=$(elapsed "$t" "$u")"
t=$(ns); "$base/product-bin" --selfcheck; u=$(ns)
echo "selfcheck_seconds=$(elapsed "$t" "$u")"
python3 - <<'PY'
import json
with open('harness.lock.json') as f:
    print('upgrade_command='+json.load(f)['upgrade'])
PY
t=$(ns); "$FAK" harness init --dir . --module example.test/calibration4; u=$(ns)
echo "rerun_seconds=$(elapsed "$t" "$u")"
user_after=$(sha256sum product/config.go | cut -d' ' -f1)
echo "user_sha_before=$user_before"
echo "user_sha_after=$user_after"
[ "$user_before" = "$user_after" ]
stop=$(ns)
echo "stop=$(stamp)"
echo "elapsed_seconds=$(elapsed "$start" "$stop")"
echo "outcome=success"
# This hashes the transcript prefix. Hash the closed file after the process exits.
echo "transcript_prefix_sha256=$(sha256sum "$transcript" | cut -d' ' -f1)"
echo "TRANSCRIPT=$transcript"
