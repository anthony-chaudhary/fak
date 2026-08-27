#!/bin/bash
set -euo pipefail

usage() { echo "usage: package-helper.sh --identity ID --team-id TEAM --bundle-id ID --output DIR [--notary-profile PROFILE]" >&2; exit 64; }
identity= team_id= bundle_id= output= notary_profile=
while (($#)); do
  case "$1" in
    --identity) identity=${2:-}; shift 2;;
    --team-id) team_id=${2:-}; shift 2;;
    --bundle-id) bundle_id=${2:-}; shift 2;;
    --output) output=${2:-}; shift 2;;
    --notary-profile) notary_profile=${2:-}; shift 2;;
    *) usage;;
  esac
done
[[ -n "$identity" && -n "$team_id" && -n "$bundle_id" && -n "$output" ]] || usage
for tool in swift codesign ditto shasum; do command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 69; }; done
[[ -z "$notary_profile" ]] || command -v xcrun >/dev/null || { echo "notary profile requires xcrun" >&2; exit 69; }

root=$(cd "$(dirname "$0")/.." && pwd)
stage=$(mktemp -d "${TMPDIR:-/tmp}/fak-localapp-helper.XXXXXX")
trap 'rm -rf "$stage"' EXIT
mkdir -p "$output"
cd "$root"
swift build -c release --product FakLocalAppHelper
bin=$(swift build -c release --show-bin-path)/FakLocalAppHelper
app="$stage/FakLocalAppHelper.app"
mkdir -p "$app/Contents/MacOS"
cp "$bin" "$app/Contents/MacOS/FakLocalAppHelper"
cat >"$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>FakLocalAppHelper</string>
<key>CFBundleIdentifier</key><string>${bundle_id}</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleVersion</key><string>1</string>
<key>FakExpectedTeamIdentifier</key><string>${team_id}</string>
<key>LSBackgroundOnly</key><true/>
</dict></plist>
PLIST
codesign --force --timestamp --options runtime --sign "$identity" "$app"
codesign --verify --deep --strict --verbose=2 "$app"
archive="$output/FakLocalAppHelper.zip"
ditto -c -k --keepParent "$app" "$archive"
status=SIGNED
if [[ -n "$notary_profile" ]]; then
  xcrun notarytool submit "$archive" --keychain-profile "$notary_profile" --wait
  xcrun stapler staple "$app"
  codesign --verify --deep --strict --verbose=2 "$app"
  spctl --assess --type execute --verbose=2 "$app"
  ditto -c -k --keepParent "$app" "$archive"
  status=SIGNED_AND_NOTARIZED
fi
sha=$(shasum -a 256 "$archive" | awk '{print $1}')
printf '{"schema":"fak.local-app-helper-package/1","status":"%s","team_id":"%s","bundle_id":"%s","archive":"FakLocalAppHelper.zip","sha256":"%s"}\n' "$status" "$team_id" "$bundle_id" "$sha" >"$output/receipt.json"
printf '%s\n' "$output/receipt.json"
