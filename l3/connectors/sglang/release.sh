#!/usr/bin/env bash
#
# release.sh — automate CAMA Standalone releases
#
# Usage:
#   ./release.sh patch    # 1.0.0 -> 1.0.1
#   ./release.sh minor    # 1.0.0 -> 1.1.0
#   ./release.sh major    # 1.0.0 -> 2.0.0
#   ./release.sh          # defaults to patch
#
set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()   { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── pre-flight checks ───────────────────────────────────────────────
command -v git >/dev/null || die "git is not installed"
command -v gh  >/dev/null || warn "gh (GitHub CLI) not found — GitHub release step will be skipped"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

git rev-parse --is-inside-work-tree &>/dev/null || die "not a git repository"

VERSION_FILE="$ROOT/VERSION"
CHANGELOG="$ROOT/CHANGELOG.md"
[[ -f "$VERSION_FILE" ]] || die "VERSION file not found"
[[ -f "$CHANGELOG" ]]    || die "CHANGELOG.md not found"

CURRENT="$(tr -d '[:space:]' < "$VERSION_FILE")"
[[ "$CURRENT" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "VERSION '$CURRENT' is not valid semver"

# ── compute next version ─────────────────────────────────────────────
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"
BUMP="${1:-patch}"

case "$BUMP" in
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  patch) PATCH=$((PATCH + 1)) ;;
  *) die "usage: $0 [major|minor|patch]" ;;
esac

NEXT="${MAJOR}.${MINOR}.${PATCH}"
TAG="v${NEXT}"

info "Current version : $CURRENT"
info "Bump type       : $BUMP"
info "Next version    : $NEXT  ($TAG)"
echo ""

# ── confirm ──────────────────────────────────────────────────────────
read -rp "Proceed with release $TAG? [y/N] " CONFIRM
[[ "$CONFIRM" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }
echo ""

# ── 1. update VERSION file ──────────────────────────────────────────
info "Updating VERSION → $NEXT"
echo "$NEXT" > "$VERSION_FILE"
ok "VERSION updated"

# ── 2. stamp CHANGELOG ──────────────────────────────────────────────
TODAY="$(date +%Y-%m-%d)"
if grep -q "^\## \[Unreleased\]" "$CHANGELOG" 2>/dev/null; then
  # Replace [Unreleased] header with the version + date
  sed -i "s/^## \[Unreleased\]/## [$NEXT] - $TODAY/" "$CHANGELOG"
  ok "CHANGELOG: [Unreleased] → [$NEXT] - $TODAY"
else
  info "No [Unreleased] section found in CHANGELOG — skipping auto-stamp."
  info "Make sure CHANGELOG.md is up to date before continuing."
fi

# ── 3. git commit ────────────────────────────────────────────────────
info "Staging all changes …"
git add -A
if git diff --cached --quiet; then
  warn "Nothing to commit — working tree clean"
else
  git commit -m "release: $TAG

- Bump version to $NEXT
- Update CHANGELOG for $TAG"
  ok "Committed"
fi

# ── 4. git tag ───────────────────────────────────────────────────────
if git rev-parse "$TAG" &>/dev/null; then
  warn "Tag $TAG already exists — skipping"
else
  git tag -a "$TAG" -m "Release $TAG"
  ok "Tagged $TAG"
fi

# ── 5. git push ──────────────────────────────────────────────────────
info "Pushing commits and tags …"
git push origin main
git push origin "$TAG"
ok "Pushed to origin"

# ── 6. create zip archive ───────────────────────────────────────────
ZIP_NAME="cama-standalone-${TAG}.zip"
info "Creating archive $ZIP_NAME …"
git archive --format=zip --prefix="cama-standalone-${TAG}/" "$TAG" -o "$ROOT/$ZIP_NAME"
ok "Archive created: $ZIP_NAME ($(du -h "$ROOT/$ZIP_NAME" | cut -f1))"

# ── 7. GitHub release (if gh available) ──────────────────────────────
if command -v gh &>/dev/null; then
  info "Creating GitHub release …"

  # Extract this version's notes from CHANGELOG
  NOTES=$(awk "/^## \[$NEXT\]/{found=1; next} /^## \[/{if(found) exit} found" "$CHANGELOG")
  if [[ -z "$NOTES" ]]; then
    NOTES="Release $TAG"
  fi

  gh release create "$TAG" "$ROOT/$ZIP_NAME" \
    --title "$TAG" \
    --notes "$NOTES"
  ok "GitHub release $TAG created with $ZIP_NAME attached"
else
  warn "Skipping GitHub release (install gh CLI to enable)"
fi

# ── done ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  Release $TAG complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
info "Artifacts:"
info "  • Git tag   : $TAG"
info "  • Archive   : $ZIP_NAME"
if command -v gh &>/dev/null; then
  info "  • Release   : $(gh browse -n)/releases/tag/$TAG"
fi
echo ""
