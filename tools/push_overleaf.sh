#!/usr/bin/env bash
set -euo pipefail

mode="${1:---dry-run}"
if [[ "$mode" != "--dry-run" && "$mode" != "--push" ]]; then
    printf 'Usage: %s [--dry-run|--push]\n' "$0" >&2
    exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
allowlist="$root/.overleaf-export"
remote_name="${OVERLEAF_REMOTE:-overleaf}"
remote_branch="${OVERLEAF_BRANCH:-main}"
work="$(mktemp -d "${TMPDIR:-/tmp}/space-concerns-overleaf.XXXXXX")"
trap 'rm -rf "$work"' EXIT

if [[ ! -f "$allowlist" ]]; then
    printf 'Missing export allowlist: %s\n' "$allowlist" >&2
    exit 1
fi

remote_url="$(git -C "$root" remote get-url "$remote_name")"
source_revision="$(git -C "$root" rev-parse --short HEAD)"
manifest="$work/export-manifest.txt"

while IFS= read -r pattern; do
    [[ -z "$pattern" || "$pattern" == \#* ]] && continue
    if [[ -f "$root/$pattern" ]]; then
        printf '%s\n' "$pattern"
    elif compgen -G "$root/$pattern" > /dev/null; then
        while IFS= read -r match; do
            printf '%s\n' "${match#"$root/"}"
        done < <(compgen -G "$root/$pattern")
    else
        git -C "$root" ls-files -- "$pattern"
    fi
done < "$allowlist" | sort -u > "$manifest"

while IFS= read -r path; do
    if [[ ! -f "$root/$path" ]]; then
        printf 'Allowlisted file is missing: %s\n' "$path" >&2
        exit 1
    fi
    mkdir -p "$work/export/$(dirname "$path")"
    cp -p "$root/$path" "$work/export/$path"
done < "$manifest"

file_count="$(wc -l < "$manifest" | tr -d ' ')"
printf 'Prepared %s files for Overleaf from source revision %s.\n' "$file_count" "$source_revision"
printf 'Export contains:\n'
sed 's/^/  /' "$manifest"

if [[ "$mode" == "--dry-run" ]]; then
    printf 'Dry run only. Use --push to replace Overleaf %s with this manuscript-only history.\n' "$remote_branch"
    exit 0
fi

repo="$work/repo"
git init -q -b "$remote_branch" "$repo"
git -C "$repo" remote add overleaf "$remote_url"
git -C "$repo" fetch -q overleaf "$remote_branch"
git -C "$repo" checkout -q -B "$remote_branch" FETCH_HEAD
git -C "$repo" rm -r -q .
cp -a "$work/export/." "$repo/"

git -C "$repo" config user.name "Overleaf export"
git -C "$repo" config user.email "overleaf-export@local"
git -C "$repo" add -A

if git -C "$repo" diff --cached --quiet; then
    printf 'Overleaf already matches the manuscript-only projection.\n'
    exit 0
fi

git -C "$repo" commit -q -m "Manuscript export from $source_revision"
git -C "$repo" push overleaf "$remote_branch:$remote_branch"

printf 'Published manuscript-only working tree to Overleaf %s.\n' "$remote_branch"
