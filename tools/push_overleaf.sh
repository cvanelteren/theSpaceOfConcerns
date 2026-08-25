#!/usr/bin/env bash
set -euo pipefail

mode="${1:---dry-run}"
case "$mode" in
    --dry-run|--push|--push-if-changed) ;;
    *)
        printf 'Usage: %s [--dry-run|--push|--push-if-changed]\n' "$0" >&2
        exit 2
        ;;
esac

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

mapfile -t patterns < <(sed -e '/^[[:space:]]*$/d' -e '/^[[:space:]]*#/d' "$allowlist")
if (( ${#patterns[@]} == 0 )); then
    printf 'The Overleaf export allowlist is empty: %s\n' "$allowlist" >&2
    exit 1
fi

if [[ "$mode" == "--push-if-changed" ]]; then
    changed="$work/changed-files.txt"
    if git -C "$root" rev-parse --verify HEAD^ >/dev/null 2>&1; then
        git -C "$root" diff --name-only HEAD^ HEAD > "$changed"
    else
        git -C "$root" diff-tree --root --no-commit-id --name-only -r HEAD > "$changed"
    fi

    should_push=false
    while IFS= read -r path; do
        if [[ "$path" == ".overleaf-export" ]]; then
            should_push=true
            break
        fi
        for pattern in "${patterns[@]}"; do
            if [[ "$path" == $pattern ]]; then
                should_push=true
                break 2
            fi
        done
    done < "$changed"

    if [[ "$should_push" == false ]]; then
        exit 0
    fi
    mode="--push"
fi

remote_url="$(git -C "$root" remote get-url "$remote_name")"
source_revision="$(git -C "$root" rev-parse --short HEAD)"
source_subject="$(git -C "$root" log -1 --format=%s HEAD)"
tree_files="$work/source-tree.txt"
manifest="$work/export-manifest.txt"
git -C "$root" ls-tree -r --name-only HEAD > "$tree_files"
: > "$manifest"

for pattern in "${patterns[@]}"; do
    matched=false
    while IFS= read -r path; do
        if [[ "$path" == $pattern ]]; then
            printf '%s\n' "$path" >> "$manifest"
            matched=true
        fi
    done < "$tree_files"
    if [[ "$matched" == false ]]; then
        printf 'No committed file at HEAD matches Overleaf pattern: %s\n' "$pattern" >&2
        exit 1
    fi
done
sort -u -o "$manifest" "$manifest"

mapfile -t export_paths < "$manifest"
mkdir -p "$work/export"
git -C "$root" archive --format=tar HEAD -- "${export_paths[@]}" | tar -xf - -C "$work/export"

file_count="${#export_paths[@]}"
printf 'Prepared %s committed files for Overleaf from %s.\n' "$file_count" "$source_revision"

if [[ "$mode" == "--dry-run" ]]; then
    printf 'Export contains:\n'
    sed 's/^/  /' "$manifest"
    printf 'Dry run only. Use --push to publish this projection.\n'
    exit 0
fi

repo="$work/repo"
git init -q -b "$remote_branch" "$repo"
git -C "$repo" remote add overleaf "$remote_url"
git -C "$repo" fetch -q overleaf "$remote_branch"
git -C "$repo" checkout -q -B "$remote_branch" FETCH_HEAD
git -C "$repo" rm -r -q --ignore-unmatch .
cp -a "$work/export/." "$repo/"

git -C "$repo" config user.name "Overleaf export"
git -C "$repo" config user.email "overleaf-export@local"
git -C "$repo" add -A

if git -C "$repo" diff --cached --quiet; then
    printf 'Overleaf already matches committed revision %s.\n' "$source_revision"
    exit 0
fi

git -C "$repo" commit -q -m "Manuscript export from $source_revision: $source_subject"
git -C "$repo" push overleaf "$remote_branch:$remote_branch"

printf 'Published committed manuscript revision %s to Overleaf %s.\n' "$source_revision" "$remote_branch"
