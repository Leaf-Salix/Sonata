#!/bin/bash
# Apply Sonata patches to upstream TMARB source tree.
# Called before NPU build to insert the sonata_orchestrate_with_schedule
# branch into the AICPU orchestrator thread.
#
# Usage: apply.sh <upstream_aicpu_dir>
#   <upstream_aicpu_dir> = path to tensormap_and_ringbuffer/aicpu/
#
# Example:
#   ./patches/apply.sh ../../upstream/pypto/runtime/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/

set -euo pipefail

PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="${1:?Usage: $0 <upstream_aicpu_dir>}"
TARGET_FILE="$TARGET_DIR/aicpu_executor.cpp"

if [ ! -f "$TARGET_FILE" ]; then
    echo "ERROR: target file not found: $TARGET_FILE"
    exit 1
fi

echo "Applying Sonata patches to $TARGET_FILE ..."

# Apply each patch in order
for patch in "$PATCH_DIR"/*.patch; do
    [ -f "$patch" ] || continue
    echo "  applying $(basename "$patch")"
    if ! patch -p1 -d "$TARGET_DIR/../.." < "$patch" 2>/dev/null; then
        # Check if already applied (patch -N for reverse check)
        if ! patch -p1 -R --dry-run -d "$TARGET_DIR/../.." < "$patch" >/dev/null 2>&1; then
            echo "WARN: patch failed — may be a conflict. Check $TARGET_FILE"
            exit 1
        fi
        echo "  already applied, skipping"
    fi
done

echo "Done."
