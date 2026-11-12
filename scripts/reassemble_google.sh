#!/usr/bin/env bash
set -euo pipefail
PARTS="${1:?Usage: scripts/reassemble_google.sh /directory/with/six-parts /output/directory}"
OUTPUT="${2:?Missing output directory}"
for n in 000 001 002 003 004 005; do test -s "$PARTS/speech_commands.part-$n"; done
mkdir -p "$OUTPUT"
cat "$PARTS"/speech_commands.part-{000..005} | gzip -t
cat "$PARTS"/speech_commands.part-{000..005} | tar -tzf - > "$OUTPUT/archive_manifest.txt"
cat "$PARTS"/speech_commands.part-{000..005} | tar -xzf - -C "$OUTPUT" --no-same-owner --no-same-permissions
echo "Archive extracted. Run the Google benchmark inventory and split next."
