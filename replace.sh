#!/bin/bash

# Usage message
usage() {
  echo "Usage: $0 <full_video> <intro_video> <timestamp (mm:ss)>"
  echo "Example: $0 episode.mp4 intro.mp4 00:35"
  exit 1
}

# Check for correct number of arguments
if [ "$#" -ne 3 ]; then
  usage
fi

FULL_VIDEO="$1"
INTRO_VIDEO="$2"
TIMESTAMP="$3"

# Validate timestamp format (mm:ss)
if ! [[ "$TIMESTAMP" =~ ^[0-9]{2}:[0-5][0-9]$ ]]; then
  echo "❌ Invalid timestamp format. Use mm:ss (e.g. 00:35)"
  exit 1
fi

# Convert mm:ss to total seconds
IFS=":" read -r MM SS <<< "$TIMESTAMP"
START_SECONDS=$((10#$MM * 60 + 10#$SS))

# Get intro duration in seconds using ffprobe
INTRO_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INTRO_VIDEO")
if [ -z "$INTRO_DURATION" ]; then
  echo "❌ Failed to get intro video duration."
  exit 1
fi

INTRO_DURATION=${INTRO_DURATION%.*}  # Strip decimal part
END_SECONDS=$((START_SECONDS + INTRO_DURATION))

OUTPUT="output_replaced_audio.mp4"

echo "🎬 Replacing audio in '$FULL_VIDEO'"
echo "🔁 From ${START_SECONDS}s to ${END_SECONDS}s with audio from '$INTRO_VIDEO' (duration: ${INTRO_DURATION}s)"

# Run ffmpeg
ffmpeg \
  -i "$FULL_VIDEO" \
  -i "$INTRO_VIDEO" \
  -filter_complex "
    [0:a]atrim=0:${START_SECONDS},asetpts=PTS-STARTPTS[a1];
    [1:a]atrim=0:${INTRO_DURATION},asetpts=PTS-STARTPTS[a2];
    [0:a]atrim=${END_SECONDS},asetpts=PTS-STARTPTS[a3];
    [a1][a2][a3]concat=n=3:v=0:a=1[outa]
  " \
  -map 0:v \
  -map "[outa]" \
  -c:v copy -c:a aac -shortest \
  "$OUTPUT"

echo "✅ Done! Output saved to: $OUTPUT"

