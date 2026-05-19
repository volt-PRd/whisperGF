#!/bin/bash
# Build and publish a new release of whisperGF to GitHub
# Usage: ./build-release.sh <version> [commit_message]
# Example: ./build-release.sh v1.1.0 "Added new features"

VERSION=${1:-"v1.0.0"}
COMMIT_MSG=${2:-"Release ${VERSION}"}
REPO="volt-PRd/whisperGF"
TOKEN="${GITHUB_TOKEN:-$TOKEN}"

echo "=== Building release ${VERSION} ==="

# Step 1: Build the AAR (if needed - uncomment and adjust paths)
# echo "Building AAR..."
# ./gradlew :lib:assembleRelease
# cp lib/build/outputs/aar/lib-release.aar release/aar/whisper-android.aar
# cd release && zip -r whisper-android.zip whisper-android-package/ && cd ..

# Step 2: Commit changes
echo "Committing changes..."
git add -A
git commit -m "${COMMIT_MSG}"

# Step 3: Push to GitHub
echo "Pushing to GitHub..."
git push origin main

# Step 4: Create GitHub Release
echo "Creating GitHub Release..."
curl -s -X POST \
  "https://api.github.com/repos/${REPO}/releases" \
  -H "Authorization: token ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{
    \"tag_name\": \"${VERSION}\",
    \"name\": \"whisperGF ${VERSION}\",
    \"body\": \"${COMMIT_MSG}\",
    \"draft\": false,
    \"prerelease\": false
  }" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Release:', d.get('html_url', d.get('message', 'error')))"

# Step 5: Upload AAR to release
echo "Uploading AAR asset..."
RELEASE_ID=$(curl -s "https://api.github.com/repos/${REPO}/releases/tags/${VERSION}" \
  -H "Authorization: token ${TOKEN}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))")

if [ -n "$RELEASE_ID" ] && [ "$RELEASE_ID" != "" ]; then
  curl -s -X POST \
    "https://uploads.github.com/repos/${REPO}/releases/${RELEASE_ID}/assets?name=whisper-android.aar" \
    -H "Authorization: token ${TOKEN}" \
    -H "Content-Type: application/octet-stream" \
    --data-binary "@release/aar/whisper-android.aar" > /dev/null

  curl -s -X POST \
    "https://uploads.github.com/repos/${REPO}/releases/${RELEASE_ID}/assets?name=whisper-android.zip" \
    -H "Authorization: token ${TOKEN}" \
    -H "Content-Type: application/zip" \
    --data-binary "@release/whisper-android.zip" > /dev/null

  echo "Assets uploaded successfully!"
fi

echo "=== Release ${VERSION} complete! ==="
echo "View at: https://github.com/${REPO}/releases/tag/${VERSION}"
