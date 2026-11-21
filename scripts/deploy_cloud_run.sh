#!/usr/bin/env bash
set -euo pipefail

registry="${IMAGE%%/*}"
gcloud auth configure-docker "$registry" --quiet
docker push "$IMAGE"
image_digest=$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE")

scope=(--project "$GCP_PROJECT" --region "$CLOUD_RUN_REGION" --quiet)
tag=ci-candidate
revision_suffix="gha-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"

deployment=$(gcloud run deploy "$CLOUD_RUN_SERVICE" "${scope[@]}" \
  --image "$image_digest" --service-account "$RUNTIME_ACCOUNT" \
  --no-traffic --tag "$tag" --revision-suffix "$revision_suffix" --format=json)
revision=$(jq -er '.status.latestCreatedRevisionName' <<< "$deployment")
candidate_url=$(jq -er --arg tag "$tag" '.status.traffic[] | select(.tag == $tag) | .url' <<< "$deployment")

python3 scripts/smoke_forecast.py --url "$candidate_url"
gcloud run services update-traffic "$CLOUD_RUN_SERVICE" "${scope[@]}" \
  --to-revisions "$revision=100" --remove-tags "$tag"
printf 'Serving revision %s\n' "$revision"
