#!/usr/bin/env bash
set -euo pipefail

: "${CLIENT_ID:?Set CLIENT_ID}"
: "${CLIENT_SECRET:?Set CLIENT_SECRET}"
: "${TENANT_ALIAS:?Set TENANT_ALIAS}"

AUTH_URL="${AUTH_URL:-https://api.appprodus.platform.navify.com/api/v1/auth/protocols/oidc/token}"

curl -fsS --location --request POST "$AUTH_URL" \
  --header "Content-Type: application/x-www-form-urlencoded" \
  --user "${CLIENT_ID}:${CLIENT_SECRET}" \
  --data-urlencode "grant_type=client_credentials" \
  --data-urlencode "scope=default navify:tenant:${TENANT_ALIAS}" \
  | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
try:
    print(payload["access_token"])
except KeyError as exc:
    raise SystemExit("Token response did not include access_token") from exc
'
