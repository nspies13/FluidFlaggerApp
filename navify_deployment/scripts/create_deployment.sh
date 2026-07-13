#!/usr/bin/env bash
set -euo pipefail

: "${TOKEN:?Set TOKEN to a Navify bearer token}"
: "${TENANT_ID:?Set TENANT_ID}"
: "${TENANT_ALIAS:?Set TENANT_ALIAS}"

TAG="${TAG:-1.0.0}"
API_BASE="${API_BASE:-https://api.prod.algosuite.navify.com}"
TMP_JSON="$(mktemp)"

cd "$(dirname "$0")/.."

python3 - "$TENANT_ALIAS" "$TAG" "$TMP_JSON" <<'PY'
import sys
from pathlib import Path

tenant_alias, tag, output = sys.argv[1:4]
template = Path("deployment.json.template").read_text()
body = template.replace("{tenantAlias}", tenant_alias).replace("{tag}", tag)
Path(output).write_text(body)
PY

curl -fsS -X POST "${API_BASE}/tenants/${TENANT_ID}/deployments" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary "@${TMP_JSON}"

rm -f "$TMP_JSON"
