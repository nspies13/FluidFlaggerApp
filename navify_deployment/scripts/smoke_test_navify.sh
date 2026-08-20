#!/usr/bin/env bash
set -euo pipefail

TENANT_ID="${TENANT_ID:-e785ce6d-2098-4edc-9af2-4879481d433c}"
ALGORITHM_ID="${ALGORITHM_ID:-washu.fluidflagger}"
API_BASE="${API_BASE:-https://api.us.prod.algosuite.navify.com}"
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-10}"
EXECUTION_TIMEOUT_SECONDS="${EXECUTION_TIMEOUT_SECONDS:-120}"

: "${TOKEN:?Set TOKEN to a fresh Navify bearer token}"

cd "$(dirname "$0")/.."

EXECUTION_URL="${EXECUTION_URL:-${API_BASE%/}/tenants/${TENANT_ID}/algorithms/${ALGORITHM_ID}/executions}"
umask 077
RESPONSE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fluidflagger-navify-smoke.XXXXXX")"
AUTH_HEADER_FILE="${RESPONSE_DIR}/authorization_header"
printf 'Authorization: Bearer %s\n' "$TOKEN" >"$AUTH_HEADER_FILE"
unset TOKEN

cleanup() {
  rm -f \
    "$AUTH_HEADER_FILE" \
    "${RESPONSE_DIR}/realtime_response.json" \
    "${RESPONSE_DIR}/retrospective_response.json"
  rmdir "$RESPONSE_DIR"
}
trap cleanup EXIT

print_error_response() {
  local response_file="$1"

  if [[ ! -s "$response_file" ]]; then
    echo "Navify returned an empty response body." >&2
    return
  fi

  python3 - "$response_file" >&2 <<'PY'
import json
import sys
from pathlib import Path

response_path = Path(sys.argv[1])
body = response_path.read_text(errors="replace")
try:
    parsed = json.loads(body)
except json.JSONDecodeError:
    print(body)
else:
    print(json.dumps(parsed, indent=2, sort_keys=True))
PY
}

validate_response() {
  local request_file="$1"
  local response_file="$2"
  local mode="$3"

  python3 - "$request_file" "$response_file" "$mode" <<'PY'
import json
import math
import sys
from pathlib import Path

request_path = Path(sys.argv[1])
response_path = Path(sys.argv[2])
mode = sys.argv[3]
schema_path = Path("schemas/fluidflagger-bmp-response.schema.json")

try:
    request_payload = json.loads(request_path.read_text())
except json.JSONDecodeError as exc:
    raise SystemExit(f"{request_path} is not valid JSON: {exc}") from exc
try:
    response_payload = json.loads(response_path.read_text())
except json.JSONDecodeError as exc:
    raise SystemExit(f"Navify response is not valid JSON: {exc}") from exc
response_schema = json.loads(schema_path.read_text())

if not isinstance(request_payload, dict):
    raise SystemExit(f"{request_path} must contain one JSON object")
if not isinstance(response_payload, dict):
    raise SystemExit("Navify response must be one JSON object")

result = response_payload

properties = response_schema["properties"]
unexpected_fields = sorted(set(result) - set(properties))
if unexpected_fields:
    raise SystemExit(
        "Navify result contains fields outside the response schema: "
        + ", ".join(unexpected_fields)
    )

missing_fields = [field for field in response_schema["required"] if field not in result]
if missing_fields:
    raise SystemExit(
        "Navify result is missing required response fields: "
        + ", ".join(missing_fields)
    )

def matches_json_type(value, expected):
    expected_types = expected if isinstance(expected, list) else [expected]
    for expected_type in expected_types:
        if expected_type == "null" and value is None:
            return True
        if expected_type == "string" and isinstance(value, str):
            return True
        if expected_type == "boolean" and isinstance(value, bool):
            return True
        if (
            expected_type == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            return True
    return False


for field, value in result.items():
    if not matches_json_type(value, properties[field]["type"]):
        raise SystemExit(
            f"Navify result field {field!r} does not match the response schema"
        )

for field, expected in request_payload.items():
    if field not in result:
        raise SystemExit(f"Navify result is missing input field {field!r}")
    if result[field] != expected:
        raise SystemExit(
            f"Navify result changed input field {field!r}: "
            f"expected {expected!r}, received {result[field]!r}"
        )

def require_unit_interval(field):
    value = result[field]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise SystemExit(f"{field} must be a finite number between 0 and 1")


for field in result:
    if result[field] is not None and (
        field.startswith("prob_")
        or field.startswith("mix_ratio_")
        or (field.startswith("max_") and isinstance(result[field], (int, float)))
    ):
        require_unit_interval(field)

if mode == "retrospective":
    fluids = (
        "NS",
        "LR",
        "D5NS",
        "D5LR",
        "D5W",
        "D5halfNSwK",
        "D5halfNS",
        "halfNS",
        "Water",
    )
    required_retrospective_fields = {
        field
        for fluid in fluids
        for field in (
            f"prob_{fluid}_Retrospective",
            f"pred_{fluid}_Retrospective",
            f"mix_ratio_{fluid}",
        )
    }
    required_retrospective_fields.update(
        {
            "any_retrospective_pred",
            "any_retrospective_pred_with_LR",
            "max_retrospective_prob",
            "max_prob_fluid_retrospective",
            "max_retrospective_prob_with_LR",
            "max_mix_ratio",
            "max_mix_ratio_with_LR",
        }
    )
    missing_retrospective_fields = sorted(required_retrospective_fields - set(result))
    if missing_retrospective_fields:
        raise SystemExit(
            "Navify result is missing retrospective fields: "
            + ", ".join(missing_retrospective_fields)
        )
    null_retrospective_fields = sorted(
        field for field in required_retrospective_fields if result[field] is None
    )
    if null_retrospective_fields:
        raise SystemExit(
            "Navify retrospective result contains null computed fields: "
            + ", ".join(null_retrospective_fields)
        )
else:
    post_dependent_fields = {
        field for field in properties if field.endswith("_post")
    }
    post_dependent_fields.update(
        field
        for field in properties
        if field.endswith("_Retrospective") or field.startswith("mix_ratio_")
    )
    post_dependent_fields.update(
        {
            "any_retrospective_pred",
            "any_retrospective_pred_with_LR",
            "max_retrospective_prob",
            "max_prob_fluid_retrospective",
            "max_retrospective_prob_with_LR",
            "max_mix_ratio",
            "max_mix_ratio_with_LR",
        }
    )
    nonnull_post_dependent_fields = sorted(
        field for field in post_dependent_fields if result[field] is not None
    )
    if nonnull_post_dependent_fields:
        raise SystemExit(
            "Realtime result contains non-null post-dependent fields: "
            + ", ".join(nonnull_post_dependent_fields)
        )
PY
}

run_smoke_case() {
  local mode="$1"
  local label
  local request_file="examples/${mode}_request.json"
  local response_file="${RESPONSE_DIR}/${mode}_response.json"
  local http_status

  case "$mode" in
    realtime) label="Realtime" ;;
    retrospective) label="Retrospective" ;;
    *) echo "Unsupported smoke-test mode: $mode" >&2; return 1 ;;
  esac

  if ! http_status="$(curl --silent --show-error \
    --connect-timeout "$CONNECT_TIMEOUT_SECONDS" \
    --max-time "$EXECUTION_TIMEOUT_SECONDS" \
    --output "$response_file" \
    --write-out '%{http_code}' \
    --request POST \
    --header "@${AUTH_HEADER_FILE}" \
    --header "Accept: application/json" \
    --header "Content-Type: application/json" \
    --data-binary "@${request_file}" \
    "$EXECUTION_URL")"; then
    echo "${label} execution failed before Navify returned an HTTP response." >&2
    return 1
  fi

  if [[ "$http_status" != 2* ]]; then
    echo "${label} execution failed with HTTP ${http_status}." >&2
    print_error_response "$response_file"
    return 1
  fi

  if ! validate_response "$request_file" "$response_file" "$mode"; then
    echo "${label} execution returned an invalid algorithm response." >&2
    print_error_response "$response_file"
    return 1
  fi

  echo "${label} execution passed (HTTP ${http_status})."
}

run_smoke_case realtime
run_smoke_case retrospective

echo "Navify smoke test passed for ${ALGORITHM_ID} at ${EXECUTION_URL}"
