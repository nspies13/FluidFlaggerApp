# FluidFlagger BMP Navify Deployment

This folder is a self-contained BMP-only deployment bundle for Navify Algorithm Suite Host and Connect.

It intentionally does not depend on Hugging Face or the parent application at runtime. The image bakes in the 27 BMP `.joblib` model artifacts and loads them from `/app/models`.

## WashU US Tenant Configuration

The current tenant values are stored in `.env.example`:

```text
TENANT_ID=e785ce6d-2098-4edc-9af2-4879481d433c
TENANT_ALIAS=washu
CLIENT_ID=f9a664ec-21fd-4fd3-b445-fe04cc8e7677
```

Copy it to `.env` and fill in `CLIENT_SECRET` when it is provided. `.env` is ignored by Git. The bundle uses the US Navify endpoints specified in the US onboarding guide.

```bash
cd navify_deployment
cp .env.example .env
set -a
source .env
set +a
```

## Get an Auth Token

After loading `.env`, obtain a bearer token with the client-credentials grant:

```bash
export TOKEN="$(./scripts/get_auth_token.sh)"
```

The helper uses `https://api.appprodus.platform.navify.com/api/v1/auth/protocols/oidc/token` and the required scope `default navify:tenant:${TENANT_ALIAS}`. The token is valid for the duration returned by the API (currently one hour).

Always source `.env` before requesting a fresh token. Do not persist a generated `TOKEN` in `.env`; sourcing an expired value later can turn an authenticated request into a bodyless HTTP `401` response.

## Local Build

```bash
cd navify_deployment
./scripts/build_local.sh
./scripts/smoke_test.sh
```

The build script verifies:

- non-root image user
- exactly one exposed port, `8080/tcp`
- `linux/amd64` image architecture
- model count, checksums, and loadability during the Docker build

## Navify-Hosted Execution Smoke Test

Request a fresh token and run the hosted smoke test after registering the algorithm:

```bash
cd navify_deployment
set -a
source .env
set +a
export TOKEN="$(./scripts/get_auth_token.sh)"
export EXECUTION_URL="${API_BASE}/tenants/${TENANT_ID}/algorithms/washu.fluidflagger/executions"
./scripts/smoke_test_navify.sh
```

The onboarding guide defines the synchronous private-algorithm execution endpoint as:

```text
POST https://api.us.prod.algosuite.navify.com/tenants/e785ce6d-2098-4edc-9af2-4879481d433c/algorithms/washu.fluidflagger/executions
```

The request requires `Authorization: Bearer ${TOKEN}` and `Content-Type: application/json`. An `x-correlation-id` header may be supplied to correlate an execution with the calculation endpoint. The request body is the JSON payload expected by FluidFlagger.

The documented responses are:

- `200`: successful execution with an `application/json` result
- `400`: invalid request
- `403`: invalid token or insufficient execution permission
- `404`: tenant or algorithm not found
- `502`: communication failure between AlgoSuite and the algorithm

HTTP `500` is not part of the onboarding guide's execution contract. Both smoke-test calls must return HTTP `200` and a valid FluidFlagger prediction response. For another tenant or algorithm, construct `EXECUTION_URL` with the same `/tenants/{tenantId}/algorithms/{algorithmId}/executions` path structure.

## Push To Navify Registry

```bash
cd navify_deployment
set -a; source .env; set +a
export TOKEN="$(./scripts/get_auth_token.sh)"
export TAG="1.0.0"
./scripts/push_navify.sh
```

The pushed image is:

```text
acr.us.prod.algosuite.navify.com/${TENANT_ALIAS}/fluidflagger-bmp:${TAG}
```

For the currently hosted release, this resolves to:

```text
acr.us.prod.algosuite.navify.com/washu/fluidflagger-bmp:1.0.0
```

## Current Navify-Hosted Deployment

The WashU tenant deployment API reported the following active deployment on 2026-07-17:

| Field | Current API value |
| --- | --- |
| Deployment ID | `019f6b6f-979f-7630-a256-bd59d8699d37` |
| Name | `fluidflagger` |
| Image | `acr.us.prod.algosuite.navify.com/washu/fluidflagger-bmp:1.0.0` |
| Version | `1.0.0` |
| Container port | `8080` |
| Owner | `e785ce6d-2098-4edc-9af2-4879481d433c` (`washu`) |
| Liveness check | `/health/live`, 30-second initial delay |
| Readiness check | `/health/ready`, 30-second initial delay |
| Environment variables | none |
| Deployment status | `Deployed` |
| Application state | `Healthy` |
| Created | `2026-07-16T14:58:23.262475+00:00` |
| Host URL | `http://fluidflagger.washu/` |

`Deployed` and `Healthy` mean that the application is running and passing health checks in the Navify AlgoSuite cluster. The `hostUrl` is a cluster-internal Navify service address, not a local Docker or workstation address.

## Verify the Current Deployment

Request a fresh token immediately before checking the collection and the specific deployment:

```bash
cd navify_deployment
set -a
source .env
set +a
export TOKEN="$(./scripts/get_auth_token.sh)"
export DEPLOYMENT_ID="019f6b6f-979f-7630-a256-bd59d8699d37"

curl --fail-with-body --silent --show-error \
  --write-out '\nHTTP %{http_code}\n' \
  "${API_BASE}/tenants/${TENANT_ID}/deployments" \
  --header "Authorization: Bearer ${TOKEN}"

curl --fail-with-body --silent --show-error \
  --write-out '\nHTTP %{http_code}\n' \
  "${API_BASE}/tenants/${TENANT_ID}/deployments/${DEPLOYMENT_ID}" \
  --header "Authorization: Bearer ${TOKEN}"
```

Both calls should return HTTP `200`. The list should contain the deployment ID above, and the details response should report `status: "Deployed"` and `applicationState: "Healthy"`. Printing the HTTP status prevents an empty-body authentication error from being mistaken for an empty deployment list.

## Create a New Deployment From the Template

```bash
cd navify_deployment
set -a
source .env
set +a
export TOKEN="$(./scripts/get_auth_token.sh)"
export TAG="1.0.0"
./scripts/create_deployment.sh
```

This command posts `deployment.json.template`. The current template names the requested deployment `fluidflagger-bmp-v1`; it therefore targets a different deployment name and does not update the active `fluidflagger` deployment or its ID. The API already contains a deleted `fluidflagger-bmp-v1` record, so another `POST` may conflict with that name. Use the deployment-specific `PUT` endpoint to update `fluidflagger`, and use `POST` only when intentionally creating a separate deployment.

## Runtime Contract

- `GET /health/live`
- `GET /health/ready`
- `POST /predict`

`POST /predict` accepts `application/json` only. The Navify request body is one BMP object, and the response is one complete JSON object containing every registered response property. For realtime-only requests, unavailable post-specimen, retrospective, and mix-ratio properties are `null`. A one-element request array remains accepted for local compatibility, but the response is still a single object.

Required realtime fields are all current BMP values and all `_prior` values:

```text
sodium, chloride, potassium_plas, co2_totl, bun, creatinine, calcium, glucose
sodium_prior, chloride_prior, potassium_plas_prior, co2_totl_prior, bun_prior, creatinine_prior, calcium_prior, glucose_prior
```

Retrospective predictions are enabled only when all `_post` values are present. Partial post-specimen input returns HTTP 400.

## Register the Hosted Algorithm

Open the US production registration page:

```text
https://ui.us.prod.algosuite.navify.com/en/registeralgorithm
```

Use the separately provisioned Navify UI email/password account. The current registration uses the external-host form as a workaround for the empty Navify deployment picker, with this configuration:

- Algorithm name: `fluidflagger`
- Executable algorithm ID: `washu.fluidflagger`
- Hosting option: `Hosted externally`
- Host: `http://fluidflagger.washu`
- Calculation endpoint path: `/predict`
- Synchronization: `Synchronous`
- User interface: `No`
- Endpoint authentication: `No authentication`
- Request media type: `application/json`
- Response media type: `application/json`

Despite the registration form's `Hosted externally` selection, `fluidflagger.washu` remains the cluster-internal address of the Navify-hosted deployment. `No authentication` applies to the hop from the AlgoSuite gateway to that calculation endpoint; callers must still authenticate to the AlgoSuite executions API.

If the Navify-hosted deployment picker is empty, inspect its deployment-list request in the browser developer tools:

- The path should contain the WashU tenant ID above.
- HTTP `200` should return an array containing deployment ID `019f6b6f-979f-7630-a256-bd59d8699d37`.
- HTTP `401` or `403` indicates a UI-user authentication or permission problem.
- HTTP `200` containing the deployment while the picker remains empty indicates a registration UI filtering problem. Capture that response and the browser console output for Navify support.
- A request made with a different tenant ID, or HTTP `200` with an empty array, indicates that the UI user is mapped to a different tenant.

The API client credentials in `.env` and the email/password UI account are separate identities. The UI account must be assigned to the WashU tenant and allowed to read deployments and create algorithms.
