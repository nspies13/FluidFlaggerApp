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

## Create Deployment

```bash
cd navify_deployment
set -a; source .env; set +a
export TOKEN="$(./scripts/get_auth_token.sh)"
export TAG="1.0.0"
./scripts/create_deployment.sh
```

Poll the deployment details endpoint until `status` is `Deployed` and `applicationState` is `Healthy`.

## Runtime Contract

- `GET /health/live`
- `GET /health/ready`
- `POST /predict`

`POST /predict` accepts `application/json` only. The request body may be one object or an array of objects; the response is always a JSON array.

Required realtime fields are all current BMP values and all `_prior` values:

```text
sodium, chloride, potassium_plas, co2_totl, bun, creatinine, calcium, glucose
sodium_prior, chloride_prior, potassium_plas_prior, co2_totl_prior, bun_prior, creatinine_prior, calcium_prior, glucose_prior
```

Retrospective predictions are enabled only when all `_post` values are present. Partial post-specimen input returns HTTP 400.
