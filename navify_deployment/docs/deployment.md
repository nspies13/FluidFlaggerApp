# FluidFlagger BMP Navify Deployment

This folder is a self-contained BMP-only deployment bundle for Navify Algorithm Suite Host and Connect.

It intentionally does not depend on Hugging Face or the parent application at runtime. The image bakes in the 27 BMP `.joblib` model artifacts and loads them from `/app/models`.

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
export TENANT_ALIAS="your-tenant-alias"
export TOKEN="your-navify-bearer-token"
export TAG="1.0.0"
./scripts/push_navify.sh
```

The pushed image is:

```text
acr.prod.algosuite.navify.com/${TENANT_ALIAS}/fluidflagger-bmp:${TAG}
```

## Create Deployment

```bash
cd navify_deployment
export TENANT_ID="your-tenant-id"
export TENANT_ALIAS="your-tenant-alias"
export TOKEN="your-navify-bearer-token"
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
