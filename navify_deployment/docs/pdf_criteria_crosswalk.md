# Navify PDF Criteria Crosswalk

The confidential onboarding PDF is not copied into this repository. This crosswalk records the deployment criteria implemented by this folder.

| PDF criterion | Implementation |
| --- | --- |
| Calculation endpoint accepts `application/json` | `POST /predict` accepts only `application/json`. |
| Successful algorithm execution returns HTTP 200 | Valid BMP requests return HTTP 200 with a JSON array. |
| Invalid input returns HTTP 400 | Missing fields, partial post fields, malformed JSON, nonnumeric values, and wrong content type return HTTP 400. |
| Liveness endpoint | `GET /health/live`. |
| Readiness endpoint | `GET /health/ready`; returns 200 only after all 27 BMP models load. |
| Non-root container user | Dockerfile creates and uses UID `1000`. |
| Exactly one exposed port | Dockerfile exposes only `8080`. |
| Target architecture is `linux/amd64` | Build and push scripts use `--platform linux/amd64`. |
| OCI image manifest | Build helper creates a `linux/amd64` image and the standard Docker tag/push workflow publishes an OCI manifest (verified in the Navify registry). |
| Registry image naming | Push script uses `acr.us.prod.algosuite.navify.com/${TENANT_ALIAS}/fluidflagger-bmp:${TAG}`. |
| Deployment body includes image, version, port, health checks | `deployment.json.template` supplies these fields. |
| Register algorithm after healthy deployment | `docs/algorithm_registration.md` records registration values. |
