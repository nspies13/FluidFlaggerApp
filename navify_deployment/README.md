# FluidFlagger BMP Navify Bundle

This directory contains the self-contained BMP-only FluidFlagger deployment for Navify Algorithm Suite.

Start here:

```bash
cd navify_deployment
python3 scripts/verify_models.py
pytest tests
./scripts/build_local.sh
./scripts/smoke_test.sh
```

After the algorithm is registered in Navify, exercise both hosted prediction modes with:

```bash
set -a
source .env
set +a
export TOKEN="$(./scripts/get_auth_token.sh)"
export EXECUTION_URL="${API_BASE}/tenants/${TENANT_ID}/algorithms/washu.fluidflagger/executions"
./scripts/smoke_test_navify.sh
```

The hosted smoke test sends `examples/realtime_request.json` and `examples/retrospective_request.json` directly to the synchronous private-algorithm execution endpoint documented in the onboarding guide and validates the returned predictions.

For manual testing in the Navify UI, paste one of those request files as the raw JSON body. Matching expected outputs are in `examples/realtime_response.json` and `examples/retrospective_response.json`; see `examples/README.md` for the complete fixture guide.

See `docs/deployment.md` for the registry push and deployment workflow.
