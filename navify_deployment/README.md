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

See `docs/deployment.md` for the registry push and deployment workflow.
