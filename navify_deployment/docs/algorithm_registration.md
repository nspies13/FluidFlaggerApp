# Algorithm Registration Notes

After the deployment is healthy in Navify Algorithm Suite, register the algorithm in the Navify UI.

Use these values for the first BMP-only release:

- Algorithm ID: `fluidflagger-bmp`
- Deployment: `fluidflagger-bmp-v1`
- Calculation endpoint path: `/predict`
- Request media type: `application/json`
- Response media type: `application/json`
- Liveness path: `/health/live`
- Readiness path: `/health/ready`
- Container port: `8080`

Execution can be tested through:

```text
POST /tenants/{tenantId}/algorithms/fluidflagger-bmp/executions
```

Use `examples/realtime_request.json` and `examples/retrospective_request.json` as initial payloads.
