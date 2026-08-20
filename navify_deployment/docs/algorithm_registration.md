# Algorithm Registration Notes

After the deployment is healthy in Navify Algorithm Suite, register the algorithm in the Navify UI.

Use these values for the current BMP-only release:

- Algorithm name: `fluidflagger`
- Executable algorithm ID: `washu.fluidflagger`
- Navify-hosted deployment: `fluidflagger`
- Hosting option: `Hosted externally` (deployment-picker workaround)
- Host: `http://fluidflagger.washu`
- Calculation endpoint path: `/predict`
- Request media type: `application/json`
- Response media type: `application/json`
- Liveness path: `/health/live`
- Readiness path: `/health/ready`
- Container port: `8080`

Register `schemas/fluidflagger-bmp-response.schema.json` as `ResponseOk`. Its fixed 81-property object shape is intentional: realtime-only executions return `null` for the 42 unavailable post-specimen, retrospective, and mix-ratio properties because Navify's hosted execution path does not reliably handle omitted response properties.

Execution can be tested through:

```text
POST https://api.us.prod.algosuite.navify.com/tenants/e785ce6d-2098-4edc-9af2-4879481d433c/algorithms/washu.fluidflagger/executions
```

The request must include a bearer token and `Content-Type: application/json`; `x-correlation-id` is optional. The documented responses are HTTP `200`, `400`, `403`, `404`, and `502`. A `502` indicates a communication problem between AlgoSuite and the calculation endpoint; HTTP `500` is not part of the documented execution contract.

Run `scripts/smoke_test_navify.sh` with a fresh bearer token and set `EXECUTION_URL` to the endpoint above to execute and validate both `examples/realtime_request.json` and `examples/retrospective_request.json`.

For the Navify UI execution tester, paste the contents of either request file directly, without an array or batch envelope. `examples/realtime_response.json` and `examples/retrospective_response.json` provide schema-compatible expected results.
