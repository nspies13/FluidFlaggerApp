# Navify UI Examples

Paste the raw contents of either request file into the Navify synchronous execution UI. The request body must be one JSON object; do not wrap it in an array or a `BatchRefId`/`Payload` envelope.

- `realtime_request.json`: current and prior BMP values; produces realtime predictions and `null` post-dependent response values.
- `retrospective_request.json`: current, prior, and complete post BMP values; produces realtime, retrospective, and mix-ratio predictions.
- `realtime_response.json`: expected fixed-shape response for the realtime request.
- `retrospective_response.json`: expected fixed-shape response for the retrospective request. Use this when the UI accepts only one response example because every response property has a concrete value.
- `batch_request.json`: a raw request-schema-compatible object retained for manual UI testing; it is not a Navify batch-upload envelope.
- `invalid_partial_post_request.json`: a negative test that matches the JSON request schema but should return HTTP 400 because post values must be supplied as a complete set.

The request examples correspond to `../schemas/fluidflagger-bmp-request.schema.json`; the response examples correspond to `../schemas/fluidflagger-bmp-response.schema.json`.
