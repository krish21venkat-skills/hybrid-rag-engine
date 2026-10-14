# Platform Technical Notes

## Service Overview
The Orders API is a REST service written in Python and deployed as containers on Kubernetes. It handles order creation, status updates and cancellations. The service stores data in PostgreSQL and uses Redis for caching. Each request is expected to complete within 800 milliseconds at the 95th percentile.

## Authentication
All endpoints require a bearer token issued by the identity service. Tokens expire after 60 minutes. Clients must refresh the token using the refresh endpoint before expiry, and refresh tokens are valid for 7 days. Requests with an expired token receive HTTP 401 with the error code E-4010.

## Rate Limiting
Each API key is limited to 100 requests per minute. When the limit is exceeded, the service returns HTTP 429 with the error code E-4290 and a Retry-After header. Batch endpoints are limited to 10 requests per minute and accept at most 500 orders per request.

## Common Errors
Error E-4021 means the order payload failed schema validation, usually because the customer_id field is missing or the quantity is not a positive integer. Error E-5003 indicates a database connection timeout. The first remedy is to check the connection pool, which is sized at 20 connections per instance. Error E-5009 means the downstream payment gateway is unavailable, and the service retries 3 times with exponential backoff before failing.

## Deployment Process
Releases are deployed every Tuesday and Thursday at 2:00 PM IST using a rolling update strategy. Each release must pass unit tests, integration tests and a security scan. If the error rate exceeds 2 percent within 15 minutes of a release, the pipeline automatically rolls back to the previous version. Hotfixes can be deployed outside the schedule with approval from the on-call engineering manager.