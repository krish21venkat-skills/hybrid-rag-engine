# Production Incident Runbook

## Severity Levels
SEV1 means a full outage or data loss affecting all customers. SEV2 means a major feature is degraded for many customers. SEV3 means a minor issue with a workaround available. SEV4 is a cosmetic or low-impact defect.

## Response Targets
For SEV1, the on-call engineer must acknowledge within 5 minutes and an incident commander must be assigned within 15 minutes. Status updates go to the status page every 30 minutes. For SEV2, acknowledgement is required within 15 minutes with updates every 60 minutes. SEV3 incidents are handled within the next business day.

## Declaring an Incident
Open the incident channel named with the pattern inc-YYYYMMDD-short-name. Page the on-call engineer through the paging tool. The incident commander owns communication and decisions, while the operations lead owns technical investigation. Do not make production changes without the commander's approval.

## Rollback Procedure
First check the deployment dashboard for the latest release. If the incident began within 60 minutes of a release, roll back using the pipeline's rollback button. Verify that the error rate returns below 1 percent before closing. If rollback does not help, escalate to the database team.

## Post-Incident Review
A blameless review must be held within 5 working days of every SEV1 or SEV2 incident. The report must include a timeline, root cause, customer impact and at least 3 action items with named owners. Action items are tracked until completion and reviewed monthly.