# Source Boundary

Allowed sources:

- User-provided market-data CSV/Parquet files and point-in-time fundamental manifests that the user has rights to use.
- Public papers, documentation, and examples with attribution.
- Synthetic or toy data included in this repository for mechanical validation.
- User-provided seed factor definitions and parameter files.

Not allowed unless the user has rights and explicitly provides them:

- Private datasets, leaked data, paywalled data exports, or confidential documents.
- API keys, private tokens, account credentials, or `.env` files.
- Claims that generated factors are safe, guaranteed profitable, production-ready, or investment advice.

When publishing examples, avoid local absolute paths, workflow logs, private data, or long-lived experiment artifacts.
