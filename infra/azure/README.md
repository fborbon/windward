# Azure resources — historical (decommissioned 2026-09-09)

This documents what was actually provisioned and run during the project's original Azure MLflow skill-demonstration phase. See [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) for why the project migrated off this to AWS. **The resource group below has been deleted** — kept here as a record of what was really built and verified, not a live setup guide.

| Resource | Name | Purpose | Cost note |
|---|---|---|---|
| Resource group | `rg-windward` | Container for everything below | Free |
| Region | `swedencentral` | `westeurope` rejected new subscriptions on this account at creation time | — |
| Azure ML workspace | `windward-mlflow` | MLflow tracking server + model registry | Free to hold; pay only for compute/storage actually used. No compute cluster attached — training ran locally. |
| Storage account | auto-created (`windwardstorage...`) | Backs the ML workspace | ~cents/month at this scale, within free-tier storage allowance |
| Key Vault | auto-created (`windwardkeyvault...`) | Backs the ML workspace | Free tier covers this usage |
| Log Analytics + App Insights | auto-created | Backs the ML workspace | Minimal, within free allowance |
| AKS | `windward-aks` (`Standard_B2s_v2`) | Real deployment, verified via `/health` through `kubectl port-forward`, then deleted the same session | ~$0.01–0.02 for the ~30 minutes it existed |
| ACR | `windwardacr...` | Temporary image registry for the AKS deployment | Deleted immediately after the AKS demo |

Auth was `az login` (interactive, browser-based) + `DefaultAzureCredential` in code — no API keys stored anywhere.

**Never provisioned:** Azure OpenAI (not part of the free tier; Bedrock Nova covered the same role throughout).

## What replaced it

Self-hosted MLflow + S3 artifact store on the existing AWS EC2 fleet. See [infra/aws/README.md](../aws/README.md).
