# Shared Azure storage

This Terraform root provisions the persistent Azure Blob Storage handoff shared by disposable tlmtc runners. It creates one resource group, one storage account, and private containers for pipeline inputs and outputs.

## Prerequisites

- Terraform 1.10 or later
- Azure CLI
- An Azure subscription in which the signed-in identity can create resource groups, storage resources, and role assignments

## Deploy

From the repository or deployment bundle root:

```bash
cd deploy/shared/terraform/azure

az login
az account set --subscription "<subscription-id-or-name>"

export TF_VAR_storage_account_name="<globally-unique-storage-account-name>"
export TF_VAR_operator_principal_id="$(az ad signed-in-user show --query id --output tsv)"

terraform init
terraform plan -out=shared.tfplan
terraform apply shared.tfplan
```

The storage account name must contain 3 to 24 lowercase letters or numbers. The operator principal receives `Storage Blob Data Contributor` access to the storage account. Set `TF_VAR_operator_principal_id` explicitly to use a group or service principal instead of the signed-in user. Region, resource group name, and replication type have defaults and can also be overridden with `TF_VAR_` environment variables or Terraform CLI options.

Show the deployed resource identifiers and container URLs:

```bash
terraform output
```

## Destroy

```bash
terraform destroy
```

> [!CAUTION]
> Destroying this Terraform root deletes the shared storage account and its pipeline data. Destroy disposable runner infrastructure separately.
