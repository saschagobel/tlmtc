# HPO GPU runner

This Terraform root provisions a disposable Azure GPU runner for tlmtc HPO. It creates one resource group, an isolated virtual network with an SSH-only NSG rule, a static public IP, one GPU VM with a system-assigned identity, a data disk, cloud-init GPU-host bootstrap wiring, and least-privilege Blob access.

## Prerequisites

- Terraform 1.10 or later
- Azure CLI
- An Azure subscription in which the signed-in identity can create compute, network, and role-assignment resources
- Sufficient GPU quota in the target region
- Shared storage deployed from `deploy/shared/terraform/azure`
- An SSH key pair

## Deploy

From the repository or deployment bundle root:

```bash
cd deploy/hpo-gpu/terraform/azure

az login
az account set --subscription "<subscription-id-or-name>"
```

Provide the three required values via `terraform.tfvars` or directly as environment variables:

```bash
cp terraform.tfvars.example terraform.tfvars
# Fill in shared_storage_account_name, ssh_source_address_prefix ("<your-ip>/32"), admin_ssh_public_key.
```

Or

```bash
export TF_VAR_shared_storage_account_name="<storage-account-name>"
export TF_VAR_ssh_source_address_prefix="<your-public-ip>/32"
export TF_VAR_admin_ssh_public_key="$(cat "<path-to-public-key>")"
```

Then:

```bash
terraform init
terraform plan -out=hpo.tfplan
terraform apply hpo.tfplan
```

Region, resource group names, network prefixes, admin username, VM size, and data-disk size have defaults. Override them with `TF_VAR_` environment variables, `terraform.tfvars`, or Terraform CLI options.

The T4 configuration is the default example. Other NVIDIA GPU sizes, including A100 and H100 families, can be selected with `vm_size` when available and covered by the subscription's quota.

Terraform can finish before cloud-init. The VM needs several minutes to bootstrap and reboots once before it is ready for HPO.

Connect using the matching private key. Replace `azureuser` if you changed `admin_username`.

```bash
ssh -i "<path-to-private-key>" \
  "azureuser@$(terraform output -raw public_ip_address)"
```

After connecting, wait for cloud-init to finish. The command returns immediately if it is already complete and exits nonzero on errors:

```bash
cloud-init status --wait
```

The VM reboots once during bootstrap. If SSH disconnects, reconnect when it is available again, then verify the GPU:

```bash
nvidia-smi
```

## Run HPO

From a local shell, upload the labeled data to the `training-inputs` container. The blob path is arbitrary:

```bash
az storage blob upload \
  --account-name "<storage-account-name>" \
  --container-name training-inputs \
  --name "<blob-path>" \
  --file "<path-to-local-training-data>" \
  --auth-mode login \
  --overwrite true
```

On the VM, run the HPO image. Use `/inputs/<blob-path>` for data uploaded under `<blob-path>`:

```bash
sudo run-hpo \
  "<storage-account-name>" \
  "<hpo-image-ref>" \
  --labeled-data "/inputs/<blob-path>" \
  --run-id hpo-01 \
  --tuning-trials 5
```

`run-hpo` downloads the Blob inputs, runs the container, and uploads the workspace using the VM's managed identity. Use a new run ID for each experiment, or reuse one to restore its Optuna workspace and add the requested tuning trials. Confirm that `run-hpo` completes successfully before destroying the runner.

Run the following lifecycle commands locally from `deploy/hpo-gpu/terraform/azure`.

## Pause

Deallocate the VM whenever work pauses:

```bash
az vm deallocate \
  --resource-group "$(terraform output -raw resource_group_name)" \
  --name "$(terraform output -raw vm_name)"
```

Deallocation stops the VM compute charge. The managed disks, static public IP, and usage-based Blob Storage remain billable while retained.

Resume with `az vm start` using the same resource group and VM name.

## Destroy

```bash
terraform destroy
```

> [!CAUTION]
> Destroying this Terraform root deletes the HPO VM, its disks, network, and public IP. Shared storage and Blob data are unaffected.
