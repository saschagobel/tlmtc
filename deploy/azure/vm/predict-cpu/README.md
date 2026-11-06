# Predict CPU runner

The Terraform configuration in `terraform/` provisions a disposable Azure CPU runner for tlmtc ONNX batch prediction. It creates one resource group, an isolated virtual network with an SSH-only NSG rule, a static public IP, one CPU VM with a system-assigned identity, a data disk, cloud-init CPU-host bootstrap wiring, and least-privilege Blob access.

## Prerequisites

- Terraform 1.10 or later
- Azure CLI
- An Azure subscription in which the signed-in identity can create compute, network, and role-assignment resources
- Sufficient vCPU quota for the selected VM size in the target region
- Shared storage deployed from `deploy/azure/shared/terraform`
- A completed training run with ONNX export published to the `training-outputs` container
- An SSH key pair

## Deploy

From the repository or deployment bundle root:

```bash
cd deploy/azure/vm/predict-cpu/terraform

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
terraform plan -out=predict.tfplan
terraform apply predict.tfplan
```

Region, resource group names, network prefixes, admin username, VM size, and data-disk size have defaults. Override them with `TF_VAR_` environment variables, `terraform.tfvars`, or Terraform CLI options.

Terraform can finish before cloud-init. The VM needs several minutes to bootstrap before it is ready for prediction.

Connect using the matching private key. Replace `azureuser` if you changed `admin_username`.

```bash
ssh -i "<path-to-private-key>" \
  "azureuser@$(terraform output -raw public_ip_address)"
```

After connecting, wait for cloud-init to finish. The command returns immediately if it is already complete and exits nonzero on errors:

```bash
cloud-init status --wait
```

## Run prediction

`run-predict` downloads the selected training run and the prediction inputs, runs batch prediction with the ONNX backend on CPU, and uploads the prediction outputs using the VM's managed identity. An explicit `--run-id` is required.

From a local shell, upload the unlabeled data to `prediction-inputs` if it is not already there. The blob path is arbitrary:

```bash
az storage blob upload \
  --account-name "<storage-account-name>" \
  --container-name prediction-inputs \
  --name "<blob-path>" \
  --file "<path-to-local-prediction-data>" \
  --auth-mode login \
  --overwrite true
```

On the VM, run prediction against a published training run. Use `/inputs/<blob-path>` for data uploaded under `<blob-path>`:

```bash
sudo run-predict \
  "<storage-account-name>" \
  "<predict-image-ref>" \
  --unlabeled-data "/inputs/<blob-path>" \
  --run-id <run-id>
```

`run-predict` restores `training-outputs/<run-id>/` to the local work directory, enforces `--inference-backend onnx --use-cpu --work-dir /workspace`, and uploads `prediction_outputs/<run-id>/` to the `prediction-outputs` container. Confirm that `run-predict` completes successfully before destroying the runner.

Repeating prediction with the same run ID replaces its prediction results locally and in Blob Storage; copy results elsewhere first if you need to retain multiple batches.

Run the following lifecycle commands locally from `deploy/azure/vm/predict-cpu/terraform`.

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

> [!NOTE]
> Destroying this Terraform configuration deletes the predict VM, its disks, network, and public IP. Shared storage and Blob data are unaffected.
