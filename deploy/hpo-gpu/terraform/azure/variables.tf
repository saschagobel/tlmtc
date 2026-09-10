variable "location" {
  type        = string
  description = "Azure region for the HPO runner."
  default     = "germanywestcentral"
}

variable "resource_group_name" {
  type        = string
  description = "Name of the resource group for the disposable HPO runner."
  default     = "rg-tlmtc-hpo"
}

variable "shared_storage_resource_group_name" {
  type        = string
  description = "Name of the resource group containing the shared storage account."
  default     = "rg-tlmtc"
}

variable "shared_storage_account_name" {
  type        = string
  description = "Name of the shared storage account used for HPO inputs and workspaces."
}

variable "virtual_network_address_space" {
  type        = list(string)
  description = "Address space of the HPO runner virtual network."
  default     = ["10.42.0.0/16"]
}

variable "subnet_address_prefix" {
  type        = string
  description = "Address prefix of the HPO runner subnet."
  default     = "10.42.1.0/24"
}

variable "ssh_source_address_prefix" {
  type        = string
  description = "Source address prefix allowed to connect to the HPO runner over SSH."
}

variable "admin_username" {
  type        = string
  description = "Administrator username for the HPO runner."
  default     = "azureuser"
}

variable "admin_ssh_public_key" {
  type        = string
  description = "SSH public key for the HPO runner administrator."
}

variable "vm_size" {
  type        = string
  description = "Azure NVIDIA CUDA GPU VM size for the HPO runner."
  default     = "Standard_NC4as_T4_v3"
}

variable "accelerated_networking_enabled" {
  type        = bool
  description = "Whether accelerated networking is enabled on the HPO runner NIC."
  default     = true
}

variable "data_disk_size_gb" {
  type        = number
  description = "Size of the HPO runner data disk in GiB."
  default     = 128
}
