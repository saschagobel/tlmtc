variable "location" {
  type        = string
  description = "Azure region for the train runner."
  default     = "germanywestcentral"
}

variable "resource_group_name" {
  type        = string
  description = "Name of the resource group for the disposable train runner."
  default     = "rg-tlmtc-train"
}

variable "shared_storage_resource_group_name" {
  type        = string
  description = "Name of the resource group containing the shared storage account."
  default     = "rg-tlmtc"
}

variable "shared_storage_account_name" {
  type        = string
  description = "Name of the shared storage account used for training inputs, HPO workspaces, and training outputs."
}

variable "virtual_network_address_space" {
  type        = list(string)
  description = "Address space of the train runner virtual network."
  default     = ["10.43.0.0/16"]
}

variable "subnet_address_prefix" {
  type        = string
  description = "Address prefix of the train runner subnet."
  default     = "10.43.1.0/24"
}

variable "ssh_source_address_prefix" {
  type        = string
  description = "Source address prefix allowed to connect to the train runner over SSH."
}

variable "admin_username" {
  type        = string
  description = "Administrator username for the train runner."
  default     = "azureuser"
}

variable "admin_ssh_public_key" {
  type        = string
  description = "SSH public key for the train runner administrator."
}

variable "vm_size" {
  type        = string
  description = "Azure NVIDIA CUDA GPU VM size for the train runner."
  default     = "Standard_NC4as_T4_v3"
}

variable "accelerated_networking_enabled" {
  type        = bool
  description = "Whether accelerated networking is enabled on the train runner NIC."
  default     = true
}

variable "data_disk_size_gb" {
  type        = number
  description = "Size of the train runner data disk in GiB."
  default     = 128
}
