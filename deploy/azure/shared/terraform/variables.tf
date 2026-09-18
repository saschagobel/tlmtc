variable "location" {
  type        = string
  description = "Azure region for the shared resources."
  default     = "germanywestcentral"
}

variable "operator_principal_id" {
  type        = string
  description = "Object ID of the principal that manages shared blob data."
}

variable "resource_group_name" {
  type        = string
  description = "Name of the resource group for the shared resources."
  default     = "rg-tlmtc"
}

variable "storage_account_name" {
  type        = string
  description = "Globally unique name of the shared storage account."

  validation {
    condition     = can(regex("^[a-z0-9]{3,24}$", var.storage_account_name))
    error_message = "storage_account_name must contain 3 to 24 lowercase letters or numbers."
  }
}

variable "storage_account_replication_type" {
  type        = string
  description = "Replication type for the shared storage account."
  default     = "LRS"
}
