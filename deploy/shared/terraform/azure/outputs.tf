output "container_ids" {
  description = "Resource IDs of the shared blob containers, keyed by container name."
  value = {
    for name, container in azurerm_storage_container.shared : name => container.id
  }
}

output "container_urls" {
  description = "URLs of the shared blob containers, keyed by container name."
  value = {
    for name, container in azurerm_storage_container.shared : name => container.url
  }
}

output "resource_group_name" {
  description = "Name of the resource group containing the shared resources."
  value       = azurerm_resource_group.shared.name
}

output "storage_account_id" {
  description = "Resource ID of the shared storage account."
  value       = azurerm_storage_account.shared.id
}

output "storage_account_name" {
  description = "Name of the shared storage account."
  value       = azurerm_storage_account.shared.name
}
