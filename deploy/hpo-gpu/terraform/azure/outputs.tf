output "resource_group_name" {
  description = "Name of the disposable HPO resource group."
  value       = azurerm_resource_group.hpo.name
}

output "vm_name" {
  description = "Name of the HPO runner virtual machine."
  value       = azurerm_linux_virtual_machine.hpo.name
}

output "public_ip_address" {
  description = "Public IPv4 address of the HPO runner."
  value       = azurerm_public_ip.hpo.ip_address
}

output "ssh_command" {
  description = "SSH command for connecting to the HPO runner."
  value       = "ssh ${var.admin_username}@${azurerm_public_ip.hpo.ip_address}"
}
