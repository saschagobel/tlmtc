output "resource_group_name" {
  description = "Name of the disposable predict resource group."
  value       = azurerm_resource_group.predict.name
}

output "vm_name" {
  description = "Name of the predict runner virtual machine."
  value       = azurerm_linux_virtual_machine.predict.name
}

output "public_ip_address" {
  description = "Public IPv4 address of the predict runner."
  value       = azurerm_public_ip.predict.ip_address
}

output "ssh_command" {
  description = "SSH command for connecting to the predict runner."
  value       = "ssh ${var.admin_username}@${azurerm_public_ip.predict.ip_address}"
}
