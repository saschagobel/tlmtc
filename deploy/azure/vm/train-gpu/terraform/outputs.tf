output "resource_group_name" {
  description = "Name of the disposable train resource group."
  value       = azurerm_resource_group.train.name
}

output "vm_name" {
  description = "Name of the train runner virtual machine."
  value       = azurerm_linux_virtual_machine.train.name
}

output "public_ip_address" {
  description = "Public IPv4 address of the train runner."
  value       = azurerm_public_ip.train.ip_address
}

output "ssh_command" {
  description = "SSH command for connecting to the train runner."
  value       = "ssh ${var.admin_username}@${azurerm_public_ip.train.ip_address}"
}
