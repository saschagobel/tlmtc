locals {
  os_disk_size_gb = 64

  vnet_name      = "vnet-tlmtc-hpo"
  subnet_name    = "snet-tlmtc-hpo"
  nsg_name       = "nsg-tlmtc-hpo"
  public_ip_name = "pip-tlmtc-hpo"
  nic_name       = "nic-tlmtc-hpo"
  vm_name        = "vm-tlmtc-hpo"
  data_disk_name = "disk-tlmtc-hpo-data"
}

data "azurerm_storage_account" "shared_storage" {
  name                = var.shared_storage_account_name
  resource_group_name = var.shared_storage_resource_group_name
}

resource "azurerm_resource_group" "hpo" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_network_security_group" "hpo" {
  name                = local.nsg_name
  location            = azurerm_resource_group.hpo.location
  resource_group_name = azurerm_resource_group.hpo.name

  security_rule {
    name                       = "AllowSshFromTester"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = var.ssh_source_address_prefix
    destination_address_prefix = "*"
  }
}

resource "azurerm_virtual_network" "hpo" {
  name                = local.vnet_name
  location            = azurerm_resource_group.hpo.location
  resource_group_name = azurerm_resource_group.hpo.name
  address_space       = var.virtual_network_address_space
}

resource "azurerm_subnet" "hpo" {
  name                            = local.subnet_name
  resource_group_name             = azurerm_resource_group.hpo.name
  virtual_network_name            = azurerm_virtual_network.hpo.name
  address_prefixes                = [var.subnet_address_prefix]
  default_outbound_access_enabled = false
}

resource "azurerm_subnet_network_security_group_association" "hpo" {
  subnet_id                 = azurerm_subnet.hpo.id
  network_security_group_id = azurerm_network_security_group.hpo.id
}

resource "azurerm_public_ip" "hpo" {
  name                = local.public_ip_name
  location            = azurerm_resource_group.hpo.location
  resource_group_name = azurerm_resource_group.hpo.name
  allocation_method   = "Static"
  sku                 = "Standard"
}

resource "azurerm_network_interface" "hpo" {
  name                = local.nic_name
  location            = azurerm_resource_group.hpo.location
  resource_group_name = azurerm_resource_group.hpo.name

  ip_configuration {
    name                          = "ipconfig"
    subnet_id                     = azurerm_subnet.hpo.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.hpo.id
  }

  accelerated_networking_enabled = var.accelerated_networking_enabled
}

resource "azurerm_managed_disk" "hpo" {
  name                 = local.data_disk_name
  location             = azurerm_resource_group.hpo.location
  resource_group_name  = azurerm_resource_group.hpo.name
  storage_account_type = "StandardSSD_LRS"
  create_option        = "Empty"
  disk_size_gb         = var.data_disk_size_gb
}

resource "azurerm_linux_virtual_machine" "hpo" {
  name                  = local.vm_name
  location              = azurerm_resource_group.hpo.location
  resource_group_name   = azurerm_resource_group.hpo.name
  size                  = var.vm_size
  admin_username        = var.admin_username
  network_interface_ids = [azurerm_network_interface.hpo.id]

  admin_ssh_key {
    username   = var.admin_username
    public_key = var.admin_ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "StandardSSD_LRS"
    disk_size_gb         = local.os_disk_size_gb
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "ubuntu-24_04-lts"
    sku       = "server"
    version   = "latest"
  }

  identity {
    type = "SystemAssigned"
  }

  # The bootstrap installs the NVIDIA driver without Secure Boot enrollment.
  secure_boot_enabled = false
  vtpm_enabled        = false

  custom_data = base64encode(templatefile("${path.module}/cloud-init.tftpl", {
    bootstrap_gpu_host_script_b64 = filebase64("${path.module}/../../scripts/bootstrap-gpu-host.sh")
    run_hpo_script_b64            = filebase64("${path.module}/../../scripts/run-hpo.sh")
  }))

  boot_diagnostics {}
}

resource "azurerm_virtual_machine_data_disk_attachment" "hpo" {
  managed_disk_id    = azurerm_managed_disk.hpo.id
  virtual_machine_id = azurerm_linux_virtual_machine.hpo.id
  lun                = "0"
  caching            = "None"
}

resource "azurerm_role_assignment" "training_inputs_reader" {
  scope                            = "${data.azurerm_storage_account.shared_storage.id}/blobServices/default/containers/training-inputs"
  role_definition_name             = "Storage Blob Data Reader"
  principal_id                     = azurerm_linux_virtual_machine.hpo.identity[0].principal_id
  principal_type                   = "ServicePrincipal"
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "hpo_workspaces_contributor" {
  scope                            = "${data.azurerm_storage_account.shared_storage.id}/blobServices/default/containers/hpo-workspaces"
  role_definition_name             = "Storage Blob Data Contributor"
  principal_id                     = azurerm_linux_virtual_machine.hpo.identity[0].principal_id
  principal_type                   = "ServicePrincipal"
  skip_service_principal_aad_check = true
}
