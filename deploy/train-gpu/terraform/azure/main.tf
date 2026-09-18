locals {
  os_disk_size_gb = 64

  vnet_name      = "vnet-tlmtc-train"
  subnet_name    = "snet-tlmtc-train"
  nsg_name       = "nsg-tlmtc-train"
  public_ip_name = "pip-tlmtc-train"
  nic_name       = "nic-tlmtc-train"
  vm_name        = "vm-tlmtc-train"
  data_disk_name = "disk-tlmtc-train-data"
}

data "azurerm_storage_account" "shared_storage" {
  name                = var.shared_storage_account_name
  resource_group_name = var.shared_storage_resource_group_name
}

resource "azurerm_resource_group" "train" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_network_security_group" "train" {
  name                = local.nsg_name
  location            = azurerm_resource_group.train.location
  resource_group_name = azurerm_resource_group.train.name

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

resource "azurerm_virtual_network" "train" {
  name                = local.vnet_name
  location            = azurerm_resource_group.train.location
  resource_group_name = azurerm_resource_group.train.name
  address_space       = var.virtual_network_address_space
}

resource "azurerm_subnet" "train" {
  name                            = local.subnet_name
  resource_group_name             = azurerm_resource_group.train.name
  virtual_network_name            = azurerm_virtual_network.train.name
  address_prefixes                = [var.subnet_address_prefix]
  default_outbound_access_enabled = false
}

resource "azurerm_subnet_network_security_group_association" "train" {
  subnet_id                 = azurerm_subnet.train.id
  network_security_group_id = azurerm_network_security_group.train.id
}

resource "azurerm_public_ip" "train" {
  name                = local.public_ip_name
  location            = azurerm_resource_group.train.location
  resource_group_name = azurerm_resource_group.train.name
  allocation_method   = "Static"
  sku                 = "Standard"
}

resource "azurerm_network_interface" "train" {
  name                = local.nic_name
  location            = azurerm_resource_group.train.location
  resource_group_name = azurerm_resource_group.train.name

  ip_configuration {
    name                          = "ipconfig"
    subnet_id                     = azurerm_subnet.train.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.train.id
  }

  accelerated_networking_enabled = var.accelerated_networking_enabled
}

resource "azurerm_managed_disk" "train" {
  name                 = local.data_disk_name
  location             = azurerm_resource_group.train.location
  resource_group_name  = azurerm_resource_group.train.name
  storage_account_type = "StandardSSD_LRS"
  create_option        = "Empty"
  disk_size_gb         = var.data_disk_size_gb
}

resource "azurerm_linux_virtual_machine" "train" {
  name                  = local.vm_name
  location              = azurerm_resource_group.train.location
  resource_group_name   = azurerm_resource_group.train.name
  size                  = var.vm_size
  admin_username        = var.admin_username
  network_interface_ids = [azurerm_network_interface.train.id]

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
    bootstrap_gpu_host_script_b64 = filebase64("${path.module}/../../../shared/scripts/bootstrap-gpu-host.sh")
    run_train_script_b64          = filebase64("${path.module}/../../scripts/run-train.sh")
  }))

  boot_diagnostics {}
}

resource "azurerm_virtual_machine_data_disk_attachment" "train" {
  managed_disk_id    = azurerm_managed_disk.train.id
  virtual_machine_id = azurerm_linux_virtual_machine.train.id
  lun                = "0"
  caching            = "None"
}

resource "azurerm_role_assignment" "training_inputs_reader" {
  scope                            = "${data.azurerm_storage_account.shared_storage.id}/blobServices/default/containers/training-inputs"
  role_definition_name             = "Storage Blob Data Reader"
  principal_id                     = azurerm_linux_virtual_machine.train.identity[0].principal_id
  principal_type                   = "ServicePrincipal"
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "hpo_workspaces_reader" {
  scope                            = "${data.azurerm_storage_account.shared_storage.id}/blobServices/default/containers/hpo-workspaces"
  role_definition_name             = "Storage Blob Data Reader"
  principal_id                     = azurerm_linux_virtual_machine.train.identity[0].principal_id
  principal_type                   = "ServicePrincipal"
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "training_outputs_contributor" {
  scope                            = "${data.azurerm_storage_account.shared_storage.id}/blobServices/default/containers/training-outputs"
  role_definition_name             = "Storage Blob Data Contributor"
  principal_id                     = azurerm_linux_virtual_machine.train.identity[0].principal_id
  principal_type                   = "ServicePrincipal"
  skip_service_principal_aad_check = true
}
