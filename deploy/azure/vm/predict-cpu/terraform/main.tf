locals {
  os_disk_size_gb = 64

  vnet_name      = "vnet-tlmtc-predict"
  subnet_name    = "snet-tlmtc-predict"
  nsg_name       = "nsg-tlmtc-predict"
  public_ip_name = "pip-tlmtc-predict"
  nic_name       = "nic-tlmtc-predict"
  vm_name        = "vm-tlmtc-predict"
  data_disk_name = "disk-tlmtc-predict-data"
}

data "azurerm_storage_account" "shared_storage" {
  name                = var.shared_storage_account_name
  resource_group_name = var.shared_storage_resource_group_name
}

resource "azurerm_resource_group" "predict" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_network_security_group" "predict" {
  name                = local.nsg_name
  location            = azurerm_resource_group.predict.location
  resource_group_name = azurerm_resource_group.predict.name

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

resource "azurerm_virtual_network" "predict" {
  name                = local.vnet_name
  location            = azurerm_resource_group.predict.location
  resource_group_name = azurerm_resource_group.predict.name
  address_space       = var.virtual_network_address_space
}

resource "azurerm_subnet" "predict" {
  name                            = local.subnet_name
  resource_group_name             = azurerm_resource_group.predict.name
  virtual_network_name            = azurerm_virtual_network.predict.name
  address_prefixes                = [var.subnet_address_prefix]
  default_outbound_access_enabled = false
}

resource "azurerm_subnet_network_security_group_association" "predict" {
  subnet_id                 = azurerm_subnet.predict.id
  network_security_group_id = azurerm_network_security_group.predict.id
}

resource "azurerm_public_ip" "predict" {
  name                = local.public_ip_name
  location            = azurerm_resource_group.predict.location
  resource_group_name = azurerm_resource_group.predict.name
  allocation_method   = "Static"
  sku                 = "Standard"
}

resource "azurerm_network_interface" "predict" {
  name                = local.nic_name
  location            = azurerm_resource_group.predict.location
  resource_group_name = azurerm_resource_group.predict.name

  ip_configuration {
    name                          = "ipconfig"
    subnet_id                     = azurerm_subnet.predict.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.predict.id
  }

  accelerated_networking_enabled = var.accelerated_networking_enabled
}

resource "azurerm_managed_disk" "predict" {
  name                 = local.data_disk_name
  location             = azurerm_resource_group.predict.location
  resource_group_name  = azurerm_resource_group.predict.name
  storage_account_type = "StandardSSD_LRS"
  create_option        = "Empty"
  disk_size_gb         = var.data_disk_size_gb
}

resource "azurerm_linux_virtual_machine" "predict" {
  name                  = local.vm_name
  location              = azurerm_resource_group.predict.location
  resource_group_name   = azurerm_resource_group.predict.name
  size                  = var.vm_size
  admin_username        = var.admin_username
  network_interface_ids = [azurerm_network_interface.predict.id]

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

  custom_data = base64encode(templatefile("${path.module}/cloud-init.tftpl", {
    bootstrap_cpu_host_script_b64 = filebase64("${path.module}/../scripts/bootstrap-cpu-host.sh")
    run_predict_script_b64        = filebase64("${path.module}/../scripts/run-predict.sh")
  }))

  boot_diagnostics {}
}

resource "azurerm_virtual_machine_data_disk_attachment" "predict" {
  managed_disk_id    = azurerm_managed_disk.predict.id
  virtual_machine_id = azurerm_linux_virtual_machine.predict.id
  lun                = "0"
  caching            = "None"
}

resource "azurerm_role_assignment" "training_outputs_reader" {
  scope                            = "${data.azurerm_storage_account.shared_storage.id}/blobServices/default/containers/training-outputs"
  role_definition_name             = "Storage Blob Data Reader"
  principal_id                     = azurerm_linux_virtual_machine.predict.identity[0].principal_id
  principal_type                   = "ServicePrincipal"
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "prediction_inputs_reader" {
  scope                            = "${data.azurerm_storage_account.shared_storage.id}/blobServices/default/containers/prediction-inputs"
  role_definition_name             = "Storage Blob Data Reader"
  principal_id                     = azurerm_linux_virtual_machine.predict.identity[0].principal_id
  principal_type                   = "ServicePrincipal"
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "prediction_outputs_contributor" {
  scope                            = "${data.azurerm_storage_account.shared_storage.id}/blobServices/default/containers/prediction-outputs"
  role_definition_name             = "Storage Blob Data Contributor"
  principal_id                     = azurerm_linux_virtual_machine.predict.identity[0].principal_id
  principal_type                   = "ServicePrincipal"
  skip_service_principal_aad_check = true
}
