resource "azurerm_resource_group" "shared" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_storage_account" "shared" {
  name                      = var.storage_account_name
  resource_group_name       = azurerm_resource_group.shared.name
  location                  = azurerm_resource_group.shared.location
  account_kind              = "StorageV2"
  account_tier              = "Standard"
  account_replication_type  = var.storage_account_replication_type
  access_tier               = "Hot"
  shared_access_key_enabled = false
}

resource "azurerm_storage_container" "shared" {
  for_each = toset([
    "training-inputs",
    "hpo-workspaces",
    "training-outputs",
    "prediction-inputs",
    "prediction-outputs",
  ])

  name                  = each.value
  storage_account_id    = azurerm_storage_account.shared.id
  container_access_type = "private"
}

resource "azurerm_role_assignment" "operator_blob_data" {
  scope                = azurerm_storage_account.shared.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = var.operator_principal_id
}
