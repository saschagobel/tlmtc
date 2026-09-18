provider "azurerm" {
  features {}

  resource_providers_to_register = [
    "Microsoft.Authorization",
    "Microsoft.Storage",
  ]
  storage_use_azuread = true
}
