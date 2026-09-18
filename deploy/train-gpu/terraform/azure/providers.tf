provider "azurerm" {
  features {}

  resource_providers_to_register = [
    "Microsoft.Authorization",
    "Microsoft.Compute",
    "Microsoft.Network",
  ]
}
