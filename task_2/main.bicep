param prefix string = 'myfirstkv'
param location string = resourceGroup().location

// Generowanie unikalnej nazwy Key Vaulta
var keyVaultName = '${prefix}${uniqueString(resourceGroup().id)}'

// Przycinanie do maksymalnie 24 znaków
var keyVaultNameTrimmed = substring(keyVaultName, 0, 24)

resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: keyVaultNameTrimmed
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      name: 'standard'
      family: 'A'
    }
    accessPolicies: [] // Możesz dodać odpowiednie polityki dostępu
    enableSoftDelete: true
  }
}

output vaultName string = keyVault.name
