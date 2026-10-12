# data/inventario_julio_2025_limpio.csv

Stock snapshot of Display Internacional SRL, exported July 2025 and cleaned (duplicates removed,
hidden negative stock recovered, empty location column dropped, descriptions normalised).
2,098 rows, one per SKU.

| Column | Meaning | Use in model |
|---|---|---|
| Código | SKU code; alphabetic prefix often encodes a product line | feature (prefix) |
| Descripción | free-text description, Spanish with English/abbrev mix | feature (parsed) |
| Tipo | Producto / Servicio (keyword-derived) | filter: Producto only |
| Stock | units on hand (negative = sold more than received) | **excluded — leakage/irrelevant** |
| Estado stock | Negativo / Sin stock / Bajo mínimo / En stock | **excluded** |
| Costo | average unit cost, DOP (assumed) | **TARGET** (as log) |
| Precio detalle | retail price | **excluded — markup × cost** |
| Precio mayor 1/2/3 | wholesale tiers | **excluded — markup × cost** |
| Precio mínimo | floor price | **excluded** |
| Margen % | (Precio − Costo)/Precio | **excluded** |
| Valor inventario | Stock × Costo | **excluded** |
| Empaque | units per pack | feature |
| Suplidor (ID) | supplier code (no names available) | feature (freq-encoded) |
| Área (ID) | area code | feature (freq-encoded) |
| Dpto (ID) | department code | feature (freq + top-N one-hot) |
| Punto reorden / Cant. reorden | reorder point / order qty | **excluded** |
| Alertas | data-quality flags | **excluded** (use to filter if needed) |
