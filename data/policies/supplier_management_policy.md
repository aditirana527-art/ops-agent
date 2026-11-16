Title: Supplier Management Policy
Effective: 2025-01-15
Status: active

## Source of truth for supplier terms
The contracted supplier price list (supplier_price_list) is the source of truth for contract unit cost, lead time and minimum order quantity. The supplier master record and product unit_cost in the database may lag behind and must not override the price list. A price list line is valid only until its valid_until date. Expired lines are unreliable: they may be used as an estimate only, and the item must be flagged for review before ordering.

## Supplier status
Only suppliers whose status is active may receive new purchase orders. Suspended or inactive suppliers must not receive new purchase orders. Purchase orders already sent to a supplier remain valid.

## Alternative suppliers
If a product's default supplier cannot be used, an alternative is any active supplier that has a valid price-list line for the same SKU. Prefer the lowest lead time unless an alternative costs more than 10 percent above the default supplier. If no alternative exists, flag a supply risk for human review.

## Late supplier deliveries
A purchase order is overdue when today's date is later than its created_at date plus the contracted lead time from the price list. For overdue purchase orders, send the supplier an update request and inform the procurement team (internal).
