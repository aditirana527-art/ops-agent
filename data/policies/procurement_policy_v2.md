Title: Procurement and Replenishment Policy (v2)
Effective: 2025-07-01
Status: active (replaces v1)

## Reorder trigger
Replenishment is based on AVAILABLE stock, not on-hand stock. Available stock = stock_on_hand - reserved_qty. Reorder when available stock plus the quantity on open purchase orders (status sent or confirmed, not yet received) is below the product's reorder_level. Where a warehouse cycle count is more recent than the product's last_updated date and the figures differ, use the LOWER of the system stock and the counted quantity when computing available stock, and raise a stock discrepancy review (see the Inventory Accuracy policy).

## Order quantity
Order the product's reorder_qty, unless the supplier's minimum order quantity (min_order_qty) in the contracted price list is higher, in which case order the minimum order quantity. Never order products whose status is discontinued.

## Products without a reorder level
If reorder_level is blank the product has no approved replenishment parameters. Do not raise a purchase order for it. Flag it for human review so that a reorder level can be set.

## Purchase order approval
Purchase orders with an estimated cost above 2,500 USD require procurement manager sign-off. The agent must not create these directly. Instead it flags the item for human review including the draft PO (supplier, SKU, quantity, unit cost, estimated cost) for the procurement manager. Purchase orders at or below 2,500 USD may be created by the agent, subject to operator approval in the system.

## Supplier selection
Use the product's default supplier if it is active and has a valid price-list line. Otherwise follow the Supplier Management policy for alternatives.
