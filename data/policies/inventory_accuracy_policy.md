Title: Inventory Accuracy and Cycle Count Policy
Effective: 2025-03-01
Status: active

## Discrepancy handling
Warehouse cycle counts are physical counts. If counted_qty differs from the system stock_on_hand by more than 5 percent, raise a human review item. Severity is high if the gap is more than 20 percent, otherwise medium. Agents must never change stock records directly. Corrections are made by the warehouse after review.

## Promises to customers
While a discrepancy exists, promise customers only the lower of the system quantity and the counted quantity.

## Reserved stock
reserved_qty equals the total quantity on open orders (status pending, confirmed, on_hold or backordered). Cancelling an open order releases its reservation.
