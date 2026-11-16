Title: Order Fulfilment and Customer Service Policy
Effective: 2025-02-01
Status: active

## Customer tiers
Gold customers receive priority allocation when stock is short, then Silver, then Standard. Within the same tier, the earlier order_date is served first.

## Credit hold
Orders from customers whose credit_status is hold must not be fulfilled. Set the order status to on_hold and notify finance (internal). Customer messages may only say the order is on hold pending an account review. Never mention balances or credit limits to the customer.

## Late orders
An order is late when today is after its promised_date and it has not shipped. For late orders, notify the customer of the delay with a revised estimate. The revised estimate is the expected arrival of the relevant incoming stock (open purchase order created_at plus contracted lead time) plus 2 working days of handling. Silver and Gold customers should additionally be offered partial shipment of the quantities currently available.

## Backorders
If available stock cannot cover an order and no incoming stock will arrive before the promised date, set the order status to backordered and notify the customer of the expected date.

## Cancellations
A customer may cancel any order that has not shipped (status pending, confirmed, on_hold or backordered). The cancellation request must be evidenced in the system, for example by a note on the order. Cancelling releases reserved stock. Cancelling is irreversible and needs human approval. Notify the customer to confirm the cancellation.

## Unknown customers
Orders whose customer_id does not exist in the customer master must not be processed. Flag them for human review with severity medium.

## Discontinued products
Orders containing discontinued products need review. Flag them for human review; do not cancel them unilaterally.
