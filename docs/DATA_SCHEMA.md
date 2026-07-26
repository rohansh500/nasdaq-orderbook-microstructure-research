# LOBSTER data schema

## Message file

The file is headerless and contains six columns:

1. `time_seconds`
2. `event_type`
3. `order_id`
4. `size`
5. `price`
6. `direction`

Event types:

- `1`: new limit-order submission
- `2`: partial cancellation
- `3`: full deletion
- `4`: visible execution
- `5`: hidden execution
- `6`: cross trade
- `7`: trading-halt indicator

Direction:

- `1`: buy limit order
- `-1`: sell limit order

An execution against a resting sell order represents buyer-initiated trading, so trade-pressure sign is the negative of the resting-order direction.

Prices are stored as dollar prices multiplied by 10,000.

## Order-book file

For each occupied depth level:

```text
ask_price_1, ask_size_1, bid_price_1, bid_size_1,
ask_price_2, ask_size_2, bid_price_2, bid_size_2, ...
```

The `k`-th message causes the transition from order-book row `k-1` to row `k`. Hidden executions and certain administrative events duplicate the prior displayed-book state.

## Dummy levels

When fewer occupied levels exist than requested, dummy prices and zero sizes fill the output. The loader replaces dummy levels with missing values.

## Public-mirror caveat

The raw CSV files have no header. Generic CSV dataset viewers may incorrectly treat the first row as column names. This project always loads them with an explicit schema.
