# Raw Nasdaq TotalView-ITCH reconstruction protocol

## Scope

Phase E adds an independent, order-ID-level reconstruction path for Nasdaq
TotalView-ITCH 5.0 BinaryFILE sessions. It does not treat LOBSTER snapshots
as the source of truth.

The implementation:

1. streams two-byte, big-endian BinaryFILE length prefixes;
2. parses the ITCH 5.0 order-lifecycle messages required for a displayed book;
3. maintains day-unique order-reference state for one selected symbol;
4. aggregates full visible depth independently on bid and ask sides;
5. exports the top N levels after each target-symbol book-changing message;
6. checks order-to-level conservation, timestamp ordering, underflows, duplicate
   references, and locked/crossed states;
7. records throughput and bounded-memory output metrics.

## Supported book messages

- `A`: Add Order
- `F`: Add Order with MPID Attribution
- `E`: Order Executed
- `C`: Order Executed with Price
- `X`: Order Cancel
- `D`: Order Delete
- `U`: Order Replace

`P` non-cross Trade messages are counted but do not modify the displayed book.
Administrative and other trade messages are counted and otherwise ignored by
the reconstruction engine.

## Replace handling

Order Replace is treated as one atomic source message. The old order is removed
and the replacement order is inserted using the original side, symbol, and
attribution. Exported rows use research `event_type = 8` and retain old and new
order references, sizes, and prices.

The Phase E feature adapter maps both replace legs into rolling add and cancel
pressure without changing the established 35-feature research schema.

## Data handling

Official daily files can be several gigabytes while compressed. The parser reads
`.gz` files directly and writes reconstructed rows in compressed Parquet batches;
it does not require full decompression or a full-session in-memory DataFrame.

Raw ITCH files and large generated Parquet files remain local and must not be
committed or redistributed without checking the applicable data terms.

## Validation boundary

A successful reconstruction demonstrates protocol parsing, state management,
book invariants, and engineering throughput. It does not establish predictive
model generalisation across days. Cross-day and cross-symbol research validation
is a separate extension after the raw parser has been verified.

## Complete-file termination

The official `01302019.NASDAQ_ITCH50.gz` sample was consumed to clean gzip
EOF without truncated records, payload-length mismatches or decompression
errors. The file did not expose a zero-length BinaryFILE terminator, so
`end_marker_seen` is false even though processing was not stopped early.

For this sample, complete processing is established by clean stream
exhaustion, `stopped_early = false`, `stop_reason = null`, and successful
processing of the exact downloaded input file.

## Technical references

- Nasdaq TotalView-ITCH 5.0 specification:
  https://assets.ctfassets.net/mx0rke14e5yt/5Uz6MGJxbo4wRPou8KveFs/4d76437c8e57694acee9d767587a8dfa/6-11-26_TVITCH_5.0_1.pdf
- Nasdaq BinaryFILE specification:
  https://www.nasdaqtrader.com/content/technicalSupport/specifications/dataproducts/binaryfile.pdf
- Official Nasdaq ITCH sample directory:
  https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/
