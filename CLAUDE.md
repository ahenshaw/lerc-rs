# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo build        # Build
cargo test         # Run all tests
cargo test <name>  # Run a single test
cargo check        # Type-check without producing a binary
cargo clippy       # Lint
cargo fmt          # Format
```

## Project

`lerc` is a decode-only Rust implementation of the [LERC](https://github.com/Esri/lerc) (Limited Error Raster Compression) library. No external dependencies. The public API is two functions in `src/lib.rs`:

- `lerc::decode(blob: &[u8]) -> Result<DecodedData, LercError>` — full decode
- `lerc::get_lerc_info(blob: &[u8]) -> Result<LercInfo, LercError>` — metadata only, no decode

Lerc1 (CntZImage format) is not supported.

## Architecture

All decode logic is in `src/lerc2.rs`, which calls into the other modules:

| Module | Purpose |
|--------|---------|
| `src/lerc2.rs` | Top-level decode: header parsing, checksum, mask, tile/Huffman dispatch, multi-band loop |
| `src/huffman.rs` | `HuffmanDecoder` — builds a 12-bit LUT + optional binary tree from a ReadCodeTable blob; `decode_one` decodes one symbol |
| `src/bitstuffer.rs` | `decode()` — BitStuffer2 (v3+ LSB-first and pre-v3 MSB-first bit packing) |
| `src/lossless_float.rs` | `decode_lossless_f32/f64` — DeltaDeltaHuffman lossless float decoder (v6): byte-plane decompress → inverse predictor → undo float bit transform |
| `src/bitmask.rs` | `BitMask` — compact bit array matching the C++ layout (bit k at byte k>>3, bit 7-(k&7)) |
| `src/rle.rs` | `rle_decompress` — signed i16 LE run-length decode for the validity mask |
| `src/types.rs` | Public types: `DataType`, `LercInfo`, `LercData`, `DecodedData` |
| `src/error.rs` | `LercError` enum |

### Lerc2 binary format (per single-band blob)

1. 6-byte magic `"Lerc2 "` + 4-byte version (i32) + 4-byte checksum (u32, v3+)
2. Header integers: nRows, nCols, [nDepth (v4+)], nValidPixel, microBlockSize, blobSize, dt, [nBlobsMore (v6+)], [4 flag bytes (v6+)]
3. Header doubles: maxZError, zMin, zMax, [noDataVal, noDataValOrig (v6+)]
4. 4-byte mask byte count + RLE-compressed BitMask bytes
5. [v4+] Per-depth min/max ranges (nDepth typed T values for mins, then nDepth for maxs) — present for all nDepth ≥ 1
6. 1-byte flag: 0 = tiled/Huffman encode, non-zero = raw one-sweep
7. [if tiled/Huffman, TryHuffmanInt/Flt] 1-byte imageEncodeMode (0=Tiling, 1=DeltaHuffman, 2=Huffman, 3=DeltaDeltaHuffman)
8. Pixel data (tiles, Huffman stream, or raw sweep)

### Key decode decisions

**Tile comprFlag byte**: bits[1:0] = mode (0=raw, 1=bit-stuffed, 2=const-zero, 3=const-value); bit 2 = bDiffEnc (v5+, diff from prev depth); bits[7:6] = type-reduction code for `get_data_type_used`.

**get_data_type_used(dt, tc)**: maps (DT, tc∈0..3) → smaller DT for the tile offset field. Mirrors `Lerc2::GetDataTypeUsed` exactly (signed ints subtract tc, unsigned subtract 2*tc, float/double hardcoded).

**Huffman leaves**: encoded as `-(k+2)` in child pointers so symbol k=0 doesn't collide with the -1 "no child" sentinel. Detection: `child < -1`; extraction: `-(child+2)`.

**Multi-band**: blobs are concatenated. Navigation uses `blob_size` from each header. Loop terminates when `version <= 5 || nBlobsMore > 0` is false for the last-scanned header.

**LercScalar trait**: generic pixel type abstraction used by all inner decode functions. Implemented for all 8 primitive types (i8, u8, i16, u16, i32, u32, f32, f64).
