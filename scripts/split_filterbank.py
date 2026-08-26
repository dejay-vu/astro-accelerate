#!/usr/bin/env python3
"""Split a SIGPROC .fil filterbank into smaller chunks."""
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np

STRING_KEYS = {
    "source_name",
    "rawdatafile",
    "project_id",
    "observer",
    "telescope",
    "backend",
}

FLOAT_KEYS = {
    "az_start",
    "za_start",
    "src_raj",
    "src_dej",
    "raj",
    "dej",
    "tstart",
    "tsamp",
    "fch1",
    "foff",
    "refdm",
    "period",
}


def read_string(fp) -> str:
    (length,) = struct.unpack('<i', fp.read(4))
    if length <= 0:
        return ''
    return fp.read(length).decode('ascii')


def read_header(fp) -> Tuple[List[Tuple[str, object, str]], int]:
    entries: List[Tuple[str, object, str]] = []
    header_size = 0
    start = read_string(fp)
    if start != 'HEADER_START':
        raise ValueError('Missing HEADER_START')
    header_size += 4 + len(start)

    while True:
        key = read_string(fp)
        header_size += 4 + len(key)
        if key == 'HEADER_END':
            break
        if key in STRING_KEYS:
            (slen,) = struct.unpack('<i', fp.read(4))
            header_size += 4
            value = fp.read(slen).decode('ascii')
            header_size += slen
            entries.append((key, value, 'str'))
        elif key in FLOAT_KEYS:
            (value,) = struct.unpack('<d', fp.read(8))
            header_size += 8
            entries.append((key, value, 'float'))
        else:
            (value,) = struct.unpack('<i', fp.read(4))
            header_size += 4
            entries.append((key, value, 'int'))
    return entries, header_size


def write_string(fp, text: str) -> None:
    data = text.encode('ascii')
    fp.write(struct.pack('<i', len(data)))
    fp.write(data)


def write_header(fp, entries: List[Tuple[str, object, str]]) -> None:
    write_string(fp, 'HEADER_START')
    for key, value, kind in entries:
        write_string(fp, key)
        if kind == 'str':
            write_string(fp, str(value))
        elif kind == 'float':
            fp.write(struct.pack('<d', float(value)))
        else:
            fp.write(struct.pack('<i', int(value)))
    write_string(fp, 'HEADER_END')


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a SIGPROC .fil file into smaller chunks")
    parser.add_argument('input', type=Path)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--chunk-samples', type=int, default=None,
                        help='Samples per output chunk (overrides --chunk-seconds)')
    parser.add_argument('--chunk-seconds', type=float, default=100.0,
                        help='Desired duration per chunk in seconds (default 100)')
    parser.add_argument('--prefix', type=str, default=None,
                        help='Output filename prefix (default: input stem)')
    args = parser.parse_args()

    input_path = args.input.resolve()
    out_dir = (args.out_dir or input_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open('rb') as fp:
        entries, header_size = read_header(fp)
        metadata = {k: v for k, v, _ in entries}
        nbits = metadata.get('nbits')
        nchans = metadata.get('nchans')
        tsamp = metadata.get('tsamp')

        if nbits not in (1, 2, 4, 8, 16, 32):
            raise ValueError(f'Unsupported nbits: {nbits}')
        if nchans is None or tsamp is None:
            raise ValueError('Missing nchans or tsamp in header')

        dtype_map = {
            8: 'u1',
            16: '<u2',
            32: '<f4',
        }
        if nbits not in dtype_map:
            raise ValueError('This splitter currently supports 8, 16, or 32-bit data')

        dtype = np.dtype(dtype_map[nbits])
        bytes_per_sample = dtype.itemsize * nchans
        fp.seek(0, 2)
        total_bytes = fp.tell() - header_size
        total_samples = total_bytes // bytes_per_sample
        fp.seek(header_size)

        chunk_samples = args.chunk_samples or int(args.chunk_seconds / tsamp)
        if chunk_samples <= 0:
            chunk_samples = total_samples
        n_chunks = math.ceil(total_samples / chunk_samples)

        base_tstart = metadata.get('tstart', 0.0)
        prefix = args.prefix or input_path.stem

        print(f'Input samples: {total_samples}, chunks: {n_chunks}, samples/chunk: {chunk_samples}')

        for chunk_idx in range(n_chunks):
            start_sample = chunk_idx * chunk_samples
            remaining = total_samples - start_sample
            current_samples = min(chunk_samples, remaining)
            if current_samples <= 0:
                break

            offset_days = (start_sample * tsamp) / 86400.0
            chunk_entries = []
            for key, value, kind in entries:
                if key == 'nsamples':
                    chunk_entries.append((key, current_samples, kind))
                elif key == 'tstart':
                    chunk_entries.append((key, base_tstart + offset_days, kind))
                else:
                    chunk_entries.append((key, value, kind))

            out_path = out_dir / f"{prefix}_part{chunk_idx:03d}.fil"
            with out_path.open('wb') as out_fp:
                write_header(out_fp, chunk_entries)
                bytes_to_copy = current_samples * bytes_per_sample
                remaining_bytes = bytes_to_copy
                buffer_size = 4 * 1024 * 1024
                while remaining_bytes > 0:
                    chunk = fp.read(min(buffer_size, remaining_bytes))
                    if not chunk:
                        raise EOFError('Unexpected end-of-file while reading data')
                    out_fp.write(chunk)
                    remaining_bytes -= len(chunk)
            print(f'Wrote {out_path} ({current_samples} samples)')

        leftover = total_samples - n_chunks * chunk_samples
        if leftover > 0:
            print(f'Remaining samples not written: {leftover}')

if __name__ == '__main__':  # pragma: no cover
    main()
