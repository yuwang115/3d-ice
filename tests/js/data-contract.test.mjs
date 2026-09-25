/**
 * The binary + metadata contract between the Python preparation scripts and the browser.
 *
 * Every data package is a `.bin` payload paired with a `.meta.json` that says where each
 * field lives, how it is encoded, and what the Python side measured before quantizing it.
 * The unit tests pin the decoder; the package tests then decode every committed package
 * with the exact code the browser runs and hold it to what Python recorded.
 */

import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  decodeFieldToFloat32,
  decodeQuantizedInt16,
  DTYPE_BYTE_SIZES,
  getFieldDefinition,
  INT16_FILL_VALUE,
  parseField,
  resolveInt16Quantization,
} from "../../static/tools/js/data-contract.js";

const here = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(here, "..", "..", "static", "tools", "data");
const INT16_MAX_CODE = 32767;

function bufferWith(byteLength, write) {
  const buffer = new ArrayBuffer(byteLength);
  write(new DataView(buffer));
  return buffer;
}

function metaFor(fields, quantization) {
  return quantization ? { fields, quantization } : { fields };
}

// ---------------------------------------------------------------- parseField

test("parseField reads every supported dtype little-endian", () => {
  const buffer = bufferWith(13, (view) => {
    view.setInt16(0, -1234, true);
    view.setUint16(2, 54321, true);
    view.setInt32(4, -123456789, true);
    view.setFloat32(8, 3.5, true);
    view.setUint8(12, 7);
  });
  const meta = metaFor([
    { name: "a", dtype: "int16", byte_offset: 0, byte_length: 2 },
    { name: "b", dtype: "uint16", byte_offset: 2, byte_length: 2 },
    { name: "c", dtype: "int32", byte_offset: 4, byte_length: 4 },
    { name: "d", dtype: "float32", byte_offset: 8, byte_length: 4 },
    { name: "e", dtype: "uint8", byte_offset: 12, byte_length: 1 },
  ]);

  assert.deepEqual([...parseField(meta, buffer, "a")], [-1234]);
  assert.deepEqual([...parseField(meta, buffer, "b")], [54321]);
  assert.deepEqual([...parseField(meta, buffer, "c")], [-123456789]);
  assert.deepEqual([...parseField(meta, buffer, "d")], [3.5]);
  assert.deepEqual([...parseField(meta, buffer, "e")], [7]);
});

test("parseField decodes multi-byte fields that start at unaligned offsets", () => {
  // A leading uint8 mask pushes every later field onto an odd byte offset, which is how
  // the RISE packages are laid out. TypedArray constructors throw on such offsets.
  const buffer = bufferWith(17, (view) => {
    view.setUint8(0, 1);
    view.setInt16(1, -2, true);
    view.setInt16(3, 300, true);
    view.setUint16(5, 65535, true);
    view.setInt32(7, 2147483647, true);
    view.setFloat32(11, -0.25, true);
    view.setUint16(15, 9, true);
  });
  const meta = metaFor([
    { name: "mask", dtype: "uint8", byte_offset: 0, byte_length: 1 },
    { name: "i16", dtype: "int16", byte_offset: 1, byte_length: 4 },
    { name: "u16", dtype: "uint16", byte_offset: 5, byte_length: 2 },
    { name: "i32", dtype: "int32", byte_offset: 7, byte_length: 4 },
    { name: "f32", dtype: "float32", byte_offset: 11, byte_length: 4 },
    { name: "tail", dtype: "uint16", byte_offset: 15, byte_length: 2 },
  ]);

  assert.deepEqual([...parseField(meta, buffer, "i16")], [-2, 300]);
  assert.deepEqual([...parseField(meta, buffer, "u16")], [65535]);
  assert.deepEqual([...parseField(meta, buffer, "i32")], [2147483647]);
  assert.deepEqual([...parseField(meta, buffer, "f32")], [-0.25]);
  assert.deepEqual([...parseField(meta, buffer, "tail")], [9]);
});

test("parseField returns views for aligned fields and copies for unaligned ones", () => {
  const buffer = bufferWith(8, (view) => {
    view.setInt16(0, 5, true);
    view.setInt16(3, 6, true);
  });
  const meta = metaFor([
    { name: "aligned", dtype: "int16", byte_offset: 0, byte_length: 2 },
    { name: "unaligned", dtype: "int16", byte_offset: 3, byte_length: 2 },
  ]);

  assert.equal(parseField(meta, buffer, "aligned").buffer, buffer);
  assert.notEqual(parseField(meta, buffer, "unaligned").buffer, buffer);
});

test("parseField rejects missing fields and unsupported dtypes", () => {
  const meta = metaFor([{ name: "wide", dtype: "float64", byte_offset: 0, byte_length: 8 }]);
  const buffer = new ArrayBuffer(8);

  assert.throws(() => getFieldDefinition(meta, "absent"), /Missing field: absent/);
  assert.throws(() => parseField(meta, buffer, "absent"), /Missing field: absent/);
  assert.throws(() => parseField(meta, buffer, "wide"), /Unsupported dtype for wide: float64/);
});

// ---------------------------------------------------------------- quantization

test("decodeQuantizedInt16 maps the fill code to NaN and applies scale and offset", () => {
  const decoded = decodeQuantizedInt16(Int16Array.of(INT16_FILL_VALUE, 0, 10, -10), {
    fillValue: INT16_FILL_VALUE,
    scale: 0.5,
    offset: 100,
  });

  assert.ok(decoded instanceof Float32Array);
  assert.ok(Number.isNaN(decoded[0]));
  assert.deepEqual([...decoded.slice(1)], [100, 105, 95]);
});

test("resolveInt16Quantization prefers the field, then the package, then the defaults", () => {
  const packageLevel = { scale: 2, offset: 3, int16_fill_value: -1 };
  const perField = { name: "x", dtype: "int16", scale: 0.001, offset: 0, fill_value: -32768 };
  const bare = { name: "x", dtype: "int16" };

  assert.deepEqual(resolveInt16Quantization(metaFor([perField], packageLevel), perField), {
    scale: 0.001,
    offset: 0,
    fillValue: -32768,
  });
  assert.deepEqual(resolveInt16Quantization(metaFor([bare], packageLevel), bare), {
    scale: 2,
    offset: 3,
    fillValue: -1,
  });
  assert.deepEqual(resolveInt16Quantization(metaFor([bare]), bare), {
    scale: 1,
    offset: 0,
    fillValue: INT16_FILL_VALUE,
  });
});

test("decodeFieldToFloat32 passes float32 through, decodes int16, and refuses other dtypes", () => {
  const buffer = bufferWith(9, (view) => {
    view.setFloat32(0, 1.5, true);
    view.setInt16(4, 250, true);
    view.setInt16(6, INT16_FILL_VALUE, true);
    view.setUint8(8, 3);
  });
  const meta = metaFor(
    [
      { name: "f", dtype: "float32", byte_offset: 0, byte_length: 4 },
      { name: "q", dtype: "int16", byte_offset: 4, byte_length: 4 },
      { name: "flag", dtype: "uint8", byte_offset: 8, byte_length: 1 },
    ],
    { scale: 0.1, offset: -5, int16_fill_value: INT16_FILL_VALUE },
  );

  assert.deepEqual([...decodeFieldToFloat32(meta, buffer, "f")], [1.5]);
  const q = decodeFieldToFloat32(meta, buffer, "q");
  assert.ok(Math.abs(q[0] - 20) < 1e-5);
  assert.ok(Number.isNaN(q[1]));
  assert.throws(() => decodeFieldToFloat32(meta, buffer, "flag"), /not a decodable float field/);
});

// ---------------------------------------------------------------- committed packages

const PACKAGE_NAMES = readdirSync(DATA_DIR)
  .filter((name) => name.endsWith(".meta.json"))
  .map((name) => name.slice(0, -".meta.json".length))
  .sort();

function loadPackage(name) {
  const meta = JSON.parse(readFileSync(resolve(DATA_DIR, `${name}.meta.json`), "utf8"));
  const bytes = readFileSync(resolve(DATA_DIR, `${name}.bin`));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return { meta, buffer };
}

/** Entries with a dtype are stored arrays; the rest are summaries of derived quantities. */
function storedFields(meta) {
  return meta.fields.filter((field) => typeof field.dtype === "string");
}

function recordedStats(field) {
  const key = Object.keys(field).find((name) => name.startsWith("stats"));
  const stats = key ? field[key] : null;
  return stats && Number.isFinite(stats.min) && Number.isFinite(stats.max) ? stats : null;
}

function finiteSummary(values) {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let count = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
    count += 1;
  }
  return { min, max, mean: sum / count, count };
}

test("every committed payload has its metadata, and every metadata file its payload", () => {
  const payloads = readdirSync(DATA_DIR)
    .filter((name) => name.endsWith(".bin"))
    .map((name) => name.slice(0, -".bin".length))
    .sort();

  assert.ok(PACKAGE_NAMES.length >= 30, `expected the full data set, found ${PACKAGE_NAMES.length}`);
  assert.deepEqual(payloads, PACKAGE_NAMES);
});

test("stored fields tile each payload exactly, in declared order", async (t) => {
  for (const name of PACKAGE_NAMES) {
    await t.test(name, () => {
      const { meta, buffer } = loadPackage(name);
      let cursor = 0;
      for (const field of storedFields(meta)) {
        const size = DTYPE_BYTE_SIZES[field.dtype];
        assert.ok(size, `${field.name}: unknown dtype ${field.dtype}`);
        assert.equal(field.byte_offset, cursor, `${field.name} should start where the previous field ends`);
        assert.equal(field.byte_length % size, 0, `${field.name}: byte_length is not a whole number of values`);
        assert.equal(parseField(meta, buffer, field.name).length, field.byte_length / size);
        cursor += field.byte_length;
      }
      assert.equal(cursor, buffer.byteLength, "the payload has bytes no field accounts for");
    });
  }
});

test("gridded int16 fields and masks cover the declared grid", async (t) => {
  for (const name of PACKAGE_NAMES) {
    const { meta } = loadPackage(name);
    if (!Number.isInteger(meta.grid?.nx) || !Number.isInteger(meta.grid?.ny)) continue;
    await t.test(name, () => {
      const cells = meta.grid.nx * meta.grid.ny;
      for (const field of storedFields(meta)) {
        if (field.dtype !== "int16" && field.name !== "mask") continue;
        assert.equal(field.byte_length / DTYPE_BYTE_SIZES[field.dtype], cells, `${field.name} vs grid`);
      }
    });
  }
});

test("decoded int16 fields reproduce the statistics Python recorded before quantizing", async (t) => {
  for (const name of PACKAGE_NAMES) {
    const { meta, buffer } = loadPackage(name);
    const quantized = storedFields(meta).filter((field) => field.dtype === "int16" && recordedStats(field));
    if (quantized.length === 0) continue;

    await t.test(name, () => {
      for (const field of quantized) {
        const stats = recordedStats(field);
        const { scale, offset } = resolveInt16Quantization(meta, field);
        // Rounding moves any value, and so the mean, by at most half a step. The recorded
        // range must also fit the int16 codes, or the payload silently clipped it.
        const halfStep = Math.abs(scale) / 2;
        const tolerance = (reference) => halfStep + 1e-5 * Math.max(1, Math.abs(reference));
        assert.ok(stats.max <= INT16_MAX_CODE * scale + offset, `${field.name}: recorded max exceeds the int16 range`);
        assert.ok(stats.min >= -INT16_MAX_CODE * scale + offset, `${field.name}: recorded min exceeds the int16 range`);

        const decoded = finiteSummary(decodeFieldToFloat32(meta, buffer, field.name));
        assert.ok(decoded.count > 0, `${field.name} decoded to no valid cells`);
        for (const key of ["min", "max", "mean"]) {
          const delta = Math.abs(decoded[key] - stats[key]);
          assert.ok(
            delta <= tolerance(stats[key]),
            `${field.name}.${key}: decoded ${decoded[key]} vs recorded ${stats[key]} (scale ${scale})`,
          );
        }
      }
    });
  }
});

test("hydrology packages keep their legacy quantization keys in step with the field's", () => {
  // The geometry worker still reads the package-level keys; the generic decoder reads the
  // field. The two copies must never drift apart.
  const hydrology = PACKAGE_NAMES.filter((name) => name.includes("subglacial_hydrology"));
  assert.ok(hydrology.length >= 4, `expected four hydrology packages, found ${hydrology.length}`);
  for (const name of hydrology) {
    const meta = JSON.parse(readFileSync(resolve(DATA_DIR, `${name}.meta.json`), "utf8"));
    const resolved = resolveInt16Quantization(meta, getFieldDefinition(meta, "effective_pressure"));
    assert.equal(resolved.scale, Number(meta.quantization.effective_pressure_scale_pa_per_int16), name);
    assert.equal(resolved.offset, Number(meta.quantization.effective_pressure_offset_pa ?? 0), name);
    assert.equal(resolved.fillValue, Number(meta.quantization.int16_fill_value), name);
  }
});
