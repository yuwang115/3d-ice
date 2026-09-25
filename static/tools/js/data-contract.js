/**
 * Decoder for 3D ICE data packages: a little-endian `.bin` payload plus a `.meta.json`
 * that locates and describes every field in it.
 *
 * This is the browser half of the contract the Python preparation scripts write (see
 * docs/data-contract.md). It has no DOM or Three.js dependency, so the explorer page, the
 * geometry worker and the Node tests all run exactly the same code.
 */

export const INT16_FILL_VALUE = -32768;

export const DTYPE_BYTE_SIZES = Object.freeze({
  uint8: 1,
  int16: 2,
  uint16: 2,
  int32: 4,
  float32: 4,
});

const TYPED_ARRAYS = Object.freeze({
  uint8: Uint8Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  float32: Float32Array,
});

const DATA_VIEW_READERS = Object.freeze({
  int16: "getInt16",
  uint16: "getUint16",
  int32: "getInt32",
  float32: "getFloat32",
});

export function getFieldDefinition(meta, name) {
  const field = meta.fields.find((item) => item.name === name);
  if (!field) {
    throw new Error(`Missing field: ${name}`);
  }
  return field;
}

/**
 * Typed array over one field. Aligned fields are zero-copy views into `arrayBuffer`.
 * Some packages place multi-byte fields at odd offsets, which TypedArray constructors
 * reject, so those are copied out through a DataView instead.
 */
export function parseField(meta, arrayBuffer, name) {
  const field = getFieldDefinition(meta, name);
  const TypedArray = TYPED_ARRAYS[field.dtype];
  if (!TypedArray) {
    throw new Error(`Unsupported dtype for ${name}: ${field.dtype}`);
  }

  const size = TypedArray.BYTES_PER_ELEMENT;
  const count = field.byte_length / size;
  if (field.byte_offset % size === 0) {
    return new TypedArray(arrayBuffer, field.byte_offset, count);
  }

  const view = new DataView(arrayBuffer, field.byte_offset, field.byte_length);
  const read = DATA_VIEW_READERS[field.dtype];
  const out = new TypedArray(count);
  for (let i = 0; i < count; i += 1) {
    out[i] = view[read](i * size, true);
  }
  return out;
}

/** Scale, offset and fill code of an int16 field; the field's own keys win over the package's. */
export function resolveInt16Quantization(meta, field) {
  const packageLevel = meta.quantization ?? {};
  return {
    scale: Number(field.scale ?? packageLevel.scale ?? 1),
    offset: Number(field.offset ?? packageLevel.offset ?? 0),
    fillValue: Number(field.fill_value ?? packageLevel.int16_fill_value ?? INT16_FILL_VALUE),
  };
}

export function decodeQuantizedInt16(intArray, { fillValue, scale = 1, offset = 0 }) {
  const out = new Float32Array(intArray.length);
  for (let i = 0; i < intArray.length; i += 1) {
    out[i] = intArray[i] === fillValue ? Number.NaN : intArray[i] * scale + offset;
  }
  return out;
}

export function decodeFieldToFloat32(meta, arrayBuffer, name) {
  const field = getFieldDefinition(meta, name);
  if (field.dtype === "float32") {
    return parseField(meta, arrayBuffer, name);
  }
  if (field.dtype === "int16") {
    return decodeQuantizedInt16(parseField(meta, arrayBuffer, name), resolveInt16Quantization(meta, field));
  }
  throw new Error(`Field ${name} is not a decodable float field.`);
}
