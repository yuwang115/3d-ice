/**
 * Minimal in-place iterative radix-2 complex FFT for square-power-of-two-axis 2-D grids.
 *
 * The 3D ICE isostatic-rebound layer solves the thin-plate flexure equation in the
 * spectral domain, which needs one forward and one inverse 2-D transform per Picard
 * iteration. Only real-valued input is ever transformed, so the imaginary plane is
 * zero-filled by the caller; keeping a plain complex transform costs 2x the arithmetic
 * of a real-to-complex variant but keeps this module small enough to verify by hand.
 *
 * Both axes must be powers of two. Plans cache the bit-reversal permutation and the
 * twiddle factors so repeated transforms of the same size are allocation-free.
 */

function assertPowerOfTwo(value, label) {
  if (!Number.isInteger(value) || value < 2 || (value & (value - 1)) !== 0) {
    throw new Error(`${label} must be an integer power of two >= 2 (received ${String(value)}).`);
  }
}

export function nextPowerOfTwo(value) {
  let power = 1;
  while (power < value) power <<= 1;
  return power;
}

function buildBitReversal(size) {
  const reversal = new Uint32Array(size);
  const bits = Math.round(Math.log2(size));
  for (let index = 0; index < size; index += 1) {
    let remaining = index;
    let reversed = 0;
    for (let bit = 0; bit < bits; bit += 1) {
      reversed = (reversed << 1) | (remaining & 1);
      remaining >>= 1;
    }
    reversal[index] = reversed;
  }
  return reversal;
}

function buildTwiddles(size, sign) {
  const half = size >> 1;
  const cos = new Float64Array(half);
  const sin = new Float64Array(half);
  for (let index = 0; index < half; index += 1) {
    const angle = (sign * 2 * Math.PI * index) / size;
    cos[index] = Math.cos(angle);
    sin[index] = Math.sin(angle);
  }
  return { cos, sin };
}

function transformLine(real, imaginary, offset, stride, size, reversal, twiddles) {
  for (let index = 0; index < size; index += 1) {
    const target = reversal[index];
    if (target <= index) continue;
    const a = offset + index * stride;
    const b = offset + target * stride;
    let swap = real[a];
    real[a] = real[b];
    real[b] = swap;
    swap = imaginary[a];
    imaginary[a] = imaginary[b];
    imaginary[b] = swap;
  }

  for (let span = 2; span <= size; span <<= 1) {
    const half = span >> 1;
    const twiddleStep = size / span;
    for (let base = 0; base < size; base += span) {
      for (let k = 0; k < half; k += 1) {
        const twiddleIndex = k * twiddleStep;
        const cos = twiddles.cos[twiddleIndex];
        const sin = twiddles.sin[twiddleIndex];
        const a = offset + (base + k) * stride;
        const b = offset + (base + k + half) * stride;
        const productReal = real[b] * cos - imaginary[b] * sin;
        const productImaginary = real[b] * sin + imaginary[b] * cos;
        real[b] = real[a] - productReal;
        imaginary[b] = imaginary[a] - productImaginary;
        real[a] += productReal;
        imaginary[a] += productImaginary;
      }
    }
  }
}

/**
 * Allocate a reusable 2-D transform plan. `real` and `imaginary` are the working
 * planes: callers write input into them and read the transform back out of them.
 */
export function createFft2dPlan(sizeX, sizeY) {
  assertPowerOfTwo(sizeX, "FFT x size");
  assertPowerOfTwo(sizeY, "FFT y size");
  return {
    sizeX,
    sizeY,
    real: new Float64Array(sizeX * sizeY),
    imaginary: new Float64Array(sizeX * sizeY),
    forward: {
      reversalX: buildBitReversal(sizeX),
      twiddlesX: buildTwiddles(sizeX, -1),
      reversalY: buildBitReversal(sizeY),
      twiddlesY: buildTwiddles(sizeY, -1),
    },
    inverse: {
      reversalX: buildBitReversal(sizeX),
      twiddlesX: buildTwiddles(sizeX, 1),
      reversalY: buildBitReversal(sizeY),
      twiddlesY: buildTwiddles(sizeY, 1),
    },
  };
}

/**
 * Transform `plan.real` / `plan.imaginary` in place. `inverse` selects the sign of the
 * exponent and applies the 1 / (sizeX * sizeY) normalisation, so forward followed by
 * inverse is the identity.
 */
export function transformFft2dInPlace(plan, { inverse = false } = {}) {
  const { sizeX, sizeY, real, imaginary } = plan;
  const tables = inverse ? plan.inverse : plan.forward;

  for (let row = 0; row < sizeY; row += 1) {
    transformLine(real, imaginary, row * sizeX, 1, sizeX, tables.reversalX, tables.twiddlesX);
  }
  for (let column = 0; column < sizeX; column += 1) {
    transformLine(real, imaginary, column, sizeX, sizeY, tables.reversalY, tables.twiddlesY);
  }

  if (inverse) {
    const normalisation = 1 / (sizeX * sizeY);
    for (let index = 0; index < real.length; index += 1) {
      real[index] *= normalisation;
      imaginary[index] *= normalisation;
    }
  }
  return plan;
}

/**
 * Signed angular wavenumbers for a plan axis, ordered to match the transform layout
 * (0, positive frequencies, then negative frequencies).
 */
export function angularWavenumbers(size, spacing) {
  const wavenumbers = new Float64Array(size);
  for (let index = 0; index < size; index += 1) {
    const mode = index <= size / 2 ? index : index - size;
    wavenumbers[index] = (2 * Math.PI * mode) / (size * spacing);
  }
  return wavenumbers;
}
