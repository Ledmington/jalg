/*
 * jalg - Linear Algebra with java
 * Copyright (C) 2024-2024 Filippo Barbari <filippo.barbari@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.ledmington.jalg;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public final class Jalg {

	private Jalg() {}

	public static void set(final double[] x, final int start, final int incX, final int n, final double value) {
		if (n < 0) {
			throw new IllegalArgumentException();
		}
		if (incX <= 0) {
			throw new IllegalArgumentException();
		}
		if (x == null) {
			throw new NullPointerException();
		}
		if (x.length < n) {
			throw new IllegalArgumentException();
		}
		if (n == 0) {
			return;
		}
		if (incX == 1) {
			Arrays.fill(x, start, start + n, value);
		} else {
			final int end = start + n;
			for (int i = start; i < end; i += incX) {
				x[i] = value;
			}
		}
	}

	public static void flipSign(final double[] x, final int start, final int incX, final int n) {
		if (n < 0) {
			throw new IllegalArgumentException();
		}
		if (incX <= 0) {
			throw new IllegalArgumentException();
		}
		if (x == null) {
			throw new NullPointerException();
		}
		if (x.length < n) {
			throw new IllegalArgumentException();
		}
		if (n == 0) {
			return;
		}
		if (incX == 1) {
			final int end = start + n;
			for (int i = start; i < end; i++) {
				x[i] = -x[i];
			}
		} else {
			final int end = start + n;
			for (int i = start; i < end; i += incX) {
				x[i] = -x[i];
			}
		}
	}

	public static void mul(final double[] x, final int start, final int incX, final int n, final double alpha) {
		if (n < 0) {
			throw new IllegalArgumentException();
		}
		if (incX <= 0) {
			throw new IllegalArgumentException();
		}
		if (x == null) {
			throw new NullPointerException();
		}
		if (x.length < n) {
			throw new IllegalArgumentException();
		}
		if (n == 0) {
			return;
		}
		if (alpha == 1.0) {
			return;
		}
		if (alpha == 0.0) {
			set(x, 0, n, incX, 0.0);
		} else if (alpha == -1.0) {
			flipSign(x, start, incX, n);
		} else if (incX == 1) {
			final int end = start + n;
			for (int i = start; i < end; i++) {
				x[i] *= alpha;
			}
		} else {
			final int end = start + n;
			for (int i = start; i < end; i += incX) {
				x[i] *= alpha;
			}
		}
	}

	public static void div(final double[] x, final int start, final int incX, final int n, final double alpha) {
		if (n < 0) {
			throw new IllegalArgumentException();
		}
		if (incX <= 0) {
			throw new IllegalArgumentException();
		}
		if (x == null) {
			throw new NullPointerException();
		}
		if (x.length < n) {
			throw new IllegalArgumentException();
		}
		if (n == 0) {
			return;
		}
		if (alpha == 1.0) {
			return;
		}
		if (alpha == -1.0) {
			flipSign(x, start, incX, n);
		} else if (incX == 1) {
			final int end = start + n;
			for (int i = start; i < end; i++) {
				x[i] /= alpha;
			}
		} else {
			final int end = start + n;
			for (int i = start; i < end; i += incX) {
				x[i] /= alpha;
			}
		}
	}

	public static void axpy(
			final double[] x, final int startX, final int startY, final int inc, final int n, final double alpha) {
		if (n < 0) {
			throw new IllegalArgumentException();
		}
		if (inc <= 0) {
			throw new IllegalArgumentException();
		}
		if (x == null) {
			throw new NullPointerException();
		}
		if (x.length < n) {
			throw new IllegalArgumentException();
		}
		if (startX < 0 || startX + n > x.length) {
			throw new IllegalArgumentException(String.format(
					"Cannot iterate on [%,d; %,d) with array of length %,d.", startX, startX + n, x.length));
		}
		if (startY < 0 || startY + n > x.length) {
			throw new IllegalArgumentException(String.format(
					"Cannot iterate on [%,d; %,d) with array of length %,d.", startY, startY + n, x.length));
		}
		if (n == 0) {
			return;
		}
		if (alpha == 0.0) {
			return;
		}
		if (n == 1) {
			x[startX] += alpha * x[startY];
			return;
		}

		if (inc == 1) {
			// System.out.printf("axpy(x, %,d, %,d, %,d, %,d, %.6f)\n", startX, startY, inc, n, alpha);
			int i = startX;
			int j = startY;
			for (int k = 0; k < n; k++) {
				x[i++] += alpha * x[j++];
			}
		} else {
			for (int i = 0; i < n; i += inc) {
				x[startX + i] += alpha * x[startY + i];
			}
		}
	}

	public static double[] randomMatrix(final int rows, final int columns, final double low, final double high) {
		if (rows < 0 || columns < 0) {
			throw new IllegalArgumentException();
		}
		if (low > high) {
			throw new IllegalArgumentException();
		}
		final double[] m = new double[rows * columns];
		if (low == high) {
			set(m, 0, 1, m.length, low);
		} else {
			final RandomGenerator rng = RandomGeneratorFactory.getDefault().create(System.nanoTime());
			for (int i = 0; i < m.length; i++) {
				m[i] = rng.nextDouble(low, high);
			}
		}
		return m;
	}

	public static double[] gaussJordan(final double[] m, final int rows, final int columns) {
		if (rows < 0 || columns < 0) {
			throw new IllegalArgumentException();
		}
		if (m == null) {
			throw new NullPointerException();
		}
		final int size = rows * columns;
		if (m.length < size) {
			throw new IllegalArgumentException();
		}
		if (rows != columns) {
			throw new IllegalArgumentException();
		}
		final double[] out = new double[size];

		// Copy the matrix
		System.arraycopy(m, 0, out, 0, size);

		for (int i = 0; i < rows; i++) {
			final double c = out[i * columns + i];
			for (int j = i + 1; j < rows; j++) {
				final double factor = out[j * columns + i] / c;
				set(out, j * columns, 1, i + 1, 0.0);
				axpy(out, j * columns + i, i * columns + i, 1, rows - i - 1, -factor);
			}
		}

		return out;
	}

	public static double determinant(final double[] m, final int rows, final int columns) {
		if (rows < 0 || columns < 0) {
			throw new IllegalArgumentException();
		}
		if (m == null) {
			throw new NullPointerException();
		}
		final int size = rows * columns;
		if (m.length < size) {
			throw new IllegalArgumentException();
		}
		if (rows != columns) {
			throw new IllegalArgumentException();
		}
		final double[] tmp = gaussJordan(m, rows, columns);

		// FIXME: prod(size, tmp, 0, columns+1)?
		double det = 1.0;
		for (int i = 0; i < rows; i++) {
			det *= tmp[i * columns + i];
		}
		return det;
	}

	public static double[] invert(final double[] m, final int rows, final int columns) {
		if (rows < 0 || columns < 0) {
			throw new IllegalArgumentException();
		}
		if (m == null) {
			throw new NullPointerException();
		}
		final int size = rows * columns;
		if (m.length < size) {
			throw new IllegalArgumentException();
		}
		if (rows != columns) {
			throw new IllegalArgumentException();
		}
		final double det = determinant(m, rows, columns);
		if (det == 0.0) {
			throw new IllegalArgumentException("Matrix is singular, not-invertible.");
		}
		if (size == 1) {
			return new double[] {1.0 / m[0]};
		}

		// prepare copy of current matrix
		final double[] v = new double[size];
		System.arraycopy(m, 0, v, 0, size);

		// prepare identity matrix
		final double[] id = new double[size];
		set(id, 0, columns + 1, size, 1.0);

		for (int i = 0; i < rows; i++) {
			final double elem = v[i * columns + i];

			div(v, i * columns, 1, rows, elem);
			div(id, i * columns, 1, rows, elem);

			int j;
			for (j = 0; j < i; j++) {
				final double factor = v[j * columns + i];

				axpy(v, j * columns, i * columns, 1, rows, -factor);
				axpy(id, j * columns, i * columns, 1, rows, -factor);
			}

			// Skip when i == j
			j++;

			for (; j < rows; j++) {
				final double factor = v[j * columns + i];

				axpy(v, j * columns, i * columns, 1, rows, -factor);
				axpy(id, j * columns, i * columns, 1, rows, -factor);
			}
		}

		return id;
	}
}
