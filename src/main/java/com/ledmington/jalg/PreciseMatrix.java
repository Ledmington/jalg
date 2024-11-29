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

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public final class PreciseMatrix implements Matrix<BigDecimal> {

	private static final MathContext ctx = new MathContext(100);

	public static PreciseMatrix random(final int rows, final int columns, final double low, final double high) {
		final RandomGenerator rng = RandomGeneratorFactory.getDefault().create(System.nanoTime());
		final BigDecimal[][] v = new BigDecimal[rows][columns];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				v[i][j] = BigDecimal.valueOf(rng.nextDouble(low, high));
			}
		}
		return new PreciseMatrix(v);
	}

	public static PreciseMatrix randomSymmetric(final int size, final double low, final double high) {
		final RandomGenerator rng = RandomGeneratorFactory.getDefault().create(System.nanoTime());
		final BigDecimal[][] v = new BigDecimal[size][size];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < i; j++) {
				final BigDecimal x = BigDecimal.valueOf(rng.nextDouble(low, high));
				v[i][j] = x;
				v[j][i] = x;
			}
			v[i][i] = BigDecimal.valueOf(rng.nextDouble());
		}
		return new PreciseMatrix(v);
	}

	public static PreciseMatrix upperTriangular(final int size, final double low, final double high) {
		final RandomGenerator rng = RandomGeneratorFactory.getDefault().create(System.nanoTime());
		final BigDecimal[][] v = new BigDecimal[size][size];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (j < i) {
					v[i][j] = BigDecimal.ZERO;
				} else {
					v[i][j] = BigDecimal.valueOf(rng.nextDouble(low, high));
				}
			}
		}
		return new PreciseMatrix(v);
	}

	public static PreciseMatrix identity(final int size) {
		if (size < 1) {
			throw new IllegalArgumentException("Invalid size.");
		}
		final BigDecimal[][] v = new BigDecimal[size][size];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (i == j) {
					v[i][j] = BigDecimal.ONE;
				} else {
					v[i][j] = BigDecimal.ZERO;
				}
			}
		}
		return new PreciseMatrix(v);
	}

	private final int rows;
	private final int columns;
	private final BigDecimal[] m;

	public PreciseMatrix(final BigDecimal[][] v) {
		Objects.requireNonNull(v);
		if (v.length < 1) {
			throw new IllegalArgumentException("Invalid number of rows.");
		}
		this.rows = v.length;

		if (v[0].length < 1) {
			throw new IllegalArgumentException("Invalid number of columns.");
		}
		this.columns = v[0].length;

		this.m = new BigDecimal[rows * columns];

		for (int i = 0; i < rows; i++) {
			if (v[i].length != columns) {
				throw new IllegalArgumentException("Invalid number of columns.");
			}
			System.arraycopy(v[i], 0, this.m, i * columns, columns);
		}
	}

	@Override
	public int getNumRows() {
		return rows;
	}

	@Override
	public int getNumColumns() {
		return columns;
	}

	@Override
	public BigDecimal conditionNumber() {
		return this.norm().multiply(this.getInverse().norm(), ctx);
	}

	@Override
	public boolean isUpperTriangular() {
		if (!isSquare()) {
			throw new IllegalArgumentException("A rectangular matrix cannot be triangular.");
		}
		for (int i = 0; i < getNumRows(); i++) {
			for (int j = 0; j < i; j++) {
				if (get(i, j).compareTo(BigDecimal.ZERO) != 0) {
					return false;
				}
			}
		}
		return true;
	}

	@Override
	public boolean isLowerTriangular() {
		if (!isSquare()) {
			throw new IllegalArgumentException("A rectangular matrix cannot be triangular.");
		}
		for (int i = 0; i < getNumRows(); i++) {
			for (int j = i + 1; j < getNumColumns(); j++) {
				if (get(i, j).compareTo(BigDecimal.ZERO) != 0) {
					return false;
				}
			}
		}
		return true;
	}

	private void assertCorrectIndex(final int row, final int column) {
		if (row < 0 || row >= rows || column < 0 || column >= columns) {
			throw new IllegalArgumentException(
					String.format("A %,d x %,d matrix has no element in (%,d; %,d).", rows, column, row, column));
		}
	}

	@Override
	public BigDecimal get(final int row, final int column) {
		assertCorrectIndex(row, column);
		return this.m[row * columns + column];
	}

	@Override
	public Matrix<BigDecimal> multiply(final Matrix<BigDecimal> other) {
		if (this.columns != other.getNumRows()) {
			throw new IllegalArgumentException("Invalid rows and columns.");
		}
		final BigDecimal[][] v = new BigDecimal[this.rows][other.getNumColumns()];
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < other.getNumColumns(); j++) {
				v[i][j] = BigDecimal.ZERO;
				for (int k = 0; k < this.columns; k++) {
					v[i][j] = v[i][j].add(this.get(i, k).multiply(other.get(k, j), ctx), ctx);
				}
			}
		}

		return new PreciseMatrix(v);
	}

	@Override
	public BigDecimal getDeterminant() {
		if (!isSquare()) {
			throw new IllegalArgumentException("Matrix must be square.");
		}

		final Matrix<BigDecimal> triangular = this.gaussJordan();
		BigDecimal det = BigDecimal.ONE;
		for (int i = 0; i < getNumRows(); i++) {
			det = det.multiply(triangular.get(i, i), ctx);
		}

		return det;
	}

	@Override
	public List<BigDecimal> getEigenvalues() {
		final Matrix<BigDecimal> triangular = this.gaussJordan();
		final List<BigDecimal> ev = new ArrayList<>();
		for (int i = 0; i < getNumRows(); i++) {
			ev.add(triangular.get(i, i));
		}
		return ev;
	}

	@Override
	public Matrix<BigDecimal> subtract(final Matrix<BigDecimal> other) {
		if (this.getNumRows() != other.getNumRows() || this.getNumColumns() != other.getNumColumns()) {
			throw new IllegalArgumentException("Different shapes.");
		}

		final BigDecimal[][] v = new BigDecimal[getNumRows()][getNumColumns()];
		for (int i = 0; i < getNumRows(); i++) {
			for (int j = 0; j < getNumColumns(); j++) {
				v[i][j] = this.get(i, j).subtract(other.get(i, j), ctx);
			}
		}

		return new PreciseMatrix(v);
	}

	@Override
	public BigDecimal norm() {
		return this.getTranspose().multiply(this).get(0, 0);
	}

	@Override
	public boolean isInvertible() {
		return getDeterminant().compareTo(BigDecimal.ZERO) != 0;
	}

	@Override
	public boolean isPositiveDefinite() {
		return getEigenvalues().stream().allMatch(x -> x.compareTo(BigDecimal.ZERO) >= 0);
	}

	@Override
	public Matrix<BigDecimal> getInverse() {
		final BigDecimal det = this.getDeterminant();
		if (det.compareTo(BigDecimal.ZERO) == 0) {
			throw new IllegalArgumentException("Matrix is singular, not-invertible.");
		}

		if (rows == 1 && columns == 1) {
			return new PreciseMatrix(new BigDecimal[][] {{BigDecimal.ONE.divide(m[0], ctx)}});
		}

		final int n = getNumRows();

		// prepare copy of current matrix
		final BigDecimal[][] v = new BigDecimal[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				v[i][j] = get(i, j);
			}
		}

		// prepare identity matrix
		final BigDecimal[][] id = new BigDecimal[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i == j) {
					id[i][j] = BigDecimal.ONE;
				} else {
					id[i][j] = BigDecimal.ZERO;
				}
			}
		}

		for (int i = 0; i < n; i++) {
			final BigDecimal elem = v[i][i];
			for (int j = 0; j < n; j++) {
				v[i][j] = v[i][j].divide(elem, ctx);
				id[i][j] = id[i][j].divide(elem, ctx);
			}

			for (int j = 0; j < n; j++) {
				if (j == i) {
					continue;
				}

				final BigDecimal factor = v[j][i];
				for (int k = 0; k < n; k++) {
					v[j][k] = v[j][k].subtract(factor.multiply(v[i][k], ctx), ctx);
					id[j][k] = id[j][k].subtract(factor.multiply(id[i][k], ctx), ctx);
				}
			}
		}

		return new PreciseMatrix(id);
	}

	@Override
	public Matrix<BigDecimal> getTranspose() {
		final BigDecimal[][] v = new BigDecimal[columns][rows];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				v[j][i] = this.get(i, j);
			}
		}
		return new PreciseMatrix(v);
	}

	@Override
	public Matrix<BigDecimal> gaussJordan() {
		if (!isSquare()) {
			throw new IllegalArgumentException("Matrix must be square.");
		}

		final int n = getNumRows();
		final BigDecimal[][] v = new BigDecimal[n][n];

		// Copy the matrix
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				v[i][j] = get(i, j);
			}
		}

		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				final BigDecimal factor = v[j][i].divide(v[i][i], ctx);
				for (int k = 0; k < n; k++) {
					if (k <= i) {
						v[j][k] = BigDecimal.ZERO;
					} else {
						v[j][k] = v[j][k].subtract(factor.multiply(v[i][k], ctx), ctx);
					}
				}
			}
		}

		return new PreciseMatrix(v);
	}

	@Override
	public boolean equals(final Matrix<BigDecimal> other, final double eps) {
		if (eps < 0.0) {
			throw new IllegalArgumentException("Negative epsilon.");
		}
		if (other == null) {
			return false;
		}
		if (this.getNumRows() != other.getNumRows() || this.getNumColumns() != other.getNumColumns()) {
			return false;
		}
		for (int i = 0; i < this.getNumRows(); i++) {
			for (int j = 0; j < this.getNumColumns(); j++) {
				if (this.get(i, j).subtract(other.get(i, j), ctx).abs().compareTo(BigDecimal.valueOf(eps)) > 0) {
					return false;
				}
			}
		}
		return true;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder();
		sb.append(getNumRows()).append('x').append(getNumColumns()).append('\n');
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				sb.append(String.format("%+.6e", m[i * columns + j]));
				if (j < columns - 1) {
					sb.append(", ");
				}
			}
			if (i < rows - 1) {
				sb.append('\n');
			}
		}
		return sb.toString();
	}

	@Override
	public int hashCode() {
		int h = 17;
		for (final BigDecimal x : m) {
			h = 31 * h + x.hashCode();
		}
		return h;
	}

	@Override
	@SuppressWarnings("unchecked")
	public boolean equals(final Object other) {
		if (other == null) {
			return false;
		}
		if (this == other) {
			return true;
		}
		if (!this.getClass().equals(other.getClass())) {
			return false;
		}
		return this.equals((Matrix<BigDecimal>) other, 0.0);
	}
}
