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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public final class DenseMatrix implements Matrix {

	public static DenseMatrix random(final int rows, final int columns, final double low, final double high) {
		final RandomGenerator rng = RandomGeneratorFactory.getDefault().create(System.nanoTime());
		final double[][] v = new double[rows][columns];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				v[i][j] = rng.nextDouble(low, high);
			}
		}
		return new DenseMatrix(v);
	}

	public static DenseMatrix upperTriangular(final int size, final double low, final double high) {
		final RandomGenerator rng = RandomGeneratorFactory.getDefault().create(System.nanoTime());
		final double[][] v = new double[size][size];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (j < i) {
					continue;
				}
				v[i][j] = rng.nextDouble(low, high);
			}
		}
		return new DenseMatrix(v);
	}

	public static DenseMatrix identity(final int size) {
		if (size < 1) {
			throw new IllegalArgumentException("Invalid size.");
		}
		final double[][] v = new double[size][size];
		for (int i = 0; i < size; i++) {
			v[i][i] = 1.0;
		}
		return new DenseMatrix(v);
	}

	private final int rows;
	private final int columns;
	private final double[] m;

	public DenseMatrix(final double[][] v) {
		Objects.requireNonNull(v);
		if (v.length < 1) {
			throw new IllegalArgumentException("Invalid number of rows.");
		}
		this.rows = v.length;

		if (v[0].length < 1) {
			throw new IllegalArgumentException("Invalid number of columns.");
		}
		this.columns = v[0].length;

		this.m = new double[rows * columns];

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

	private void assertCorrectIndex(final int row, final int column) {
		if (row < 0 || row >= rows || column < 0 || column >= columns) {
			throw new IllegalArgumentException(
					String.format("A %,d x %,d matrix has no element in (%,d; %,d).", rows, column, row, column));
		}
	}

	@Override
	public double get(final int row, final int column) {
		assertCorrectIndex(row, column);
		return this.m[row * columns + column];
	}

	@Override
	public Matrix multiply(final Matrix other) {
		if (this.columns != other.getNumRows()) {
			throw new IllegalArgumentException("Invalid rows and columns.");
		}
		final double[][] v = new double[this.rows][other.getNumColumns()];
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < other.getNumColumns(); j++) {
				for (int k = 0; k < this.columns; k++) {
					v[i][j] += this.get(i, k) * other.get(k, j);
				}
			}
		}

		return new DenseMatrix(v);
	}

	@Override
	public double getDeterminant() {
		if (!isSquare()) {
			throw new IllegalArgumentException("Matrix must be square.");
		}

		final Matrix triangular = this.gaussJordan();
		double det = 1.0;
		for (int i = 0; i < getNumRows(); i++) {
			det *= triangular.get(i, i);
		}

		return det;
	}

	@Override
	public List<Double> getEigenvalues() {
		final Matrix triangular = this.gaussJordan();
		final List<Double> ev = new ArrayList<>();
		for (int i = 0; i < getNumRows(); i++) {
			ev.add(triangular.get(i, i));
		}
		return ev;
	}

	@Override
	public Matrix subtract(final Matrix other) {
		if (this.getNumRows() != other.getNumRows() || this.getNumColumns() != other.getNumColumns()) {
			throw new IllegalArgumentException("Different shapes.");
		}

		final double[][] v = new double[getNumRows()][getNumColumns()];
		for (int i = 0; i < getNumRows(); i++) {
			for (int j = 0; j < getNumColumns(); j++) {
				v[i][j] = this.get(i, j) - other.get(i, j);
			}
		}

		return new DenseMatrix(v);
	}

	@Override
	public double norm() {
		return this.getTranspose().multiply(this).get(0, 0);
	}

	@Override
	public Matrix getInverse() {
		final double det = this.getDeterminant();
		if (det == 0.0) {
			throw new IllegalArgumentException("Matrix is singular, not-invertible.");
		}

		if (rows == 1 && columns == 1) {
			return new DenseMatrix(new double[][] {{1.0 / m[0]}});
		}

		final int n = getNumRows();

		// prepare copy of current matrix
		final double[][] v = new double[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				v[i][j] = get(i, j);
			}
		}

		// prepare identity matrix
		final double[][] id = new double[n][n];
		for (int i = 0; i < n; i++) {
			id[i][i] = 1.0;
		}

		for (int i = 0; i < n; i++) {
			final double elem = v[i][i];
			for (int j = 0; j < n; j++) {
				v[i][j] /= elem;
				id[i][j] /= elem;
			}

			for (int j = 0; j < n; j++) {
				if (j == i) {
					continue;
				}

				final double factor = v[j][i];
				for (int k = 0; k < n; k++) {
					v[j][k] -= factor * v[i][k];
					id[j][k] -= factor * id[i][k];
				}
			}
		}

		return new DenseMatrix(id);
	}

	@Override
	public Matrix getTranspose() {
		final double[][] v = new double[columns][rows];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				v[j][i] = this.get(i, j);
			}
		}
		return new DenseMatrix(v);
	}

	@Override
	public Matrix gaussJordan() {
		if (!isSquare()) {
			throw new IllegalArgumentException("Matrix must be square.");
		}

		final int n = getNumRows();
		final double[][] v = new double[n][n];

		// Copy the matrix
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				v[i][j] = get(i, j);
			}
		}

		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				final double factor = v[j][i] / v[i][i];
				for (int k = 0; k < n; k++) {
					if (k <= i) {
						v[j][k] = 0.0;
					} else {
						v[j][k] -= factor * v[i][k];
					}
				}
			}
		}

		return new DenseMatrix(v);
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
		for (final double x : m) {
			h = 31 * h + (int) (Double.doubleToLongBits(x) >>> 32);
			h = 31 * h + (int) (Double.doubleToLongBits(x) & 0x00000000ffffffffL);
		}
		return h;
	}

	@Override
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
		return this.equals((Matrix) other, 0.0);
	}
}
