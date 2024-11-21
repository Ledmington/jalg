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

import static org.junit.jupiter.api.Assertions.*;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

public final class TestDenseMatrix {

	private static double relativeError(final double expected, final double actual) {
		return Math.abs((actual - expected) / expected);
	}

	@Test
	void cannotBuildWithZeroRows() {
		assertThrows(IllegalArgumentException.class, () -> new DenseMatrix(new double[0][10]));
	}

	@Test
	void cannotBuildWithZeroColumns() {
		assertThrows(IllegalArgumentException.class, () -> new DenseMatrix(new double[10][0]));
	}

	private static Stream<Arguments> invalidIndices() {
		return Stream.of(
				Arguments.of(-1, 0),
				Arguments.of(2, 0),
				Arguments.of(0, -1),
				Arguments.of(0, 2),
				Arguments.of(-1, -1),
				Arguments.of(-1, 2),
				Arguments.of(2, -1),
				Arguments.of(2, 2));
	}

	@ParameterizedTest
	@MethodSource("invalidIndices")
	void invalidAccess(final int row, final int column) {
		final Matrix m = new DenseMatrix(new double[2][2]);
		assertThrows(IllegalArgumentException.class, () -> m.get(row, column));
	}

	@Test
	void initialization() {
		final RandomGenerator rng = RandomGeneratorFactory.getDefault().create(System.nanoTime());
		final int rows = 5;
		final int columns = 5;
		final double[][] v = new double[rows][columns];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				v[i][j] = rng.nextDouble(-10.0, 10.0);
			}
		}
		final Matrix m = new DenseMatrix(v);
		assertEquals(rows, m.getNumRows());
		assertEquals(columns, m.getNumColumns());
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				assertEquals(v[i][j], m.get(i, j));
			}
		}
	}

	@ParameterizedTest
	@ValueSource(ints = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	void testTranspose(final int size) {
		final Matrix m = DenseMatrix.random(size, size, -10.0, 10.0);
		final Matrix other = m.getTranspose().getTranspose();
		assertEquals(m, other, () -> {
			final StringBuilder sb = new StringBuilder();
			for (int i = 0; i < m.getNumRows(); i++) {
				for (int j = 0; j < m.getNumColumns(); j++) {
					if (Math.abs(m.get(i, j) - other.get(i, j)) > 0.0) {
						sb.append(String.format(
								"Elements at (%d, %d) were %+.6f and %+.6f, respectively.\n",
								i, j, m.get(i, j), other.get(i, j)));
					}
				}
			}
			return sb.toString();
		});
	}

	@ParameterizedTest
	@ValueSource(ints = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	void identity(final int size) {
		final Matrix m = DenseMatrix.identity(size);
		assertEquals(size, m.getNumRows());
		assertEquals(size, m.getNumColumns());
		for (int i = 0; i < m.getNumRows(); i++) {
			for (int j = 0; j < m.getNumColumns(); j++) {
				if (i == j) {
					assertEquals(1.0, m.get(i, j));
				} else {
					assertEquals(0.0, m.get(i, j));
				}
			}
		}
		assertTrue(m.isSymmetric());
		assertTrue(m.isUpperTriangular());
		assertTrue(m.isLowerTriangular());
		assertTrue(m.isTriangular());
		assertTrue(m.isDiagonal());
	}

	@ParameterizedTest
	@ValueSource(ints = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	void multiplication(final int size) {
		final Matrix m = DenseMatrix.random(size, size, -10.0, 10.0);
		final Matrix i = DenseMatrix.identity(size);
		assertEquals(m, m.multiply(i));
		assertEquals(m, i.multiply(m));
	}

	@ParameterizedTest
	@ValueSource(ints = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	void determinant(final int size) {
		final Matrix m = DenseMatrix.upperTriangular(size, -10.0, 10.0);
		final double det =
				IntStream.range(0, size).mapToDouble(i -> m.get(i, i)).reduce(1.0, (a, b) -> a * b);
		assertTrue(relativeError(det, m.getDeterminant()) < 1e-12);
	}

	@ParameterizedTest
	@ValueSource(ints = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	void inversion(final int size) {
		final Matrix m = DenseMatrix.random(size, size, -10.0, 10.0);
		final Matrix inv = m.getInverse();
		assertEquals(m.getNumRows(), inv.getNumRows());
		assertEquals(m.getNumColumns(), inv.getNumColumns());
		final Matrix i = DenseMatrix.identity(size);
		assertTrue(i.equals(m.multiply(inv), 1e-11));
		assertTrue(i.equals(inv.multiply(m), 1e-11));
	}

	@ParameterizedTest
	@ValueSource(ints = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	void gaussJordan(final int size) {
		final Matrix m = DenseMatrix.random(size, size, -10.0, 10.0);
		final Matrix d = m.gaussJordan();
		assertTrue(d.isUpperTriangular());
		assertTrue(d.isTriangular());
		assertTrue(relativeError(m.getDeterminant(), d.getDeterminant()) < 1e-12);
		assertTrue(relativeError(m.getDeterminant(), m.getEigenvalues().stream().reduce(1.0, (a, b) -> a * b)) < 1e-12);
	}

	@ParameterizedTest
	@ValueSource(ints = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	void conditionOfIdentity(final int size) {
		assertEquals(1.0, DenseMatrix.identity(size).conditionNumber());
	}

	@ParameterizedTest
	@ValueSource(ints = {2, 3, 4, 5, 6, 7, 8, 9, 10})
	void conditionNumber(final int size) {
		final Matrix m = DenseMatrix.random(size, size, -10.0, 10.0);
		final double k = m.conditionNumber();
		assertTrue(k >= 1.0, () -> String.format("Expected condition number of %s to be >= 1 but was %+.6e.", m, k));
	}
}
