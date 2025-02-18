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
import java.util.stream.IntStream;

public final class Solvers {

	private Solvers() {}

	public static Matrix<Double> jacobi(final Matrix<Double> A, final Matrix<Double> b) {
		if (!A.isSquare()) {
			throw new IllegalArgumentException("Matrix A is not a square.");
		}
		if (b.getNumColumns() != 1 || b.getNumRows() != A.getNumRows()) {
			throw new IllegalArgumentException("Vector b is not a column vector.");
		}

		final int n = A.getNumRows();
		final double[][] x = new double[n][1]; // x0 = 0
		final double[][] xNew = new double[n][1];

		final int maxit = 100;
		final double tolerance = 1e-8;
		for (int k = 0; k < maxit; k++) {
			for (int i = 0; i < n; i++) {
				double s = 0.0;
				for (int j = 0; j < n; j++) {
					if (j == i) {
						continue;
					}
					s += A.get(i, j) * x[j][0];
				}
				xNew[i][0] = (b.get(i, 0) - s) / A.get(i, i);
			}

			if (IntStream.range(0, n)
							.mapToDouble(i -> Math.abs(xNew[i][0] - x[i][0]))
							.max()
							.orElseThrow()
					<= tolerance) {
				break;
			}

			for (int i = 0; i < n; i++) {
				x[i][0] = xNew[i][0];
			}

			// System.out.println(A.multiply(new DenseMatrix(x)).subtract(b).norm());
		}

		return new DenseMatrix(xNew);
	}

	public static Matrix<BigDecimal> precisJeacobi(final Matrix<BigDecimal> A, final Matrix<BigDecimal> b) {
		if (!A.isSquare()) {
			throw new IllegalArgumentException("Matrix A is not a square.");
		}
		if (b.getNumColumns() != 1 || b.getNumRows() != A.getNumRows()) {
			throw new IllegalArgumentException("Vector b is not a column vector.");
		}

		final MathContext ctx = new MathContext(100);
		final int n = A.getNumRows();
		final BigDecimal[][] x = new BigDecimal[n][1];
		final BigDecimal[][] xNew = new BigDecimal[n][1];

		for (int i = 0; i < n; i++) {
			x[i][0] = BigDecimal.ZERO;
		}

		final int maxit = 100;
		final BigDecimal tolerance = BigDecimal.valueOf(1e-8);
		for (int k = 0; k < maxit; k++) {
			for (int i = 0; i < n; i++) {
				BigDecimal s = BigDecimal.ZERO;
				for (int j = 0; j < n; j++) {
					if (j == i) {
						continue;
					}
					s = s.add(A.get(i, j).multiply(x[j][0], ctx), ctx);
				}
				xNew[i][0] = b.get(i, 0).subtract(s, ctx).divide(A.get(i, i), ctx);
			}

			if (IntStream.range(0, n)
							.mapToObj(i -> xNew[i][0].subtract(x[i][0], ctx).abs())
							.max(BigDecimal::compareTo)
							.orElseThrow()
							.compareTo(tolerance)
					<= 0) {
				break;
			}

			for (int i = 0; i < n; i++) {
				x[i][0] = xNew[i][0];
			}

			// System.out.println(A.multiply(new DenseMatrix(x)).subtract(b).norm());
		}

		return new PreciseMatrix(xNew);
	}
}
