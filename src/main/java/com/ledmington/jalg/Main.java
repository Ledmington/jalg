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

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public final class Main {

	private static void benchInverse() {
		final long n = 1000L;
		final long flops = n * (2L * n + (n - 1L) * 2L * n);
		System.out.printf("Total FLOPs: %,d\n", flops);
		for (int i = 0; i < 10; i++) {
			final Matrix<Double> m = DenseMatrix.random((int) n, (int) n, -1.0, 1.0);
			final long start = System.nanoTime();
			m.getInverse();
			final long end = System.nanoTime();
			final long ns = end - start;
			final double s = (double) ns / 1_000_000_000.0;
			System.out.printf(
					" %,d -> %,d ns (%.6f s) -> %.6f GFLOPs/s\n", n, ns, s, ((double) flops / s) / 1_000_000_000.0);
		}
	}

	public static void main(final String[] args) {
		final RandomGenerator rng = RandomGeneratorFactory.getDefault().create(System.nanoTime());
		final Matrix<Double> A;
		{
			final double[][] m = new double[10][10];
			for (int i = 0; i < 10; i++) {
				for (int j = 0; j < 10; j++) {
					if (Math.abs(i - j) <= 1) {
						m[i][j] = rng.nextDouble(-1.0, 1.0);
					}
				}
			}
			A = new DenseMatrix(m);
		}

		final Matrix<Double> b = DenseMatrix.random(10, 1, -1.0, 1.0);
		System.out.printf("K(A) = %.6f\n", A.conditionNumber());
		final Matrix<Double> x = Solvers.jacobi(A, b);
		System.out.println(x);

		System.out.println(A.multiply(x).subtract(b).norm());
	}
}
