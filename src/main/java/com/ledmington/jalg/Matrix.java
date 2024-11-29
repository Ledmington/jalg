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

import java.util.List;

public interface Matrix<X> {

	int getNumRows();

	int getNumColumns();

	X get(final int row, final int column);

	default boolean isSquare() {
		return getNumRows() == getNumColumns();
	}

	X conditionNumber();

	X norm();

	Matrix<X> getTranspose();

	Matrix<X> multiply(final Matrix<X> m);

	X getDeterminant();

	boolean isInvertible();

	Matrix<X> getInverse();

	boolean isPositiveDefinite();

	Matrix<X> gaussJordan();

	List<X> getEigenvalues();

	Matrix<X> subtract(final Matrix<X> other);

	default boolean isSymmetric() {
		return this.equals(this.getTranspose());
	}

	boolean isUpperTriangular();

	boolean isLowerTriangular();

	default boolean isTriangular() {
		return isLowerTriangular() || isUpperTriangular();
	}

	default boolean isDiagonal() {
		return isLowerTriangular() && isUpperTriangular();
	}

	boolean equals(final Matrix<X> other, final double eps);
}
