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

public interface Matrix {

	int getNumRows();

	int getNumColumns();

	double get(final int row, final int column);

	default boolean isSquare() {
		return getNumRows() == getNumColumns();
	}

	default double conditionNumber() {
		return this.norm() * this.getInverse().norm();
	}

	double norm();

	Matrix getTranspose();

	Matrix multiply(final Matrix m);

	double getDeterminant();

	Matrix getInverse();

	Matrix gaussJordan();

	List<Double> getEigenvalues();

	Matrix subtract(final Matrix other);

	default boolean isSymmetric() {
		return this.equals(this.getTranspose());
	}

	default boolean isUpperTriangular() {
		if (!isSquare()) {
			throw new IllegalArgumentException("A rectangular matrix cannot be triangular.");
		}
		for (int i = 0; i < getNumRows(); i++) {
			for (int j = 0; j < i; j++) {
				if (get(i, j) != 0.0) {
					return false;
				}
			}
		}
		return true;
	}

	default boolean isLowerTriangular() {
		if (!isSquare()) {
			throw new IllegalArgumentException("A rectangular matrix cannot be triangular.");
		}
		for (int i = 0; i < getNumRows(); i++) {
			for (int j = i + 1; j < getNumColumns(); j++) {
				if (get(i, j) != 0.0) {
					return false;
				}
			}
		}
		return true;
	}

	default boolean isTriangular() {
		return isLowerTriangular() || isUpperTriangular();
	}

	default boolean isDiagonal() {
		return isLowerTriangular() && isUpperTriangular();
	}

	default boolean equals(final Matrix other, final double eps) {
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
				if (Math.abs(this.get(i, j) - other.get(i, j)) > eps) {
					return false;
				}
			}
		}
		return true;
	}
}
