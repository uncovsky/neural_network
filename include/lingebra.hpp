#pragma once

#include <cassert>
#include <vector>
#include <iostream>
#include <tuple>

/*
 *
	Column major matrices
	Matrix multiplication

	Add column vector to each column (bias)

	Apply function on whole matrix (activation for FP, derivative for BP)
		can use this to multiply with scalar as well (learning rate for
		adjusting parameters after BP)

	Componentwise (hadamard) product with another matrix (for BP - product with sigma'(potential))

	Transposition / or multiplication with matrix transpose for BP

	RowReduce - 1/p sum of derivative L_j/y_i for a given y_i, sum and divide row values of a matrix

    Random initializations somehow
 */

class Matrix {
    std::vector< float > _data;

public:
    size_t rows;
    size_t cols;

/*
 *  constructors and helper functions
 */

    Matrix() : _data({}), rows(0), cols(0) {}

    // Zero initialized matrix
    Matrix( size_t rows, size_t cols ) : _data({}), rows(rows), cols(cols) {
        _data = std::vector< float >( rows * cols, 0.0 );
    }

    Matrix( const Matrix& rhs ) : _data( rhs.data() ), rows(rhs.rows), cols(rhs.cols) {}
    Matrix( Matrix&& rhs ) : _data( std::move(rhs.data()) ), rows(rhs.rows), cols(rhs.cols){}


    Matrix( std::vector< float >&& data, size_t rows, size_t cols ) : _data( std::move( data ) )
                                                                    , rows( rows )
                                                                    , cols( cols ) {}

    Matrix( std::vector< float >& data, size_t rows, size_t cols ) : _data( data )
                                                                   , rows( rows )
                                                                   , cols( cols ) {}

    Matrix& operator=(const Matrix& rhs){
        _data = rhs.data();
        rows = rhs.rows;
        cols = rhs.cols;
        return *this;
    }

    Matrix& operator=(Matrix&& rhs){
        _data = std::move( rhs.data() );
        rows = rhs.rows;
        cols = rhs.cols;
        return *this;
    }

    std::vector< float > & data() {
        return _data;
    }

    const std::vector< float > & data() const {
        return _data;
    }


    // column-major storage
    float at( size_t row, size_t col ) const{
        return _data[col * rows + row];
    }

    float& at( size_t row, size_t col ){
        return _data[col * rows + row];
    }

    void print() const{

        std::cout << "Rows " << rows << " Columns " << cols << "\n";
        for ( size_t row = 0; row < rows; row++ ) {
            std::cout << "| ";
            for ( size_t col = 0; col < cols; col++ ){
                std::cout << at(row, col) << " | ";
            }

            std::cout << std::endl;
        }

        std::cout << "\n";
    }


/*
 *  Main interface
 */
    Matrix mult( const Matrix &rhs ) {

        assert( cols == rhs.rows );

        Matrix res(rows, rhs.cols);


        for ( size_t col1 = 0; col1 < cols; col1++ ){
            for (size_t col2 = 0; col2 < rhs.cols; col2++){
                for ( size_t row1 = 0; row1 < rows; row1++ ){
                    res.at(row1, col2) += at(row1, col1) * rhs.at(col1, col2);
                }
            }
        }


        return res;
    }


    Matrix transpose() const {

        Matrix res(cols, rows);

        for ( size_t col1 = 0; col1 < cols; col1++ ){
            for ( size_t row1 = 0; row1 < rows; row1++ ){
                res.at(col1, row1) = at(row1, col1);
            }
        }

        return res;
    }

    // Apply function on this matrix in place
    template < typename func >
    Matrix& apply( func f ) {
        for ( auto &x : _data ) {
            x = f(x);
        }

        return *this;
    }

    Matrix& add_scalar( float n ){
        for ( auto &x : _data ) {
            x += n;
        }

        return *this;
    }

    Matrix& multiply_scalar( float n ){
        for ( auto &x : _data ) {
            x *= n;
        }

        return *this;
    }

    /*
     * Component-wise operations on matrices.
     * The rhs matrix must have the same number of rows as *this,
     * but may have a different number of columns. If rhs has less columns, the
     * last column is repeated to accomodate the size difference.
     */

    // Component-wise (Hadamard) product with rhs of same dimensions
    Matrix& cwise_product( const Matrix& rhs ){

        for ( size_t col1 = 0; col1 < cols; col1++ ){
            for ( size_t row1 = 0; row1 < rows; row1++ ){
                at(row1, col1) *= rhs.at(row1, std::min(rhs.cols - 1, col1));
            }
        }

        return *this;
    }

    // Component-wise addition of matrices, supports different 
    Matrix& cwise_add( const Matrix& rhs ){

        for ( size_t col1 = 0; col1 < cols; col1++ ){
            for ( size_t row1 = 0; row1 < rows; row1++ ){
                at(row1, col1) += rhs.at(row1, std::min(rhs.cols - 1, col1));
            }
        }

        return *this;
    }

    // Sum rows of this matrix into single elements
    Matrix& row_reduce(){

        std::vector< float > new_data( rows, 0.f );

        for ( size_t col1 = 0; col1 < cols; col1++ ){
            for ( size_t row1 = 0; row1 < rows; row1++ ){
                new_data[row1] += at(row1, col1);
            }
        }
        _data = std::move( new_data );

        // set cols to correct number
        cols = 1;

        return *this;
    }
};
