
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Sameer Gaherwar, John Miller
 *  @version 2.0
 *  @date    Sun Sep 16 14:09:25 EDT 2012
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Cholesky Matrix Factorization
 */

package scalation
package mathstat

import scala.math.sqrt

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Fac_Cholesky` class provides methods to factor an n-by-n symmetric,
 *  positive definite matrix a into the product of two matrices:
 *      l   - an n-by-n left lower triangular matrix
 *      l.t - an n-by-n right upper triangular matrix - transpose of l
 *  such that a = l * l.t.
 *  @param a  the symmetric, positive definite matrix to be factor
 */
class Fac_Cholesky (a: MatrixD)
      extends Factorization:
 
    private val flaw = flawf ("Fac_Cholesky")        // flaw function
    private val n    = a.dim                         // the matrix should be n-by-n

    if n != a.dim2     then flaw ("init", "matrix a must be square")
    if ! a.isSymmetric then flaw ("init", "matrix a must be symmetric")

    private val l    = new MatrixD (n, n)            // for factored lower triangular matrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Factor matrix a into the product of l and l.t using Cholesky
     *  Factorization a = l * l.t, where l.t is l's transpose.
     *  It uses the Cholesky–Banachiewicz algorithm.
     *  @see introcs.cs.princeton.edu/java/95linear
     */
    def factor (): Fac_Cholesky =
        if factored then return this                  // matrix a is already factored

        for i <- 0 until n; j <- 0 to i do
            var sum = 0.0
            for k <- 0 until j do sum += l(i, k) * l(j, k)

            val diff = a(i, j) - sum
            if i == j then
                if diff < 0.0 then flaw ("factor", s"sqrt of negative diff = $diff")
                l(j, j) = sqrt (diff)
            else
                val l_jj = l(j, j)
                if l_jj == 0.0 then flaw ("factor", s"divide by zero l($j, $j) = $l_jj")
                l(i, j) = diff / l_jj
            end if
        end for
        factored = true
        this
    end factor

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Factor matrix a into the product of l and l.t using Cholesky
     *  Factorization a = l * l.t, where l.t is l's transpose.
     *  It uses the Cholesky–Crout algorithm.
     */
    def factor_ (): Fac_Cholesky =
        if factored then return this                  // matrix a is already factored

        var sum, diff = 0.0
        for j <- 0 until n do
            sum = 0.0
            for k <- 0 until j do sum += l(j, k) * l(j, k)
            diff = a(j, j) - sum
            if diff < 0.0 then flaw ("factor", s"sqrt of negative diff = $diff")
            l(j, j) = sqrt (diff)

            for i <- j + 1 until n do
                sum = 0.0
                for k <- 0 until j do sum += l(i, k) * l(j, k)
                diff = a(i, j) - sum
                l(i, j) = (1.0 / l(j, j) * diff)
            end for
        end for
        factored = true
        this
    end factor_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return both the lower triangular matrix l and its transpose l.t.
     */
    def factors: (MatrixD, MatrixD) = (l, l.transpose)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use the lower triangular matrix l from the Cholesky Factorization to
     *  solve a system of equations a * x = b. Return the solution x using
     *  forward and backward substitution.
     *  @param b  the constant vector
     */
    def solve (b: VectorD): VectorD =
        val y = new VectorD (n)
        for k <- 0 until n do                         // solve for y in l*y = b
            y(k) = (b(k) - (l(k) dot y)) / l(k, k)
        end for

        val x = new VectorD (n)
        for k <- n-1 to 0 by -1 do                    // solve for x in l.t*x = y
            x(k) = (y(k) - (l(?, k) dot x)) / l(k, k)
        end for
        x
    end solve

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Efficient calculation of inverse matrix a^-1 from existing factorization.
     *      a * a^-1 = I
     */
    def inverse: MatrixD =
//      factor ()                                     // uses Cholesky–Banachiewicz algorithm
        factor_ ()                                    // uses Cholesky–Crout algorithm
        val l_inv = Fac_Inverse.inverse_lt (l)
        l_inv.transpose * l_inv                       //  l^-1^t * l^-1
    end inverse

end Fac_Cholesky


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_CholeskyTest` main function is used to test the `Fac_Cholesky` class.
 *  @see ece.uwaterloo.ca/~dwharder/NumericalAnalysis/04LinearAlgebra/cholesky
 *  > runMain scalation.mathstat.fac_CholeskyTest
 */
@main def fac_CholeskyTest (): Unit =

    val a = MatrixD ((4, 4), 4.0,  0.4,   0.8, -0.2,
                             0.4,  1.04, -0.12, 0.28,
                             0.8, -0.12,  9.2,  1.4,
                            -0.2,  0.28,  1.4,  4.35)

    val b = VectorD (-0.2, -0.32, 13.52, 14.17)

    println (s"a = $a")
    println (s"b = $b")

    banner ("Cholesky Factorization")
    val chol = new Fac_Cholesky (a)
    chol.factor ()
    println ("factors = " + chol.factors)
    println ("solve   = " + chol.solve (b))

    banner ("LU Factorization")
    val lu = new Fac_LU (a)
    lu.factor ()
    println ("factors = " + lu.factors)
    println ("solve   = " + lu.solve (b))

end fac_CholeskyTest

