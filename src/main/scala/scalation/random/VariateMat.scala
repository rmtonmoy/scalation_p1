
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Mar  9 19:19:53 EST 2013
 *  @see     LICENSE (MIT style license file).
 */

package scalation
package random

import scala.math.{abs, exp, Pi, round, sqrt}

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VariateMat` abstract class serves as a base class for all the random
 *  variate matrix (RVM) generators. They use one of the Random Number Generators
 *  (RNG's) from Random.scala to generate numbers following their particular
 *  multivariate distribution.
 *-----------------------------------------------------------------------------
 *  @param stream  the random number stream
 */
abstract class VariateMat (stream: Int = 0):

    protected val flaw = flawf ("VariateMat")

    /** Random number stream selected by the stream number
     */
    protected val r = Random (stream)

    /** Indicates whether the distribution is discrete or continuous (default)
     */
    protected var _discrete = false

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the distribution is discrete or continuous.
     */
    def discrete: Boolean = _discrete

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the matrix mean for the particular distribution.
     */
    def mean: MatrixD

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the probability function (pf):
     *  The probability density function (pdf) for continuous RVM's or
     *  the probability mass function (pmf) for discrete RVM's.
     *  @param z  the mass point/matrix whose probability is sought
     */
    def pf (z: MatrixD): Double

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random double matrix for the particular distribution.
     */
    def gen: MatrixD

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random integer matrix for the particular distribution.
     *  It is only valid for discrete random variates.
     */
//  def igen: MatrixI

end VariateMat


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VariateMat` companion object provides a method to add correlation to
 *  a matrix.
 */
object VariateMat:

    def corTransform (x: MatrixD, cov: MatrixD): MatrixD =
        val fac1 = new Fac_Cholesky (cov).factor1 ()           // LL^t = Cov
        fac1 * x                                               // LX
    end corTransform

end VariateMat


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomMatD` class generates a random matrix of doubles.
 *  @param dim      the number of rows in the matrix
 *  @param dim2     the number of columns in the matrix
 *  @param max      generate integers in the range 0 (inclusive) to max (inclusive)
 *  @param min      generate integers in the range 0 (inclusive) to max (inclusive)
 *  @param density  sparsity basis = 1 - density
 *  @param stream   the random number stream
 */
case class RandomMatD (dim: Int = 5, dim2: Int = 10, max: Double = 20.0, min: Double = 0.0,
                       density: Double = 1.0, stream: Int = 0)
     extends VariateMat (stream):

    private val mu   = (min + max) / 2.0                                      // mean
    private val rvec = RandomVecD (dim2, max, min, density, stream = stream)  // random vector generator
    
    def mean: MatrixD = MatrixD.fill (dim, dim2, mu)

    def pf (z: MatrixD): Double = 1.0 / (max - min) ~^ (dim + dim2)

    def gen: MatrixD = MatrixD (for i <- 0 until dim yield rvec.gen)

//  def igen: MatrixI = gen.toInt

end RandomMatD


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `variateMatTest` main function is used to test the Random Variate Matrix (RVM)
 *  generators from the classes derived from `VariateMat`.
 *  > runMain scalation.random.variateMatTest
 */
@main def variateMatTest (): Unit =

     import VariateMat.corTransform

     var rvm: VariateMat = null                                // variate matrix

     banner ("Test: RandomMatD random matrix generation")
     rvm = RandomMatD (2, 100)                                 // random matrix generator
     println ("mean = " + rvm.mean)
     for k <- 0 until 10 do println (rvm.gen)

     val cor = MatrixD ((2, 2), 1.0, 0.9,                      // covariance/correlation matrix
                                0.9, 1.0)

     banner ("Test: RandomMatD random correlated matrix generation")
     val x = rvm.gen
     println ("x = " + x)
     val z = corTransform (x, cor)
     println ("z = " + z)
/* FIX
     val xx = x.toInt
     val zz = z.toInt

     Plot (xx(0), xx(1), null, "x uncorrelated")
     Plot (zz(0), zz(1), null, "z correlated")
*/

end variateMatTest

