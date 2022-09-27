
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Michael E. Cotterell
 *  @version 2.0
 *  @date    Sun Jan 11 19:05:20 EST 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Exponential Regression
 */

// FIX: needs improved optimization: try IRWLS

package scalation
package modeling

import scala.math.exp

import scalation.mathstat._
import scalation.optimization.BFGS

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ExpRegression` class supports exponential regression.  In this case,
 *  x is multi-dimensional [1, x_1, ... x_k].  Fit the parameter vector b in the
 *  exponential regression equation
 *      log (mu (x))  =  b dot x  =  b_0 + b_1 * x_1 + ... b_k * x_k
 *  @see www.stat.uni-muenchen.de/~leiten/Lehre/Material/GLM_0708/chapterGLM.pdf 
 *  @param x       the data/input matrix
 *  @param y       the response/output vector
 *  @param fname_  the feature/variable names
 *  @param hparam  the hyper-parameters (currently none)
 *  @param nonneg  whether to check that responses are nonnegative
 */
class ExpRegression (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                     hparam: HyperParameter = null, nonneg: Boolean = true)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfm = x.dim2 - 1, df = x.dim - x.dim2):

    if nonneg && ! y.isNonnegative then flaw ("init", "response vector y must be nonnegative")

    private val debug      = debugf ("ExpRegression", true)              // debug function
    private val flaw       = flawf ("ExpRegression")                     // flaw function
    private var n_dev      = -1.0                                        // null dev: -LL, for null model (intercept only)
    private var r_dev      = -1.0                                        // residual dev: -LL, for full model
    private var pseudo_rSq = -1.0                                        // McFaffen's pseudo R-squared

    modelName = "ExpRegression"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For a given parameter vector b, compute -2 * Log-Likelihood (-2LL).
     *  -2LL is the standard measure that follows a Chi-Square distribution. 
     *  @see www.stat.cmu.edu/~cshalizi/350/lectures/26/lecture-26.pdf
     *  @see www.statisticalhorizons.com/wp-content/uploads/Allison.StatComp.pdf
     *  @param b  the parameters to fit
     */
    def ll (b: VectorD): Double = 
        var sum = 0.0
        for i <- y.indices do
            val bx = b dot x(i)
            sum += -bx - y(i) / exp (bx)
        end for
        -2.0 * sum
    end ll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For a given parameter vector b, compute -2 * Log-Likelihood (-2LL) for
     *  the null model (the one that does not consider the effects of x(i)).
     *  @param b  the parameters to fit
     */
    def ll_null (b: VectorD): Double =
        var sum = 0.0
        for i <- y.indices do
            val bx = b(0)                                                // only use intercept
            sum += -bx - y(i) / exp (bx)
        end for
        -2.0 * sum
    end ll_null

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  exponential regression equation.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit =
        // FIX - currently only works for x_ = x and y_ = y
        train_null ()
        val b0   = new VectorD (x_.dim2)                                 // use b_0 = 0 for starting guess for parameters
        val bfgs = new BFGS (ll)                                         // minimizer for -2LL
        b        = bfgs.solve (b0)._2                                    // find optimal solution for parameters

//      e = y_ / (x_ * b)                                                // residual/error vector e
        e = y_ - (x_ * b).map (exp _)                                    // residual/error vector e
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For the null model, train the classifier by fitting the parameter vector
     *  (b-vector) in the logistic regression equation using maximum likelihood.
     *  Do this by minimizing -2l.
     */
    def train_null (): Unit =
         val b0   = new VectorD (x.dim2)                                 // use b0 = 0 for starting guess for parameters
         val bfgs = new BFGS (ll_null)                                   // minimizer for -2l
         val b_n = bfgs.solve (b0)._2                                    // find optimal solution for parameters

         n_dev   = ll_null (b_n)                                         // measure of fitness for null model
    end train_null

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD) =
        val yp = predict (x_)                                            // make predictions
        (yp, diagnose (y_, yp))                                          // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor 'x_j'
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = getX, fname_ : Array [String] = fname, b_ : VectorD = b,
                          vifs: VectorD = vif ()): String =
        super.summary (x_, fname_, b_, vifs)                             // summary from `Fit`
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = exp (b dot z),
     *  e.g., exp (b_0, b_1, b_2) dot (1, z_1, z_2).
     *  @param z  the new vector to predict
     */
    override def predict (z: VectorD): Double = exp (b dot z)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    override def buildModel (x_cols: MatrixD): ExpRegression =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new ExpRegression (x_cols, y, null, hparam, nonneg)
    end buildModel

end ExpRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ExpRegression` companion object provides factory apply functions and a testing method.
 */
object ExpRegression:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ExpRegression` object from a combined data-response matrix.
     *  The last column is assumed to be the response column.
     *  @param xy      the combined data-response matrix (predictors and response)
     *  @param fname   the feature/variable names
     *  @param hparam  the hyper-parameters
     *  @param nonneg  whether to check that responses are nonnegative
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = null, nonneg: Boolean = true)
              (col: Int = xy.dim2 - 1): ExpRegression =
        new ExpRegression (xy.not(?, col), xy(?, col), fname, hparam, nonneg)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ExpRegression` object from a data matrix and a response vector.
     *  This factory function provides data rescaling.
     *  @param x       the data/input m-by-n matrix
     *                     (augment with a first column of ones to include intercept in model)
     *  @param y       the response/output m-vector
     *  @param fname   the feature/variable names (use null for default)
     *  @param hparam  the hyper-parameters (use null for default)
     *  @param nonneg  whether to check that responses are nonnegative
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = null, nonneg: Boolean = true): ExpRegression =
        val xn = normalize ((x.mean, x.stdev)) (x)
        new ExpRegression (xn, y, fname, hparam, nonneg)
    end rescale

end ExpRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `expRegressionTest` main function tests `ExpRegression` class using the following
 *  exponential regression problem.
 *  > runMain scalation.modeling.expRegressionTest
 */
@main def expRegressionTest: Unit =

    val x = MatrixD ((5, 3), 1.0, 36.0,  66.0,                           // 5-by-3 matrix
                             1.0, 37.0,  68.0,
                             1.0, 47.0,  64.0,
                             1.0, 32.0,  53.0,
                             1.0,  1.0, 101.0)
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 1598.0)
    val z = VectorD (1.0, 20.0, 80.0)

    println (s"x = $x")

    val mod = new ExpRegression (x, y)                                   // create model with with interecept (else pass x)
    mod.trainNtest ()()                                                  // train and test the model
    println (mod.summary ())                                             // parameter/coefficient statistics

    val yp = mod.predict (x)
    println (s"y  = $y")
    println (s"yp = $yp")

    new Plot (null, y, yp, "expRegressionTest")

    val yp2 = mod.predict (z)
    println (s"predict ($z) = $yp2")

end expRegressionTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `expRegressionTest2` main function has a basic test for the `ExpRegression` class.
 *  > runMain scalation.modeling.expRegressionTest
 */
@main def expRegressionTest2 (): Unit =

    import scalation.random.{Uniform, Exponential, Random}

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test `ExpRegression` by simulating n-many observations.
     *  @param n  number of observations
     *  @param k  number of variables
     */
    def test (n: Int = 10000, k: Int = 5): Unit =
        val u = new Uniform (0, 1)                                       // uniform random
        val e = new Exponential (1)                                      // exponential error
        val r = new Random ()

        val x = new MatrixD (n, k)                                       // data matrix
        val y = new VectorD (x.dim)                                      // response vector
        val b = new VectorD (k)                                          // known coefficients

        for i <- b.indices do b(i) = 1 + r.gen * 6

        for i <- x.indices; j <- x.indices2 do x(i, j) = u.gen

        for i <- y.indices do y(i) = exp (x(i) dot b) * e.gen

        banner ("Simulated exp regression problem with $k vars")
        val mod = new ExpRegression (x, y)                                // create model with with interecept (else pass x)
        mod.trainNtest ()()                                               // train and test the model
        println (mod.summary ())                                          // parameter/coefficient statistics
    end test

    for k <- 1 to 10 do test (1000, k)

end expRegressionTest2

