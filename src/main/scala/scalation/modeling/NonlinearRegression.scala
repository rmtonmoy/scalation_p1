
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Feb 20 17:39:57 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: NonLinear Regression (nonlinear in the parameters)
 */

package scalation
package modeling

import scalation.mathstat._
import scalation.optimization.BFGS

/** Function type for f(x, b) that maps a pair of vectors to a double
 */
type FunctionP2S = (VectorD, VectorD) => Double

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NonlinearRegression` class supports Nonlinear Regression.  In this case,
 *  x can be multi-dimensional [1, x1, ... xk] and the function f is nonlinear
 *  in the parameters b.  Fit the parameter vector b in the regression equation
 *      y  =  f(x, b) + e
 *  where e represents the residuals (the part not explained by the model).
 *  Use Least-Squares (minimizing the residuals) to fit the parameter vector
 *  b by using Nonlinear Programming to minimize sum of squared errors (sse).
 *  @see www.bsos.umd.edu/socy/alan/stats/socy602_handouts/kut86916_ch13.pdf
 *  @param x       the data/input matrix augmented with a first column of ones
 *  @param y       the response/output vector
 *  @param f       the nonlinear function f(x, b) to fit
 *  @param b_init  the initial guess for the parameter vector b
 *  @param fname_  the feature/variable names
 *  @param hparam  the hyper-parameters (currently has none)
 */
class NonlinearRegression (x: MatrixD, y: VectorD, f: FunctionP2S,
                           b_init: VectorD, fname_ : Array [String] = null,
                           hparam: HyperParameter = null)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfm = x.dim2 - 1, df = x.dim - x.dim2)
         with NoSubModels:

    private val debug = debugf ("Nonlinear", true)                       // debug function
    private val flaw  = flawf ("Nonlinear")                              // flaw function

    if y != null && x.dim != y.dim then flaw ("init", "dimensions of x and y are incompatible")

    modelName = "NonlinearRegression"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  nonlinear regression equation for the response vector y_.
     *      y = f(x, b)
     *  using the least squares method.
     *  Caveat:  Optimizer may converge to an unsatisfactory local optima.
     *           If the regression can be linearized, use linear regression for
     *           starting solution.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit =
    
        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Function to compute sse for the given values for the parameter vector b.
         *  @param b  the parameter vector
         */
        def sseF (b: VectorD): Double =
            val yp = x_.map (f(_, b))                                    // predicted response vector
            (y_ - yp).normSq                                             // sum of squared errors
        end sseF

        val bfgs   = new BFGS (sseF)                                     // minimize sseF using nonlinear optimizer
        val result = bfgs.solve (b_init)                                 // result from optimizer
        val sse    = result._1                                           // optimal function value
        b          = result._2                                           // optimal parameter vector
    end train

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
        e = y_ - yp                                                      // RECORD the residuals/errors (@see `Predictor`)
        (yp, diagnose (y_, yp))                                          // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = f(z, b),
     *  i.e., (b0, b1) dot (1.0, z1).
     *  @param z  the new vector to predict
     */
    override def predict (z: VectorD): Double = f(z, b)

end NonlinearRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NonlinearRegression` companion object provides factory methods for buidling
 *  Nonlinear Regression models.
 */
object NonlinearRegression:

    private val debug = debugf ("NonlinearRegression", true)             // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NonlinearRegression` with automatic rescaling from a combined data matrix.
     *  @param xy      the combined data/input and response/output matrix
     *  @param f       the nonlinear function f(x, b) to fit
     *  @param b_init  the initial guess for the parameter vector b
     *  @param fname   the feature/variable names
     *  @param hparam  the hyper-parameters (currently has none)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, f: FunctionP2S, b_init: VectorD,
               fname: Array [String] = null, hparam: HyperParameter = null)
              (col: Int = xy.dim2 - 1): NonlinearRegression =
        var itran: FunctionV2V = null                                    // inverse transform -> original scale
        val (x, y) = (xy.not(?, col), xy(?, col))                        // assumes the last column is the response

/*                                                                       // FIX - function needs bounds
        val x_s = if rescale then rescaleX (x, f0)
                  else x
        val y_s = if f0.bounds != null then { val y_i = rescaleY (y, f0); itran = y_i._2; y_i._1 }
                  else y

*/
        val (x_s, y_s) = (x, y)
        debug ("apply", s" scaled: x = $x_s \n scaled y = $y_s")
        new NonlinearRegression (x_s, y_s, f, b_init, fname, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NonlinearRegression` with automatic rescaling from a data matrix and response vector.
     *  @param x       the data/input matrix
     *  @param y       the response/output vector
     *  @param f       the nonlinear function f(x, b) to fit
     *  @param b_init  the initial guess for the parameter vector b
     *  @param fname   the feature/variable names
     *  @param hparam  the hyper-parameters (currently has none)
     */
    def rescale (x: MatrixD, y: VectorD, f: FunctionP2S, b_init: VectorD,
                 fname: Array [String] = null,
                 hparam: HyperParameter = null): NonlinearRegression =
        var itran: FunctionV2V = null                                    // inverse transform -> original scale

/*                                                                       // FIX - function needs bounds
        val x_s = if rescale then rescaleX (x, f0)
                  else x
        val y_s = if f0.bounds != null then { val y_i = rescaleY (y, f0); itran = y_i._2; y_i._1 }
                  else y
*/
        val (x_s, y_s) = (x, y)
        debug ("rescale", s" scaled: x = $x_s \n scaled y = $y_s")
        new NonlinearRegression (x_s, y_s, f, b_init, fname, hparam)
    end rescale

end NonlinearRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nonlinearRegressionTest` object tests the `NonlinearRegression` class:
 *  y  =  f(x; b)  =  b0 + exp (b1 * x0).
 *  @see www.bsos.umd.edu/socy/alan/stats/socy602_handouts/kut86916_ch13.pdf
 *  Answers:  sse = 49.45929986243339
 *            fit = (VectorD (58.606566327280426, -0.03958645286504356), 0.9874574894685292)
 *            predict (VectorD (50.0)) = 8.09724678182599
 *  FIX: check this example
 *  > runMain scalation.modeling.nonlinearRegressionTest
 */
@main def nonlinearRegressionTest (): Unit =

    import scala.math.exp

    // 5 data points: constant term, x1 coordinate, x2 coordinate
    val x = MatrixD ((15, 1), 2.0, 5.0, 7.0, 10.0, 14.0, 19.0, 26.0, 31.0, 34.0, 38.0, 45.0, 52.0, 53.0, 60.0, 65.0)
    val y = VectorD (54.0, 50.0, 45.0, 37.0, 35.0, 25.0, 20.0, 16.0, 18.0, 13.0, 8.0, 11.0, 8.0, 4.0, 6.0)

    println ("x = " + x)
    println ("y = " + y)

    def f (x: VectorD, b: VectorD): Double = b(0) * exp (b(1) * x(0))   // nonlinear regression function
    val b_init = VectorD (4.04, .038)                                   // initial guess for parameter vector b

    val mod = new NonlinearRegression (x, y, f, b_init)
    mod.trainNtest ()()

    val z  = VectorD (1); z(0) = 50.0                                   // predict y for one point
    val yp = mod.predict (z)
    println ("predict (" + z + ") = " + yp)

end nonlinearRegressionTest

