
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Feb 20 17:39:57 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Multiple Linear Regression (linear terms, no cross-terms)
 */

package scalation
package modeling

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Regression` class supports multiple linear regression.  In this case,
 *  x is multi-dimensional [1, x_1, ... x_k].  Fit the parameter vector b in
 *  the regression equation
 *      y  =  b dot x + e  =  b_0 + b_1 * x_1 + ... b_k * x_k + e
 *  where e represents the residuals (the part not explained by the model).
 *  Use Least-Squares (minimizing the residuals) to solve the parameter vector b
 *  using the Normal Equations:
 *      x.t * x * b  =  x.t * y 
 *      b  =  fac.solve (.)
 *  Five factorization algorithms are provided:
 *      `Fac_QR`         QR Factorization: slower, more stable (default)
 *      `Fac_SVD`        Singular Value Decomposition: slowest, most robust
 *      `Fac_Cholesky`   Cholesky Factorization: faster, less stable (reasonable choice)
 *      `Fac_LU'         LU Factorization: better than Inverse
 *      `Fac_Inverse`    Inverse Factorization: textbook approach
 *  @see see.stanford.edu/materials/lsoeldsee263/05-ls.pdf
 *  Note, not intended for use when the number of degrees of freedom 'df' is negative.
 *  @see en.wikipedia.org/wiki/Degrees_of_freedom_(statistics)
 *------------------------------------------------------------------------------
 *  @param x       the data/input m-by-n matrix
 *                     (augment with a first column of ones to include intercept in model)
 *  @param y       the response/output m-vector
 *  @param fname_  the feature/variable names
 *  @param hparam  the hyper-parameters
 */
class Regression (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                  hparam: HyperParameter = Regression.hp)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfm = x.dim2 - 1, df = x.dim - x.dim2):
         // if not using an intercept df = (x.dim2, x.dim-x.dim2), correct by calling 'resetDF' method from `Fit`

    private val debug     = debugf ("Regression", false)                 // debug function
    private val flaw      = flawf ("Regression")                         // flaw function
    private val algorithm = hparam("factorization")                      // factorization algorithm
    private val n         = x.dim2                                       // number of columns

    modelName = "Regression"

    if n < 1 then flaw ("init", s"dim2 = $n of the 'x' matrix must be at least 1")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a solver for the Normal Equations using the selected factorization algorithm.
     *  @param x_  the matrix to be used by the solver
     */
    private def solver (x_ : MatrixD): Factorization =
        algorithm match                                                  // select factorization algorithm
        case "Fac_Cholesky" => new Fac_Cholesky (x_.transpose * x_)      // Cholesky Factorization
        case "Fac_LU"       => new Fac_LU (x_.transpose * x_)            // LU Factorization
        case "Fac_Inverse"  => new Fac_Inverse (x_.transpose * x_)       // Inverse Factorization
        case "Fac_SVD"      => new Fac_SVD (x_)                          // Singular Value Decomposition
        case _              => new Fac_QR (x_)                           // QR Factorization (default)
        end match
    end solver

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  multiple regression equation
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_1 , ... x_k] + e
     *  using the ordinary least squares 'OLS' method.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit =
        val fac = solver (x_)
        fac.factor ()                                                    // factor the matrix, either X or X.t * X

        b = fac match                                                    // RECORD the parameters/coefficients (@see `Predictor`)
            case fac: Fac_QR  => fac.solve (y_)
            case fac: Fac_SVD => fac.solve (y_)
            case _            => fac.solve (x_.transpose * y_)

        if b(0).isNaN then flaw ("train", s"parameter b = $b")
        debug ("train", s"$fac estimates parameter b = $b")
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
    /** Predict the value of vector y = f(x_, b).  It is overridden for speed.
     *  @param x_  the matrix to use for making predictions, one for each row
     */
    override def predict (x_ : MatrixD): VectorD = x_ * b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    override def buildModel (x_cols: MatrixD): Regression =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new Regression (x_cols, y, null, hparam)
    end buildModel

end Regression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Regression` companion object provides factory apply functions and a testing method.
 */
object Regression:

    /** Base hyper-parameter specification for `Regression`
     */
    val hp = new HyperParameter; hp += ("factorization", "Fac_QR", "Fac_QR")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `Regression` object from a combined data-response matrix.
     *  @param xy      the combined data-response matrix (predictors and response)
     *  @param fname   the feature/variable names
     *  @param hparam  the hyper-parameters
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = hp)(col: Int = xy.dim2 - 1): Regression = 
        new Regression (xy.not(?, col), xy(?, col), fname, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `Regression` object from a data matrix and a response vector.
     *  This factory function provides data rescaling.
     *  @param x       the data/input m-by-n matrix
     *                     (augment with a first column of ones to include intercept in model)
     *  @param y       the response/output m-vector
     *  @param fname   the feature/variable names (use null for default)
     *  @param hparam  the hyper-parameters (use Regression.hp for default)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = hp): Regression = 
        val xn = normalize ((x.mean, x.stdev)) (x)
        new Regression (xn, y, fname, hparam)
    end rescale

end Regression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTest` main function tests `Regression` class using the following
 *  regression equation.
 *      y  =  b dot x  =  b_0 + b_1*x_1 + b_2*x_2.
 *  @see statmaster.sdu.dk/courses/st111/module03/index.html
 *  > runMain scalation.modeling.regressionTest
 */
@main def regressionTest (): Unit =

    // 5 data points: constant term, x_1 coordinate, x_2 coordinate

    val x = MatrixD ((5, 3), 1.0, 36.0,  66.0,                     // 5-by-3 matrix
                             1.0, 37.0,  68.0,
                             1.0, 47.0,  64.0,
                             1.0, 32.0,  53.0,
                             1.0,  1.0, 101.0)
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 1598.0)
    val z = VectorD (1.0, 20.0, 80.0)

    println ("model: y = b_0 + b_1*x_1 + b_2*x_2")

//  Pick one of the factorization algorithms via the hyper-parameter

    Regression.hp("factorization") = "Fac_QR"
//  Regression.hp("factorization") = "Fac_SVD"
//  Regression.hp("factorization") = "Fac_Cholesky"
//  Regression.hp("factorization") = "Fac_LU"
//  Regression.hp("factorization") = "Fac_Inverse"

    val mod = new Regression (x, y)                                // create a regression model
    mod.train ()                                                   // train the model
    println (mod.report (mod.test ()._2))                          // test the model and report the results
    println (s"predict ($z) = ${mod.predict (z)}")                 // make an out-of-sample prediction

    val mod2 = Regression (x :^+ y)()                              // create model from combined matrix
                                                                   // () -> use last column for response
    mod2.train ()                                                  // train the model
    println (mod2.report (mod2.test ()._2))                        // test the model and report the results
    println (s"predict ($z) = ${mod2.predict (z)}")                // make an out-of-sample prediction

end regressionTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTest2` main function is used to test the correctness of the
 *  factorization algorithms used to solve for the parameters b in `Regression`.
 *  @see `scalation.mathstat.Fac_QRTest2`
 *  > runMain scalation.modeling.regressionTest2
 */
@main def regressionTest2 (): Unit =

    // Blood Pressure Dataset
    // 20 data points:   Constant     x_1     x_2    x_3      x_4
    //                                Age  Weight    Dur   Stress
    val x = MatrixD ((20, 5), 1.0,   47.0,   85.4,   5.1,    33.0,
                              1.0,   49.0,   94.2,   3.8,    14.0,
                              1.0,   49.0,   95.3,   8.2,    10.0,
                              1.0,   50.0,   94.7,   5.8,    99.0,
                              1.0,   51.0,   89.4,   7.0,    95.0,
                              1.0,   48.0,   99.5,   9.3,    10.0,
                              1.0,   49.0,   99.8,   2.5,    42.0,
                              1.0,   47.0,   90.9,   6.2,     8.0,
                              1.0,   49.0,   89.2,   7.1,    62.0,
                              1.0,   48.0,   92.7,   5.6,    35.0,
                              1.0,   47.0,   94.4,   5.3,    90.0,
                              1.0,   49.0,   94.1,   5.6,    21.0,
                              1.0,   50.0,   91.6,  10.2,    47.0,
                              1.0,   45.0,   87.1,   5.6,    80.0,
                              1.0,   52.0,  101.3,  10.0,    98.0,
                              1.0,   46.0,   94.5,   7.4,    95.0,
                              1.0,   46.0,   87.0,   3.6,    18.0,
                              1.0,   46.0,   94.5,   4.3,    12.0,
                              1.0,   48.0,   90.5,   9.0,    99.0,
                              1.0,   56.0,   95.7,   7.0,    99.0)
    //  response BP
    val y = VectorD (105.0, 115.0, 116.0, 117.0, 112.0, 121.0, 121.0, 110.0, 110.0, 114.0,
                     114.0, 115.0, 114.0, 106.0, 125.0, 114.0, 106.0, 113.0, 110.0, 122.0)

    println ("model: y = b_0 + b_1*x1 + b_2*x_ + b3*x3 + b4*x42")
//  println ("model: y = b₀ + b₁∙x₁ + b₂∙x₂ + b₃∙x₃ + b₄∙x₄")
    println (s"x = $x")
    println (s"y = $y")

    val xtx = x.transpose * x
    val xty = x.transpose * y

    var fac: Factorization = null                             // factorization algorithm
    var mod: Regression = null                                // regression model
    var yp:  VectorD = null                                   // y-predicted (predicted response)

// Test QR Factorization -------------------------------------------------------

    banner ("Direct Application of QR Factorization")
    fac = new Fac_QR (x)                                      // input = X
    fac.factor ()
    println (s"parameters b = ${fac.solve (y)}")              // compute the b vector by using solve of `Fac_QR`

    banner ("Application of Factorization via Regression")
    Regression.hp("factorization") = "Fac_QR"
    mod = new Regression (x, y)                               // create a regression model
    mod.train ()                                              // train the model
    println (mod.report (mod.test ()._2))                     // test the model and report the results

// Test SVD Factorization ------------------------------------------------------

    banner ("Direct Application of SVD Factorization")
    fac = new Fac_SVD (x)                                     // input = X
    fac.factor ()
    println (s"parameters b = ${fac.solve (y)}")              // compute the b vector by using solve of `Fac_QR`

    banner ("Application of Factorization via Regression")
    Regression.hp("factorization") = "Fac_SVD"
    mod = new Regression (x, y)                               // create a regression model
    mod.train ()                                              // train the model
    println (mod.report (mod.test ()._2))                     // test the model and report the results

// Test Cholesky Factorization -------------------------------------------------

    banner ("Direct Application of Cholesky Factorization")
    fac = new Fac_Cholesky (xtx)                              // input = X^t * X
    fac.factor ()
    println (s"parameters b = ${fac.solve (xty)}")            // compute the b vector by using solve of `Fac_Cholesky`

    banner ("Application of Factorization via Regression")
    Regression.hp("factorization") = "Fac_Cholesky"
    mod = new Regression (x, y)                               // create a regression model
    mod.train ()                                              // train the model
    println (mod.report (mod.test ()._2))                     // test the model and report the results

// Test LU Factorization -------------------------------------------------------

    banner ("Direct Application of LU Factorization")
    fac = new Fac_LU (xtx)                                    // input = X^t * X
    fac.factor ()
    println (s"parameters b = ${fac.solve (xty)}")            // compute the b vector by using solve of `Fac_LU`

    banner ("Application of Factorization via Regression")
    Regression.hp("factorization") = "Fac_LU"
    mod = new Regression (x, y)                               // create a regression model
    mod.train ()                                              // train the model
    println (mod.report (mod.test ()._2))                     // test the model and report the results

// Test Inverse Factorization -------------------------------------------------------

    banner ("Direct Application of Inverse Factorization")
    fac = new Fac_LU (xtx)                                    // input = X^t * X
    fac.factor ()
    println (s"parameters b = ${fac.solve (xty)}")            // compute the b vector by using solve of `Fac_LU`

    banner ("Application of Factorization via Regression")
    Regression.hp("factorization") = "Fac_Inverse"
    mod = new Regression (x, y)                               // create a regression model
    mod.train ()                                              // train the model
    println (mod.report (mod.test ()._2))                     // test the model and report the results

end regressionTest2

import Example_AutoMPG._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTest3` main function tests the `Regression` class using the AutoMPG
 *  dataset.  Assumes no missing values.  It test cross validation.
 *  > runMain scalation.modeling.regressionTest3
 */
@main def regressionTest3 (): Unit =

//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    banner ("AutoMPG Regression")
    val mod = new Regression (ox, y, ox_fname)                // create model with intercept (else pass x)
    mod.trainNtest ()()                                       // train and test the model
    println (mod.summary ())                                  // parameter/coefficient statistics

    banner ("AutoMPG Validation Test")
    mod.validate ()()

    banner ("AutoMPG Cross-Validation Test")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end regressionTest3
 

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTest4` main function tests the `Regression` class using the AutoMPG
 *  dataset.  Assumes no missing values.  It tests forward selection.
 *  > runMain scalation.modeling.regressionTest4
 */
@main def regressionTest4 (): Unit =

//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    banner ("AutoMPG Regression")
    val mod = new Regression (ox, y, ox_fname)                // create model with intercept (else pass x)
    mod.trainNtest ()()                                       // train and test the model
    println (mod.summary ())                                  // parameter/coefficient statistics

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                    // R^2, R^2 bar, R^2 cv
//  val (cols, rSq) = mod.backwardElimAll ()                  // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Regression", lines = true)
    println (s"rSq = $rSq")

end regressionTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTest5` main function tests the `Regression` class using the AutoMPG
 *  dataset.  Assumes no missing values.  It tests forward, backward and stepwise selection.
 *  > runMain scalation.modeling.regressionTest5
 */
@main def regressionTest5 (): Unit =

//  println (s"ox = $ox")
//  println (s"y  = $y")

    banner ("AutoMPG Regression")
    val mod = new Regression (ox, y, ox_fname)                // create model with intercept (else pass x)
    mod.trainNtest ()()                                       // train and test the model
    println (mod.summary ())                                  // parameter/coefficient statistics

    banner ("Cross-Validation")
    FitM.showQofStatTable (mod.crossValidate ())

    println (s"ox_fname = ${stringOf (ox_fname)}")

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)           // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Regression with $tech", lines = true)
        banner ("Feature Importance")
        println (s"$tech: rSq = $rSq")
        val imp = mod.importance (cols.toArray, rSq)
        for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r") 
    end for

end regressionTest5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTest6` main function tests the `Regression` class using the following
 *  regression equation.
 *      y = b dot x = b_0 + b_1*x1 + b_2*x_2.
 *  Show effects of increasing collinearity.
 *  > runMain scalation.modeling.regressionTest6
 */
@main def regressionTest6 (): Unit =
//                         one x1 x2
    val x = MatrixD ((4, 3), 1, 1, 1,
                             1, 2, 2,
                             1, 3, 3,
                             1, 4, 0)                             // change 0 by .5 to 4
    val y = VectorD (1, 3, 3, 4)

    val v = x(?, 0 until 2)
    banner (s"Test without column x2")
    println (s"v = $v")
    var mod = new Regression (v, y)
    mod.trainNtest ()()
    println (mod.summary ())

    for i <- 0 to 8 do
        banner (s"Test Increasing Collinearity: x_32 = ${x(3, 2)}")
        println (s"x = $x")
        println (s"x.corr = ${x.corr}")
        mod = new Regression (x, y)
        mod.trainNtest ()()
        println (mod.summary ())
        x(3, 2) += 0.5
    end for

end regressionTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTest7` main function trains a regression model on a small dataset of
 *  temperatures from counties in Texas where the variables/factors to consider
 *  are Latitude (x1), Elevation (x2) and Longitude (x3).  The model equation
 *  is the following:
 *      y  =  b dot x  =  b0 + b1*x1 + b2*x2 + b3*x3
 *  > runMain scalation.modeling.regressionTest7
 */
@main def regressionTest7 (): Unit =

    // 16 data points:         one      x1      x2       x3     y
    //                                 Lat    Elev     Long  Temp        County
    val xy = MatrixD ((16, 5), 1.0, 29.767,   41.0,  95.367, 56.0,    // Harris
                               1.0, 32.850,  440.0,  96.850, 48.0,    // Dallas
                               1.0, 26.933,   25.0,  97.800, 60.0,    // Kennedy
                               1.0, 31.950, 2851.0, 102.183, 46.0,    // Midland
                               1.0, 34.800, 3840.0, 102.467, 38.0,    // Deaf Smith
                               1.0, 33.450, 1461.0,  99.633, 46.0,    // Knox
                               1.0, 28.700,  815.0, 100.483, 53.0,    // Maverick
                               1.0, 32.450, 2380.0, 100.533, 46.0,    // Nolan
                               1.0, 31.800, 3918.0, 106.400, 44.0,    // El Paso
                               1.0, 34.850, 2040.0, 100.217, 41.0,    // Collington
                               1.0, 30.867, 3000.0, 102.900, 47.0,    // Pecos
                               1.0, 36.350, 3693.0, 102.083, 36.0,    // Sherman
                               1.0, 30.300,  597.0,  97.700, 52.0,    // Travis
                               1.0, 26.900,  315.0,  99.283, 60.0,    // Zapata
                               1.0, 28.450,  459.0,  99.217, 56.0,    // Lasalle
                               1.0, 25.900,   19.0,  97.433, 62.0)    // Cameron

    banner ("Texas Temperatures Regression")
    val mod = Regression (xy)()                               // create model with intercept (else pass x)
    mod.trainNtest ()()                                       // train and test the model
    println (mod.summary ())                                  // parameter/coefficient statistics

end regressionTest7

