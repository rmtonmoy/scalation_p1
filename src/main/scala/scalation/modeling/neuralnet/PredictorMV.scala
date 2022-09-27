
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Jan 21 19:16:18 EST 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model Framework: Predictor for Matrix Input, Matrix Output
 */

package scalation
package modeling
package neuralnet

import scala.collection.mutable.{ArrayBuffer, IndexedSeq, LinkedHashSet, Set}
import scala.math.{cbrt, sqrt}
import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat._
import scalation.random.PermutedVecI

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PredictorMV` trait provides a framwork for multiple predictive analytics
 *  techniques, e.g., Neural Netoworks.  x is multi-dimensional [1, x_1, ... x_k].
 *  Fit the parameter vector b in for example the regression equation
 *      y  =  f(b dot x) + e
 *  @param x       the input/data m-by-n matrix
 *                     (augment with a first column of ones to include intercept in model)
 *  @param y       the response/output m-by-ny matrix
 *  @param fname   the feature/variable names (if null, use x_j's)
 *  @param hparam  the hyper-parameters for the model/network
 */
trait PredictorMV (x: MatrixD, y: MatrixD, protected var fname: Array [String],
                   hparam: HyperParameter)
      extends Model:

    protected val DO_PLOT = true                                             // whether to plot y vs yp
    private   val debug   = debugf ("PredictorMV", true)                     // debug function
    private   val flaw    = flawf ("PredictorMV")                            // flaw function

    if x != null then
        if x.dim != y.dim then flaw ("init", "row dimensions of x and y are incompatible")
        if x.dim <= x.dim2 then
            flaw ("init", s"PredictorMV requires more rows ${x.dim} than columns ${x.dim2}")
    end if

    private val MIN_FOLDS = 3                                                // minimum number of folds for cross validation
    private val stream    = 0                                                // random number stream to use
    private val permGen   = PermutedVecI (VectorI.range (0, y.dim), stream)  // permutation generator

    protected var b: MatrixD | NetParam = null                               // parameter/coefficient matrix - init in class
    protected var bb: Array [NetParam]  = null                               // array of network parameters  - init in class
    protected var e: MatrixD            = null                               // residual/error matrix

    if x != null && fname == null then fname = x.indices2.map ("x" + _).toArray  // default feature/variable names

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used data matrix x.  Mainly for derived classes where x is expanded
     *  from the given columns in x_.
     */
    def getX: MatrixD = x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used response matrix y.  Mainly for derived classes where y is
     *  transformed.
     */
    def getY: MatrixD = y

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the feature/variable names.
     */
    def getFname: Array [String] = fname

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of terms/parameters in the model, e.g., b_0 + b_1 x_1 + b_2 x_2 
     *  has three terms.
     */
    def numTerms: Int = getX.dim2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train a predictive model y_ = f(x_) + e where x_ is the data/input
     *  matrix and y_ is the response/output matrix.  These arguments default
     *  to the full dataset x and y, but may be restricted to a training
     *  dataset.  Training involves estimating the model parameters b.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output matrix (defaults to full y)
     */
    def train (x_ : MatrixD = x, y_ : MatrixD = y): Unit

    def train (x_ : MatrixD, y_ : VectorD): Unit =                           // first column only
        train (x_, MatrixD (y_).transpose)
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The train2 method should work like the train method, but should also
     *  optimize hyper-parameters (e.g., shrinkage or learning rate).
     *  Only implementing classes needing this capability should override this method.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output matrix (defaults to full y)
     */
    def train2 (x_ : MatrixD = x, y_ : MatrixD = y): Unit =
        throw new UnsupportedOperationException ("train2: not supported - no hyper-parameters to optimize")
    end train2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the predictive model y_ = f(x_) + e and return its predictions and QoF matrix.
     *  Each variable predictions and QoF values are returned in columns of respective matrices.
     *  Testing may be in-sample (on the full dataset) or out-of-sample (on the testing set)
     *  as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output matrix (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : MatrixD = y): (MatrixD, MatrixD)

    def test (x_ : MatrixD, y_ : VectorD): (VectorD, VectorD) =               // first column only
        val (yp, qof) = test (x_, MatrixD (y_).transpose)
        (yp(?, 0), qof(?, 0))
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the predictive model y_ = f(x_) + e and report its QoF
     *  and plot its predictions.
     *  FIX - currently must override if y is transformed, @see `TranRegression`
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output matrix (defaults to full y)
     *  @param xx  the testing/full data/input matrix (defaults to full x)
     *  @param yy  the testing/full response/output matrix (defaults to full y)
     */
    def trainNtest (x_ : MatrixD = x, y_ : MatrixD = y)
                   (xx: MatrixD = x, yy: MatrixD = y): (MatrixD, MatrixD) =
        train (x_, y_)
        val (yp, qof) = test (xx, yy)
        println (report (qof))
        if DO_PLOT then
            val (ryy, ryp) = orderByYY (yy, yp)                               // order by yy
            for k <- ryy.indices2 do
                new Plot (null, ryy(?, k), ryp(?, k), s"$modelName: y$k actual, predicted")
            end for
        end if
        (yp, qof)
    end trainNtest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the predictive model y_ = f(x_) + e and report its QoF
     *  and plot its predictions.  This version does auto-tuning.
     *  FIX - currently must override if y is transformed, @see `TranRegression`
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output matrix (defaults to full y)
     *  @param xx  the testing/full data/input matrix (defaults to full x)
     *  @param yy  the testing/full response/output matrix (defaults to full y)
     */
    def trainNtest2 (x_ : MatrixD = x, y_ : MatrixD = y)
                    (xx: MatrixD = x, yy: MatrixD = y): (MatrixD, MatrixD) =
        train2 (x_, y_)
        val (yp, qof) = test (xx, yy)
        println (report (qof))
        if DO_PLOT then
            val (ryy, ryp) = orderByYY (yy, yp)                               // order by yy
            for k <- ryy.indices2 do
                new Plot (null, ryy(?, k), ryp(?, k), s"$modelName: y$k actual, predicted")
            end for
        end if
        (yp, qof)
    end trainNtest2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a basic report on a trained and tested multi-variate model.
     *  @param ftMat  the matrix of qof values produced by the `Fit` trait
     */
    override def report (ftMat: MatrixD): String =
        s"""
REPORT
    ----------------------------------------------------------------------------
    modelName  mn  = $modelName
    ----------------------------------------------------------------------------
    hparameter hp  = $hparameter
    ----------------------------------------------------------------------------
    features   fn  = ${stringOf (getFname)}
    ----------------------------------------------------------------------------
    parameter  bb  = ${stringOf (parameters)}
    ----------------------------------------------------------------------------
    fitMap     qof = ${FitM.showFitMap (ftMat, QoF.values.map (_.toString))}
    ----------------------------------------------------------------------------
        """
    end report

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Order vectors y_ and yp_ accroding to the ascending order of y_.
     *  @param y_   the vector to order by (e.g., true response values)
     *  @param yp_  the vector to be order by y_ (e.g., predicted response values)
     */
    def orderByY (y_ : VectorD, yp_ : VectorD): (VectorD, VectorD) =
        val rank = y_.iqsort                                                  // rank order for vector y_
        (y_.reorder (rank), yp_.reorder (rank))                               // (y_ in ascending order, yp ordered by y_)
    end orderByY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = b dot z,
     *  e.g., (b_0, b_1, b_2) dot (1, z_1, z_2).
     *  Must override when using transformations, e.g., `ExpRegression`.
     *  @param z  the new vector to predict
     */
    def predict (z: VectorD): VectorD                                         // = b dot z

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of vector y = f(x_, b), e.g., x_ * b for `Regression`.
     *  @param x_  the matrix to use for making predictions, one for each row
     */
    def predict (x_ : MatrixD): MatrixD =
        MatrixD (for i <- x_.indices yield predict (x_(i)))
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the hyper-parameters.
     */
    def hparameter: HyperParameter = hparam

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the matrix (or `NetParam`) of parameter/coefficient values.
     */
    def parameter: MatrixD =
        if b.isInstanceOf [MatrixD] then b.asInstanceOf [MatrixD]
        else b.asInstanceOf [NetParam].toMatrixD
    end parameter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the array of network parameters (weight matrix, bias vector) bb.
     */
    def parameters: NetParams = bb

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the matrix of residuals/errors.
     */
    def residual: MatrixD = e

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  Override for models that support feature section.
     *  @param x_cols  the columns that the new model is restricted to
     */
    def buildModel (x_cols: MatrixD): PredictorMV

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `BestStep` is used to record the best improvement step found so far.
     *  Only considers the first response variable y(0) => qof(?, 0).
     *  @param col  the column/variable to ADD/REMOVE for this step
     *  @param qof  the Quality of Fit (QoF) for this step
     *  @param mod  the model including selected features/variables for this step
     */
    case class BestStep (col: Int = -1, qof: VectorD = null, mod: PredictorMV = null)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the rSq-based QoF results for the l-th iteration.
     *  @param rSq    the matrix contain information about r-Sq-based QoF measures
     *  @param l      the l-th iteration
     *  @param cross  indicator of whether cross-validation are to be included
     *  @param fit_l  the fit vector for the l-th iteration
     *  @param mod_l  the predictive model for the l-th iteration
     */
    private def updateQoF (rSq: MatrixD, l: Int, cross: Boolean, best: BestStep): Unit =
        rSq(l) =
            if cross then
                Fit.qofVector (best.qof, best.mod.crossValidate ())           // results for model mod_l, with cross-validation
            else
                Fit.qofVector (best.qof, null)                                // results for model mod_l, no cross-validation
            end if
    end updateQoF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform feature selection to find the most predictive variables to have
     *  in the model, returning the variables added and the new Quality of Fit (QoF)
     *  measures for all steps.
     *  @see `Fit` for index of QoF measures.
     *  @param tech   the feature selection technique to apply
     *  @param idx_q  index of Quality of Fit (QoF) to use for comparing quality
     *  @param cross  whether to include the cross-validation QoF measure
     */
    def selectFeatures (tech: SelectionTech, idx_q: Int = QoF.rSqBar.ordinal, cross: Boolean = true):
                       (LinkedHashSet [Int], MatrixD) =
        tech match
        case SelectionTech.Forward  => forwardSelAll (idx_q, cross)
        case SelectionTech.Backward => backwardElimAll (idx_q, 1, cross)
        case SelectionTech.Stepwise => stepRegressionAll (idx_q, cross)
        end match
    end selectFeatures

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform forward selection to find the most predictive variable to add the
     *  existing model, returning the variable to add and the new model.
     *  May be called repeatedly.
     *  @see `Fit` for index of QoF measures.
     *  @param cols   the columns of matrix x currently included in the existing model
     *  @param idx_q  index of Quality of Fit (QoF) to use for comparing quality
     */
    def forwardSel (cols: LinkedHashSet [Int], idx_q: Int = QoF.rSqBar.ordinal): BestStep =
        var best  = BestStep ()                                               // best step so far
        var bestq = -MAX_VALUE                                                // best score so far

        for j <- x.indices2 if ! (cols contains j) do
            val cols_j = cols union LinkedHashSet (j)                         // try adding variable/column x_j
            val x_cols = x(?, cols_j)                                         // x projected onto cols_j columns
            val mod_j  = buildModel (x_cols)                                  // regress with x_j added
            mod_j.train ()                                                    // train model
            val cand = BestStep (j, mod_j.test ()._2(?, 0), mod_j)            // candidate step
            if cand.qof(idx_q) > bestq then { best = cand; bestq = cand.qof(idx_q) }
        end for

        if best.col == -1 then
            flaw ("forwardSel", "could not find a variable x_j to add: best.col = -1")
        end if
        best
    end forwardSel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform forward selection to find the most predictive variables to have
     *  in the model, returning the variables added and the new Quality of Fit (QoF)
     *  measures for all steps.
     *  @see `Fit` for index of QoF measures.
     *  @param idx_q  index of Quality of Fit (QoF) to use for comparing quality
     *  @param cross  whether to include the cross-validation QoF measure
     */
    def forwardSelAll (idx_q: Int = QoF.rSqBar.ordinal, cross: Boolean = true):
            (LinkedHashSet [Int], MatrixD) =
        val rSq  = new MatrixD (x.dim2 - 1, 3)                                // QoF: R^2, R^2 Bar, R^2 cv
        val cols = LinkedHashSet (0)                                          // start with x_0 in model

        banner (s"forwardSelAll: (l = 0) INITIAL variable (0, ${fname(0)}) => cols = $cols")

        breakable {
            for l <- 1 until x.dim2 do
                val best = forwardSel (cols, idx_q)                           // add most predictive variable
                if best.col == -1 then break ()                               // could not find variable to add
                cols += best.col                                              // add variable x_j
                updateQoF (rSq, l-1, cross, best)                             // update QoF results
                val (jj, jj_qof) = (best.col, best.qof(idx_q))
                banner (s"forwardSelAll: (l = $l) ADD variable ($jj, ${fname(jj)}) => cols = $cols @ $jj_qof")
            end for
        } // breakable

        (cols, rSq(0 until cols.size-1))
    end forwardSelAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform backward elimination to find the least predictive variable to remove
     *  from the existing model, returning the variable to eliminate, the new parameter
     *  matrix and the new Quality of Fit (QoF).  May be called repeatedly.
     *  @see `Fit` for index of QoF measures.
     *  @param cols   the columns of matrix x currently included in the existing model
     *  @param idx_q  index of Quality of Fit (QoF) to use for comparing quality
     *  @param first  first variable to consider for elimination
     *                      (default (1) assume intercept x_0 will be in any model)
     */
    def backwardElim (cols: LinkedHashSet [Int], idx_q: Int = QoF.rSqBar.ordinal, first: Int = 1): BestStep =
        var best  = BestStep ()                                               // best step so far
        var bestq = -MAX_VALUE                                                // best score so far

        for j <- first until x.dim2 if cols contains j do
            val cols_j = cols diff LinkedHashSet (j)                          // try removing variable/column x_j
            val x_cols = x(?, cols_j)                                         // x projected onto cols_j columns
            val mod_j  = buildModel (x_cols)                                  // regress with x_j added
            mod_j.train ()                                                    // train model
            val cand = BestStep (j, mod_j.test ()._2(?, 0), mod_j)            // candidate step
            if cand.qof(idx_q) > bestq then { best = cand; bestq = cand.qof(idx_q) }
        end for

        if best.col == -1 then
            flaw ("backwardElim", "could not find a variable x_j to eliminate: best.col = -1")
        end if
        best
    end backwardElim

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run the full model before variable elimination as a starting point for
     *  backward elimination.
     */
    private def fullModel: BestStep =
        val mod_a = buildModel (x)                                            // regress with all variables x_j
        mod_a.train ()                                                        // train model
        BestStep (-1, mod_a.test ()._2(?, 0), mod_a)                          // results for full model
    end fullModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform backward elimination to find the least predictive variables to remove
     *  from the full model, returning the variables left and the new Quality of Fit (QoF)
     *  measures for all steps.
     *  @see `Fit` for index of QoF measures.
     *  @param idx_q  index of Quality of Fit (QoF) to use for comparing quality
     *  @param first  first variable to consider for elimination
     *  @param cross  whether to include the cross-validation QoF measure
     */
    def backwardElimAll (idx_q: Int = QoF.rSqBar.ordinal, first: Int = 1, cross: Boolean = true):
            (LinkedHashSet [Int], MatrixD) =
        val rSq  = new MatrixD (x.dim2 - 1, 3)                                // R^2, R^2 Bar, R^2 cv
        val cols = LinkedHashSet.range (0, x.dim2)                            // start with all x_j in model

        val best0 = fullModel
        updateQoF (rSq, 0, cross, best0)                                      // update QoF results for full model
        val jj_qof = best0.qof(idx_q)
        banner (s"backwardElimAll: (l = 0) INITIAL variables (all) => cols = $cols @ $jj_qof")

        breakable {
            for l <- 1 until x.dim2 - 1 do                                    // l indicates number of variables eliminated
                val best = backwardElim (cols, idx_q, first)                  // remove least predictive variable
                if best.col == -1 then break ()                               // could not find variable to remove
                cols -= best.col                                              // remove variable x_j
                updateQoF (rSq, l, cross, best)                               // update QoF results
                val (jj, jj_qof) = (best.col, best.qof(idx_q))
                banner (s"backwardElimAll: (l = $l) REMOVE variable ($jj, ${fname(jj)}) => cols = $cols @ $jj_qof")
            end for
        } // breakable

        (cols, rSq.reverse)                                                   // reverse the order results
    end backwardElimAll 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform stepwise regression to find the most predictive variables to have
     *  in the model, returning the variables left and the new Quality of Fit (QoF)
     *  measures for all steps.  At each step it calls 'forwardSel' and 'backwardElim'
     *  and takes the best of the two actions.  Stops when neither action yields improvement.
     *  @see `Fit` for index of QoF measures.
     *  @param idx_q  index of Quality of Fit (QoF) to use for comparing quality
     *  @param cross  whether to include the cross-validation QoF measure
     */
    def stepRegressionAll (idx_q: Int = QoF.rSqBar.ordinal, cross: Boolean = true):
            (LinkedHashSet [Int], MatrixD) =
        val SWAP   = false                                                    // whether to include swapping
        val rSq    = new MatrixD (x.dim2 - 1, 3)                              // QoF: R^2, R^2 Bar, R^2 cv
        val cols   = LinkedHashSet (0)                                        // start with x_0 in model
        var last_q = -MAX_VALUE                                               // current best QoF
        val vars   = ArrayBuffer [Int]()

        banner (s"stepRegressionAll: (l = 0) INITIAL variable (0, ${fname(0)}) => cols = $cols")

        breakable {
            for l <- 1 until x.dim2 - 1 do
                val bestf = forwardSel (cols, idx_q)                          // add most predictive variable OR
                val bestb = backwardElim (cols, idx_q, 1)                     // remove least predictive variable
                debug ("stepRegressionAll", s"bestf = $bestf, bestb = $bestb")

                if (bestb.col == -1 || bestf.qof(idx_q) >= bestb.qof(idx_q)) &&   // forward as good as backward
                   (bestf.col != -1 && bestf.qof(idx_q) > last_q) then            // a better model has been found
                    vars  += bestf.col
                    cols  += bestf.col                                            // ADD variable bestf.col
                    last_q = bestf.qof(idx_q)
                    updateQoF (rSq, l, cross, bestf)                              // update QoF results
                    println (s"\nstepRegressionAll: (l = $l) ADD variable $bestf")
                    val (jj, jj_qof) = (bestf.col, last_q)
                    banner (s"stepRegressionAll: (l = $l) ADD variable ($jj, ${fname(jj)}) => cols = $cols @ $jj_qof")

                else if bestb.col != -1 && bestb.qof(idx_q) > last_q then         // a better model has been found
                    vars  += bestb.col
                    cols  -= bestb.col                                            // REMOVE variable bestb.col 
                    last_q = bestb.qof(idx_q)
                    updateQoF (rSq, l, cross, bestb)                              // update QoF results
                    println (s"\nstepRegressionAll: (l = $l) REMOVE variable $bestb")
                    val (jj, jj_qof) = (bestb.col, last_q)
                    banner (s"stepRegressionAll: (l = $l) REMOVE variable ($jj, ${fname(jj)}) => cols = $cols @ $jj_qof")

                else
                    if ! SWAP then break ()
                    val (out, in) = (bestb.col, bestf.col)
                    val bestfb = swapVars (cols, out, in)
                    if out != -1 && in != -1 && bestfb.qof(idx_q) > last_q then    // a better model has been found
                        vars  += bestb.col
                        vars  += bestf.col
                        cols  -= bestb.col                                    // REMOVE variable bestb.col (swap out)
                        cols  += bestf.col                                    // ADD variable bestf.col (swap in)
                        last_q = bestfb.qof(idx_q)
                        updateQoF (rSq, l, cross, bestfb)                     // update QoF results
                        println (s"\nstepRegressionAll: (l = $l) SWAP variable $bestb with $bestf")
                    else
                        break ()                                              // can't find a better model -> quit
                    end if
                end if
            end for
        } // breakable

        println (s"stepRegressionAll: selected features = $cols")
        println (s"stepRegressionAll: selected features = ${cols.map (fname (_))}")
        println (s"stepRegressionAll: features in/out   = $vars")

        (cols, rSq(1 until cols.size))
    end stepRegressionAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Swap out variable with in variable.
     *  @param cols  the columns of matrix x currently included in the existing model
     *  @param out   the variable to swap out
     *  @param in    the variable to swap in
     */
    private def swapVars (cols: LinkedHashSet [Int], out: Int, in: Int): BestStep =
        val cols_  = cols diff LinkedHashSet (out) union LinkedHashSet (in)   // swap out var with in var
        val x_cols = x(?, cols_)                                              // x projected onto cols_j columns
        val mod_j  = buildModel (x_cols)                                      // regress with x_out removed and x_in added
        mod_j.train ()                                                        // train model
        BestStep (in, mod_j.test ()._2(?, 0), mod_j)                          // candidate step
    end swapVars

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Variance Inflation Factor (VIF) for each variable to test
     *  for multi-collinearity by regressing x_j against the rest of the variables.
     *  A VIF over 50 indicates that over 98% of the variance of x_j can be predicted
     *  from the other variables, so x_j may be a candidate for removal from the model.
     *  Note:  override this method to use a superior regression technique.
     *  @param skip  the number of columns of x at the beginning to skip in computing VIF
     */
    def vif (skip: Int = 1): VectorD =
        val vifV = new VectorD (x.dim2 - skip)                                // VIF vector for x columns except skip columns
        for j <- skip until x.dim2 do
            val x_j   = x(?, j)                                               // column j vector
            val x_noj = x.not (?, j)                                          // all columns except j matrix                   
            val mod_j = new Regression (x_noj, x_j)                           // regress with x_j removed
            mod_j.train ()                                                    // train model
            val rSq_j = (mod_j.test ()._2)(QoF.rSq.ordinal)                   // R^2 for predicting x_j
            if rSq_j.isNaN then Fac_LU.diagnoseMat (x_noj)                    // check for problems with matrix
//          debug ("vif", s"for variable x_$j, rSq_$j = $rSq_j")
            vifV(j-1) =  1.0 / (1.0 - rSq_j)                                  // store vif for x_1 in vifV(0)
        end for
        vifV
    end vif

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the indices for the test-set.
     *  @param rando  whether to select indices randomly or in blocks
     *  @param sz     the size of test-set
     */
    protected def testIndices (rando: Boolean, sz: Int): IndexedSeq [Int] =
        (if rando then permGen.igen (0 until sz)                              // premuted indices
         else VectorI.range (0, sz)).toMuIndexedSeq                           // ordered indices
    end testIndices

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Use validation to compute test Quality of Fit (QoF) measures by dividing
     *  the full dataset into a TESTING set and a TRAINING set.
     *  The test set is defined by idx and the rest of the data is the training set.
     *  @param rando  flag indicating whether to use randomized or simple validation
     *  @param ratio  the ratio of the TESTING set to the full dataset (most common 70-30, 80-20)
     *  @param idx    the prescribed TESTING set indices
     */
    def validate (rando: Boolean = true, ratio: Double = 0.2)
                 (idx : IndexedSeq [Int] = testIndices (rando, (ratio * y.dim).toInt)): MatrixD =
        val sz  = (ratio * y.dim).toInt                                       // size of testing set

        debug ("validate", s"test set size = $sz")
        val (x_e, x_) = if x == null then (null, null) else x.split (idx)     // test, training data/input matrices
        val (y_e, y_) = y.split (idx)                                         // test, training response/output matrices

        train (x_, y_)                                                        // train model on the training set
        debug ("validate", s"parameters b = $b")
        val qof = test (x_e, y_e)._2                                          // test on test-set and get QoF measures
        if qof(QoF.sst.ordinal)(0) <= 0.0 then                                // requires variation in test-set
            flaw ("validate", "chosen testing set has no variability")
        end if
        qof
    end validate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Use k-fold cross-validation to compute test Quality of Fit (QoF) measures
     *  by iteratively dividing the full dataset into a TESTING set and a TRAINING set.
     *  Each test set is defined by idx and the rest of the data is the training set.
     *  @see showQofStatTable in `Fit` object for printing the returned stats.
     *  @param k      the number of cross-validation iterations/folds (defaults to 5x).
     *  @param rando  flag indicating whether to use randomized or simple cross-validation
     */
    def crossValidate (k: Int = 5, rando: Boolean = true): Array [Statistic] =
        if k < MIN_FOLDS then flaw ("crossValidate", s"k = $k must be at least $MIN_FOLDS")
        val stats   = Fit.qofStatTable                                        // create table for QoF measures
        val fullIdx = if rando then permGen.igen                              // premuted indices
                      else VectorI.range (0, y.dim)                           // ordered indices
        val sz      = y.dim / k                                               // size of each fold
        val ratio   = 1.0 / k                                                 // fraction of dataset used for testing

        for fold <- 0 until k do
            val idx = fullIdx (fold * sz until (fold+1) * sz).toMuIndexedSeq  // instance indices for this fold
            debug ("crossValidate", s"fold $fold: test set size = $sz")
            val qof = validate (rando, ratio)(idx)
            debug ("crossValidate", s"fold $fold: qof = $qof")
            if qof(QoF.sst.ordinal)(0) > 0.0 then                             // requires variation in test-set
                for q <- qof.indices do stats(q).tally (qof(q)(0))            // tally these QoF measures
            end if
        end for

        stats
    end crossValidate

end PredictorMV


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PredictorMV` companion object provides a method for testing predictive
 *  models.
 */
object PredictorMV:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test (in-sample) by training and testing on the FULL dataset.
     *  Test (out-of-sample) by training on the TRAINING set and testing on the TESTING set.
     *  @param mod    the model to be used
     *  @param ext    the model subtype extension (e.g., indicating the transformation function used)
     *  @param check  whether to check the assertion that the in-sample and out-of-sample results
     *                are in rough agreement (e.g., at 20%)
     */
    def test (mod: PredictorMV, ext: String = "", check: Boolean = true): Unit =
        val iq = QoF.rSq.ordinal
        banner (s"Test ${mod.modelName} $ext")
        val (yp, qof) = mod.trainNtest ()()                                   // train and test the model on full dataset (in-sample)

        println ("Validate: Out-of-Sample Testing")
        val qof2 = mod.validate ()()                                          // train on training set, test on testing set
        if check then assert (rel_diff (qof(iq)(0), qof2(iq)(0)) < 0.2)       // check agreement of in-sample and out-of-sample results
        println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))
    end test

end PredictorMV


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `predictorMVTest` main function is used to test the `PredictorMV` trait
 *  and its derived classes using the `Example_Concrete` dataset containing
 *  data matrices x, ox and response matrix y.
 *  > runMain scalation.modeling.predictorMV
 */
@main def predictorMVTest (): Unit =

    import Example_Concrete._

    println ("TBD")

end predictorMVTest

