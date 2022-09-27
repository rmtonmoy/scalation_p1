
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Feb 16 16:14:34 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Null Model Classifier
 */

package scalation
package modeling
package classifying

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NullModel` class implements a Null Model Classifier, which is a simple
 *  classifier for discrete input data.  The classifier is trained just using a
 *  classification vector y.  Picks the most frequent class.
 *  Each data instance is classified into one of k classes numbered 0, ..., k-1.
 *  Note: the train method in the super class suffices.
 *  @param y       the classification vector, where y(i) = class for instance i
 *  @param k       the number of classes
 *  @param cname_  the names for all classes
 */
class NullModel (y: VectorI, k: Int = 2, cname_ : Array [String] = Array ("No", "Yes"))
      extends Classifier (null, y, null, k, cname_, null)               // no x matrix, no hyper-parameters
         with FitC (y, k):

    private val debug = debugf ("NullModel", true)                      // debug function

    modelName = "NullModel"                                             // name of the model

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the predictive model y_ = f(x_) + e and return its predictions and QoF vector.
     *  Testing may be in-sample (on the full dataset) or out-of-sample (on the testing set)
     *  as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = null, y_ : VectorI = y): (VectorI, VectorD) =
        val c   = predictI (null)
        val yp  = VectorI.fill (y_.dim)(c)                              // prediction does not change
        val qof = diagnose (y_.toDouble, yp.toDouble)
        debug ("test", s" yp = $yp \n qof = $qof")
        (yp, qof)
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the integer value of y = f(z) by selecting the most probabble class.
     *  @param z  the new vector to predict
     */
    def predictI (z: VectorI): Int = py.argmax ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor 'x_0', 'x_1',
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = null, fname_ : Array [String] = null, b_ : VectorD = py,
                          vifs: VectorD = null): String =
        super.summary (x_, fname_, b_, vifs)                            // summary from `Fit`
    end summary

end NullModel


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nullModelTest` object is used to test the `NullModel` class.
 *  Classify whether to play tennis(1) or not (0).
 *  > runMain scalation.modeling.classifying.nullModelTest
 */
@main def nullModelTest (): Unit =

    import Example_PlayTennis._

    banner ("Tennis Example")
    val y = xy(?, 4).toInt                                              // 4th column
    println (s"y = $y")
    println ("-" * 60)

    val nm = new NullModel (y, k, cname)                                // create a classifier
    nm.trainNtest ()()                                                  // train and test the classifier
    println (nm.summary ())                                             // summary statistics

    val z = VectorI (1)                                                 // new data vector to classify
    println (s"predictI ($z) = ${nm.predictI (z)}")

    banner ("Validation")
    println ("nm test accu = " + nm.validate ()())                      // out-of-sample testing

    banner ("Cross-validation")
    FitM.showQofStatTable (nm.crossValidate ())                         // 5-fold cross-validation (14 instances typically too few)

end nullModelTest

