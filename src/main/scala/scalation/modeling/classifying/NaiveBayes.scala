
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng
 *  @version 2.0
 *  @date    Sat Sep  8 13:53:16 EDT 2012
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Integer-Based Naive Bayes Classifier
 *
 *  @see eric.univ-lyon2.fr/~ricco/tanagra/fichiers/en_Tanagra_Naive_Bayes_Classifier_Explained.pdf
 */

package scalation
package modeling
package classifying

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

import Probability.{freq, plog}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NaiveBayes` class implements an Integer-Based Naive Bayes Classifier,
 *  which is a commonly used such classifier for discrete input data.  The
 *  classifier is trained using a data matrix x and a classification vector y.
 *  Each data vector in the matrix is classified into one of k classes numbered
 *  0, ..., k-1.  Prior probabilities are calculated based on the population of
 *  each class in the training-set.  Relative posterior probabilities are computed
 *  by multiplying these by values computed using conditional probabilities.
 *  stored in Conditional Probability Tables (CPTs).
 *------------------------------------------------------------------------------
 *  The classifier is naive, because it assumes variable/feature independence and
 *  therefore simply multiplies the conditional probabilities.
 *  @param x       the input/data matrix
 *  @param y       the class vector, where y(i) = class for row i of matrix x
 *  @param fname_  the names of the features/variables
 *  @param k       the number of classes
 *  @param cname_  the names of the classes
 *  @param vc      the value count (number of distinct values) for each feature
 *  @param hparam  the hyper-parameters
 */
class NaiveBayes (x: MatrixD, y: VectorI, fname_ : Array [String] = null,
                  k: Int = 2, cname_ : Array [String] = Array ("No", "Yes"),
                  protected var vc: Array [Int] = null, hparam: HyperParameter = NaiveBayes.hp)
      extends Classifier (x, y, fname_, k, cname_, hparam)
         with FitC (y, k):

    private val debug = debugf ("NaiveBayes", true)                      // debug function
    private val flaw  = flawf ("NaiveBayes")                             // flaw function

    if cname.length != k then flaw ("init", "# class names != # classes")

    modelName = "NaiveBayes"                                             // name of the model

    if vc == null then
        shift2zero (x); vc = vc_fromData (x)                             // set value counts from data
    end if

    private val me     = hparam("me").toDouble                           // m-estimates (me == 0 => regular MLE estimates)
    private val (m, n) = (x.dim, x.dim2)                                 // number of (instances, variables)
    private val md     = m.toDouble                                      // m as a double (real number)
    private val cr     = 0 until k                                       // range of class values
    private val nu_Xy  = Array.ofDim [MatrixI] (n)                       // joint frequency tables (JFTs) for feature xj
    private val p_Xy   = Array.ofDim [MatrixD] (n)                       // Conditional Probability Tables (CPTs) for each feature xj

    for j <- x.indices2 do p_Xy(j) = new MatrixD (vc(j), k)              // CPT for each xj is distinct-values by class

    debug ("init", "distinct value count vc = " + stringOf (vc))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train a classification model y_ = f(x_) + e where x_ is the data/input
     *  matrix and y_ is the response/output vector.  These arguments default
     *  to the full dataset x and y, but may be restricted to a training
     *  dataset.  Training involves estimating the model parameters or pmf.
     *  Train the classifier by computing the probabilities for y, and the
     *  conditional probabilities for each x_j.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    override def train (x_ : MatrixD = x, y_ : VectorI = y): Unit =
        val nuy = y_.freq (k)                                            // class frequencies
        for j <- x_.indices2 do
            nu_Xy(j) = freq (x_(?, j).toInt, vc(j), y_, k)               // Joint Frequency Tables (JFTs)
            debug ("train", s"nu_Xy($j) = ${nu_Xy(j)}")
        end for

        py = nuy.toDouble / md                                           // probability for each class
        for j <- x.indices2 do                                           // for each feature xj
            val me_vc = me / vc(j)
            for c <- cr; v <- 0 until vc(j) do                           // for each value and class
                p_Xy(j)(v, c) = (nu_Xy(j)(v, c) + me_vc) / (nuy(c) + me)
            end for
        end for
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the predictive model y_ = f(x_) + e and return its predictions and QoF vector.
     *  Testing may be in-sample (on the full dataset) or out-of-sample (on the testing set)
     *  as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorI = y): (VectorI, VectorD) =
        val yp  = predictI (x_)                                          // predicted classes
        val qof = diagnose (y_.toDouble, yp.toDouble)                    // diagnose from actual and predicted
        debug ("test", s" yp = $yp \n qof = $qof")
        (yp, qof)
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the integer value of y = f(z) by computing the product of the class
     *  probabilities py and all the conditional probabilities P(X_j = z_j | y = c)
     *  and returning the class with the highest probability.
     *  @param z  the new vector to predict
     */
    override def predictI (z: VectorI): Int =
        val prob = py.copy                                               // start with class (prior) probabilities
        for j <- z.indices do                                            // P(X_j = z_j | y = c)
            val cpt = p_Xy(j)                                            // get j-th CPT
            prob   *= cpt (z(j))                                         // multiply in its v = z(j) row
        end for
        debug ("predictI", s"prob = $prob")
        prob.argmax ()                                                   // return class with highest probability
    end predictI

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a discrete data vector z, classify it returning the class number
     *  (0, ..., k-1) with the highest relative posterior probability.
     *  Return the best class, its name and its relative log-probability.
     *  This method adds "positive log probabilities" to avoids underflow.
     *  @param z  the data vector to classify
     */
    def lclassify (z: VectorI): (Int, String, Double) =
        val prob = plog (py)                                             // start with class (prior) plogs
        for j <- z.indices do                                            // P(X_j = z_j | y = c)
            val cpt = p_Xy(j)                                            // get j-th CPT
            prob   += plog (cpt (z(j)))                                  // add in its plog for v = z(j) row
        end for
        debug ("lclassify", s"prob = $prob")
        val best = prob.argmax ()                                        // class with highest relative posterior plog
        (best, cname(best), prob(best))                                  // return the best class and its name
    end lclassify

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the conditional probability tables by iterating over the features/variables.
     */
    def printCPTs (): Unit =
        for j <- x.indices2 do println (s"CPT for x$j: P(X_j = z_j | y = c) = ${p_Xy(j)}")
    end printCPTs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor x_0, x_1,
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = null, fname_ : Array [String] = null, b_ : VectorD = py,
                          vifs: VectorD = null): String =
        super.summary (x_, fname_, b_, vifs)                             // summary from `Fit`
    end summary

end NaiveBayes


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** `NaiveBayes` is the companion object for the `NaiveBayes` class.
 */
object NaiveBayes:

    val hp = new HyperParameter ()
    hp += ("me", 1, 1)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NaiveBayes` object, passing x and y together in one matrix.
     *  @param xy      the combined data-response matrix
     *  @param fname   the names of the features/variables
     *  @param k       the number of classes
     *  @param cname   the names of the classes
     *  @param vc      the value count (number of distinct values) for each feature
     *  @param hparam  the hyper-parameters
     */
    def apply (xy: MatrixI, fname: Array [String] = null, k: Int = 2,
               cname: Array [String] = Array ("No", "Yes"), vc: Array [Int] = null,
               hparam: HyperParameter = NaiveBayes.hp)
              (col: Int = xy.dim2 - 1): NaiveBayes =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)              // data matrix, response vector
        new NaiveBayes (x, y, fname, k, cname, vc, hparam)
    end apply

end NaiveBayes


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `naiveBayesTest` object is used to test the `NaiveBayes` class.
 *  > runMain scalation.modeling.classifying.naiveBayesTest
 */
@main def naiveBayesTest (): Unit =

    import Example_PlayTennis._

    banner ("Play Tennis Example")
    println (s"xy = $xy")                                           // combined data matrix [ x | y ]

    val nb = NaiveBayes (xy, fname)()                               // create a classifier
    nb.trainNtest ()()                                              // train and test the classifier
    nb.printCPTs ()                                                 // print the conditional probability tables
    println (nb.summary ())                                         // summary statistics

    val z = VectorI (2, 2, 1, 1)                                    // new data vector to classify
    banner (s"Classify $z")
    println (s"Use nb to classify ($z) = ${nb.classify (z)}")

    banner ("Validation")
    println ("nb test accu = " + nb.validate ()())                  // out-of-sample testing

    banner ("Cross-validation")
    FitM.showQofStatTable (nb.crossValidate ())                     // 5-fold cross-validation (14 instances typically too few)

end naiveBayesTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `naiveBayesTest2` object is used to test the `NaiveBayes` class.
 *  Classify whether a car is more likely to be stolen (1) or not (1).
 *  @see www.inf.u-szeged.hu/~ormandi/ai2/06-naiveBayes-example.pdf
 *  > runMain scalation.modeling.classiying.naiveBayesTest2
 */
@main def naiveBayesTest2 (): Unit =

    // x0: Color:   Red (1), Yellow (0)
    // x1: Type:    SUV (1), Sports (0)
    // x2: Origin:  Domestic (1), Imported (0)
    // x3: Mpg:     High (1), Low (0)
    // features:             x0 x1 x2 x3
    val x = MatrixI ((10, 4), 1, 0, 1, 1,                           // data matrix
                              1, 0, 1, 0,
                              1, 0, 1, 1,
                              0, 0, 1, 1,
                              0, 0, 0, 1,
                              0, 1, 0, 0,
                              0, 1, 0, 0,
                              0, 1, 1, 1,
                              1, 1, 0, 0,
                              1, 0, 0, 0)

    val y  = VectorI (1, 0, 1, 0, 1, 0, 1, 0, 0, 1)                 // classification vector: 0(No), 1(Yes))
    val fn = Array ("Color", "Type", "Origin", "Mpg")               // feature/variable names
    val cn = Array ("No", "Yes")                                    // class names

    banner ("Stolen Car Example")
    println (s"x = $x")

    val nb = new NaiveBayes (x, y, fn, 2, cn)                       // create the classifier
    nb.trainNtest ()()                                              // train and test the classifier
    nb.printCPTs ()                                                 // print the conditional probability tables
    println (nb.summary ())                                         // summary statistics

    val z1 = VectorI (1, 0, 1, 1)                                   // existing data vector to classify
    val z2 = VectorI (1, 1, 1, 0)                                   // new data vector to classify
    println (s"Use nb to classify ($z1) = ${nb.classify (z1)}")
    println (s"Use nb to classify ($z2) = ${nb.classify (z2)}")

end naiveBayesTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `naiveBayesTest3` object is used to test the `NaiveBayes` class.
 *  Given whether a person is Fast and/or Strong, classify them as making C = 1
 *  or not making C = 0 the football team.
 *  > runMain scalation.modeling.classiying.naiveBayesTest3
 */
@main def naiveBayesTest3 (): Unit =

    // x0: Fast
    // x1: Strong
    // y:  Classification (No/0, Yes/1)
    // features:              x0 x1  y
    val xy = MatrixI ((10, 3), 1, 1, 1,
                               1, 1, 1,
                               1, 0, 1,
                               1, 0, 1,
                               1, 0, 0,
                               0, 1, 0,
                               0, 1, 0,
                               0, 1, 1,
                               0, 0, 0,
                               0, 0, 0)

    val fn = Array ("Fast", "Strong")                               // feature names
    val cn = Array ("No", "Yes")                                    // class names

    banner ("Football Team  Example")
    println (s"xy = $xy")

    val nb = NaiveBayes  (xy, fn, 2, cn)()                          // create the classifier
    nb.trainNtest ()()                                              // train and test the classifier
    nb.printCPTs ()                                                 // print the conditional probability tables
    println (nb.summary ())                                         // summary statistics

    val z = VectorI (1, 0)                                          // new data vector to classify
    println (s"Use nb  to classify ($z) = ${nb.classify (z)}")

end naiveBayesTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `naiveBayesTest4` object is used to test the `NaiveBayes` class.
 *  @see archive.ics.uci.edu/ml/datasets/Lenses
 *  @see docs.roguewave.com/imsl/java/7.3/manual/api/com/imsl/datamining/NaiveBayesClassifierEx2.html
 *  > runMain scalation.modeling.classiying.naiveBayesTest4
 */
@main def naiveBayesTest4 (): Unit =

    // y:  Classification (1): hard contact lenses, (2) soft contact lenses, (3) no contact lenses
    // x0. Age of the patient: (1) young, (2) pre-presbyopic, (3) presbyopic
    // x1. Spectacle prescription:  (1) myope, (2) hypermetrope
    // x2. Astigmatic:     (1) no, (2) yes
    // x3. Tear production rate:  (1) reduced, (2) normal
    // features:              x0  x1  x2  x3   y
    var xy = MatrixI ((24, 5), 1,  1,  1,  1,  3,           // 1
                               1,  1,  1,  2,  2,           // 2
                               1,  1,  2,  1,  3,           // 3
                               1,  1,  2,  2,  1,           // 4
                               1,  2,  1,  1,  3,           // 5
                               1,  2,  1,  2,  2,           // 6
                               1,  2,  2,  1,  3,           // 7
                               1,  2,  2,  2,  1,           // 8
                               2,  1,  1,  1,  3,           // 9
                               2,  1,  1,  2,  2,           // 10
                               2,  1,  2,  1,  3,           // 11
                               2,  1,  2,  2,  1,           // 12
                               2,  2,  1,  1,  3,           // 13
                               2,  2,  1,  2,  2,           // 14
                               2,  2,  2,  1,  3,           // 15
                               2,  2,  2,  2,  3,           // 16
                               3,  1,  1,  1,  3,           // 17
                               3,  1,  1,  2,  3,           // 18
                               3,  1,  2,  1,  3,           // 19
                               3,  1,  2,  2,  1,           // 20
                               3,  2,  1,  1,  3,           // 21
                               3,  2,  1,  2,  2,           // 22
                               3,  2,  2,  1,  3,           // 23
                               3,  2,  2,  2,  3)           // 24

    xy -= 1                                                         // shift values to start at 0

    val fn = Array ("Age", "Spectacle", "Astigmatic", "Tear")       // feature names
    val cn = Array ("Hard", "Soft", "Neither")                      // class names

    banner ("Contact Leans Example")
    println (s"xy = $xy")

    val nb = NaiveBayes (xy, fn, 3, cn)()                           // create the classifier
    nb.trainNtest ()()                                              // train and test the classifier
    nb.printCPTs ()                                                 // print the conditional probability tables
    println (nb.summary ())                                         // summary statistics

    for i <- xy.indices2 do
        val z   = xy(i).not(4).toInt                                // x-values
        val y   = xy(i, 4).toInt                                    // y-value
        val yp  = nb.classify (z)                                   // y predicted
        println (s"Use nb : yp = classify ($z) = $yp,\t y = $y,\t ${cn(y)}")
    end for

end naiveBayesTest4

