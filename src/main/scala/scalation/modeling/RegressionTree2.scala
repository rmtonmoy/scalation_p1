
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Dong Yu Yu, John Miller
 *  @version 2.0
 *  @date    Wed Nov  7 17:08:17 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Regression Tree
 */

package scalation
package modeling

import scala.collection.mutable.{ArrayBuffer, Queue, Set}
import scala.math.abs
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

// FIX - rSqBar from validate is wrong

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTree2` companion object is used to count the number of leaves
 *  and provide factory functions.
 */
object RegressionTree2:

    private val debug    = debugf ("RegressionTree2", true)           // debug function
    private var nLeaves_ = 0                                          // the number of leaves in the tree

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of leaves in the tree.
     */
    def nLeaves: Int = nLeaves_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Icrement the number of leaves in the tree.
     */
    def incLeaves (): Unit = nLeaves_ += 1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the number of leaves in the tree.
     */
    def resetLeaves (): Unit = nLeaves_ = 0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTree2` object from a combined data-response matrix.
     *  @param xy      the combined data-response matrix
     *  @param fname   the names for all features/variables
     *  @param hparam  the hyper-parameters
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = RegressionTree.hp)
              (col: Int = xy.dim2 - 1): RegressionTree2 =
        new RegressionTree2 (xy.not(?, col), xy(?, col), fname, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTree2` object from a data matrix and response vector.
     *  @param x       the data matrix
     *  @param y       the response vector
     *  @param fname   the names for all features/variables
     *  @param hparam  the hyper-parameters
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
               hparam: HyperParameter = RegressionTree.hp): RegressionTree2 =
        val xn = normalize ((x.mean, x.stdev)) (x)
        new RegressionTree2 (xn, y, fname, hparam)
    end rescale

end RegressionTree2

import RegressionTree2.{nLeaves, incLeaves, resetLeaves}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTree2` class implements a Regression Tree that selects splitting features
 *  using minimal variance in children nodes.  To avoid exponential choices in the selection,
 *  supporting ordinal features currently.
 *  Note: may not split in certain cases where `RegressionTree` does.
 *  @param x            the m-by-n input/data matrix
 *  @param y            the response m-vector
 *  @param fname_       the names of the model's features/variables
 *  @param hparam       the hyper-parameters for the model
 *  @param curDepth     current depth
 *  @param branchValue  the branch value for the tree node
 *  @param feature      the feature for the tree's parent node
 */
class RegressionTree2 (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                      hparam: HyperParameter = RegressionTree.hp,
                      curDepth: Int = 0, branchValue: Int = -1, feature: Int = -1)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfm = x.dim2 - 1, df = x.dim - x.dim2):            // call resetDF once tree is built

    private val debug     = debugf ("RegressionTree2", true)          // debug function
    private val flaw      = flawf ("RegressionTree2")                 // flaw function
    private val (m, n)    = (x.dim, x.dim2)                           // matrix dimensions
    private val depth     = hparam ("maxDepth").toInt                 // the depth limit for tree
    private val thres     = hparam ("threshold").toDouble             // the threshold for the tree's parent node
    private val threshold = new Array [(Double, Double)] (n)          // store best splitting (threshold, score) for each feature

    private var root: Node = null                                     // root node   

    modelName = s"RegressionTree2 ($depth)"

    debug ("init", s"Constructing a Regression Tree 2: curDepth = $curDepth")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split gives row indices of left and right children when splitting using thresh.
     *  @param j       the column/feature to use
     *  @param thresh  the threshold for splitting (below => left, above => right) 
     */
    private def split (j: Int, thresh: Double): (IndexedSeq [Int], IndexedSeq [Int]) =
        val (sLeft, sRight) = (Set [Int] (), Set [Int] ())
        for i <- x.indices do if x(i, j) <= thresh then sLeft += i else sRight += i
        (sLeft.toIndexedSeq, sRight.toIndexedSeq)
    end split

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given feature f, use fast threshold selection to find an optimal threshold/
     *  split point in O(NlogN) time.
     *  @see people.cs.umass.edu/~domke/courses/sml/12trees.pdf
     *  @param xj  the j-th column in data matrix
     */
    def fastThreshold (xj: VectorD): (Double, Double) =
        var thres = -0.0                                                   // to hold optimal threshold (nothing marker)
        var tSSE  = Double.MaxValue                                        // total sum of squared errors
        var ref   = Array.ofDim [(Double, Int)] (y.dim)                    // pair column value with column index
        for i <- x.indices do ref(i) = (xj(i), i)                          // assign pairs
        ref = ref.sortBy (_._1)                                            // sort by column value

        val dvals = xj.distinct.sorted                                     // get distinct values from column xj & sort
        if dvals.dim <= 1 then return (thres, -1.0)                        // can't divide  => return early
        val v = dvals.mids                                                 // mid points between all values

        val (totalSum, totalSqr) = (y.sum, y.normSq)                       // total sum and sum of squares
        var sum, square, mean = 0.0                                        // left sum, square and mean
        var (row, valu) = (0, v(0))                                        // candidate split value/threshold

        for i <- ref.indices do
            if ref(i)._1 > valu then
                val n_i   = ref.size - i                                   // number of elements on left
                val rSum  = totalSum - sum                                 // right sum
                val rSqr  = totalSqr - square                              // right sum of squares
                val rMean = rSum / n_i                                     // right mean
                val lrSSE = square - 2 * sum  * mean  + i   * mean  * mean +
                            rSqr   - 2 * rSum * rMean + n_i * rMean * rMean

                if lrSSE < tSSE then { tSSE = lrSSE; thres = valu }        // update if lrSSE is smaller
                row += 1
            end if

            val yi  = y(ref(i)._2)
            sum    += yi                                                   // left sum
            square += yi * yi                                              // left sum of squares
            mean    = (yi + i * mean) / (i + 1)                            // left mean
            if row < v.dim then valu = v(row)
        end for

        println (s"(thres, tSSE) = ($thres, $tSSE)")
        (thres, tSSE)                                                      // return best split point for feature j
    end fastThreshold

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return new x matrix and y vector for next step of constructing regression tree.
     *  @param j     the feature/variable index
     *  @param side  indicator for which side of child is chosen (i.e., 0 for left child)
     */
    private def nextXY (j: Int, side: Int): (MatrixD, VectorD) =
        val (left, right) = split (j, threshold(j)._1)
        if side == 0 then (x(left), y(left)) else (x(right), y(right))
    end nextXY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check that results of the two fast threshold algorithms agree, returning
     *  true if they do and false otherwise.
     *  @param j      the column index in data matrix (xj)
     *  @param thr    the threshold selected using RegressionTree.fastThreshold
     *  @param th2    the threshold selected using this.fastThreshold
     *  @param score  the score from RegressionTree.fastThreshold
     *  @param scor2  the score from this.fastThreshold
     */
    private def check (j: Int, thr: Double, th2: Double, score: Double, scor2: Double): Boolean =
        var okay = true
        println ("-" * 70)
        println (s"check for x$j, thr = $thr, th2 = $th2, score = $score, scor2 = $scor2")
        if thr != th2 then
            println ("\n W A I T   W H A T \n")
            okay = flaw ("check", s"threshold for x$j thr = $thr != th2 = $th2")
        end if
        if abs (score - scor2) > 1E-6 then
            okay = flaw ("check", s"scores for x$j score = $score != scor2 = $scor2")
        end if
//      assert (thr == th2)
        okay
    end check

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the regression tree by selecting thresholds for the features/variables
     *  in matrix x_.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD, y_ : VectorD): Unit =
        resetLeaves ()
        val ssy = y_.normSq                                                  // sum of squared y
        for j <- x.indices2 do
            val (thr, score) = RegressionTree.fastThreshold (x(?, j), y, ssy)   // set threshold for features
            val (th2, scor2) = fastThreshold (x(?, j))                       // set threshold for features
            check (j, thr, th2, score, scor2)                                // just check
            threshold(j) = (th2, scor2)                                      // set threshold for features
        end for

        var opt = (0, threshold(0)._2)                                       // compute variance for feature 0
        debug ("train", s"for feature ${opt._1} the variance is ${opt._2}")

        for j <- 1 until x.dim2 do
            val jScore = threshold(j)._2
            debug ("train", s"for feature $j the score is $jScore")
            if jScore <= opt._2 then opt = (j, jScore)                       // save feature giving minimal variance
        end for

        debug ("train", s"optimal feature is ${opt._1} with variance of ${opt._2}")
        buildTree (opt)
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the next most distinguishing feature/attribute, extend the regression tree.
     *  @param opt  the optimal feature and the variance
     */
    private def buildTree (opt: (Int, Double)): Unit =
        root =
            if curDepth == 0 then Node (opt._1, -1, VectorD (y.mean), threshold(opt._1)._1, curDepth, -1.0, -1)
            else Node (opt._1, branchValue, VectorD (y.mean), threshold(opt._1)._1, curDepth, thres, feature)
        debug ("buildTree", s"--> Add root = ${root}")

        for i <- 0 until 2 do                                             // 0 => left, 1 => right
            val next = nextXY (opt._1, i)
            if next._2.size != 0 then
                root.child +=
                    (if curDepth == depth - 1 || next._2.size <= x.dim2 then
                        val yp = next._2.mean
                        incLeaves ()
                        Node (opt._1, root.child.length, VectorD (yp), threshold(opt._1)._1, curDepth + 1,
                              threshold(opt._1)._1, opt._1, true)
                    else
                        val hp = RegressionTree.hp.updateReturn ("threshold", threshold(opt._1)._1)
                        val subtree = new RegressionTree2 (next._1, next._2, fname, hp, curDepth + 1, i, opt._1)
                        subtree.train (next._1, next._2)
                        subtree.root)

                debug ("buildTree", s"--> Add child = ${root.child}")
//              debug ("buildTree", s"\t x \t = ${next._1} \n\t y \t = ${next._2}")
            end if
        end for
    end buildTree

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD) =
        val yp = predict (x_)                                             // make predictions
        val df1 = nLeaves                                                 // degrees of freedom model = number of leaves
        val df2 = y_.dim - df1                                            // degrees of freedom error
        resetDF ((df1, df2))
        (yp, diagnose (y_, yp))                                           // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the regression tree in Pre-Order using printT method. 
     */
    def printTree (): Unit =
        println ("Regression Tree: nLeaves = " + nLeaves)
        println ("fname = " + stringOf (fname))
        printT (root, 0)
        println ()
    end printTree

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Recursively print the regression tree nodes.
     *  @param nod     the current node
     *  @param level   the level of node nod in the tree
     */
    def printT (nod: Node, level: Int): Unit =
        println ("\t" * level + "[ " + nod + " ]")
        for cnode <-nod.child do printT (cnode, level + 1)
    end printT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print out the regression tree using Breadth First Search (BFS).
     */
    def printTree2 (): Unit =
        println ("RegressionTree2:")
        println ("fname = " + stringOf (fname))
        val queue = new Queue [Node] ()

        for cnode <- root.child do queue += cnode
        println (root)
        var level = 0

        while ! queue.isEmpty do
            val size = queue.size
            level   += 1
            for i <- 0 until size do
                val nod = queue.dequeue ()
                println ("\t" * level + "[ " + nod + " ]")
                for cnode <- nod.child do queue += cnode
            end for
            println ()
        end while
    end printTree2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a data vector z, predict the value by following the tree to the leaf.
     *  @param z  the data vector to predict
     */
    override def predict (z: VectorD): Double =
        var nd = root                                                     // current node
        while nd.child.length >= 2 do
            nd = if z(nd.j) <= nd.thresh then nd.child(0) else nd.child(1)
        end while
        nd.b(0)                                                           // b0 is the mean
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a data matrix z, predict the value by following the tree to the leaf.
     *  @param z  the data matrix to predict
     */
    override def predict (z: MatrixD = x): VectorD =
        VectorD (for i <- z.indices yield predict (z(i)))
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    override def buildModel (x_cols: MatrixD): RegressionTree2 =
        new RegressionTree2 (x_cols, y, null, hparam)
    end buildModel

end RegressionTree2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTree2Test` main function is used to test the `RegressionTree2` class.
  *  It tests a simple case that does not require a file to be read.
  *  @see translate.google.com/translate?hl=en&sl=zh-CN&u=https:
  *       //www.hrwhisper.me/machine-learning-decision-tree/&prev=search
  *  > runMain scalation.modeling.regressionTree2Test
  */
@main def regressionTree2Test (): Unit =

    val x  = MatrixD ((10, 1), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val y  = VectorD (5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05)
    val ox = VectorD.one (x.dim) +^: x
    val fname = Array ("x")

    banner (s"Regression no intercept")
    val reg = new Regression (x, y)
    reg.trainNtest ()()                                               // train and test the model

    banner (s"Regression with intercept")
    val reg2 = new Regression (ox, y)
    reg2.trainNtest ()()                                              // train and test the model

    banner (s"Quadratic Regression")
    val reg3 = SymbolicRegression.quadratic (x, y, fname)
    reg3.trainNtest ()()                                              // train and test the model

    banner (s"Perceptron sigmoid")
    val nn = Perceptron.rescale (reg3.getX, y)
    nn.trainNtest ()()                                                // train and test the model

    banner (s"Perceptron tanh")
    val nn2 = Perceptron.rescale (reg3.getX, y, f = ActivationFun.f_tanh)
    nn2.trainNtest ()()                                               // train and test the model

    for d <- 1 to 2 do
        banner (s"Regression Tree 2 with depth = $d")
        RegressionTree.hp("maxDepth") = d
        val mod = new RegressionTree2 (x, y, fname)
        mod.trainNtest ()()                                           // train and test the model
        mod.printTree ()
    end for

end regressionTree2Test


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTree2Test2` main function tests the `RegressionTree2` class using the
 *  AutoMPG dataset.  Assumes no missing values.  It tests multiple depths.
 *  > runMain scalation.modeling.regressionTree2Test2
 */
@main def regressionTree2Test2 (): Unit =

    import Example_AutoMPG._

//  println (s"x = $o")
//  println (s"y = $y")

    for d <- 1 to 5 do
        banner (s"AutoMPG Regression Tree 2 with d = $d")
        RegressionTree.hp("maxDepth") = d
        val mod = new RegressionTree (x, y, x_fname)                // create model with intercept (else pass x)
        mod.trainNtest ()()                                          // train and test the model
        mod.printTree ()                                             // print the regression tree
//      println (mod.summary ())                                     // parameter/coefficient statistics
    end for

end regressionTree2Test2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTree2Test3` main function tests the `RegressionTree2` class using the
 *  AutoMPG dataset.  Assumes no missing values.  It tests forward, backward and stepwise
 *  selection.
 *  > runMain scalation.modeling.regressionTree2Test3
 */
@main def regressionTree2Test3 (): Unit =

    import Example_AutoMPG._

    val d = 5

//  println (s"x = $x")
//  println (s"y = $y")

    banner (s"AutoMPG Regression Tree 2 with d = $d")
    RegressionTree.hp("maxDepth") = d
    val mod = new RegressionTree2 (x, y, x_fname)                    // create model with intercept (else pass x)
    mod.trainNtest ()()                                              // train and test the model
    mod.printTree ()                                                 // print the regression tree

//  banner ("Cross-Validation")
//  Fit.showQofStatTable (mod.crossValidate ())

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                   // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Regression Tree with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end regressionTree2Test3

