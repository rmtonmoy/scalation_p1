
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng
 *  @version 2.0
 *  @date    Fri Sep 30 13:37:32 EDT 2011
 *  @see     LICENSE (MIT style license file).
 *
 *  @ttile   Broyden–Fletcher–Goldfarb–Shanno (BFGS) Quasi-Newton Optimizer
 *
 *  @see The Superlinear Convergence of a Modified BFGS-Type Method for Unconstrained Optimization 
 *  @see On the Robustness of Conjugate-Gradient Methods and Quasi-Newton Methods
 *  @see Limited Memory BFGS for Nonsmooth Optimization
 *  @see http://en.wikipedia.org/wiki/BFGS_method
 *  @see http://www.personal.psu.edu/cxg286/Math555.pdf
 *  @see http://people.orie.cornell.edu/aslewis/publications/bfgs_inexactLS.pdf
 *  @see http://people.orie.cornell.edu/aslewis/publications/bfgs_exactLS.pdf
 */

package scalation
package optimization

import scala.math.{abs, max}
import scala.util.control.Breaks.{breakable, break}

import scalation.calculus.Differential.∇
import scalation.mathstat._
import scalation.random.RandomVecD

import MatrixD.{eye, outer}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BFGS` the class implements the Broyden–Fletcher–Goldfarb–Shanno (BFGS)
 *  Quasi-Newton Algorithm for solving Non-Linear Programming (NLP) problems.
 *  BFGS determines a search direction by deflecting the steepest descent direction
 *  vector (opposite the gradient) by  multiplying it by a matrix that approximates
 *  the inverse Hessian.  Note, this  implementation may be set up to work with the matrix
 *  b (approximate Hessian) or directly with the binv matrix (the inverse of b).
 *
 *  minimize    f(x)
 *  subject to  g(x) <= 0   [ optionally g(x) == 0 ]
 *
 *  @param f        the objective function to be minimized
 *  @param g        the constraint function to be satisfied, if any
 *  @param ineq     whether the constraint is treated as inequality (default) or equality
 *  @param exactLS  whether to use exact (e.g., `GoldenLS`)
 *                            or inexact (e.g., `WolfeLS`) Line Search
 */
class BFGS (f: FunctionV2S, g: FunctionV2S = null,
            ineq: Boolean = true, exactLS: Boolean = false)
      extends Minimizer:

    private val debug  = debugf ("BFGS", false)             // debug function
    private val flaw   = flawf ("BFGS")                     // flaw function
    private val WEIGHT = 1000.0                             // weight on penalty for constraint violation

    private var df: Array [FunctionV2S] = null              // array of partials
    private var b: MatrixD    = null                        // approx. Hessian matrix (use b or binv)
    private var binv: MatrixD = null                        // inverse of approx. Hessian matrix
    private var bfgs          = true                        // use BFGS (true) or Gradient Descent (false)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use the Gradient Descent algorithm rather than the default BFGS algorithm.
     */
    def setSteepest (): Unit = { bfgs = false }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the b matrix, whose inverse is used to deflect -gradient to a
     *  better direction than steepest descent (-gradient).
     *  @param s  the step vector (next point - current point)
     *  @param y  the difference in the gradients (next - current)
     */
//  def updateB (s: VectorD, y: VectorD): Unit =
//  {
//      var sy = s dot y                                    // dot product of s and y
//      if abs (sy) < TOL then sy = TOL
//      val sb = s * b
//      b += outer (y, y) / sy - outer (sb, sb) / (sb dot s)
//  } // updateB

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the binv matrix, which is used to deflect -gradient to a better
     *  search direction than steepest descent (-gradient).
     *  Compute the binv matrix directly using the Sherman–Morrison formula.
     *  @see http://en.wikipedia.org/wiki/BFGS_method
     *  @param s  the step vector (next point - current point)
     *  @param y  the difference in the gradients (next - current)
     */
    def updateBinv (s: VectorD, y: VectorD): Unit =
        var sy = s dot y                                    // dot product of s and y
        if abs (sy) < TOL then sy = TOL
        val binvy = binv * y
        binv +=  (outer (s, s) * (sy + (binvy dot y))) / (sy * sy) -
                 (outer (binvy, s) + outer (s, binvy)) / sy
    end updateBinv

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the partial derivative functions.  If these functions are available,
     *  they are more efficient and more accurate than estimating the values
     *  using difference quotients (the default approach).
     *  @param partials  the array of partial derivative functions
     */
    def setDerivatives (partials: Array [FunctionV2S]): Unit =
        if g != null then flaw ("setDerivatives", "only works for unconstrained problems")
        df = partials                           // use given functions for partial derivatives
    end setDerivatives

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The objective function f plus a weighted penalty based on the constraint
     *  function g.
     *  @param x  the coordinate values of the current point
     */
    override def fg (x: VectorD): Double =
        val f_x = f(x)
        if g == null then                                  // unconstrained
            f_x
        else                                               // constrained, g(x) <= 0
            val penalty = if ineq then max (g(x), 0.0) else abs (g(x))
            f_x + abs (f_x) * WEIGHT * penalty * penalty
        end if
    end fg

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform an exact GoldenSectionLS or inexact WolfeLS Line Search.
     *  Search in direction dir, returning the distance z to move in that direction.
     *  Default to 
     *  @param x     the current point
     *  @param dir   the direction to move in
     *  @param step  the initial step size
     */
    def lineSearch (x: VectorD, dir: VectorD, step: Double = STEP): Double =
        def f_1D (z: Double): Double = fg(x + dir * z)         // create a 1D function
        val ls = if exactLS then new GoldenSectionLS (f_1D)    // Golden Section Line Search
                 else new WolfeLS (f_1D)                       // Wolfe line search ((c1 = .0001, c2 = .9)
        ls.search (step)                                       // perform a Line Search
    end lineSearch

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the following Non-Linear Programming (NLP) problem using BFGS:
     *  min { f(x) | g(x) <= 0 }.  To use explicit functions for gradient,
     *  replace gradient (fg, x._1 + s) with gradientD (df,  x._1 + s).
     *  @param x0     the starting point 
     *  @param step_  the initial step size
     *  @param toler  the tolerance
     */
    def solve (x0: VectorD, step_ : Double = STEP, toler: Double = TOL): FuncVec =
        debug ("solve", s"x0 = $x0, step_ = $step_, toler = $toler")

        var step = step_                                      // set the current step size
        var x    = (x0, ∇ (fg, x0))                           // current (point, gradient)
        var xx:  (VectorD, VectorD) = (null, null)            // next (point, gradient)
        var dir: VectorD = null                               // initial direction is -gradient
        var s:   VectorD = null                               // step vector

        binv = eye (x0.dim, x0.dim)                           // inverse of approx. Hessian matrix

        debug ("solve", s"||gradient||^2 = ${x._2.normSq}")

        var mgn         = 0.0                                 // mean gradient normSq
        var diff        = 0.0                                 // diff between current and next point
        val diffTol     = toler * toler                       // tolerance for changes in diff
        var count       = 0                                   // number of times mgn stayed roughly same (< diffTol)
        val maxCount    = 10                                  // max number of times mgn stayed roughly same => terminate
        val n           = x0.dim                              // size of the parameter vector
        var goodGrad    = true                                // good gradient value flag (not NaN nor infinity)
        var xn: VectorD = null                                // next value for x (point)

        breakable {
            for k <- 1 to MAX_ITER do
                debug ("solve", s"start of iteration $k: step = $step, f(x) = ${fg(x._1)}")
                if goodGrad then
                    dir = if bfgs then -(binv * x._2) else -x._2
                end if
                s  = dir * lineSearch (x._1, dir, step)           // update step vector
                xn = x._1 + s                                     // next x point
                if goodGrad then
                    for xx_i <- xn if xx_i.isNaN || xx_i.isInfinite do break ()
                    diff = (xn - x._1).normSq / n                 // measure of distance moved
                end if
                xx = (xn, ∇ (fg, xn))                             // compute the next point
                mgn = xx._2.normSq / n                            // compute mean gradient normSq
                debug ("solve", s"current mean gradient normSq = $mgn")

                if mgn.isNaN || mgn.isInfinite then
                    goodGrad = false                              // gradient blew up
                    step /= 2.0                                   // halve the step size 
                else if mgn < toler || count > maxCount then { x = xx; break () }  // return when vanished gradient or haven't moved
                else if goodGrad then
                    if diff < diffTol then count += 1             // increment no movement counter
                    if step < step_   then step  *= 1.5           // increase step size by 50%
                else
                    goodGrad = true                               // gradient is currently fine
                end if

                if goodGrad then
                    if bfgs then updateBinv (s, xx._2 - x._2)     // update the deflection matrix binv
                    debug ("solve", s"(k = $k) move from ${x._1} to ${xx._1} where fg(xx._1) = ${fg(xx._1)}")
                    x = xx                                        // make the next point the current point
                end if
            end for
        } // breakable
        (fg(x._1), x._1)                                      // return functional value and current point
    end solve

end BFGS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BFGS` companion object provides factory functions.
 */
object BFGS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Steepest Descent (default) or BFGS optimizer.
     *  @param f         the objective function to be minimized
     *  @param g         the constraint function to be satisfied, if any
     *  @param ineq      whether the constraint is treated as inequality (default) or equality
     *  @param exactLS   whether to use exact (e.g., `GoldenLS`)
     *                             or inexact (e.g., `WolfeLS`) Line Search
     *  @param steepest  whether to use Steepest Descent rather than BFGS
     */
    def apply (f: FunctionV2S, g: FunctionV2S = null,
               ineq: Boolean = true, exactLS: Boolean = false,
               steepest: Boolean = true): BFGS =
        if steepest then
           val steep = new BFGS (f, f, ineq, exactLS)
           steep.setSteepest ()
           steep
        else
           new BFGS (f, f, ineq, exactLS)
        end if
    end apply

end BFGS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSTest` main function is used to test the `BFGS` class on f(x):
 *      f(x) = (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.bFGSTest
 */
@main def bFGSTest (): Unit =

    val n  = 2                                             // dimension of the search space
    val x0 = new VectorD (n)

    banner ("Minimize: (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    def f (x: VectorD): Double = (x(0) - 3.0)~^2 + (x(1) - 4.0)~^2 + 1.0

    val optimizer = new BFGS (f)
    val opt = optimizer.solve (x0)
    println (s"][ optimal solution (f(x), x) = $opt")

end bFGSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSTest2` main function is used to test the `BFGS` class on f(x):
 *      f(x) = x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.bFGSTest2
 */
@main def bFGSTest2 (): Unit =

    val n  = 2                                             // dimension of the search space
    val x0 = new VectorD (n)

    banner ("Minimize: x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    def f (x: VectorD): Double = x(0)~^4 + (x(0) - 3.0)~^2 + (x(1) - 4.0)~^2 + 1.0

    val optimizer = new BFGS (f)
    val opt = optimizer.solve (x0)
    println (s"][ optimal solution (f(x), x) = $opt")

end bFGSTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSTest3` main function is used to test the `BFGS` class on f(x):
 *      f(x) = 1/x(0) + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.bFGSTest3
 */
@main def bFGSTest3 (): Unit =  

    val n  = 2                                             // dimension of the search space
    val x0 = VectorD (0.1, 0.0)                            // starting location

    banner ("Minimize: 1/x(0) + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    def f (x: VectorD): Double = 1/x(0) + x(0)~^4 + (x(0) - 3.0)~^2 + (x(1) - 4.0)~^2 + 1.0

    val optimizer = new BFGS (f)
    var opt = optimizer.solve (x0)
    println (s"][ optimal solution (f(x), x) = $opt")

    opt = optimizer.resolve (n)
    println (s"][ optimal solution (f(x), x) = $opt")

end bFGSTest3

