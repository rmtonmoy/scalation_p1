
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng
 *  @version 2.0
 *  @date    Sun Jan 27 15:34:08 EST 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Optimization: Stopping Rule for Iterative Optimizers
 */

package scalation
package modeling
package neuralnet

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `StoppingRule` trait provides stopping rules to terminating the iterative
 *  steps in an optimization early.
 */
trait StoppingRule:

    protected val EPSILON            = 1E-7                                // number close to zero
    private   val upLimit            = Optimizer.hp("upLimit").toInt       // number of increasing steps allowed
    private   var up                 = 0                                   // consecutive steps up
    private   var b_best:  VectorD   = null                                // best parameter vector for Perceptron
    private   var bb_best: NetParams = null                                // best parameters for neural networks
    private   var sse0               = Double.MaxValue                     // previous sum of squared errors
    private   var sse_best           = Double.MaxValue                     // best sse, so far

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Stop when too many steps have the cost measure (e.g., sse) increasing.
     *  Signal a stopping condition by returning the best parameter vector, else null.
     *  @param b    the current value of the parameter vector
     *  @param sse  the current value of cost measure (e.g., sum of squared errors)
     */
    def stopWhen (b: VectorD, sse: Double): (VectorD, Double) =
        if sse > sse0 + EPSILON then  up += 1                              // getting worse
        else                                                               // getting better
            up = 0
            if sse < sse_best then { b_best = b.copy; sse_best = sse }     // lower see => save as best
        end if
        sse0 = sse                                                         // make current the previous
        if up > upLimit then (b_best, sse_best) else (null, sse_best)      // if at limit, return best
    end stopWhen

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Stop when too many steps have the cost measure (e.g., sse) increasing.
     *  Signal a stopping condition by returning the best parameter vector, else null.
     *  @param b    the current parameter value (weights and biases)
     *  @param sse  the current value of cost measure (e.g., sum of squared errors)
     */
    def stopWhen (b: NetParams, sse: Double): (NetParams, Double) =
        if sse > sse0 + EPSILON then up += 1                               // getting worse
        else                                                               // getting better
            up = 0
            if sse < sse_best then                                         // lower see => save as best
                bb_best  = (for l <- b.indices yield b(l).copy).toArray    // copy for each layer l
                sse_best = sse
            end if
        end if
        sse0 = sse                                                         // make current the previous
        if up > upLimit then (bb_best, sse_best)                           // if at limit, return best
        else (null, sse_best)                                              // return null => continue
    end stopWhen

end StoppingRule

