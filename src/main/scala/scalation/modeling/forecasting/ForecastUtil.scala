
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Feb 13 16:22:21 EST 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Utilities for Time Series Forecasting
 */

package scalation
package modeling
package forecasting

import scala.math.max

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Given a response vector y, build and return
 *  (1) an input/predictor MATRIX xx and
 *  (2) an output/multi-horizon output/response MATRIX yy.
 *  The first lag responses can't be predicted due to missing past values.
 *  The last h-1 responses can't be predicted due to missing future values.
 *  Therefore the number of rows in xx and yy is reduced to y.dim + 1 - lag - h.
 *  @param y    the given output/response vector
 *  @param lag  the maximum lag included (inclusive)
 *  @param h    the forecasting horizon (1, 2, ... h)
 */
def buildMatrix4TS (y: VectorD, lag: Int, h: Int): (MatrixD, MatrixD) =
    val xx = new MatrixD (y.dim + 1 - lag - h, lag)
    val yy = new MatrixD (y.dim + 1 - lag - h, h)
    for i <- lag to y.dim - h do
        for j <- xx.indices2 do xx(i-lag, lag - 1 - j) = y(i - 1 - j)
        for j <- yy.indices2 do yy(i-lag, j) = y(i + j)
    end for
//  println (s"buildMatrix4TS: xx = $xx \n yy = $yy")
    (xx, yy)
end buildMatrix4TS


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Given a response vector y, build and return
 *  (1) an input/predictor MATRIX xx and 
 *  (2) an output/single-horizon output/response VECTOR yy. 
 *  The first lag responses can't be predicted due to missing past values.
 *  Therefore the number of rows in xx and yy is reduced to y.dim - lag.
 *  @param y    the given output/response vector
 *  @param lag  the maximum lag included (inclusive)
 */  
def buildMatrix4TS (y: VectorD, lag: Int): (MatrixD, VectorD) =
    val xx = new MatrixD (y.dim - lag, lag)
    val yy = new VectorD (y.dim - lag)
    for i <- lag until y.dim do
        for j <- xx.indices2 do xx(i-lag, lag - 1 - j) = y(i - 1 - j)
        yy(i-lag) = y(i)
    end for 
//  println (s"buildMatrix4TS: xx = $xx \n yy = $yy")
    (xx, yy) 
end buildMatrix4TS


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Given an exogenous variable vector ex corresponding to an endogenous response
 *  vector y, build and return an input/predictor MATRIX xx.
 *  The first lag responses can't be predicted due to missing past values.
 *  Therefore the number of rows in xx is reduced to ex.dim - lag.
 *  @param ex     the exogenous variable vector
 *  @param lag    the maximum lag included (inclusive) for the endogenous variable
 *  @param elag1  the maximum lag included (inclusive) for the exogenous variable
 *  @param elag2  the maximum lag included (inclusive) for the exogenous variable
 */
def buildMatrix4TS_exo (ex: VectorD, lag: Int, elag1: Int, elag2: Int): MatrixD =
    val flaw = flawf ("top")
    val n = elag2 - elag1
    if n < 1 then flaw ("buildMatrix4TS_exo", "min exo lag must be smaller than max exo lag")
    if elag2 > lag then flaw ("buildMatrix4TS_exo", "exo lag cannot exceed endogenous lag")

    val xx = new MatrixD (ex.dim - lag, n)
    for i <- lag until ex.dim do
        for j <- xx.indices2 do xx(i-lag, n - 1 - j) = ex(i - elag1 - j)
    end for
//  println (s"buildMatrix4TS_exo: xx = $xx")
    xx
end buildMatrix4TS_exo


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Test the actual response vector vs. forecasted matrix, returning the QoF
 *  for all forecasting horizons 1 to h.
 *  @param mod  the fittable model (one that extends `Fit`)
 *  @param y    the orginal actual response vector
 *  @param yf   the forecasted response matrix
 *  @param p    the number of variables/lags used in the model
 */
def testForecast (mod: Fit, y: VectorD, yf: MatrixD, p: Int): MatrixD =
    MatrixD (for k <- yf.indices2 yield
        val y_  = y(p + k until y.dim)
        val yf_ = yf(?, k)(0 until y.dim - p - k)
        println (s"y_.dim = ${y_.dim}, yf_.dim = ${yf_.dim}")
        mod.resetDF (p, y.dim - p - (k+1))                               // reset the degrees of freedom
        mod.diagnose (y_, yf_))                                          // return the QoF of the forecasts
end testForecast

