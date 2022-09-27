
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Hao Peng, John Miller
 *  @version 2.0
 *  @date    Sun Nov 19 12:27:00 EST 2017
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model Support: Statistical Test for Time Series Stationarity
 *
 *  Kwiatkowski–Phillips–Schmidt–Shin (KPSS) Test
 *  Time Series Stationarity around a deterministic trend
 *  @see debis.deu.edu.tr/userweb//onder.hanedar/dosyalar/kpss.pdf
 *  @see github.com/olmallet81/URT
 */

package scalation
package modeling
package forecasting

import scala.collection.immutable.HashMap

import scalation.mathstat.{MatrixD, VectorD, VectorS}
import scalation.random.Normal

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The companion object for `KPSS` class, containing critical value coefficients
 *  needed in KPSS tests for Time Series Stationarity around a deterministic trend.
 */
object Stationarity_KPSS:

    val coeffKpss = HashMap (
        "c" -> HashMap (
            (0.001 -> HashMap ((0 -> VectorD (0.0167565, 0.1628078, -0.01441111)),
                               (1 -> VectorD (0.2036426, 0.1065976, -0.5134231)))),
            (0.005 -> HashMap ((0 -> VectorD (0.02160067, 0.1649124, -0.005748196)),
                               (1 -> VectorD (0.1939746, 0.180736, -0.6535936)))),
            (0.01  -> HashMap ((0 -> VectorD (0.02466286, 0.1635288, 0.04500264)),
                               (1 -> VectorD (0.1893402, 0.2089429, -0.6943796)))),
            (0.025 -> HashMap ((0 -> VectorD (0.03026605, 0.1623029, 0.09417188)),
                               (1 -> VectorD (0.1847549, 0.2238625, -0.6818411)))),
            (0.05  -> HashMap ((0 -> VectorD (0.03650665, 0.1643194, 0.118059)),
                               (1 -> VectorD (0.1819509, 0.2103519, -0.5991279)))),
            (0.1   -> HashMap ((0 -> VectorD (0.04597858, 0.1695003, 0.1046518)),
                               (1 -> VectorD (0.1809212, 0.1596069, -0.4150671)))),
            (0.2   -> HashMap ((0 -> VectorD (0.06222337, 0.1785042, 0.06728867)),
                               (1 -> VectorD (0.1794514, 0.1031811, -0.2438007)))),
            (0.5   -> HashMap ((0 -> VectorD (0.1188952, 0.2136505, -0.1988974)),
                               (1 -> VectorD (0.1738762, -0.04517147, 0.1061083)))),
            (0.8   -> HashMap ((0 -> VectorD (0.2413558, 0.2427114, -0.4925517)),
                               (1 -> VectorD (0.07161341, 0.3635483, -0.8456676)))),
            (0.9   -> HashMap ((0 -> VectorD (0.347477, 0.1815891, -0.8897294)),
                               (1 -> VectorD (-0.1026364, 0.4255538, -0.6916551)))),
            (0.95  -> HashMap ((0 -> VectorD (0.461564, -0.03241585, -0.4093091)),
                               (1 -> VectorD (-0.4579306, 1.037593, -0.9517771)))),
            (0.975 -> HashMap ((0 -> VectorD (0.5808698, -0.4198131, 1.286591)),
                               (1 -> VectorD (-1.021283, 2.743161, -2.926945)))),
            (0.99  -> HashMap ((0 -> VectorD (0.7434379, -1.282285, 7.296387)),
                               (1 -> VectorD (-2.025517, 6.584114, -8.361188)))),
            (0.995 -> HashMap ((0 -> VectorD (0.8686703, -2.108955, 13.70689)),
                               (1 -> VectorD (-3.003148, 11.11568, -15.74966)))),
            (0.999 -> HashMap ((0 -> VectorD (1.162377, -4.578843, 34.27324)),
                               (1 -> VectorD (-5.808165, 25.97152, -41.79132))))),
        "ct" -> HashMap (
            (0.001 -> HashMap ((0 -> VectorD (0.01234275, 0.1655343, 0.01672944, -1.675843)),
                               (1 -> VectorD (0.2106948, 0.03970888, -0.3368414)))),
            (0.005 -> HashMap ((0 -> VectorD (0.01526251, 0.1588324, 0.2212439, -3.176791)),
                               (1 -> VectorD (0.2015772, 0.1179992, -0.5024903)))),
            (0.01  -> HashMap ((0 -> VectorD (0.01699618, 0.1544975, 0.3604586, -4.262545)),
                               (1 -> VectorD (0.1969925, 0.1538411, -0.5702144)))),
            (0.025 -> HashMap ((0 -> VectorD (0.02005522, 0.1485785, 0.5907839, -6.211238)),
                               (1 -> VectorD (0.1880815, 0.2169062, -0.6777426)))),
            (0.05  -> HashMap ((0 -> VectorD (0.02326679, 0.1414555, 0.8224264, -7.893967)),
                               (1 -> VectorD (0.1804144, 0.2639471, -0.7486399)))),
            (0.1   -> HashMap ((0 -> VectorD (0.02781531, 0.1377224, 0.917243, -8.218599)),
                               (1 -> VectorD (0.170477, 0.3089857, -0.7857759)))),
            (0.2   -> HashMap ((0 -> VectorD (0.03489574, 0.1345627, 0.9657117, -7.923499)),
                               (1 -> VectorD (0.1579158, 0.3369968, -0.7515043)))),
            (0.5   -> HashMap ((0 -> VectorD (0.0556109, 0.1397416, 0.4620311, -1.559948)),
                               (1 -> VectorD (0.1300005, 0.2833583, -0.4283284)))),
            (0.8   -> HashMap ((0 -> VectorD (0.09159799, 0.1345566, -0.619965, 9.817577)),
                               (1 -> VectorD (0.05585951, 0.2773908, -0.1216728)))),
            (0.9   -> HashMap ((0 -> VectorD (0.1193112, 0.09514355, -1.079983, 16.87997)),
                               (1 -> VectorD (-0.03271029, 0.4379497, -0.2279858)))),
            (0.95  -> HashMap ((0 -> VectorD (0.1478756, 0.03863841, -1.89105, 30.13683)),
                               (1 -> VectorD (-0.149998, 0.6506317, -0.2305911)))),
            (0.975 -> HashMap ((0 -> VectorD (0.1773233, -0.03755152, -3.125617, 51.66143)),
                               (1 -> VectorD (-0.3091562, 1.057115, -0.4266199)))),
            (0.99  -> HashMap ((0 -> VectorD (0.2172606, -0.2049983, -4.308647, 82.65278)),
                               (1 -> VectorD (-0.5765132, 1.915267, -1.057833)))),
            (0.995 -> HashMap ((0 -> VectorD (0.2480191, -0.3967449, -4.07506, 101.2076)),
                               (1 -> VectorD (-0.8216993, 2.85189, -1.957697)))),
            (0.999 -> HashMap ((0 -> VectorD (0.3203646, -1.046171, -0.8434172, 144.566)),
                               (1 -> VectorD (-1.515429, 6.124994, -6.22971)))))
    ) // HashMap

end Stationarity_KPSS

import Stationarity_KPSS.coeffKpss

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KPSS` class provides capabilities of performing KPSS test to determine
 *  if a time series is stationary around a deterministic trend.
 *  This code is translated from the C++ code found in
 *  @see github.com/olmallet81/URT.
 *  @param yy         the original time series vector
 *  @param lags_      the number of lags to use
 *  @param lagsType_  type of lags, long or short
 *  @param trend_     type of trend to test for
 */
class Stationarity_KPSS (yy : VectorD, lags_ : Int, lagsType_ : String, trend_ : String = "c")
      extends UnitRoot ("KPSS", yy.dim, VectorS ("c", "ct"), lagsType_, lags_, trend_):

    private var y = yy.copy                                            // y may be demeaned/detrended
    private var stat: Double = -0.0                                    // test statistic

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset y to the original data.
     */
    def setData (): Unit = y = yy.copy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** OLS demeaning or detrending.
     */
    def olsDetrend (): Unit =
        var w: MatrixD = null

        if trend == "ct" then
            w = new MatrixD (nobs, 2)
            for i <- w.indices do { w(i, 0) = 1.0; w(i, 1) = i }
        else if trend == "nc" then
            throw new IllegalArgumentException
                  ("\n ERROR: olsDetrend: no detrending possible when regression trend is no constant.\n")
        end if

        if trend == "c" then
            y = y - y.mean                                             // use mean centered
        else
            val reg = new Regression (w, y)
            reg.train ()
            val yp = reg.predict (w)
            y -= yp                                                    // use residuals
        end if
    end olsDetrend

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute test statistics.
     */
    def computeStat (): Unit =
        val factor = 1.0 / nobs
        val s = new VectorD (nobs)
        s(0)  = y(0)
        for i <- 1 until nobs do s(i) = y(i) + s(i-1)
        val eta = factor * factor * (s dot s)
        var s2  = factor * (y dot y)

        var tmp1 = 0.0                                                 // estimating long run variance
        for i <- 1 to lags do
            val tmp2   = y(i until nobs) dot y(0 until nobs - i)
            val weight = 1.0 - i / (lags + 1.0)                        // computing Bartlett weight
            tmp1      += weight * tmp2
        end for

        s2  += factor * 2.0 * tmp1
        stat = eta / s2
    end computeStat

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute KPSS test statistic.
     */
    def statistic (): Double =
        setLagsType ()                                                 // set type of lags (if default type of lags value was chosen)
        setLags ()                                                     // set number of lags
        setTrend ()                                                    // set regression trend

        if newTrend then                                               // if new trend
            setData ()                                                 // set back to original data for new detrending
            olsDetrend ()                                              // detrending data by OLS
        end if

        if newTrend || newLags then                                    // if new trend or new lags
            computeStat ()                                             // computing statistic
            newTrend = false
            newLags  = false
        end if

        stat
    end statistic

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute test statistic p-value.
     */
    override def pvalue (st: Double = stat): Double =
        statistic ()                                                   // computing test statistic
        coeff = coeffKpss.getOrElse (trend, null)                      // setting critical values coefficients
        pval = 1 - super.pvalue (st)                                   // computing p-value
        pval
    end pvalue

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Output test results.
     */
    override def show (st: Double = statistic ()): Unit =
        pvalue (st)       // computing p-value
        super.show (st)   // outputting results
    end show

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check to see if H0 can be rejected. Must call pvalue first to compute pval.
     */
    def canReject (alpha: Double = 0.05): Boolean = pval < alpha

end Stationarity_KPSS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `stationarity_KPSSTest` main function is used to test the `Stationarity_KPSS` class.
 *  Test whether white noise time-series has a unit root.
 *  > runMain scalation.modeling.forecasting.stationarity_KPSSTest
 */
@main def stationarity_KPSSTest (): Unit =

    val nobs = 1000

    val noise = new Normal ()                                          // generating stationary random data
    val y = VectorD (for i <- 0 until nobs yield noise.gen)

    val test = new Stationarity_KPSS (y, 0, "short", "c")              // init KPSS test with lags of type short and constant trend
    test.show ()                                                       // outputting test results

    test.reset (5, "ct")                                               // switching to test with 5 lags and constant term
    test.show ()                                                       // outputting test results

end stationarity_KPSSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `stationarity_KPSSTest2` main function is used to test the `Stationarity_KPSS` class.
 *  Test whether the Lake Levels time-series has a unit root.
 *  > runMain scalation.modeling.forecasting.stationarity_KPSSTest2
 */
@main def stationarity_KPSSTest2 (): Unit =

    import Example_LakeLevels.y

    val test = new Stationarity_KPSS (y, 0, "short", "c")              // init KPSS test with lags of type short and constant trend
    test.show ()                                                       // outputting test results

    test.reset (5, "ct")                                               // switching to test with 5 lags and constant term
    test.show ()                                                       // outputting test results

end stationarity_KPSSTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `stationarity_KPSSTest3` main function is used to test the `Stationarity_KPSS` class.
 *  Test whether the differenced Lake Levels time-series has a unit root.
 *  > runMain scalation.modeling.forecasting.stationarity_KPSSTest3
 */
@main def stationarity_KPSSTest3 (): Unit =

    import Example_LakeLevels.y

    val yd = del (y)                                                    // take first difference of y

    val test = new Stationarity_KPSS (yd, 0, "short", "c")              // init KPSS test with lags of type short and constant trend
    test.show ()                                                        // outputting test results

    test.reset (5, "ct")                                                // switching to test with 5 lags and constant term
    test.show ()                                                        // outputting test results

end stationarity_KPSSTest3

