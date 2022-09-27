
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Aug 26 18:41:26 EDT 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Sample Statistics Collection, Confidence Intervals and Significance Tests
 */

package scalation
package mathstat

import scala.collection.mutable.ListBuffer
import scala.math.{abs, sqrt}

private val flaw = flawf ("top")                               // flaw function

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Compute the product of the critical value from the z-distribution (Standard
 *  Normal) and the standard deviation of the vector.
 *  @param sig  the standard deviation of the vector/sample
 *  @param p    the confidence level
 */
def z_sigma (sig: Double, p: Double = .95): Double =
    val pp = 1.0 - (1.0 - p) / 2.0                       // e.g., .95 --> .975 (two tails)
    val z  = random.Quantile.normalInv (pp)
    z * sig
end z_sigma

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Compute the confidence interval half-width (ihw) for the given confidence level
 *  using the z-distribution.
 *  @param sig  the standard deviation of the vector/sample
 *  @param n    the length of the vector/sample
 *  @param p    the confidence level
 */
def z_interval (sig: Double, n: Int, p: Double = .95): Double =
    z_sigma (sig, p) / sqrt (n)
end z_interval

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Test (2-tail) to see if two means (one given and one estimated) are significantly
 *  different, using the z-distribution.
 *  @param mu0   the given mean
 *  @param mu    the estimated mean of the vector/sample
 *  @param sig   the standard deviation of the vector/sample
 *  @param n     the length of the vector/sample
 *  @param p     the significance/confidence level
 *  @param show  whether to show details of the test
 */
def z_meanTest (mu0: Double, mu: Double, sig: Double, n: Int, p: Double = .95,
                show: Boolean = true): Boolean =
    val ihw = z_interval (sig, n, p)
    if show then println (s"z_meanTest: | $mu - $mu0 | <=? $ihw")
    abs (mu - mu0) <= ihw
end z_meanTest

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Compute the product of the critical value from the t-distribution and the
 *  standard deviation of the vector.
 *  @param sig  the standard deviation of the vector/sample
 *  @param df   the degrees of freedom
 *  @param p    the confidence level
 */
def t_sigma (sig: Double, df: Int, p: Double = .95): Double =
    if df < 1 then { flaw ("interval", "must have at least 2 observations"); return 0.0 }
    val pp = 1.0 - (1.0 - p) / 2.0                       // e.g., .95 --> .975 (two tails)
    val t  = random.Quantile.studentTInv (pp, df)
    t * sig
end t_sigma

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Compute the confidence interval half-width (ihw) for the given confidence level
 *  using the t-distribution.
 *  @param sig  the standard deviation of the vector/sample
 *  @param n    the length of the vector/sample
 *  @param p    the confidence level
 */
def t_interval (sig: Double, n: Int, p: Double = .95): Double =
    t_sigma (sig, n-1, p) / sqrt (n)
end t_interval

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Test (2-tail) to see if two means (one given and one estimated) are significantly
 *  different, using the t-distribution.
 *  @param mu0   the given mean
 *  @param mu    the estimated mean of the vector/sample
 *  @param sig   the standard deviation of the vector/sample
 *  @param n     the length of the vector/sample
 *  @param p     the significance/confidence level
 *  @param show  whether to show details of the test
 */
def t_meanTest (mu0: Double, mu: Double, sig: Double, n: Int, p: Double = .95,
                show: Boolean = true): Boolean =
    val ihw = t_interval (sig, n, p)
    if show then println (s"t_meanTest: | $mu - $mu0 | <=? $ihw")
    abs (mu - mu0) <= ihw
end t_meanTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Statistic` class is used to collect values and compute sample statistics
 *  on them (e.g., waiting time).  Contrast with `TimeStatistic` defined in
 *  TimeStatistic.scala.
 *  @param name      the name for this statistic (e.g., 'waitingTime')
 *  @param unbiased  whether the estimators are restricted to be unbiased
 */
class Statistic (val name: String = "stat", unbiased: Boolean = true):

    /** The number of samples
     */
    protected var n = 0

    /** Sum of the sample values
     */
    protected var sum = 0.0

    /** Sum of the sample absolute values
     */
    protected var sumAb = 0.0

    /** Sum of the sample values squared
     */
    protected var sumSq = 0.0

    /** The minimum sample value
     */
    protected var minX = MAX_VALUE

    /** The maximum sample value
     */
    protected var maxX = -MAX_VALUE

    private val flaw = flawf ("Statistic")                           // flaw function

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set accumulators.
     */
    def set (n_ : Int, sum_ : Double, sumAb_ : Double, sumSq_ : Double, minX_ : Double, maxX_ : Double): Unit =
        n     = n_
        sum   = sum_
        sumAb = sumAb_
        sumSq = sumSq_
        minX  = minX_
        maxX  = maxX_
    end set

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset accumulators.
     */
    def reset (): Unit =
        n     = 0
        sum   = 0.0
        sumAb = 0.0
        sumSq = 0.0
        minX  = MAX_VALUE
        maxX  = -MAX_VALUE
    end reset

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Tally the next value and update accumulators.
     *  @param x  the value to tally (e.g., time in sytem)
     */
    def tally (x: Double): Unit =
        n     += 1
        sum   += x
        sumAb += abs (x)
        sumSq += x * x
        if x < minX then minX = x
        if x > maxX then maxX = x
    end tally

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Tally the next vector of values and update accumulators.
     *  @param v  the vector of values to tally (e.g., time in sytem)
     */
    def tallyVec (v: VectorD): Unit = for x <- v do tally (x)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of samples.
     */
    inline def num: Int = n

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of samples as a double.
     */
    inline def nd: Double = n.toDouble

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the minimum value in sample.
     */
    inline def min: Double = if n == 0 then 0.0 else minX

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the maximum value in sample.
     */
    inline def max: Double = maxX

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute/estimate the sample mean.
     */
    def mean: Double = if n == 0 then 0.0 else sum / nd

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute/estimate the sample variance.  The denominator is one less for
     *  unbiased (n-1) vs. maximum likelihood (n) estimators.  Also use n for
     *  population variance.
     */
    def variance: Double =
        if n == 0 then 0.0 else (sumSq - sum*sum/nd) / (if unbiased then nd - 1.0 else nd)
    end variance

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute/estimate the sample standard deviation.
     */
    def stdev: Double = sqrt (variance)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute/estimate the mean square (ms), e.g., Mean Square Error (MSE).
     */
    def ms: Double = sumSq / nd

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute/estimate the mean absolue value (ma), e.g., Mean Absolute Error (MAE).
     */
    def ma: Double = sumAb / nd

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute/estimate the root mean square (rms), e.g., Root Mean Square Error (RMSE).
     */
    def rms: Double = sqrt (ms)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the confidence interval half-width for the given confidence level
     *  using the t-distribution.
     *  @param p  the confidence level
     */
    def interval (p: Double = .95): Double =
        val df = n - 1                                        // degrees of freedom
        t_sigma (stdev, df, p) / sqrt (nd)
    end interval

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the confidence interval half-width for the given confidence level
     *  using the z-distribution.
     *  @param p  the confidence level
     */
    def interval_z (p: Double = .95): Double =
        z_sigma (stdev, p) / sqrt (nd)
    end interval_z

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the values of this collector's accumulators.
     */
    def show: String = s"Statistic: $n, $sum, $sumAb, $sumSq, $minX, $maxX"

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the summary statistics as a row/Array.
     */
    def statRow: Array [Any] = Array (name, num, min, max, mean, stdev, interval ())

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate a row of statistical results as a string.
     */
    override def toString: String = 
        "| %11s | %5s | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |".format (
        name, num, min, max, mean, stdev, interval ())
    end toString

end Statistic


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Statistic` companion object provides additional values and functions.
 */
object Statistic:

    private val flaw = flawf ("Statistic")                      // flaw function

    /** The line separator
     */
    val line = "----------------------------------------------------------------------------------------"

    /** The statistical labels (column headers) as an Array
     */
    val label = Array ("name", "num", "min", "max", "mean", "stdev", "interval")

    /** The statistical labels (column headers) as a formatted String
     */
    val labels = "| %11s | %5s | %10s | %10s | %10s | %10s | %10s |".format (
                 "name", "num", "min", "max", "mean", "stdev", "interval")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a statistical object and set its accumulators and extreme values.
     *  @param n         the number of samples
     *  @param sum       sum of the sample values
     *  @param sumAb     sum of the sample absolute values
     *  @param sumSq     sum of the sample values squared
     *  @param minX      the minimum sample value
     *  @param maxX      the maximum sample value
     *  @param name      the name for this statistic (e.g., 'waitingTime')
     *  @param unbiased  whether the estimators are restricted to be unbiased
     */
    def apply (n: Int, sum: Double, sumAb: Double, sumSq: Double, minX: Double, maxX: Double,
               name: String = "stat", unbiased: Boolean = false): Statistic =
        val stat = new Statistic (name, unbiased)
        stat.set (n, sum, sumAb, sumSq, minX, maxX)
        stat
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Tally all the values and return the sum.
     *  @param x  the vector of values to tally (e.g., time in sytem)
     */
    def tallyAll (x: VectorD): Double =
        var s = 0.0
        for i <- x.indices do s += x(i)
        s
    end tallyAll

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Aggregate the sub-statistics by summing all accumulators and determining
     *  new overall extreme values.
     *  @param subStats  the list of sub-statistics
     *  @param name      the name for the aggregated statistic
     *  @param unbiased  whether the estimators are restricted to be unbiased
     */
    def aggregate (subStats: ListBuffer [Statistic], name: String = "a-stat",
                   unbiased: Boolean = false): Statistic =
        val m = subStats.size
        if m < 1 then { flaw ("aggregate", "there are no subparts to average"); return null }
        var n     = 0
        var sum   = 0.0
        var sumAb = 0.0
        var sumSq = 0.0
        var minX  = Double.MaxValue
        var maxX  = 0.0
        for s <- subStats do
            n     += s.n
            sum   += s.sum
            sumAb += s.sumAb
            sumSq += s.sumSq
            if s.minX < minX then minX = s.minX
            if s.maxX > maxX then maxX = s.maxX
        end for
        Statistic (n, sum, sumAb, sumSq, minX, maxX, name, unbiased)
    end aggregate

end Statistic


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `statisticTest` main function is used to test the `Statistic` class.
 *  > runMain scalation.mathstat.statisticTest
 */
@main def statisticTest (): Unit =

    val rv = SimpleUniform (0, 10)

    //:: Test ordinary sample statistics

    banner ("Test sample statistics")
    val stat1 = new Statistic ()
    for i <- 1 to 1000 do stat1.tally (rv.gen)

    println (Statistic.labels)
    println (stat1)
    println (stat1.show)
    println (s"ma = ${stat1.ma}, rms = ${stat1.rms}")

end statisticTest

