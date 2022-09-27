
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep  7 15:05:06 EDT 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Monitor is used for Tracing Action/Events
 */

package scalation
package simulation.agent

import scalation.database.Identifiable

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Monitor` class is used to trace the actions/events in the models.
 *  @param project  the project to be monitored
 */
case class Monitor (project: String = "simulation"):

    /** Flag indicating whether tracing is on (initially on)
     */
    private var tracing = true 

    /** Use `EasyWriter` to make it easy to switch from standard out to a (log) file
     */
    private val ew = new EasyWriter (project, "monitor.log")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Toggle output destination from default of (log) file to standard output. etc.
     */
    def toggle (): Unit = ew.toggle ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Turn tracing off.
     */
    def traceOff (): Unit = tracing = false

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Turn tracing back on.
     */
    def traceOn (): Unit = tracing = true

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Trace an action/event.
     *  @param who   who caused the action
     *  @param what  what was the action
     *  @param whom  whom did the action effect
     *  @param when  when was the action taken
     */
    def trace (who: Identifiable, what: String, whom: Identifiable, when: Double): Unit =
        if tracing then
            if whom == null then
                ew.println (s"${who.me} \t $what \t at time $when.")
            else
                ew.println (s"${who.me} \t $what \t ${whom.me} \t at time $when.")
            end if
        end if
    end trace

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Finish up by flushing and closing the log.
     */
    def finish (): Unit = ew.finish ()

end Monitor


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MonitorTest` function is used to test the `Monitor` class.
 *  > runMain scalation.simulation.agent.monitorTest
 */
@main def monitorTest (): Unit =

    object Mon extends Identifiable ("MonitorTest")

    val log = Monitor ("simulation")
    log.trace (Mon, "writes log entry", null, 0.0)
    log.finish ()

end monitorTest

