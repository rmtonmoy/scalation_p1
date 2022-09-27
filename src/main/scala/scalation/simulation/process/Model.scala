
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep  7 15:05:06 EDT 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Base Model Class for Process Simulation - Method of Independent Replications
 */

package scalation
package simulation
package process

import scala.collection.mutable.{LinkedHashMap, ListBuffer, PriorityQueue}
import scala.runtime.ScalaRunTime.stringOf

import scalation.animation.{AnimateCommand, CommandType, DgAnimator}
import scalation.animation.CommandType._
import scalation.mathstat.{Statistic, StatTable, t_interval, VectorD}
import scalation.scala2d.Colors._
import scalation.scala2d.Shape

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Model` class maintains a list of components making up the model and
 *  controls the flow of entities (`SimActor`s) through the model, following the
 *  process-interaction world-view.  It maintains a time-ordered priority queue
 *  to activate/re-activate each of the entities.  Each entity (`SimActor`) is
 *  implemented as a `Coroutine` and may be thought of as running in its own thread.
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param full       generate a full report with both sample and time-persistent statistics
 */
class Model (name: String, val reps: Int = 1, animating: Boolean = true, aniRatio: Double = 1.0,
             val full: Boolean = true)
      extends Coroutine (name) with Completion with Modelable with Component:

    initComponent (name, Array ())

    private val debug = debugf ("Model", false)                    // debug function
    private val flaw  = flawf ("Model")                            // flaw function
    private [process] val log  = Monitor ("simulation")            // log for model execution

    director = this
    debug ("constructor", s"make ${director.name} the director")

    /** The map of statistics vectors records the means of each replication
     */
    val statV = LinkedHashMap [String, VectorD] ()

    /** The stop time for the model
     */
    var stopTime = MAX_VALUE

    /** The number of actors created so far
     */
    var numActors: Int = 0

    /** The time at which the simulation is to begin
     */
    protected var startTime = 0.0

    /** The agenda of things to be done (time-ordered activation list)
     */
    protected val agenda = PriorityQueue.empty [SimActor]

    /** The currently acting actor (act one at a time)
     */
    protected var _theActor: SimActor = null

    /** List of Components making up the model
     */
    private val parts = ListBuffer [Component] ()

    /** The animation engine
     */
    private val dgAni = if animating then new DgAnimator ("Process Animator", black, white, aniRatio)
                        else null

    /** The animation engine's command queue
     */
    private val aniQ = if animating then dgAni.getCommandQueue
                       else null

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add component parts to the model.
     *  @param _parts  the component parts
     */
    def addComponent (_parts: Component*): Unit = for p <- _parts do parts += p

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add lists of component parts to the model.
     *  @param _parts  the lists of component parts
     */
    def addComponents (_parts: List [Component]*): Unit = for p <- _parts; q <- p do parts += q

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the current acting actor.
     */
    def theActor: SimActor = _theActor

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indicate whether the model has been stopped.
     */
    def stopped: Boolean = ! simulating

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the agenda and stateful components for next replication.
     */
    def reset (): Unit =
        banner ("Model.reset in progress")

        // reset the agenda - activation priority queue
        while ! agenda.isEmpty do agenda.dequeue ()             // clean out actors from agenda

        // reset stateful components
        for p <- parts do
            if p.isInstanceOf [Source] then                     // reset sources
                val s = p.asInstanceOf [Source]
                reschedule (s)
            end if
            if p.isInstanceOf [WaitQueue] then                  // reset wait queues
                val w = p.asInstanceOf [WaitQueue]
                while ! w.isEmpty do w.dequeue ()
            end if
        end for
    end reset

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset and aggregate all statistics.
     *  @param rep   the current replication (1, ... reps)
     *  @param rmax  the maximum number of replications/batches
     */
    def resetStats (rep: Int, rmax: Int = reps): Unit =
        if rep == 1 then
            for stat <- getStatistics do statV += stat.name -> new VectorD (rmax)
        end if 
        for stat <- getStatistics do
            statV (stat.name)(rep - 1) = stat.mean
            stat.reset ()
        end for
    end resetStats

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Execute the simulation (includes scheduling all Sources) returning summary
     *  statistics.
     */
    def simulate (_startTime: Double = 0.0): Unit =
        startTime = _startTime
        _clock = startTime
        log.trace (this, "starts", this, _clock)

        for p <- parts do
            log.trace (this, s"establish x = ${p.at(0)}, y = ${p.at(1)}", p, _clock)
            p.director = this
            for q <- p.subpart do q.director = this
            if p.isInstanceOf [Source] then reschedule (p.asInstanceOf [Source]) 
        end for

        start ()                                                // start the director thread/actor -> act ()
    end simulate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Cleanup the agenda and any stateful components.  Any actors left in the
     *  agenda or a wait queue must be terminated.  The model (i.e., the director)
     *  must be terminated as well.
     */
    def cleanup (): Unit =
        banner ("Model.cleanup in progress")

        println ("cleanup: agenda")
        while ! agenda.isEmpty do                               // cleanup actors left on agenda
            val a = agenda.dequeue ()
            if a != this then
                println (s"cleanup: terminate actor $a in agenda")
                a.interrupt ()                                  // terminate all actors, except director
            end if
        end while

        println ("cleanup: wait queues")
        for p <- parts do
            if p.isInstanceOf [WaitQueue] then                  // cleanup wait queues
                val w = p.asInstanceOf [WaitQueue]
                while ! w.isEmpty do
                    val a = w.dequeue ()
                    println (s"cleanup: terminate actor $a in $w")
                    a.interrupt ()                              // terminate all actors in queue
                end while
            end if
        end for
    end cleanup

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Schedule (first time) or reschedule (subsequent times) an actor to act.
     *  @param actor  the actor to be scheduled
     */
    def reschedule (actor: SimActor): Unit = agenda += actor

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The model itself is an Actor (not an ordinary `SimActor`) and may be
     *  thought of as the director.  The director iteratively manages the clock
     *  and the agenda of actors until the simulation flag becomes false
     *  or the agenda (priority queue) becomes empty.
     */
    def act (): Unit =
        log.trace (this, s"starts model for $reps replications", null, _clock)

        for rep <- 1 to reps do                                 // LOOP THROUGH REPLICATIONS
            _clock = startTime
            if rep == 1 && animating then display ()            // turn animation on (true) off (false)

            log.trace (this, s"starts rep $rep", null, _clock)
            simulating = _clock <= stopTime                     // simulate unless past stop time

            while simulating && ! agenda.isEmpty do             // INNER SCHEDULING LOOP
                _theActor = agenda.dequeue ()                   // next from priority queue
                if _theActor.actTime < clock then               // out of order execution => QUIT
                    flaw ("act", s"actor $_theActor activation time < $_clock")
                    println ("QUIT")
                    simulating = false
                else
                    _clock    = _theActor.actTime                   // advance the time
                    debug ("act", s"${this.me} resumes ${_theActor.me} at $clock")
                    log.trace (this, "resumes", _theActor, _clock)
                    yyield (_theActor)                              // director yields to actor
                end if
            end while

            simulating = false
            log.trace (this, s"ends rep $rep", null, _clock)

            fini (rep)                                          // post-run results
            if rep < reps then reset ()                         // reset for next replication
            resetStats (rep)                                    // reset and aggregate statistics
        end for

        cleanup ()
        reportV ()
        println (s"coroutine counts = $counts")
        log.trace (this, "terminates model", null, _clock)
        hasFinished ()                                          // signal via semaphore that simulation is finished
        yyield (null, true)                                     // yield and terminate the director
    end act

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the statistical results of the simulation (statistics for each part).
     *  This includes the sample/duration statistics and if 'full', time persistent
     *  statistics as well.
     */
    def getStatistics: ListBuffer [Statistic] =
        val stats = new ListBuffer [Statistic] ()
        for p <- parts do
            if p.composite then p.aggregate ()
            stats += p.durationStat
        end for
        if full then
            for p <- parts if p.persistentStat != null do
                stats += p.persistentStat
            end for
        end if
//      for st <- stats do println (s"getStatistics: ${st.show}")
        stats
    end getStatistics

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Put the components on the animation engine's queue.
     */
    def display (): Unit = for p <- parts do p.display ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Put a node/token command on the animation queue.
     *  @param who    who is being animated
     *  @param what   what animation command
     *  @param color  the color the node/token
     *  @param shape  the shape of the node/token
     *  @param at     the location of the node/token
     */
    def animate (who: Identifiable, what: CommandType, color: Color, shape: Shape,
                 at: Array [Double]): Unit =
        if animating then
            val eid   = who.id
            val label = who.name
            debug ("animate", s"$label.$eid, $what, $color, $shape, ${stringOf (at)}")
            aniQ.add (AnimateCommand (what, eid, shape, label, true, color, at, _clock))
        end if
    end animate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Put a edge command on the animation queue.
     *  @param who    who is being animated
     *  @param what   what animation command
     *  @param color  the color the edge
     *  @param shape  the shape of the edge
     *  @param from   the location of the origination node
     *  @param to     the location of the destination node
     *  @param at     the location of the edge (empty array => implicitly determined)
     */
    def animate (who: Identifiable, what: CommandType, color: Color, shape: Shape,
                 from: Component, to: Component, at: Array [Double] = Array ()): Unit =
        if animating then
            val eid   = who.id
            val label = who.name
            debug ("animate", s"$label.$eid, $what, $color,+ $shape, ${from.me} ${to.me}, ${stringOf (at)}")
            aniQ.add (AnimateCommand (what, eid, shape, label, true, color, at, _clock, from.id, to.id))
        end if
    end animate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare the order of actors based on their 'actTime's.
     *  @param actor1  the first actor in comparison
     */
    private def orderedActor (actor1: SimActor): Ordered [SimActor] =
        new Ordered [SimActor]
            { def compare (actor2: SimActor) = actor1.actTime compare actor2.actTime }
    end orderedActor

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Finish by producing statistical reports and optionally animation.
     *  Typically animation and reports in pop up window turned off for high
     *  replications and/or simulation optimization.
     *  @param rep  the replication number (1, ... reps)
     */
    protected def fini (rep: Int): Unit =
        report ()                                               // report in terminal
        if animating then
            reportF ()                                          // report in new window/frame
            if rep == 1 then dgAni.animate (0, 100000)          // only animate first rep
            dgAni.saveImage (DATA_DIR + name + ".png")
        end if
    end fini

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Report on the statistical results of a simulation run.
     */
    protected def report (): Unit =
        println (Statistic.line)
        println (Statistic.labels)
        println (Statistic.line)
        for stat <- getStatistics do println (stat)
        println (Statistic.line)
    end report

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Report on the statistical results of the overall simulation as recorded
     *  in statV (may include multiple replications/batches).
     *  @param showMeans  whether to show the individual run/batch means
     */
    protected def reportV (showMeans: Boolean = false): Unit =
        banner (s"Summary over replications/batches")
        println (Statistic.line)
        println (Statistic.labels)
        println (Statistic.line)
        if showMeans then println (s"reportV: MEANS statV = $statV")
        for (k, v) <- statV do
            val aStat = new Statistic (k)
            aStat.tallyVec (v)
            println (aStat)
        end for
        println (Statistic.line)
    end reportV

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Report on the statistical results of the simulation in a new GUI window/frame.
     */
    protected def reportF (): Unit = new StatTable (s"$name statistics", getStatistics)

end Model


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Model` companion object provides a shutdown method.
 */
object Model:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Shutdown the Model execution infrastructure (WARNING: this method should
     *  only be called right before program termination).  Make sure all threads
     *  have finished (e.g., call `waitFinished`), not just the main thread.
     *  If `shutdown` is not called, the application may hang.
     */
    def shutdown (): Unit = Coroutine.shutdown ()

end Model

