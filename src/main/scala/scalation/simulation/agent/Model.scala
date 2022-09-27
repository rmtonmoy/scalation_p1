/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep  7 15:05:06 EDT 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Base Model Class for Agent-Based Simulation
 */

package scalation
package simulation.agent

import scala.collection.mutable.{ArrayBuffer => VEC, PriorityQueue}

import scalation.animation.{AnimateCommand, CommandType}
import scalation.animation.CommandType.{CreateToken, MoveToken, DestroyToken}
import scalation.database.Identifiable
import scalation.database.graph.{EdgeType, PGraph, VertexType}
import scalation.mathstat.{Statistic, VectorD}
import scalation.simulation.{Completion, Coroutine}
import scalation.scala2d.Colors._
import scalation.scala2d.{RectangularShape, Shape}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Model` class maintains a property graph making up the model and
 *  controls the flow of entities (`SimAgent`s) through the model, following the
 *  agent-based simulation world-view.  It maintains a time-ordered priority queue
 *  to activate/re-activate each of the entities.  Each entity (`SimAgent`) is
 *  implemented as a `Coroutine` and may be thought of as running in its own thread.
 *  @param _name      the name of this simulation model
 *  @param reps       the number of independent replications
 *  @param startSim   the start time of this simulation
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 */
class Model (_name: String, reps: Int = 1, startSim: Double = 0.0,
             animating: Boolean = true, aniRatio: Double = 10.0)
      extends Coroutine (_name)
         with Identifiable (_name)
         with Completion:

    protected val graphMod = PGraph (name, Model.vertexTypes, Model.edgeTypes,
                                     animating, aniRatio)             // the graph model

    private val debug  = debugf ("Model", true)                       // debug function
    private val flaw   = flawf ("Model")                              // flaw function
    private val agenda = PriorityQueue.empty [SimAgent]               // time-ordered activation list

    private [agent] var clock      = startSim                         // the simulation clock
    private [agent] var simulating = false                            // the simulation clock
    private [agent] val log        = Monitor ("simulation")           // log for model execution

    val director = this

    debug ("init", s"name = $name, startSim = $startSim")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Execute the simulation (includes scheduling all Sources) returning summary
     *  statistics.
     */
    def simulate (): Unit =
        banner (s"start simulation $name at $startSim")
        graphMod.print ()
        if animating then graphMod.display (100000)                   // FIX - should be adaptive
//      return                                                        // end before simulating to only examine initial graph
        log.trace (this, "starts", this, clock)
        for source <- Source.sources do schedule (source)             // put all sources on agenda
        start ()                                                      // start the director thread/agent -> act ()
    end simulate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Cleanup the agenda and any stateful components.  Any agent left in the
     *  agenda or a wait queue must be terminated.  The model (i.e., the director)
     *  must be terminated as well.
     */
    def cleanup (): Unit =
        banner ("Model.cleanup in progress")

        println ("cleanup: agenda")
        while ! agenda.isEmpty do                                     // cleanup agents left on agenda
            val a = agenda.dequeue ()
            if a != this then
                println (s"cleanup: terminate agent $a in agenda")
                a.interrupt ()                                        // terminate all agents, except director
            end if
        end while

/*
        println ("cleanup: wait queues")
        for p <- parts do
            if p.isInstanceOf [WaitQueue] then                        // cleanup wait queues
                val w = p.asInstanceOf [WaitQueue]
                while ! w.isEmpty do
                    val a = w.dequeue ()
                    println (s"cleanup: terminate agent $a in $w")
                    a.interrupt ()                                    // terminate all agents in queue
                end while
            end if
        end for
*/
    end cleanup

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Schedule the agent to act (be activated) at agent.time (optionally delayed).
     *  @param agent  the agent to be scheduled
     *  @param delay  the amount of time to delay the agent's activation time
     */
    def schedule (agent: SimAgent, delay: Double = 0.0): Unit =
        if delay < 0.0 then
           flaw ("schedule", s"agent $agent delay time is negative: $delay")
           banner ("WARN")
        end if
        agent.time = clock + delay
        if agent.time < clock then                                    // out of order scheduling => WARN
           flaw ("schedule", s"agent $agent activation time < $clock")
           banner ("WARN")
        end if
//      debug ("schedule", s"now = $clock: schedule agent $agent")
        log.trace (this, "schedules agent", agent, clock)
        agenda += agent
    end schedule

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The model itself is an Agent (not an ordinary `SimAgent`) and may be
     *  thought of as the director.  The director iteratively manages the clock
     *  and the agenda of agents until the the agenda (priority queue) becomes empty
     *  or the number of live agents is greater than zero (gates are considered live).
     */
    def act (): Unit =
        simulating = true
        debug ("act", s"agenda = $agenda")

        while ! agenda.isEmpty && SimAgent.nAgents > 0 do             // scheduling loop
            val agent = agenda.dequeue ()                             // next from priority queue
            if agent.time < clock then                                // out of order execution => QUIT
               flaw ("act", s"agent $agent activation time < $clock")
               banner ("QUIT")
               return
            end if
            clock = agent.time                                        // advance the time
//          debug ("act", s"${this.me} resumes ${agent.me} at $clock")
            log.trace (this, "resumes", agent, clock)
            yyield (agent)                                            // director yields to agent
        end while

        log.trace (this, s"ends", null, clock)
        report ()                                                     // report results
        cleanup ()
        println (s"coroutine counts = $counts")
        log.trace (this, "terminates model", null, clock)
        simulating = false
        hasFinished ()                                                // signal via semaphore that simulation is finished
        yyield (null, true)                                           // yield and terminate the director
    end act

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Put a token command (CreateToken, MoveToken or DestroyToken) on the animation queue.
     *  @param agent  who is being animated
     *  @param what   what animation command
     *  @param color  the color the token
     *  @param shape  the shape of the token
     */
    def animate (agent: SimAgent, what: CommandType, color: Color = null,
                 shape: Shape = null): Unit =
        var eid   = agent.id
        if agent.isInstanceOf [Gate] then eid += 1                    // FIX - Gate's vertex is one more
        val label = agent.name
        val apos  = if what == MoveToken then agent.pos(0 to 2)       // agent's position (x, y)
                    else agent.pos                                    // (x, y, w, h)
//      debug ("animate", s">>> $label.$eid, $what, $color, $shape, $apos")
        if animating then graphMod.add_aniQ (AnimateCommand (what, eid, shape, label, true, color,
                                             apos.toArray, clock))
    end animate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare the order of agents based on their activation times.
     *  @param agent  the first agent in comparison
    private def orderedAgent (agent1: SimAgent): Ordered [SimAgent] =
        new Ordered [SimAgent]
            { def compare (agent2: SimAgent) = agent2.time compare agent1.time }
    end orderedAgent
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the statistical results of the simulation (statistics for each vertex).
     *  Includes both sample and time-persistent statistics.
     */
    def getStatistics: VEC [Statistic] =
        val stats = VEC [Statistic] ()
//      for v <- graphMod.vt(0).verts do v.addStats (stats)   // FIX
        stats
    end getStatistics

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Report on the statistical results of the simulation.
     */
    private def report (): Unit =
        println (Statistic.line)
        println (Statistic.labels)
        println (Statistic.line)
        for stat <- getStatistics do println (stat)
        println (Statistic.line)
    end report

end Model


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Model` companion object provides a shutdown method and methods to add
 *  vertex/edge types to the model.
 */
object Model:

    private val vertexTypes = VEC [VertexType] ()                     // collection of vertex types
    private val edgeTypes   = VEC [EdgeType] ()                       // collection of edge types

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add vertex type vt to the collection of vertex types.
     *  @param vt  the vertex type to add
     */
    def add (vt: VertexType): Unit = vertexTypes += vt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add edge type et to the collection of edge types.
     *  @param et  the edge type to add
     */
    def add (et: EdgeType): Unit = edgeTypes += et

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Shutdown the Model execution infrastructure (WARNING: this method should
     *  only be called right before program termination).  Make sure all threads
     *  have finished (e.g., call `waitFinished`), not just the main thread.
     *  If `shutdown` is not called, the application may hang.
     */
    def shutdown (): Unit = Coroutine.shutdown ()

end Model

