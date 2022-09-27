
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep 20 15:47:16 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Active Entity/Simulation Agent
 */

package scalation
package simulation.agent

import scala.math.{cos, sin}

import scalation.database.{Identifiable, Spatial, Temporal}
import scalation.database.graph.{Edge, Element, Topological}
import scalation.mathstat.VectorD

import simulation.Coroutine

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimAgent` companion object keeps track of the number of live agents.
 */
object SimAgent:

    private val wh       = (10.0, 10.0)                             // default display size
    private var _nAgents = 0                                        // current number of live agents

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the current number of live agents.  `Gate`s are not considered live,
     *  since they simply cycle until the simulation ends.  `Source`s are considered
     *  live, since they produce application agents and terminate when their production
     *  finishes.
     */
    def nAgents: Int = _nAgents

end SimAgent

import SimAgent._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimAgent` abstract class represents entities that are active in the model.
 *  The act abstract method, which specifies entity behavior, must be defined
 *  for each subclass.  Each `SimAgent` extends extends `Coroutine` and may be
 *  roughly thought of as running in its own thread.
 *  SimAgent adds knowledge of its own properties, the agents it follows, and the
 *  component it is currently at.
 *  @param _name     the name of this simulation agent (name from `Identifiable`)
 *  @param _time     the activation time for this agent
 *  @param director  the director controlling the model
 *  @param _pos      the position (Euclidean coordinates) of this agent
 *  @param loc       the location (graph coordinates) of this agent
 *  @param prop      the properties (Map) for this agent, e.g., color, weight
 */
abstract class SimAgent (_name: String, _time: Double, director: Model,
                         _pos: VectorD = VectorD (0, 0, wh._1, wh._2),
                         var loc: (Element, Double) = (null, 0.0),
                         val prop: Map [String, ValueType] = null)
         extends Coroutine (_name)
            with Identifiable (_name)
            with Temporal (_time)
            with Spatial (_pos)
            with Topological (loc._1, loc._2):                      // (element, distance)
//            with Ordered [SimAgent]:                              // ordered in time
//            with PartiallyOrdered [SimAgent]:                     // partially ordered in space

    var subtype = 0                                                 // indicator of subtype of this agent

    private val debug = debugf ("SimAgent", true)                   // debug function
    private val flaw  = flawf ("SimAgent")                          // flaw function

    private [agent] var fore: SimAgent = null                       // the SimAgent ahead
    private [agent] var aft: SimAgent  = null                       // the SimAgent behind
    private [agent] val arrivalT       = time                       // time agent started/arrived at the model
    private [agent] var nextTransport: Transport = null             // the next transport to move along

    if ! this.isInstanceOf [Gate] then _nAgents += 1                // increment the number of live agents

    debug ("init", s" <-- SimAgent $me created at time $time = ${director.clock}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare the activation times of the two agents, this and other.
     *  Their activation times are used to order them in the director's agenda
     *  (a time-based HPF priority queue).
     *  @param other  the other agent to compare with this
    override def compare (other: SimAgent): Int = other.time compare time
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the initial position of this agent using the given x and y coordinates.
     *  Add half width and height to position center rather than top-left.
     *  @param x  the x-coordinate
     *  @param y  the y-coordinate
     */
    def setPos (x: Double, y: Double): Unit =
        pos(0) = x + wh._1 / 2.0                                    // x + half width
        pos(1) = y + wh._2 / 2.0                                    // y + half height
    end setPos

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the position of this agent by moving distance units in the direction
     *  determined by the angle.
     *  @param distance  the incremental distance to move
     *  @param angle     the angle/direction to move in (0 radians => move right)
     */
    def updatePos (distance: Double, angle: Double): Unit =
        pos(0) += cos (angle) * distance
        pos(1) += sin (angle) * distance
    end updatePos

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the location of this agent by moving distance units along the
     *  given edge.
     *  @param distance  the incremental distance to move along the edge
     *  @param edge      the current edge (typically the same one it was on)
     */
    def updateLoc (distance: Double, edge: Edge = null): Unit =
        loc = if edge == null then (loc._1, loc._2 + distance)      // same edge
        else (edge, distance)                                       // new edge
    end updateLoc

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The abstract method, act, is defined in each subclass to provide specific
     *  behavior.
     */
    def act (): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare two spatial objects based on their space coordinates.
     *  @param other  the other item to compare with this item
     */
    override def tryCompareTo [B >: SimAgent: AsPartiallyOrdered] (other: B): Option [Int] =
        val oth = other.asInstanceOf [SimAgent]
        pos tryCompareTo oth.pos
    end tryCompareTo

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the `SimAgent` ahead (e.g., to follow).  Must be nearest ahead in your element.
     *  @param d  the maximum allowed distance to be considered in the neighborhood
     */
    def findFore (d: Double): SimAgent =
        var minDiff = Double.MaxValue
        var minA: SimAgent = null 

        val agents = if d > 0.0 then neighbors (d) else neighbors
        for a <- agents do
            val diff = a.dist - dist
            if diff > 0 && diff < minDiff then
               minDiff = diff
               minA    = a.asInstanceOf [SimAgent]
            end if
        end for
        minA
    end findFore

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reorient the simulation agent behind when making a change (e.g., exit,
     *  turn or lane change).
     */
    def reorient (): Unit = aft.fore = fore

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Yield control to the director so the director can take the next action.
     *  @param quit  the flag indicating whether this agent is done
     */
    def yieldToDirector (quit: Boolean = false): Unit =
        director.log.trace (this, "resumes", director, director.clock)
        if quit && ! this.isInstanceOf [Gate] then _nAgents -= 1               // decrement the number of live agents
        yyield (director, quit)
    end yieldToDirector

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the simulation agent to a string.
     */
    override def toString: String = s"SimAgent ($me, $time, $pos, $loc)"

end SimAgent


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simAgentTest` function tests the `SimAgent` class.
 *  > runMain scalation.simulation.agent.simAgentTest
 */
@main def simAgentTest (): Unit =

   val director = Model ("TestModel")

   case class TestAgent () extends SimAgent ("TestAgent", 0.0, director):
        def act (): Unit =
            println ("act")
            yieldToDirector (true)
        end act
   end TestAgent

   val myAgent = TestAgent ()
   println (s"myAgent = $myAgent")
   myAgent.start ()

end simAgentTest

