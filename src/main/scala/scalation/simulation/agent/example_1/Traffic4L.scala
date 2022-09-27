
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep 20 15:47:16 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Example Model: Traffic4L (Four-Lane) for Agent-Based Simulation
 */

package scalation
package simulation.agent
package example_1

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.random.{Bernoulli, Sharp, Uniform}
import scalation.random.RandomSeeds.N_STREAMS

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runTraffic4L` function is used to launch the `Traffic4LModel` class.
 *  > runMain scalation.simulation.agent.example_1.runTraffic4L
 */
@main def runTraffic4L (): Unit = new Traffic4LModel ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Traffic4LModel` class simulates an intersection with four traffic lights
 *  `Gates` and four roads.  Each road consists of two routes with one in each
 *  direction.  Each `Route` has two lanes (`Transport`s).
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param startSim   the start time of the simulation
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param nStop      the number arrivals before stopping
 *  @param stream     the base random number stream (0 to 999)
 */
class Traffic4LModel (name: String = "Traffic4L", reps: Int = 1, startSim: Double = 0.0,
                    animating: Boolean = true, aniRatio: Double = 8.0,
                    nStop: Int = 20, stream: Int = 0)
      extends Model (name, reps, startSim, animating, aniRatio):

    //--------------------------------------------------
    // Initialize Model Constants

    val iaTime  = (4000.0, 6000.0)                    // (lower, upper) on inter-arrival time
    val onTime  = 8000.0                              // on (green-light) time for North-South traffic
    val offTime = 6000.0                              // off (red-light) time for North-South traffic
    val mvTime  = (2900.0, 3100.0)                    // (lower, upper) on move time

    //--------------------------------------------------
    // Create Random Variates (RVs)

    val iArrivalRV = Uniform (iaTime, stream)
    val onTimeRV   = Sharp (onTime, (stream + 1) % N_STREAMS)
    val offTimeRV  = Sharp (offTime, (stream + 2) % N_STREAMS)
    val moveRV     = Uniform (mvTime, (stream + 3) % N_STREAMS)
    val laneRV     = Bernoulli ((stream + 4) % N_STREAMS)

    //--------------------------------------------------
    // Create the Graph Model: Vertices and Edges

    val base = (800.0, 400.0)

    val source = Source.group (this, 0.0, () => Car (), nStop, null, base,
                              ("s1N", iArrivalRV, 0, (-16, -250)),             // from North
                              ("s1E", iArrivalRV, 1, (250, -16)),
                              ("s1S", iArrivalRV, 2, (16, 250)),
                              ("s1W", iArrivalRV, 3, (-250, 16)))

    val queue = WaitQueue.group (this, Int.MaxValue, null, base,
                                ("q1N", (-16, -40)),                         // before North light 
                                ("q1E", (40, -16)),
                                ("q1S", (16, 40)),
                                ("q1W", (-40, 16)))

    val light = Gate.group (this, 0.0, onTimeRV, offTimeRV, 15, null, base,
                           ("l1N", queue(0), (-16, 16)),                    // traffic from North
                           ("l1E", queue(1), (-16, -16)),
                           ("l1S", queue(2), (16, -16)),
                           ("l1W", queue(3), (16, 16)))

    val sink = Sink.group (this, null, base,
                          ("k1N", (16, -250)),
                          ("k1E", (250, 16)),
                          ("k1S", (-16, 250)),                           // end for North traffic
                          ("k1W", (-250, -16)))

    val road = VEC [Route] ()
    for i <- source.indices do
        road += Route (s"ra$i", this, 2, source(i).vert, queue(i), moveRV)
    end for
    for i <- source.indices do
        road += Route (s"rb$i", this, 2, light(i).vert, sink((i + 2) % 4), moveRV)
    end for

    //--------------------------------------------------
    // Specify Scripts for each Type of Simulation Agent

    case class Car () extends SimAgent ("c", director.clock, this):

        def act (): Unit =
            println (s"Car $me running")
            val i = subtype                                // from North (0), East (1), South (2), West (3)
            val l = laneRV.igen                            // randomly select lane l
            println (s"Car $me move on road $i lane $l")
            road(i).lane(l).move (this)                    // move half way down lane l
            if light(i).open (this) then
                println (s"Car $me skips queue $i")
                queue(i).noWait (this)                     // skip the queue
            else
                println (s"Car $me wait in queue $i")
                queue(i).waitIn (this)                     // stop and wait for red light
            end if
            println (s"Car $me move on road ${i+4} lane $l")
            road(i + 4).lane(l).move (this)                // add 4 for next segment
            sink((i + 2) % 4).leave (this)                 // end at this sink
        end act

    end Car

/*
            road(i).lane(l).move (this, 0.5)               // move half way down lane l
            val l2 = (l + 1) % 2                           // index of other lane
            road(i).changeLane (this, l, l2)               // change to lane l2
            road(i).lane(l2).move (this, 0.5)              // move the rest of the way down lane l2
            if light(i).shut (this) then queue(i).waitIn (this)   // stop and wait for red light
            road(i + 4).lane(l2).move (this)               // add 4 for next segment
            sink((i + 2) % 4).leave (this)                 // end at this sink

    case class Car (prop: Map [String, ValueType]) extends SimAgent ("c", this, prop):
    val turnRV     = Randi (-1, 1, (stream + 5) % N_STREAMS)
    def prop_gen = Map ("turn" -> turnRV.gen)
    val source = { val p = prop_gen; Source.group (this, () => Car (p), nStop, (800, 250),
            prop("turn") match
            case -1 => road(i + 3).lane(l).move ()         // left turn
                       sink((i + 1) % 4).leave ()
            case  0 => road(i + 4).lane(l).move ()         // straight
                       sink((i + 2) % 4).leave ()
            case  _ => road((i + 5) % 8).lane(l).move ()   // right turn
                       sink((i + 3) % 4).leave ()
            end match
*/

    simulate ()
    waitFinished ()
    Model.shutdown ()

end Traffic4LModel

