
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Sep 26 15:00:24 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Example Model: Bank for Process-Interaction Simulation
 */

package scalation
package simulation
package process
package example_1                                     // One-Shot

import scalation.random.{Exponential, Uniform}
import scalation.random.RandomSeeds.N_STREAMS

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runBank` function is used to launch the `BankModel` class.
 *  > runMain scalation.simulation.process.example_1.runBank
 */
@main def runBank (): Unit = new BankModel ()

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BankModel` class defines a simple process-interaction model of a bank
 *  where service is provided by one or more tellers.
 *  Caveat: must add 'from' and 'to' components before transport!!
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param nStop      the number arrivals before stopping
 *  @param stream     the base random number stream (0 to 999)
 */
class BankModel (name: String = "Bank", reps: Int = 1, animating: Boolean = true,
                 aniRatio: Double = 50.0, nStop: Int = 100, stream: Int = 0)
      extends Model (name, reps, animating, aniRatio):

    //--------------------------------------------------
    // Initialize Model Constants

    val lambda   = 12.0  // 6.0                       // customer arrival rate (per hour)
    val mu       = 7.5                                // customer service rate (per hour)
    val nTellers = 2     // 1                         // the number of bank tellers (servers)

    //--------------------------------------------------
    // Create Random Variables (RVs)

    val iArrivalRV = Exponential (HOUR / lambda, stream)
    val serviceRV  = Exponential (HOUR / mu, (stream + 1) % N_STREAMS)
    val moveRV     = Uniform (4 * MINUTE, 6 * MINUTE, (stream + 2) % N_STREAMS)

    //--------------------------------------------------
    // Create Model Components

    val entry     = Source ("entry", this, () => Customer (), 0, nStop, iArrivalRV, (100, 290))
    val tellerQ   = WaitQueue ("tellerQ", (330, 290))
    val teller    = Resource ("teller", tellerQ, nTellers, serviceRV, (350, 285))
    val door      = Sink ("door", (600, 290))
    val toTellerQ = Transport ("toTellerQ", entry, tellerQ, moveRV)
    val toDoor    = Transport ("toDoor", teller, door, moveRV)

    addComponent (entry, tellerQ, teller, door, toTellerQ, toDoor)

    //--------------------------------------------------
    // Specify Scripts for each Type of Simulation Actor

    case class Customer () extends SimActor ("c", this):

        def act (): Unit =
            toTellerQ.move ()
            if teller.busy then tellerQ.waitIn () else tellerQ.noWait ()
            teller.utilize ()
            teller.release ()
            toDoor.move ()
            door.leave ()
        end act

    end Customer

    simulate ()
    waitFinished ()
    Model.shutdown ()

end BankModel

