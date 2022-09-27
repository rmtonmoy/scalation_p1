
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Sep 26 15:00:24 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Example Model: Emergency Department for Process-Interaction Simulation
 */

package scalation
package simulation
package process
package example_1                                     // One-Shot

import scalation.random.Uniform
import scalation.random.RandomSeeds.N_STREAMS

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runEmerDept` function is used to launch the `EmerDeptModel` class.
 *  > runMain scalation.simulation.process.example_1.runEmerDept
 */
@main def runEmerDept (): Unit = new EmerDeptModel ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `EmerDeptModel` class defines a simple process-interaction model of an
 *  Emergency Department model where service is provided by one or more nurses and
 *  one or more doctors.   A patient will first see a nurse and then a doctor.
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param nStop      the number arrivals before stopping
 *  @param stream     the base random number stream (0 to 999)
 */
class EmerDeptModel (name: String = "EmerDept", reps: Int = 1, animating: Boolean = true,
                     aniRatio: Double = 2.0, nStop: Int = 100, stream: Int = 0)
      extends Model (name, reps, animating, aniRatio):

    //--------------------------------------------------
    // Initialize Model Constants

    val ia_time    = (2000.0, 4000.0)                 // patient inter-arrival time range
    val nurse_ser  = (7000.0, 9000.0)                 // nurse service time range
    val doctor_ser = (5000.0, 7000.0)                 // doctor service time range
    val mv_time    = (900.0, 1100.0)                  // move time range
    val nurses     = 3                                // number of nurses
    val doctors    = 2                                // number of doctors

    //--------------------------------------------------
    // Create Random Variables (RVs)

    val iArrivalRV = Uniform (ia_time, stream)
    val nurseRV    = Uniform (nurse_ser, (stream + 1) % N_STREAMS)
    val doctorRV   = Uniform (doctor_ser, (stream + 2) % N_STREAMS)
    val moveRV     = Uniform (mv_time, (stream + 3) % N_STREAMS)

    //--------------------------------------------------
    // Create Model Components

    val entry     = Source ("entry", this, () => Patient (), 0, nStop, iArrivalRV, (70, 340))
    val nurseQ    = WaitQueue ("nurseQ", (230, 340))
    val nurse     = Resource ("nurse", nurseQ, nurses, nurseRV, (250, 335))
    val doctorQ   = WaitQueue ("doctorQ", (440, 340))
    val doctor    = Resource ("doctor", doctorQ, doctors, doctorRV, (460, 335))
    val door      = Sink ("door", (640, 340))
    val toNurseQ  = Transport ("toNurseQ", entry, nurseQ, moveRV)
    val toDoctorQ = Transport ("toDoctorQ", nurse, doctorQ, moveRV)
    val toDoor    = Transport ("toDoor", doctor, door, moveRV)

    addComponent (entry, nurseQ, nurse, doctorQ, doctor, door, toNurseQ, toDoctorQ, toDoor)

    //--------------------------------------------------
    // Specify Scripts for each Type of Simulation Actor

    case class Patient () extends SimActor ("p", this):

        def act (): Unit =
            toNurseQ.move ()
            if nurse.busy then nurseQ.waitIn ()
            nurse.utilize ()
            nurse.release ()
            toDoctorQ.move ()
            if doctor.busy then doctorQ.waitIn ()
            doctor.utilize ()
            doctor.release ()
            toDoor.move ()
            door.leave ()
        end act

    end Patient

    simulate ()
    waitFinished ()
    Model.shutdown ()

end EmerDeptModel

