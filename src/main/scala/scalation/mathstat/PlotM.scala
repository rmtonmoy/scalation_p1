
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Michael Cotterell
 *  @version 2.0
 *  @date    Sun Nov 15 15:05:06 EDT 2009
 *  @see     LICENSE (MIT style license file). 
 *
 *  @title   Plot Rows in Matrix y versus x
 */

package scalation
package mathstat

import scala.math.{ceil, floor, min, pow, round}

import scalation.scala2d._
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PlotM` class takes an x vector and a y matrix of data values and plots
 *  the (x, y_i) data points for each row y_i of the matrix.
 *  @param x_      the x vector of data values (horizontal)
 *  @param y_      the y matrix of data values where y(i) is the i-th vector (vertical)
 *  @param label   the label/legend/key for each curve in the plot
 *  @param _title  the title of the plot
 *  @param lines   flag for generating a line plot
 */
class PlotM (x_ : VectorD, y_ : MatrixD, var label: Array [String] = null,
            _title: String = "PlotM y_i vs. x for each i", lines: Boolean = false)
      extends VizFrame (_title, null):

    val xa: VectorD = if x_ == null then VectorD.range (0, y_.dim2) else x_

    private val EPSILON   = 1E-9
    private val frameW    = getW
    private val frameH    = getH
    private val offset    = 70
    private val baseX     = offset
    private val baseY     = frameH - offset
    private val stepsX    = 10
    private val stepsY    = 10
    private val minX      = floor (xa.min)
    private val maxX      = ceil (xa.max + EPSILON)
    private val minY      = floor (y_.mmin)
    private val maxY      = ceil (y_.mmax)
//  private val maxY      = ceil (y_.mmax + EPSILON)
    private val deltaX    = maxX - minX
    private val deltaY    = maxY - minY
    private val diameter  = 4
    private val dot       = Ellipse ()
    private val axis      = Line (0, 0, 0, 0)

    if label == null then label = defaultLabels

    println (s"x-axis: minX = $minX, maxX = $maxX")
    println (s"y-axis: minY = $minY, maxY = $maxY")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return default labels for y-vector.
     */
    def defaultLabels: Array [String] =
        val l = new Array [String] (y_.dim)
        for i <- y_.indices do l(i) = "Vector" + i
        l
    end defaultLabels

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a canvas on which to draw the plot.
     */
    class CanvasP extends Panel:
        setBackground (white)
        val colors = Array (red, green, blue, black, yellow, cyan, magenta)

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Paint the canvas by plotting the data points.
         *  @param gr  low-resolution graphics environment
         */
        override def paintComponent (gr: Graphics): Unit =
            super.paintComponent (gr)
            val g2d = gr.asInstanceOf [Graphics2D]            // use hi-res

            var x_pos = 0
            var y_pos = 0
            var step  = 0.0

            //:: Draw the axes

            g2d.setPaint (black)
            g2d.setStroke (new BasicStroke (2.0f))
            axis.setLine (baseX - 1, baseY + 1, baseX + 15 + frameW - 2 * offset, baseY + 1)
            g2d.draw (axis)
            axis.setLine (baseX - 1, offset - 15, baseX - 1, baseY + 1)
            g2d.draw (axis)

            //:: Draw the labels on the axes

            y_pos = baseY + 15
            step  = deltaX / stepsX.asInstanceOf [Double]       // for x-axis
            for j <- 0 to stepsX do
                val x_val = clip (minX + j * step)
                x_pos = offset - 8 + j * (frameW - 2 * offset) / stepsX
                g2d.drawString (x_val, x_pos, y_pos)
            end for

            x_pos = baseX - 30
            step  = deltaY / stepsY.asInstanceOf [Double]       // for y-axis
            for j <- 0 to stepsY do
                val y_val = clip (maxY - j * step)
                y_pos = offset + 2 + j * (frameH - 2 * offset) / stepsY
                g2d.drawString (y_val, x_pos, y_pos)
            end for

            //:: Draw the color keys below the x-axis

            g2d.drawString ("Key:", offset, frameH - 30)

            //:: Draw the dots for the data points being plotted

            for i <- 0 until y_.dim do
                val y_i = y_(i)
//              val color = randomColor (i)
                val color = colors (i % colors.size)
                g2d.setPaint (color)
                if i < label.length then g2d.drawString (label(i), offset * (i + 2), frameH - 30)

                var px_pos = 0           // previous x
                var py_pos = 0           // previous y

                for j <- xa.indices do
                    val xx = round ((xa(j) - minX) * (frameW - 2 * offset).asInstanceOf [Double])
                    x_pos = (xx / deltaX).asInstanceOf [Int] + offset
                    val yy = round ((maxY - y_i(j)) * (frameH - 2 * offset).asInstanceOf [Double])
                    y_pos = (yy / deltaY).asInstanceOf [Int] + offset
                    dot.setFrame (x_pos, y_pos, diameter, diameter)         // x, y, w, h
                    g2d.fill (dot)

                    // connect with lines
                    if j != 0 && lines then
                        g2d.setStroke (new BasicStroke (1.0f))
                        g2d.drawLine (px_pos+1, py_pos+1, x_pos+1, y_pos+1)
                    end if

                    px_pos = x_pos // update previous x
                    py_pos = y_pos // update previous y

                end for
            end for
        end paintComponent

    end CanvasP

/*
    import java.awt.image.BufferedImage
    import java.io.File
    import javax.imageio.ImageIO
*/

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Save the plot in a file as an image.
     *  @see stackoverflow.com/questions/5655908/export-jpanel-graphics-to-png-or-gif-or-jpg/39490801#39490801
     *  @param fname  the name of the file to save the plot in
    def saveImage (fname: String): Unit =
        val bimg = new BufferedImage (getSize ().width, getSize ().height, BufferedImage.TYPE_INT_ARGB)
        val gr   = bimg.createGraphics ()
        paint (gr)
        gr.dispose ()
        ImageIO.write (bimg, "png", new File (fname))
    end saveImage
     */

    {
        getContentPane ().add (new CanvasP ())
        setVisible (true)
    } // primary constructor

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert value to string and cut out the first four characters.
     *  @param x  the value to convert and cut
     */
    def clip (x: Double): String =
        val s = x.toString 
        s.substring (0, min (s.length, 4))
    end clip

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this `PlotM` object to a string.
     */
    override def toString: String = s"PlotM (y = $y_ vs. x = $xa)"

end PlotM


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PlotM` companion object provides a builder method for plotting several
 *  y vectors versus an x vector.
 */
object PlotM:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a plot of several y vectors versus an x vector.
     *  @param x  the x vector of data values (horizontal)
     *  @param y  one or more vectors of values where y(i) is the i-th vector (vertical)
     */
    def apply (x: VectorD, y: VectorD*): PlotM =
        val yy = new MatrixD (y.length, x.dim)
        for i <- 0 until y.length do yy(i) = y(i)
        new PlotM (x, yy)
    end apply

end PlotM


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `plotMTest` main functionb is used to test the `PlotM` class.
 *  > runMain scalation.mathstat.plotMTest
 */
@main def plotMTest (): Unit =

    val x = new VectorD (200)
    val y = new MatrixD (5, 200)

    for i <- 0 until 200 do
        x(i)    = (i - 100) / 10.0
        y(0, i) = 10.0 * x(i)
        y(1, i) = pow (x(i), 2)
        y(2, i) = .1 * pow (x(i), 3)
        y(3, i) = .01 * pow (x(i), 4)
        y(4, i) = .001 * pow (x(i), 5)
    end for
    val plot = new PlotM (x, y, Array ("Linear", "Quadratic", "Cubic", "Quartic", "Quintic"))
    println (s"plot = $plot")

end plotMTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `plotMTest2` main function is used to test the `PlotM` class.  This
 *  version also plots lines connecting the points.
 *  > runMain scalation.mathstat.plotMTest2
 */
@main def plotMTest2 (): Unit =

    val x = new VectorD (200)
    val y = new MatrixD (6, 200)

    for i <- 0 until 200 do
        x(i)    = (i - 100) / 10.0
        y(0, i) = 10.0 * x(i)
        y(1, i) = pow (x(i), 2)
        y(2, i) = .1 * pow (x(i), 3)
        y(3, i) = .01 * pow (x(i), 4)
        y(4, i) = .001 * pow (x(i), 5)
        y(5, i) = .0001 * pow (x(i), 6)
    end for
    val plot = new PlotM (x, y, Array ("Linear", "Quadratic", "Cubic", "Quartic", "Quintic"), lines = true)
    println (s"plot = $plot")

    writeImage (DATA_DIR + "plotm.png", plot)

end plotMTest2

