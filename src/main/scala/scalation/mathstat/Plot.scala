
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Michael Cotterell, Aiman Munir
 *  @version 2.0
 *  @date    Sun Nov 15 15:05:06 EDT 2009
 *  @see     LICENSE (MIT style license file). 
 *
 *  @title   Plot Vectors y and z vs. x
 */

package scalation
package mathstat

import scala.math.{ceil, floor, min, pow, round}

import scalation.scala2d._
import scalation.scala2d.BorderLayout._
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Plot` class takes 'x' and 'y' vectors of data values and plots the '(x, y)'
 *  data points.  Optionally, a 'z' vector may be plotted with 'y'.  Note, axes are
 *  determined by the 'x' and 'y' vectors only.  For more vertical vectors use `PlotM`.
 *------------------------------------------------------------------------------
 *  Zoom functionality has two options:
 *  When clicked on the plot label the value on that label will be selected as min/max value.
 *  By default, the clicked value on x and y axis will be chosen as min value.
 *  To change the value to max the resetLabel with title "Switch min and max value" can be used.
 *------------------------------------------------------------------------------
 *  @param x       the x vector of data values (horizontal), use null to use y's index
 *  @param y       the y vector of data values (primary vertical)
 *  @param z       the z vector of data values (secondary vertical) to compare with y
 *  @param _title  the title of the plot
 *  @param lines   flag for generating a line plot
 */
class Plot (x: VectorD, y: VectorD, z: VectorD = null, _title: String = "Plot y vs. x", lines: Boolean = false)
      extends VizFrame (_title, null):

    val resetLabel = new Label ()
    resetLabel.setText ("Reset Plot")
        
    val maxLabel = new Label ()
    maxLabel.setText ("Switch max and min value")

    val xx: VectorD = if x == null then VectorD.range (0, y.dim) else x
    val canvas      = new Canvas (xx, y, z, getW, getH, lines)
    getContentPane.add (resetLabel, BorderLayout.NORTH)
    getContentPane.add (maxLabel, BorderLayout.AFTER_LAST_LINE)
    getContentPane.add (canvas, BorderLayout.CENTER)
    setVisible (true)

    // reset plot to original values
    resetLabel.addMouseListener (new MouseListener () {
        override def mouseClicked  (mouseEvent: MouseEvent): Unit = canvas.resetMinMax ()
        override def mousePressed  (mouseEvent: MouseEvent): Unit = {}
        override def mouseReleased (mouseEvent: MouseEvent): Unit = {}
        override def mouseEntered  (mouseEvent: MouseEvent): Unit = {}
        override def mouseExited   (mouseEvent: MouseEvent): Unit = {}
    })

    // mouse listener that switches max and min value
    maxLabel.addMouseListener (new MouseListener () {
        override def mouseClicked  (mouseEvent: MouseEvent): Unit = canvas.setMaxMinValue ()
        override def mousePressed  (mouseEvent: MouseEvent): Unit = {}
        override def mouseReleased (mouseEvent: MouseEvent): Unit = {}
        override def mouseEntered  (mouseEvent: MouseEvent): Unit = {}
        override def mouseExited   (mouseEvent: MouseEvent): Unit = {}
    })

end Plot


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Plot` companion object provides factory methods for creating plots.
 */
object Plot:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a plot object from integer vectors.
     *  @param x       the x vector of data values (horizontal)
     *  @param y       the y vector of data values (primary vertical)
     *  @param z       the z vector of data values (secondary vertical) to compare with y
     *  @param _title  the title of the plot
     *  @param lines   flag for generating a line plot
     */
    def apply (x: VectorI, y: VectorI, z: VectorI = null, _title: String, lines: Boolean = false): Plot =
        new Plot (x.toDouble, y.toDouble, if z == null then null else z.toDouble, _title, lines)
    end apply

end Plot


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FramelessPlot` class is used for embedded applications.
 *  @param x       the x vector of data values (horizontal)
 *  @param y       the y vector of data values (primary vertical)
 *  @param z       the z vector of data values (secondary vertical) to compare with y
 *  @param width   the width
 *  @param height  the height
 */
class FramelessPlot (x: VectorD, y: VectorD, z: VectorD = null, var width: Int = 840, var height: Int = 480):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Dynamically create and return a drawing canvas.
     */
    def canvas: Canvas = new Canvas (x, y, z, width, height)

end FramelessPlot
 

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Canvas` class provides a canvas on which to draw the plot.
 *  @param x       the x vector of data values (horizontal)
 *  @param y       the y vector of data values (primary vertical)
 *  @param z       the z vector of data values (secondary vertical) to compare with y
 *  @param width   the width
 *  @param height  the height
 *  @param lines   flag for generating a line plot
 */
class Canvas (x: VectorD, y: VectorD, z: VectorD, width: Int, height: Int, lines: Boolean = false)
      extends Panel:

    private val EPSILON   = 1E-9
    private val SCALE     = 10                             // FIX - pass as a parameter
    private val frameW    = width
    private val frameH    = height
    private val offset    = 80
    private val baseX     = offset
    private val baseY     = frameH - offset
    private val stepsX    = 10
    private val stepsY    = 10

    private var minX      = floor (SCALE * x.min) / SCALE.toDouble
    private var maxX      = ceil (x.max + EPSILON)
    private var minY      = floor (SCALE * y.min) / SCALE.toDouble
    private var maxY      = ceil (y.max)
//  private var maxY      = ceil (y.max + EPSILON)

    private var deltaX    = maxX - minX
    private var deltaY    = maxY - minY

    private val diameter  = 4
    private val dot       = Ellipse ()
    private val axis      = Line (0, 0, 0, 0)

    private var setMin    = true
    private val origMinX  = minX
    private val origMaxX  = maxX
    private val origMinY  = minY
    private val origMaxY  = maxY
    private var setMax    = false

    setBackground (white)

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
        axis.setLine (baseX - 1, baseY + 1, baseX + 10 + frameW - 2 * offset, baseY + 1)
        val linex = axis.getBounds
        g2d.draw (axis)
        axis.setLine (baseX - 1, offset - 10, baseX - 1, baseY + 1)
        val liney = axis.getBounds
        g2d.draw (axis)

        //:: Draw the labels on the axes

        var xlabels = List [Rectangle2D] ()
        var xValues = List [String] ()
        y_pos = baseY + 15
        step  = deltaX / stepsX.asInstanceOf [Double]       // for x-axis
        for j <- 0 to stepsX do
            val x_val = clip (minX + j * step)
            x_pos = offset - 8 + j * (frameW - 2 * offset) / stepsX
            g2d.drawString (x_val, x_pos, y_pos)

            xValues = xValues.::(x_val)                // store  the postion of x y labels to know the postion of click
            var resumeRect = g2d.getFontMetrics.getStringBounds (x_val, g2d)
            resumeRect.setRect (x_pos, y_pos - g2d.getFontMetrics ().getAscent (),
                                resumeRect.getWidth (), resumeRect.getHeight ())
            xlabels = xlabels.::(resumeRect)
        end for

        var ylabels = List [Rectangle2D] ()
        var yValues = List [String] ()
        x_pos = baseX - 30
        step  = deltaY / stepsY.asInstanceOf [Double]       // for y-axis
        for j <- 0 to stepsY do
            val y_val = clip (maxY - j * step)
            y_pos = offset + 2 + j * (frameH - 2 * offset) / stepsY
            g2d.drawString (y_val, x_pos, y_pos)

            yValues = yValues.::(y_val)
            var resumeRect = g2d.getFontMetrics.getStringBounds (y_val, g2d)
            resumeRect.setRect (x_pos, y_pos - g2d.getFontMetrics ().getAscent (),
                                resumeRect.getWidth (), resumeRect.getHeight ())
            ylabels = ylabels.::(resumeRect)
        end for

        //:: Draw the dots for the data points being plotted

        var px_pos = 0                 // previous x
        var py_pos = 0                 // previous y

        for i <- 0 until y.dim do
            val xx = round ((x(i) - minX) * (frameW - 2 * offset).asInstanceOf [Double])
            x_pos = (xx / deltaX).asInstanceOf [Int] + offset
            val yy = round ((maxY - y(i)) * (frameH - 2 * offset).asInstanceOf [Double])
            y_pos = (yy / deltaY).asInstanceOf [Int] + offset
            dot.setFrame (x_pos, y_pos, diameter, diameter)         // x, y, w, h

            g2d.setPaint (black)
            g2d.fill (dot)

            // connect with lines
            if i != 0 && lines then
                g2d.setStroke (new BasicStroke (1.0f))
                g2d.drawLine (px_pos+1, py_pos+1, x_pos+1, y_pos+1)
            end if

            px_pos = x_pos             // update previous x
            py_pos = y_pos             // update previous y
        end for

        g2d.setStroke (new BasicStroke (2.0f))

        if z != null then
            for i <- 0 until min (y.dim, z.dim) do
                val xx = round ((x(i) - minX) * (frameW - 2 * offset).asInstanceOf [Double])
                x_pos = (xx / deltaX).asInstanceOf [Int] + offset
                val yy = round ((maxY - z(i)) * (frameH - 2 * offset).asInstanceOf [Double])
                y_pos = (yy / deltaY).asInstanceOf [Int] + offset
                dot.setFrame (x_pos, y_pos, diameter, diameter)         // x, z, w, h
                g2d.setPaint (red)
                g2d.fill (dot)

                // connect with lines
                if i != 0 && lines then
                    g2d.setStroke (new BasicStroke (1.0f))
                    g2d.drawLine (px_pos+1, py_pos+1, x_pos+1, y_pos+1)
                end if

                px_pos = x_pos         // update previous x
                py_pos = y_pos         // update previous y
            end for
        end if

        // mouse listener to identify axis click
        addMouseListener (new MouseListener {
            //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            /** Handle mouse clicked event
             *  @param mouseEvent  the mouse clicked event
             */
            override def mouseClicked (mouseEvent: MouseEvent): Unit =
                var x = mouseEvent.getX
                var y = mouseEvent.getY

                // give a gap of 4 points so click can be easily recognized
                if linex.getY > y-4 && linex.getY < y+4 then
                    var pointFound = false
                    if x < linex.getMaxX && x > linex.getX then
                        for a <- 0 until xlabels.size do
                            if a == 0 then
                                if xlabels(a).getMaxX > x && xlabels(a+1).getMinX < x then
                                    pointFound = true
                                    if setMin then      minX = xValues(a).toDouble
                                    else if setMax then maxX = xValues(a).toDouble
                                    deltaX = maxX - minX
                                    repaint ()
                                end if
                            else if a == xlabels.size-1 && ! pointFound then
                                if xlabels(a).getMinX < x && xlabels(a-1).getMinX > x then
                                    pointFound = true
                                    if setMin then      minX = xValues(a).toDouble
                                    else if setMax then maxX = xValues(a).toDouble
                                    deltaX = maxX - minX
                                    repaint ()
                                end if
                            else if ! pointFound then
                                if xlabels(a).getMinX < x && xlabels(a-1).getMinX > x then
                                    pointFound = true
                                    if setMin then      minX = xValues(a).toDouble
                                    else if setMax then maxX = xValues(a).toDouble
                                    deltaX = maxX - minX
                                    repaint ()
                                end if
                            end if
                        end for
                    end if
                end if

                if liney.getX > mouseEvent.getX -4 && liney.getX < x+4 then
                    var pointFound = false
                    if y < liney.getMaxY && y > liney.getY then
                        for a <- 0 until ylabels.size do
                            if ! pointFound then
                                if ylabels(a).getMaxY > y && ylabels(a+1).getMinY < y then
                                    pointFound = true
                                    if setMin then      minY = yValues(a).toDouble
                                    else if setMax then maxY = yValues(a).toDouble
                                    deltaY = maxY - minY
                                    repaint ()
                                end if
                            end if
                        end for
                    end if
                end if

            end mouseClicked

            override def mousePressed  (mouseEvent: MouseEvent): Unit = {}
            override def mouseReleased (mouseEvent: MouseEvent): Unit = {}
            override def mouseEntered  (mouseEvent: MouseEvent): Unit = {}
            override def mouseExited   (mouseEvent: MouseEvent): Unit = {}
        })
    end paintComponent

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert value to string and cut out the first four characters.
     *  @param x  the value to convert and cut
     */
    def clip (x: Double): String =
        val s = x.toString 
        s.substring (0, min (s.length, 4))
    end clip

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset all plot values to their original.
     */
    def resetMinMax (): Unit =
        minX   = origMinX
        maxX   = origMinX
        deltaX = origMaxX - origMinX
        minY   = origMinX
        maxY   = origMaxY
        deltaY = origMaxY - origMinY
        repaint ()
    end resetMinMax

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Switch clicked location to be set as max or min.
     */
    def setMaxMinValue (): Unit =
        if setMin then
            setMin = false
            setMax = true
        else
            setMin = true
            setMax = false
        end if
    end setMaxMinValue

end Canvas


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `plotTest` main function is used to test the `Plot` class.
 *  > runMain scalation.mathstat.plotTest
 */
@main def plotTest (): Unit =

    val x = new VectorD (100)
    val y = new VectorD (100)
    for i <- 0 until 100 do { x(i) = i / 10.0; y(i) = pow (x(i) - 5, 2) }
    new Plot (x, y, null, "plot1", lines = true)
    val plot = new Plot (null, y, null, "plot2", lines = true)

    writeImage (DATA_DIR + "plot.png", plot)

end plotTest

