
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu May 19 14:31:51 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Turn Off Building Sub-Models and Feature Selection
 */

package scalation
package modeling


import scalation.mathstat.MatrixD

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NoSubModel` trait is for modeling techniques that do not support
 *  building sub-models/feature selection (adding/remove variables from the model).
 *  For example, `SimpleRegression` only has one feature/predictor variable, so
 *  feature selection makes no sense for this modeling technique.
 */
trait NoSubModels:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  Must be implemented for models that support feature selection.
     *  NOT SUPPORTED for this model, so throw an EXCEPTION.
     *  @param x_cols  the columns that the new model is restricted to
     */
    def buildModel (x_cols: MatrixD): Predictor =
        throw new UnsupportedOperationException
              ("buildModel: this model does not support building sub-models for feature selection")
    end buildModel

end NoSubModels

