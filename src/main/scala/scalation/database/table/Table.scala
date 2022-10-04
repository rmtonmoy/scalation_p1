
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Jun 17 11:19:14 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Relational Algebra (RA) for Row-Oriented Relational DBMS
 *
 *  RA Operators: rename, project, select, union, minus, intersect, product, join,
 *                leftJoin, divide, groupBy, aggregate, orderBy
 *
 *  Most of the RA Operators have Unicode versions: @see `scalation.UnicodeTest`
 *
 *  Types of Indices (for Unique, Non-Unique Indices):
 *      LinHashMap, LinHashMultiMap                           // ScalaTion's Linear Hash Maps
 *      HashMap,    HashMultiMap                              // Scala's Hash Maps
 *      JHashMap,   JHashMultiMap                             // Java's Hash Maps
 *      BpTreeMap,  BpTreeMultiMap                            // ScalaTion's B+Tree Maps
 *      TreeMap,    TreeMultiMap                              // Scala's Tree Maps
 *      JTreeMap,   JTreeMultiMap                             // Java's Tree Maps
 */

package scalation
package database
package table

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

import java.io.{FileInputStream, FileOutputStream}
import java.io.{ObjectInputStream, ObjectOutputStream, PrintWriter}

// pick a type of Map for Unique `IndexMap` and for Non-Unique `MIndexMap`

//import scalation.database.{LinHashMap => IndexMap}
//import scalation.database.{LinHashMultiMap => MIndexMap}

  import scala.collection.mutable.{HashMap => IndexMap}
  import scalation.database.{HashMultiMap => MIndexMap}

//import scalaTion.database.{JHashMap => IndexMap}
//import scalaTion.database.{JHashMultiMap => MIndexMap}

//import scalation.database.{BpTreeMap => IndexMap}
//import scalation.database.{BpTreeMultiMap => MIndexMap}

//import scala.collection.mutable.{TreeMap => IndexMap}
//import scalation.database.{TreeMultiMap => MIndexMap}

//import scalation.database.{JTreeMap => IndexMap}
//import scalation.database.{JTreeMultiMap => MIndexMap}

import scala.collection.mutable.{ArrayBuffer => Bag, IndexedSeq, Map}
import scala.math.{max, min}
import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{breakable, break}

import scalation.mathstat.{MatrixD, VectorD, VectorI, VectorL, VectorS, VectorT}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Table` companion object provides factory functions for creating tables.
 *  Supported domains/data-types are 'D'ouble, 'I'nt, 'L'ong, 'S'tring, and 'T'imeNum.
 *  Note 'X' is for Long String (a formatting issue).
 */
object Table:

    private val debug = debugf ("Table", true)                              // debug function
    private val flaw  = flawf ("Table")                                     // flaw function
    private val cntr  = Counter ()                                          // counter for generating unique names

    private var useFullPath = false                                         // defaults to using relative file paths
    private var limit       = -1                                            // limit on number of lines to read

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the full-path flag to the value of parameter fullPath.
     *  @param fullPath  flag indicating whether full or relative paths should be used
     */
    def setFullPath (fullPath: Boolean = true): Unit = { useFullPath = fullPath }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the limit on the number of lines to read to lim.
     *  @param lim  the limit on the number of lines to read (<= 0 => unlimited)
     */
    def setLimit (lim: Int): Unit = { limit = lim }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a table given convenient string specifications.
     *  @param name     the name of the table
     *  @param schema   the attributes for the table
     *  @param domain_  the domains/data-types for attributes ('D', 'I', 'L', 'S', 'X', 'T')
     *  @param key      the attributes forming the primary key
     */
    def apply (name: String, schema: String, domain_ : String, key: String): Table =
        new Table (name, strim (schema), strim (domain_).map (_.head), strim (key))
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read the table with the given name into memory loading its columns with data from
     *  the CSV file named fileName.  The attribute names are read from the FIRST LINE.
     *  @param fileName  the file name (or file-path) of the data file
     *  @param name      the name of the table
     *  @param domain_   the domains/data-types (as one string) for attributes ('D', 'I', 'L', 'S', 'X', 'T')
     *  @param key       the attributes forming the primary key
     *  @param pos_      the sequence of column positions in the input file to be used (null => select all)
     *  @param sep       the element separation string/regex (e.g., "," ";" " +")
     */
    def load (fileName: String, name: String, domain_ : String, key: String,
              pos: Array [Int], sep: String): Table =
        load (fileName, name, strim (domain_).map (_.head), key, pos, sep)
    end load

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read the table with the given name into memory loading its columns with data from
     *  the CSV file named fileName.  The attribute names are read from the FIRST LINE.
     *  @see scalation.readFileIntoArray
     *  @param fileName  the file name (or file-path) of the data file
     *  @param name      the name of the table
     *  @param domain    the domains/data-types for attributes ('D', 'I', 'L', 'S', 'X', 'T')
     *  @param key       the attributes forming the primary key
     *  @param pos_      the sequence of column positions in the input file to be used (null => select all)
     *  @param sep       the element separation string/regex (e.g., "," ";" " +")
     */
    def load (fileName: String, name: String, domain: Domain, key: String,
              pos_ : Array [Int] = null, sep: String = ","): Table =

        debug ("load", s"""fileName = $fileName, name = $name, domain = ${stringOf (domain)}, key = $key,
                       pos_ = $pos_, sep = '$sep'; useFullPath = $useFullPath, limit = $limit""")
                         
        val pos    = if pos_ == null then Array.range (0, domain.size) else pos_
        val schema = Array.ofDim [String] (domain.size)

        if pos.size != domain.size then flaw ("apply", "pos size should be same as domain size")

        var s: Table = null                     // new Table (name, schema, domain, strim (key))

//      val lines = getFromURL_File (fileName)                              // read the CSV file
        val lines = readFileIntoArray (fileName, useFullPath, limit)        // read the CSV file
        var l_no  = 0                                                       // the line number

        println (s"lines(0) = ${lines(0)}")

        for ln <- lines do                                                  // iterate by lines in file

            if l_no == 0 then                                               // FIRST LINE - for schema
                val header = ln.split (sep, -1).map (_.trim)                // array of column names
                debug ("load", s"header = ${stringOf (header)}")
                for j <- pos.indices do schema(j) = header(pos(j))          // use those at positions in pos
                s = new Table (name, schema, domain, strim (key))           // make table after schema is formed

            else                                                            // REMAINING LINES
                val token = ln.split (sep, -1).map (_.trim)                 // array of token strings
                s.tuples += makeTuple (token, domain, pos)
            end if

            l_no += 1
        end for
        s
    end load

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read the table with the given name into memory loading its columns with data from
     *  the CSV file named fileName.  The attribute names are read from the FIRST LINE.
     *  Use a short-cut (not reliable) to determines the column domains, by applying
     *  the 'tuple2type' method to the SECOND LINE.
     *  Note: safer to pull a row without missing or zero values from the middle of the dataset
     *  @see `tableTest3`
     *  @see scalation.readFileIntoArray
     *  @param fileName  the file name (or file-path) of the data file
     *  @param name      the name of the table
     *  @param mumCol    the number of columns
     *  @param key       the attributes forming the primary key
     */
    def load (fileName: String, name: String, numCol: Int, key: String): Table =

        val pos = Array.range (0, numCol)
        val sep = ","
        debug ("load", s"""fileName = $fileName, name = $name, numCol = $numCol, key = $key,
                       pos = $pos, sep = '$sep'; useFullPath = $useFullPath, limit = $limit""")

        val schema = Array.ofDim [String] (numCol)
        val domain = Array.ofDim [Char] (numCol)

        var s: Table = null                     // new Table (name, schema, domain, strim (key))

//      val lines = getFromURL_File (fileName)                              // read the CSV file
        val lines = readFileIntoArray (fileName, useFullPath, limit)        // read the CSV file
        var l_no  = 0                                                       // the line number

        for ln <- lines do                                                  // iterate by lines in file

            if l_no == 0 then                                               // FIRST LINE - for schema
                val header = ln.split (sep, -1).map (_.trim)                // array of column names
                debug ("load", s"header = ${stringOf (header)}")
                for j <- 0 until numCol do schema(j) = header(j)            // collect from header
                s = new Table (name, schema, domain, strim (key))           // make table after schema is formed

            else if l_no == 1 then                                          // SECOND LINE - for domains
                val token = ln.split (sep, -1).map (_.trim)                 // array of token strings
                val dom   = tuple2type (token)                              // guess domains from first data row
                debug ("load", s"dom = ${stringOf (dom)}")
                for j <- 0 until numCol do domain(j) = dom(j)               // collect from dom
                s.tuples += makeTuple (token, domain, pos)

            else                                                            // REMAINING LINES
                val token = ln.split (sep, -1).map (_.trim)                 // array of token strings
                s.tuples += makeTuple (token, domain, pos)
            end if

            l_no += 1
        end for
        s
    end load

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make a tuple from an array of token strings, converting each each token
     *  according the corresponding domain specification.  Use only the tokens
     *  in the array at the pos positions.
     *  @param token   the array of token strings, e.g., Array ("5.0", "12", "Smith")
     *  @param domain  the domains/data-types for attributes ('D', 'I', 'L', 'S', 'X', 'T')
     *  @param pos     the positions in the token array to be used, e.g., Array (0, 2)
     */
    def makeTuple (token: Array [String], domain: Domain, pos: Array [Int]): Tuple =
        if token.size < pos.max then
            flaw ("makeTuple", "not enough tokens for positions given in pos")
            return null
        end if

        val tup = Array.ofDim [ValueType] (token.size)
        for j <- pos.indices do
            val nextToken = token(pos(j))                                   // get j-th token according to pos
            tup(j) = domain(j) match
            case 'D' =>       nextToken.mkDouble                            // Double
            case 'I' =>       nextToken.toInt                               // Int
            case 'L' =>       nextToken.toLong                              // Long
            case 'S' | 'X' => nextToken                                     // String or Long-String
            case 'T' =>       TimeNum (nextToken)                           // TimeNum
            case _   =>     { flaw ("makeTuple", s"domain($j) = ${domain(j)} not supported"); "?" }
        end for
        tup
    end makeTuple

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given an array of strings (e.g., read from a file) with unknown domains,
     *  return the data-types (domains) by the lexical form of the strings.
     *  @see `scalation.typeOfStr` (in ValueType.scala)
     *  Caveat:  may not be reliable since a column of doubles may start: 5, 7, 9.2, ...
     *  @param tup  the type un-differentiated tuple as an array of strings
     */
    def tuple2type (tup: Array [String]): Domain =
        val dom = Array.ofDim [Char] (tup.size)
        for j <- dom.indices do dom(j) = typeOfStr (tup(j))
        dom
    end tuple2type

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** LOAD/Read the table with the given name into memory using serialization.
     *  @see save in `Table` class.
     *  @param name  the name of the table to load
     */
    def load (name: String): Table =
        val ois = new ObjectInputStream (new FileInputStream (STORE_DIR + name + SER))
        val tab = ois.readObject.asInstanceOf [Table]
        ois.close ()
        tab
    end load

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** LOAD/Read the table with the given name into memory from a JSON file.
     *  @param fileName  the file name of the JSON file
     *  @param name      the name of the table to load
     */
    def load (fileName: String, name: String): Table =
        val jsonArr = readFileIntoArray (fileName)
        val nlines  = jsonArr.size 
        val jsonStr: String = jsonArr(0)
        debug ("load", s"jsonStr = ${jsonStr.slice (0, 5000)}")
        var tab: Table = null
        val gson = new Gson ()
//      val tableType  = new TypeToken [Table] ().getType                  // FIX - fails
//      tab = gson.fromJson (jsonStr, tableType)
        tab
    end load

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a table from a matrix of doubles.
     *  @see the `toMatrix` mathod
     *  @param x       the matrix containing the data
     *  @param name    the name of the table
     *  @param schema  the attribute/column names
     *  @param key     the attributes forming the primary key
     */
    def fromMatrix (x: MatrixD, name: String, schema: Schema, key: String): Table =
        val domain = Array.fill (x.dim2)('D')                               // domain is all 'D'
        val s = new Table (name, schema, domain, strim (key))

        for i <- x.indices do s.tuples += x(i).toArray                      // i-th vector to tuple
        s
    end fromMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return basic statistics on the given column corresponding to SQL's
     *  aggregate functions:  count, countd, min, max, sum, avg.
     *  @param cname  the given column name
     *  @param colj   the given column
     */
    def stats (cname: String, colj: Array [ValueType]): Array [ValueType] =
        Array (cname, count (colj), countd (colj), min (colj).toString, max (colj).toString, sum (colj), avg (colj))
    end stats

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the total number of elements in a column.
     *  @param colj  the given column
     */
    def count (colj: Array [ValueType]): ValueType = colj.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of distinct elements in a column.
     *  @param colj  the given column
     */
    def countd (colj: Array [ValueType]): ValueType = colj.distinct.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the minimum value of all the elements in a column.
     *  @param colj  the given column
     */
    def min (colj: Array [ValueType]): ValueType = colj.min (ValueTypeOrd)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the maximum value of all the elements in a column.
     *  @param colj  the given column
     */
    def max (colj: Array [ValueType]): ValueType = colj.max (ValueTypeOrd)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the average of all the elements for a numeric column or 0 otherwise.
     *  @param colj  the given column
     */
    def avg (colj: Array [ValueType]): ValueType = sum (colj).toDouble / colj.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the sum of all the elements for a numeric column or 0 otherwise.
     *  @param colj  the given column
     */
    def sum (colj: Array [ValueType]): ValueType =
        colj(0) match
        case _: Double  => var s = 0.0; for x <- colj do s += x.toDouble; s
        case _: Int     => var s = 0.0; for x <- colj do s += x.toDouble; s
        case _: Long    => var s = 0.0; for x <- colj do s += x.toDouble; s
        case _: String  => -0.0
        case _: TimeNum => -0.0
        case null       => -0.0
        end match
    end sum

    def π (x: String)(r: Table): Table = r.project (strim (x))
    def σ (condition: String)(r: Table): Table = r.select (condition)
    def σ (predicate: Predicate)(r: Table): Table = r.select (predicate)

end Table

import Table.{cntr, debug, flaw}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Table` class stores relational data and implements relational algebra operators.
 *  Supported domains/data-types are 'D'ouble, 'I'nt, 'L'ong, 'S'tring, and 'T'imeNum.
 *  @param name    the name of the table
 *  @param schema  the attributes for the table
 *  @param domain  the domains/data-types for the attributes ('D', 'I', 'L', 'S', 'X', 'T')
 *  @param key     the attributes forming the primary key
 */
class Table (override val name: String, override val schema: Schema,
             override val domain: Domain, override val key: Schema)
     extends Tabular [Table] (name, schema, domain, key)
        with Serializable:

    private [table] val tuples    = Bag [Tuple] ()                                 // storage of tuples
    private [table] val linkTypes = Map [String, Table] ()                         // link types for foreign keys
    private [table] val index     = IndexMap [KeyType, Tuple] ()                   // index on primary key
    private [table] var hasIndex  = false                                          // whether the primary index has been built
    private [table] val sindex    = Map [String, IndexMap [ValueType, Tuple]] ()   // map of secondary unique indices 
    private [table] val mindex    = Map [String, MIndexMap [ValueType, Tuple]] ()  // map of secondary non-unique indices
    private val groupMap          = Map [ValueType, Bag [Tuple]] ()                // map from group key to collection of tuples

    protected val countX          = domain.count ((c: Char) => c == 'X')           // count the number of eXtended Strings

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the cardinality (number of tuples) in this table.
     */
    def rows: Int = tuples.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the j-th column in this table (or the passed in tuples) as an array of value-type.
     *  @param j     the column to return
     *  @param tups  the collection of tuples to use (defaults to all tuples in this table)
     */
    def col (j: Int, tups: Bag [Tuple] = tuples): Array [ValueType] =

        if j >= schema.size then
            flaw ("col", s"column index j = $j exceeds the number of columns")
        end if

        val c = Array.ofDim [ValueType] (tups.size)
        for i <- c.indices do c(i) = tups(i)(j)
        c
    end col

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this table contains tuple u.
     *  @param u  the tuple to look for
     */
    def contains (u: Tuple): Boolean = tuples.exists (_ sameElements u)

    // I N T E G R I T Y   C H E C K S

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add LINKAGE from this table to the refTab, by adding a FOREIGN KEY CONSTRAINT 
     *  to this table specifying the foreign key attribute fkey and the table it
     *  references refTab.  If refTab does not have a primary index already, make one.
     *  Caveat:  a foreign key may not be composite.
     *  @param fkey    the foreign key attribute
     *  @param refTab  the table being referenced (to its primary key)
     */
    def addLinkage (fkey: String, refTab: Table): Unit =
        if ! refTab.hasIndex then refTab.create_index ()
        linkTypes += fkey -> refTab
    end addLinkage

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check that all the foreign keys values in tuple t satisfy their
     *  REFERENTIAL INTEGRITY CONSTRAINTS.
     *  @param t  the tuple being checked for referential integrity
     */
    def referenceCheck (t: Tuple): Boolean =
        var satisfy = true
        for (fkey, refTab) <- linkTypes do
            val fkeyVal = new KeyType (pull (t, fkey))
            if refTab.hasIndex && refTab.index.getOrElse (fkeyVal, null) == null then
                flaw ("referenceCheck", s"foreign key $fkey = $fkeyVal is not in table ${refTab.name}")
                satisfy = false
            end if
        end for
        satisfy
    end referenceCheck

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the i-th primary key.
     *  @param i  the index in the tuples/row index
     */
    def getPkey (i: Int): KeyType = new KeyType (pull (tuples(i), key))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** CREATE/recreate the primary INDEX that maps the primary key to the tuple
     *  containing it.  Warning, creating an index will remove DUPLICATES based
     *  on maintaining UNIQUENESS CONSTRAINT of primary key values.
     *  @param rebuild  if rebuild is true, use old index to build new index; otherwise, create new index
     */
    def create_index (rebuild: Boolean = false): Unit =
        debug ("create_index", s"create an index of type ${index.getClass.getName}")
        if rebuild then flaw ("create_index", "rebuilding off old primary key index has not yet been implemented")
        index.clear ()
        val toRemove = Bag [Tuple] ()
        for t <- tuples do
            val pkey = new KeyType (pull (t, key))                           // primary key
            if index.getOrElse (pkey, null) == null then index += pkey -> t
            else toRemove += t
        end for
        debug ("create_index", s"remove duplicate tuples = ${showT (toRemove)}")
        tuples --= toRemove
        hasIndex = true
    end create_index

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** CREATE a secondary unique INDEX that maps a secondary key to the tuple
     *  containing it.  Has no effect on duplicates; should first create a primary
     *  index to remove duplicates, otherwise, this index may skip tuples.
     *  @param atr  the attribute/column to create the index on
     */
    def create_sindex (atr: String): Unit =
        debug ("create_sindex", s"create a secondary unique index of type ${index.getClass.getName}")
        if ! hasIndex then flaw ("create_sindex", "should first create a primary index to eliminate duplicates")
        val newIndex = IndexMap [ValueType, Tuple] ()
        for t <- tuples do
            val skey = (pull (t, atr))                                       // secondary (non-composite) key
            newIndex += skey -> t                                            // add key-value pair into new index
        end for
        sindex += atr -> newIndex                                            // add new index into the sindex map
    end create_sindex

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** CREATE a non-unique INDEX (multi-valued) that maps a non-unique attribute
     *  to the tuple containing it.
     *  @see `scalation.database.MultiMap`
     *  @param atr  the attribute/column to create the non-unique index on
     */
    def create_mindex (atr: String): Unit =
        debug ("create_mindex", s"create a non-unique index of type ${index.getClass.getName}")
        val newIndex = MIndexMap [ValueType, Tuple] ()
        for t <- tuples do
            val t_atr = (pull (t, atr))                                      // non-unique attribute
            newIndex.addOne1 (t_atr, t)                                      // add key-value pair into new index
        end for
        mindex += atr -> newIndex                                            // add new index into the mindex map
    end create_mindex

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** DROP the primary INDEX that maps the primary key to the tuple containing it.
     */
    def drop_index (): Unit =
        index.clear ()
        hasIndex = false
    end drop_index

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** DROP a secondary INDEX that maps a secondary key to the tuple containing it.
     */
    def drop_sindex (atr: String): Unit =
        val oldIndex = sindex.getOrElse (atr, null)
        if oldIndex != null then 
            oldIndex.clear ()
            sindex -= atr
        else
            flaw ("drop_sindex", s"no index found for attribute = $atr")
        end if
    end drop_sindex

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** DROP a non-unique INDEX that maps a non-unique attribute to the tuple containing it.
     */
    def drop_mindex (atr: String): Unit =
        val oldIndex = mindex.getOrElse (atr, null)
        if oldIndex != null then 
            oldIndex.clear ()
            mindex -= atr
        else
            flaw ("drop_mindex", s"no index found for attribute = $atr")
        end if
    end drop_mindex

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the table restricted to the given range of rows.
     *  @param r  the given range of rows
     */
    def apply (r: Range): Table =
        val s = new Table (s"${name}_a_${cntr.inc ()}", schema, domain, key)

        s.tuples ++= (for i <- r yield tuples(i))
        s
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the table restricted to the given collection of rows.
     *  @param pos  the given collection of rows
     */
    def apply (pos: collection.immutable.IndexedSeq [Int]): Table = 
        val s = new Table (s"${name}_a_${cntr.inc ()}", schema, domain, key)

        s.tuples ++= (for i <- pos yield tuples(i))
        s
    end apply

    // R E L A T I O N   A L G E B R A   O P E R A T O R S

    // ================================================================== RENAME

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** RENAME this table, returning a shallow copy of this table.
     *  Usage:  customer rename "client"
     *--------------------------------------------------------------------------
     *  @param newName  the new name for the table
     */
    def rename (newName: String): Table =
        val s = new Table (newName, schema, domain, key)
        s.tuples ++= tuples                                      // shallow copy
        s
    end rename

    // ================================================================= PROJECT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** PROJECT the tuples in this table onto the given attribute names.
     *  Uaage:  customer project (Array ("street", "ccity"))
     *--------------------------------------------------------------------------
     *  @param x  the schema/attribute names to project onto
     */
    def project (x: Schema): Table =
        val newKey = if subset (key, x) then key else x
        val s = new Table (s"${name}_p_${cntr.inc ()}", x, pull (x), newKey)

        s.tuples ++= (for t <- tuples yield pull (t, x))
        s
    end project

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** PROJECT onto the columns with the given column positions (first is column 0).
     *  Uaage: customer.project (Array (1, 2))
     *--------------------------------------------------------------------------
     *  @param cPos  the column positions to project onto
     */
    def project (cPos: IndexedSeq [Int]): Table = 
        val mxPos = cPos.max
        if mxPos >= cols then flaw ("project", s"mxPos = $mxPos is too large for the number of columns")

        val newAtrs = (for c <- cPos yield schema(c)).toArray 
        val newKey  = if subset (key, newAtrs) then key else newAtrs
        val s = new Table (s"${name}_p_${cntr.inc ()}", newAtrs, pull (cPos), newKey)

        s.tuples ++= (for t <- tuples yield pull (t, cPos))
        s
    end project

    // ========================================================== PROJECT-SELECT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT elements from column a in this table that satisfy the atomic
     *  predicate apred and PROJECT onto that column.
     *  Usage:  customer selproject ("ccity", _ > "Athens")
     *--------------------------------------------------------------------------
     *  @param a      the attribute name of the column used for selection
     *  @param apred  the atomic predicate (`Boolean` function) to be satisfied
     */
    def selproject (a: String, apred: APredicate): Table =
        val newAtr = Array (a) 
        val newDom = Array (domain(on(a)))
        val s = new Table (s"${name}_s_${cntr.inc ()}", newAtr, newDom, newAtr)

        for t <- tuples do 
            val ta = pull (t, a)
            if apred (ta) then s.tuples += Array (ta)
        end for
        s
    end selproject

    // ================================================================== SELECT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT the tuples in this table that satisfy the atomic predicate on column a.
     *  Usage:  customer select ("ccity", _ == "Athens")
     *--------------------------------------------------------------------------
     *  @param a      the attribute name of the column used for selection
     *  @param apred  the atomic predicate (`Boolean` function) to be satisfied
     */
    def select (a: String, apred: APredicate): Table =
        val s = new Table (s"${name}_s_${cntr.inc ()}", schema, domain, key)

        for t <- tuples if apred (pull (t, a)) do s.tuples += t
        s
    end select

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT the tuples in this table that satisfy the predicate.
     *  Usage:  customer select (t => t(customer.on("ccity")) == "Athens")
     *--------------------------------------------------------------------------
     *  @param predicate  the predicate (`Boolean` function) to be satisfied
     */
    def select (predicate: Predicate): Table =
        val s = new Table (s"${name}_s_${cntr.inc ()}", schema, domain, key)

        s.tuples ++= tuples.filter (predicate)
        s
    end select

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT the tuples in this table that satisfy the given simple (3 token) condition.
     *  Usage:  customer select ("ccity == 'Athens'")
     *--------------------------------------------------------------------------
     *  @param condition  the simple condition string "a1 op a2" to be satisfied, where
     *                    a1 is attribute, op is comparison operator, a2 is attribute or value
     */
    def select (condition: String): Table =
        val s = new Table (s"${name}_s_${cntr.inc ()}", schema, domain, key)

        val (tok, twoAtrs) = parseCond (condition)
        val (a1, op, a2) = (tok(0), tok(1), tok(2))    
        debug ("select", s"condition: (a1, op, a2) = ($a1, $op, $a2), twoAtrs = $twoAtrs")

        s.tuples ++= selectTups (a1, op, a2, twoAtrs)
        s
    end select

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT via the INDEX the tuple with the given primary key value pkey.
     *  Returns an empty table if the primary index has not been created.
     *  Usage:  customer select (new KeyType ("Mary"))
     *--------------------------------------------------------------------------
     *  @param pkey  the primary key value
     */
    def select (pkey: KeyType): Table =
        val s = new Table (s"${name}_s_${cntr.inc ()}", schema, domain, key)

        if hasIndex then
            debug ("select", s"primary key pkey = $pkey")
            val t = index.getOrElse (pkey, null)
            if t != null then s.tuples += t
        else
            flaw ("select", s"must call 'create_index' before using indexed-select on table $name")
        end if
        s
    end select

    // =================================================================== UNION

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** UNION this table and r2.  Check that the two tables are compatible.
     *  If they are not, return the first table.
     *  Caveat:  Assumes the key from the first table still works (@see create_index)
     *  Acts like union-all, so to remove duplicates call create_index after union.
     *  Usage:  deposit union loan
     *--------------------------------------------------------------------------
     *  @param r2  the second table
     */
    def union (r2: Table): Table =
        if incompatible (r2) then return this
        val s = new Table (s"${name}_u_${cntr.inc ()}", schema, domain, key)

        s.tuples ++= tuples ++ r2.tuples
        s
    end union

    // =================================================================== MINUS

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute this table MINUS (set difference) table r2 (this - r2).  Check that
     *  the two tables are compatible.  If they are not, return the first table.
     *  Usage:  account minus loan
     *--------------------------------------------------------------------------
     *  @param r2  the second table
     */
    def minus (r2: Table): Table =
        if incompatible (r2) then return this
        val s = new Table (s"${name}_m_${cntr.inc ()}", schema, domain, key)

                if subset (key, schema) then
        
            if r2.hasIndex then 
                
                for t <- tuples do
                    
                    val t_fkey = new KeyType (pull (t, key))
                    val u = r2.index.getOrElse (t_fkey, null)
                    if u == null then s.tuples += t

                end for 
                
            else

                for t <- tuples do if ! (r2 contains t) then s.tuples += t

            end if

        else 
        
            for t <- tuples do if ! (r2 contains t) then s.tuples += t

        end if 
        s
    end minus


    // =============================================================== INTERSECT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** INTERSECT this table and r2.  Check that the two tables are compatible.
     *  If they are not, return the first table.
     *  Usage:  account intersect loan
     *--------------------------------------------------------------------------
     *  @param r2  the second table
     */

    def intersect (r2: Table): Table =
        if incompatible (r2) then return this
        val s = new Table (s"${name}_i_${cntr.inc ()}", schema, domain, key)

        if subset (key, schema) then
            if r2.hasIndex then
                for t <- tuples do
                    
                    val t_fkey = new KeyType (pull (t, key))
                    val u = r2.index.getOrElse (t_fkey, null)
                    if u != null then s.tuples += t

                end for 
                
            else 
                for t <- tuples do if r2 contains t then s.tuples += t
        
            end if 

        else            
            for t <- tuples do if r2 contains t then s.tuples += t
        
        end if
        s
    end intersect

    // ================================================================= PRODUCT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the CARTESIAN PRODUCT of this table and r2 (this × r2).
     *  Usage:  customer product deposit
     *--------------------------------------------------------------------------
     *  @param r2  the second table
     */
    def product (r2: Table): Table =
        val newKey = key ++ r2.key                                          // requires keys from both tables
        val s = new Table (s"${name}_x_${cntr.inc ()}", disambiguate (schema, r2.schema),
                       domain ++ r2.domain, newKey)

        for t <- tuples; u <- r2.tuples do
            s.tuples += t ++ u
        end for
        s
    end product

    // ==================================================================== JOIN

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** JOIN this table and r2 keeping concatenated tuples that satisfy the predicate.
     *  Caveat:  Assumes both keys are needed for the new key (depending on the
     *           predicate both may not be required).
     *  Usage:  customer join ((t, u) => t(customer.on("cname")) == u(deposit.on("cname")), deposit)
     *--------------------------------------------------------------------------
     *  @param predicate  the join predicate to be satisfied
     *  @param r2         the second table
     */
    def join (predicate: Predicate2, r2: Table): Table =
        val newKey = key ++ r2.key                                          // requires keys from both tables
        val s = new Table (s"${name}_j_${cntr.inc ()}", disambiguate (schema, r2.schema),
                       domain ++ r2.domain, newKey)

        for t <- tuples; u <- r2.tuples do
            if predicate (t, u) then s.tuples += t ++ u
        end for
        s
    end join

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the THETA-JOIN of this table and r2 keeping concatenated tuples that
     *  satisfy the given simple (3 token) condition.
     *  Usage:  customer join ("cname == cname"), deposit)
     *--------------------------------------------------------------------------
     *  @param condition  the simple condition "a1 op a2"
     *  @param r2         the second table
     */
    def join (condition: String, r2: Table): Table =
        val tok = parseCond (condition)._1
        val (a1, op, a2) = (tok(0), tok(1), tok(2))
//      debug ("join", s"(a1, op, a2) = ($a1, $op, $a2)")

        val newKey = key ++ r2.key                                          // requires keys from both tables

        val s = new Table (s"${name}_j_${cntr.inc ()}", disambiguate (schema, r2.schema),
                       domain ++ r2.domain, newKey)

        s.tuples ++= 
        (op match
        case "==" => tJoinTups (a1, equ, a2, r2)
        case "!=" => tJoinTups (a1, neq, a2, r2)
        case "<"  => tJoinTups (a1, <,   a2, r2)
        case "<=" => tJoinTups (a1, <=,  a2, r2)
        case ">"  => tJoinTups (a1, >,   a2, r2)
        case ">=" => tJoinTups (a1, >=,  a2, r2)
        case _    => flaw ("join", s"$op is an unrecognized operator"); null)
        s
    end join

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the EQUI-JOIN of this table and r2 keeping concatenated tuples that
     *  are equal on specified attributes.
     *  Usage:  customer join (Array ("cname"), Array ("cname"), deposit)
     *--------------------------------------------------------------------------
     *  @param x   the subschema/attributes for the first/this table
     *  @param y   the subschema/attributes for the second table
     *  @param r2  the second table
     */
    def join (x: Schema, y: Schema, r2: Table): Table =
        val newKey = if subset (x, key) then r2.key                         // three possibilities for new key
                     else if subset (y, r2.key) then key
                     else key ++ r2.key

        val s = new Table (s"${name}_j_${cntr.inc ()}", disambiguate (schema, r2.schema),
                           domain ++ r2.domain, newKey)

        for t <- tuples; u <- r2.tuples do
            if pull (t, x) sameElements r2.pull (u, y) then s.tuples += t ++ u
        end for
        s
    end join

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the EQUI-JOIN via the INDEX of this table and the referenced table keeping
     *  concatenated tuples that are equal on the primary key and foreign key attributes.
     *  Caveat:  Requires the foreign key table to be first [ fkey_table join ((fkey, pkey_table) ].
     *  Usage:   deposit join (("cname", customer))
     *--------------------------------------------------------------------------
     *  @param ref  the foreign key reference (foreign key attribute, referenced table)
     */
    def join (ref: (String, Table)): Table =
        val (fkey, refTab) = ref                                            // foreign key, referenced table

        val s = new Table (s"${name}_j_${cntr.inc ()}", disambiguate (schema, refTab.schema),
                           domain ++ refTab.domain, key)

        if refTab.hasIndex then
            for t <- tuples do                                              // iterate over fkey table
                val t_fkey = new KeyType (pull (t, fkey))
                debug ("join", s"foreign key t_fkey = $t_fkey")
                val u = refTab.index.getOrElse (t_fkey, null)               // get pkey from refTab
                if u != null then s.tuples += t ++ u                        // add concatenated tuples
            end for
        else
            flaw ("join", s"must call 'create_index' before using indexed-join on ${refTab.name}")
        end if
        s
    end join

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the NATURAL JOIN of this table and r2 keeping concatenated tuples
     *  that agree on the common attributes.
     *  Usage:  customer join deposit
     *--------------------------------------------------------------------------
     *  @param r2  the second table
     */
    def join (r2: Table): Table =
//      val common = schema intersect r2.schema                             // common attributes
        val common = meet (schema, r2.schema)                               // common attributes
        debug ("join", s"common = ${stringOf (common)}")
        val rest   = r2.schema diff common
        val newKey = if subset (common, key) then r2.key                    // three possibilities for new key
                     else if subset (common, r2.key) then key
                     else key ++ r2.key

        val s = new Table (s"${name}_j_${cntr.inc ()}", schema ++ rest,
                           domain ++ r2.pull (rest), newKey)

        for t <- tuples; u <- r2.tuples do
            if pull (t, common) sameElements r2.pull (u, common) then
                s.tuples += t ++ r2.pull (u, rest)
            end if
        end for
        s
    end join

    // =============================================================== LEFT-JOIN

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the LEFT-EQUI-JOIN of this table and r2 keeping concatenated tuples
     *  that are equal on specified attributes.  Also, keep all tuples in the left
     *  table padding the missing attributes with null.
     *  For right-join swap table1 and table2, e.g., table1.leftJoin (... table2)
     *  Usage:  customer leftJoin (Array ("cname"), Array ("cname"), deposit)
     *--------------------------------------------------------------------------
     *  @param x   the subschema/attributes for the left/first/this table
     *  @param y   the subschema/attributes for the right/second table
     *  @param r2  the second table
     */
    def leftJoin (x: Schema, y: Schema, r2: Table): Table =
        val s = join (x, y, r2)

        val absentTuple = nullTuple (r2.domain)
        val ss = s.project (schema)                        // join projected onto original schema
        for t <- tuples if ! (ss contains t) do
            s.tuples += t ++ absentTuple
        end for
        s
    end leftJoin

    
    //usage deposit leftjoin (("cname", customer))

    def leftJoin (ref: (String, Table)): Table =

             val (fkey, refTab) = ref
             val s = join(ref)

             val absentTuple = nullTuple (refTab.domain)
             val ss = s.project (schema)                        // join projected onto original schema
             for t <- tuples if ! (ss contains t) do
                 s.tuples += t ++ absentTuple
             end for


             s
    end leftJoin


    // ================================================================== DIVIDE

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** DIVIDE this table by table r2.  Requires a tuple in the quotient part of
     *  this table to be paired with all tuples in table r2.
     *  Usage:  deposit.project ("cname, bname") divide branch.project ("bname")
     *--------------------------------------------------------------------------
     *  @param r2  the second table
     */  
    def divide (r2: Table): Table =
        val divisor  = r2.schema
        if ! subset (divisor, schema) then flaw ("divide", "divisor schema must be a subset of schema")
        val quotient = schema diff divisor
        val newKey   = if subset (key, quotient) then key else quotient
        val s = new Table (s"${name}_d_${cntr.inc ()}", quotient, pull (quotient), newKey)

        val q    = project (quotient)
        var keep = false
        for t <- q.tuples do
            keep = true
            breakable {
                for u <- r2.tuples do
                    if ! (this contains t ++ u) then { keep = false; break () }
                end for
            } // breakable
            if keep then s.tuples += t
        end for
        s
    end divide

    // ================================================================ GROUP-BY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** GROUP this table BY the specified attribute, returning this table.
     *  Each value for attribute ag will be mapped to a collection of tuples.
     *  Usage:  deposit groupBy "bname"
     *--------------------------------------------------------------------------
     *  @param ag  the attribute to group by
     */
    def groupBy (ag: String): Table =
        if ! (schema contains ag) then
            flaw ("groupBy", s"ag = $ag is not contained in schema")
        end if

        val col = on(ag)                                                    // the column number for ag 
        for t <- tuples do
            val gkey  = t(col)
            val group = groupMap.getOrElseUpdate (gkey, Bag [Tuple] ())
            group += t                                                      // add tuple t to gkey's group
        end for
        this
    end groupBy

    // =============================================================== AGGREGATE

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Assuming this table has been grouped by attribute ag, create a table 
     *  where the first column is ag and the rest are AGGREGATE FUNCTIONs applied
     *  to their corresponding attributes.
     *  Usage:  deposit aggregate ("bname", (avg, "balance"))
     *--------------------------------------------------------------------------
     *  @param ag    the attribute the table has been grouped on
     *  @param f_as  the aggregate function and the attribute to apply it to (as varargs)
     */
    def aggregate (ag: String, f_as: (AggFunction, String)*): Table =
        val n = f_as.size + 1
        val cols = Array.ofDim [Int] (n - 1)
        val schm = Array.ofDim [String] (n)
        schm(0) = ag
        for j <- f_as.indices do
            cols(j)   = on(f_as(j)._2)                                      // the column number for atr j
            schm(j+1) = f_as(j)._2
        end for
        val s = new Table (s"${name}_a_${cntr.inc ()}", schm, pull (schm), Array (ag))

        for (gkey, tups) <- groupMap do
            val t = Array.ofDim [ValueType] (n)
            t(0) = gkey
            for j <- f_as.indices do t(j+1) = f_as(j)._1 (col (cols(j), tups))
            s.tuples += t
        end for
        s
    end aggregate

    // ================================================================ ORDER-BY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** ORDER-BY the given attributes, i.e., reorder the tuples in this table into
     *  'ascending' order.  A stable sorting is used to allow sorting on multiple attributes.
     *  Usage:  deposit orderBy "bname"
     *--------------------------------------------------------------------------
     *  @param x  the subschema/attributes to order by
     */
    def orderBy (x: String*): Table =
        val s    = new Table (s"${name}_o_${cntr.inc ()}", schema, domain, key)

        val perm = rankOrder (x :_*)
        for i <- perm do s.tuples += tuples(i)
        s
    end orderBy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** ORDER-BY-DESC the given attributes, i.e., reorder the tuples in this table into
     *  'descending' order.  A stable sorting is used to allow sorting on multiple attributes.
     *  Usage:  deposit orderByDesc "bname"
     *--------------------------------------------------------------------------
     *  @param x  the subschema/attributes to order by
     */
    def orderByDesc (x: String*): Table =
        val s    = new Table (s"${name}_o_${cntr.inc ()}", schema, domain, key)

        val perm = rankOrder (x :_*)
        for i <- perm.reverse do s.tuples += tuples(i)
        s
    end orderByDesc

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the basic statistics for each column of this table.
     */
    def stats: Table =
        val s = new Table (s"${name}_stats",
                           Array ("column", "count", "countd", "min", "max", "sum", "avg"),
                           Array ('S', 'I', 'I', 'S', 'S', 'D', 'D'), Array ("column"))

        for j <- colIndices do s add Table.stats (schema(j), col(j))
        s
    end stats

    // D A T A   M A N I P U L A T I O N

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** ADD (insert) tuple t into this table checking to make sure the domains are correct.
     *  Also, checks referential integrity for any foreign keys in the tuple.
     *  Return true iff the tuple passes the type check and reference check.
     *  @param t  the tuple to be inserted
     */
    def add (t: Tuple): Table =
        if typeCheck (t) && referenceCheck (t) then
            if hasIndex then
                val pkey = new KeyType (pull (t, key))                      // values for primary key part of t
                if index.getOrElse (pkey, null) == null then                // check if it's a duplicate
                    index  += pkey -> t                                     // add to index map
                    tuples += t                                             // add to tuples
                else
                    flaw ("add", s"$name: tuple ${stringOf (t)} has a duplicate value for its primary key") 
                end if
            else
                tuples += t                                                 // no index - allow duplicates
            end if
        end if
        this
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** UPDATE the column with attribute name a using newVal for elements with value
     *  matchVal.  Return true iff at least one tuple is updated.
     *  @param a         the attribute name for the column to be updated
     *  @param newVal    the value used to assign updated values
     *  @param matchVal  the value to be matched to elements
     */
    def update (a: String, newVal: ValueType, matchVal: ValueType): Boolean =
        var updated = false
        if hasIndex && (key contains a) then 
            flaw ("update", "attempt to update an indexed primary key: use delete and add")
        end if

        val j = on(a)
        for i <- tuples.indices do
            if tuples(i)(j) == matchVal then
               tuples(i)(j) = newVal
               updated = true
            end if
        end for
        updated
    end update

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** UPDATE the column with attribute name a using function func for elements with
     *  value matchVal.  Return true iff at least one tuple is updated.
     *  @param a         the attribute name for the column to be updated
     *  @param func      the function used to assign updated values
     *  @param matchVal  the value to be matched to elements
     */
    def update (a: String, func: ValueType => ValueType, matchVal: ValueType): Boolean =
        var updated = false
        if hasIndex && (key contains a) then 
            flaw ("update", "attempt to update an indexed primary key: use delete and add")
        end if

        val funcVal = func (matchVal)
        val j       = on(a)
        for i <- tuples.indices do
            if tuples(i)(j) == matchVal then
               tuples(i)(j) = funcVal
               updated = true
            end if
        end for
        updated
    end update

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** DELETE all tuples in this table satisfying the deletion predicate.
     *  If there is an index, remove those tuples from the index as well.
     *  Return true iff at least one tuple is deleted.
     *  @param predicate  the predicate that specifies which tuples to delete
     */
    def delete (predicate: Predicate): Boolean =
        val rem = tuples.filter (predicate)
        for t <- rem do
            tuples -= t                                                     // remove from tuples
            if hasIndex then index -= new KeyType (pull (t, key))           // remove from index map
        end for
        rem.size > 0
    end delete

    // C O N V E R T

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** CONVERT this table to a matrix of doubles by making the necessary
     *  type transformations.
     *  @see the `fromMatrix` method
     *  @param cols  the column positions to use for forming the matrix
     */
    def toMatrix (cols: Array [Int] = Array.range (0, schema.size)): MatrixD =
        val (m, n) = (tuples.size, cols.size)

        val a = Array.ofDim [Double] (m, n)
        for j <- 0 until n do
            val jj = cols(j)
            domain(j) match
            case 'S' | 'X' => val x = VectorS.map2Int (col(jj).map (_.toString))._1 
                        for i <- 0 until m do a(i)(j) = x(i).toDouble
            case 'T' => val x = VectorT.map2Long (col(jj).map (TimeNum.fromValueType (_)))._1
                        for i <- 0 until m do a(i)(j) = x(i).toDouble
            case _   => for i <- 0 until m do a(i)(j) = tuples(i)(jj).toDouble
        end for

        new MatrixD (m, n, a)
    end toMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** CONVERT this table to a matrix and a vector of doubles by making the necessary
     *  type transformations.
     *  Usage: table -> (X, y) for linear algebra/regression problem Xb = y.
     *  @param cols  the column positions to use for forming the matrix
     *  @param colj  the column position to use for forming the vector
     */
    def toMatrixV (cols: Array [Int] = Array.range (0, schema.size-1),
                   colj: Int = schema.size-1): (MatrixD, VectorD) =
        val (m, n) = (tuples.size, cols.size)

        val a = Array.ofDim [Double] (m, n)
        for j <- 0 until n do
            val jj = cols(j)
            domain(j) match
            case 'S' | 'X' => val x = VectorS.map2Int (col(jj).map (_.toString))._1
                        for i <- 0 until m do a(i)(j) = x(i).toDouble
            case 'T' => val x = VectorT.map2Long (col(jj).map (TimeNum.fromValueType (_)))._1
                        for i <- 0 until m do a(i)(j) = x(i).toDouble
            case _   => for i <- 0 until m do a(i)(j) = tuples(i)(jj).toDouble
        end for

        val b = Array.ofDim [Double] (m)
        domain(colj) match
        case 'S' | 'X' => val x = VectorS.map2Int (col(colj).map (_.toString))._1
                    for i <- 0 until m do b(i) = x(i).toDouble
        case 'T' => val x = VectorT.map2Long (col(colj).map (TimeNum.fromValueType (_)))._1
                    for i <- 0 until m do b(i) = x(i).toDouble
        case _   => for i <- 0 until m do b(i) = tuples(i)(colj).toDouble

        (new MatrixD (m, n, a), new VectorD (m, b))
    end toMatrixV

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** CONVERT the colj column of this table into a vector of doubles, etc.
     *  @param colj  the column position to use for the vector
     */
    def toVectorD (colj: Int = 0): VectorD =
        val b = Array.ofDim [Double] (rows)
        for i <- indices do b(i) = tuples(i)(colj).toDouble
        new VectorD (rows, b)
    end toVectorD

    def toVectorI (colj: Int = 0): VectorI =
        val b = Array.ofDim [Int] (rows)
        for i <- indices do b(i) = tuples(i)(colj).toInt
        new VectorI (rows, b)
    end toVectorI

    def toVectorL (colj: Int = 0): VectorL =
        val b = Array.ofDim [Long] (rows)
        for i <- indices do b(i) = tuples(i)(colj).toLong
        new VectorL (rows, b)
    end toVectorL

    def toVectorS (colj: Int = 0): VectorS =
        val b = Array.ofDim [String] (rows)
        for i <- indices do b(i) = tuples(i)(colj).toString
        new VectorS (rows, b)
    end toVectorS

    def toVectorT (colj: Int = 0): VectorT =
        val b = Array.ofDim [TimeNum] (rows)
        for i <- indices do b(i) = tuples(i)(colj).asInstanceOf [TimeNum]
        new VectorT (rows, b)
    end toVectorT

    // O U T P U T

    private val width_ = 18                                                 // default column width
    private val width  = Array.fill (domain.size) (width_)                  // width for each column

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the width of column col to w.
     *  @param col  the column whose width is to be adjusted
     *  @param w    the new width (# chars) for column col
     */
    def resetWidth (col: Int, w: Int): Unit = width(col) = w

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SHOW/print this table, one tuple per row.
     *  Formatting: regular column is 'width_' chars wide, 'X' is double that
     *  FIX - currently only works for width_, not width
     *  @param rng  the range of tuples to show (e.g, 0 until 10), defaults to all
     */
    def show (rng: Range = tuples.indices): Unit =
        val len = width_ * (schema.size + countX)
        println (s"\n>> Table $name with ${rng.size} rows, primary key = ${stringOf (key)}")
        println ("|-" + "-" * len + "-|")
        print ("| ")
        for j <- schema.indices do
            val wj = if domain(j) == 'X' then 2 * width_ else width_
            prt (schema(j), wj)
        end for
        println (" |")
        println ("|-" + "-" * len + "-|")
        for i <- rng do
            print ("| ")
            val tuple_i = tuples(i)
            if tuple_i.size > domain.size then flaw ("show", s"tuple($i) has size ${tuple_i.size}")
            for j <- tuple_i.indices do
                val wj = if domain(j) == 'X' then 2 * width_ else width_
                prt (tuple_i(j), wj)
            end for
            println (" |")
        end for
        println ("|-" + "-" * len + "-|")
    end show

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** PRINT value-type v with a width of w.
     *  @param v  the value to be printed
     *  @param w  the width (# chars) for the column
     */
    def prt (v: ValueType, w: Int): Unit =

        //here for some valu it is giving negative number instead of null
        if v == null then // add concatenated tuples
            val str="Null"
            val w0 = str.size
            val rem = w - w0
            val lft = max (rem / 2, 0)
            val rht = max (rem - lft, 0)
            print (" " * lft + str +" " * rht)

        else if  v==NO_DOUBLE then

            val str = "Null"
            val w0 = str.size
            val rem = w - w0
            val lft = max (rem / 2, 0)
            val rht = max (rem - lft, 0)
            print (" " * lft + str + " " * rht)
        else if v == NO_INT then

            val str = "Null"
            val w0 = str.size
            val rem = w - w0
            val lft = max (rem / 2, 0)
            val rht = max (rem - lft, 0)
            print (" " * lft + str + " " * rht)
        else
            val str = v.toString
            val w0 = str.size
            val rem = w - w0
            val lft = max (rem / 2, 0)
            val rht = max (rem - lft, 0)

            print (" " * lft + str +" " * rht)
        end if
    end prt


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SHOW/print this table's primary index.
     */
    def show_index (): Unit =
        println (s"\n>> Table $name has indexed primary key = ${stringOf (key)}")
        for (k, v) <- index do println (s"index: ${stringOf (k)} -> ${stringOf (v)}")
    end show_index

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SHOW/print this table's foreign keys.
     */
    def show_foreign_keys (): Unit =
        println (s"\n>> Table $name has foreign keys")
        for lnk <- linkTypes do println ("link = $lnk")
    end show_foreign_keys

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SAVE this table in a file using serialization.
     *  @see load in `Table` object
     */
    def save (): Unit =
        val oos = new ObjectOutputStream (new FileOutputStream (STORE_DIR + name + SER))
        oos.writeObject (this)
        oos.close ()
    end save

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** WRITE this table into a Comma-Separated-Value (CSV) file with each tuple
     *  written to a line.
     *  @param fileName  the file name of the data file (defaults to "name.csv")
     */
    def writeCSV (fileName: String = name + ".csv"): Unit =
        val out = new PrintWriter (DATA_DIR + fileName)
        out.println (stringOf (schema).drop (6).dropRight (1))
        for i <- tuples.indices do
            val tuple_i = stringOf (tuples(i))
            out.println (tuple_i.drop (6).dropRight (1))
        end for
        out.close
    end writeCSV

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** WRITE this table into a JavaScript Object Notation (JSON) file.
     *  @param fileName  the file name of the data file
     */
    def writeJSON (fileName: String = name + ".json"): Unit =
        val gson    = new Gson () 
        val jsonStr = gson.toJson (this)
        debug ("writeJSON", s"jsonStr = ${jsonStr.slice (0, min (jsonStr.size, 5000))}")
        val out = new PrintWriter (DATA_DIR + fileName)
        out.println (jsonStr)
        out.close ()
    end writeJSON

    // P R I V A T E   M E T H O D S

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the tuples in this table that satisfy the given simple (3 token) condition.
     *  @param a1       the left attribute
     *  @param op       the comparison operator (==, !=, <, <=. >, >=)
     *  @param a2       the right attribute or value
     *  @param twoAtrs  the whether a2 is an attribute or value
     *  @param tups     the initial collection of tuples
     */
    protected def selectTups (a1: String, op: String, a2: String, twoAtrs: Boolean,
                              tups: Bag [Tuple] = tuples): Bag [Tuple] =
        if twoAtrs then                                                     // a1 and a2 are attributes
            val a2_ = a2.toString
            op match
            case "==" => tups.filter (t => t(on(a1)) == t(on(a2_)))
            case "!=" => tups.filter (t => t(on(a1)) != t(on(a2_)))
            case "<"  => tups.filter (t => t(on(a1)) <  t(on(a2_)))
            case "<=" => tups.filter (t => t(on(a1)) <= t(on(a2_)))
            case ">"  => tups.filter (t => t(on(a1)) >  t(on(a2_)))
            case ">=" => tups.filter (t => t(on(a1)) >= t(on(a2_)))
            case _    => flaw ("select", s"$op is an unrecognized operator"); tups
        else                                                                // a1 is attribute, a2 is value
            val col = on(a1)
            val a2_ : ValueType = string2Dom (a2, domain (col))
            op match
            case "==" => tups.filter (t => t(col) == a2_)
            case "!=" => tups.filter (t => t(col) != a2_)
            case "<"  => tups.filter (t => t(col) <  a2_)
            case "<=" => tups.filter (t => t(col) <= a2_)
            case ">"  => tups.filter (t => t(col) >  a2_)
            case ">=" => tups.filter (t => t(col) >= a2_)
            case _    => flaw ("select", s"$op is an unrecognized operator"); tups
        end if
    end selectTups

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert a `String` into a `ValueType` with the given domain.
     *  @param str  the given string
     *  @param dom  the domain/data-type to convert it into
     */
    def string2Dom (str: String, dom: Char): ValueType =
        dom match
        case 'D' => str.toDouble
        case 'I' => str.toInt
        case 'L' => str.toLong
        case 'S' | 'X' => str
        case 'T' => TimeNum (str)
    end string2Dom

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the theta-join tuples for this table and r2 keeping concatenated tuples that
     *  satisfy the comparison (theta) operator on the specified attributes.
     *  @param a1  the attribute from the first/this table
     *  @param op  the comparison operator (==, !=, <, <=. >, >=)
     *  @param a2  the attribute from the second table
     *  @param r2  the second table
     */
    private def tJoinTups (a1: String, op: (ValueType, ValueType) => Boolean, a2: String,
                           r2: Table): Bag [Tuple] =
        val tups = Bag [Tuple] ()
        for t <- tuples; u <- r2.tuples do
            if op (t(on(a1)), u(r2.on(a2))) then tups += t ++ u
        end for
        tups
    end tJoinTups

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the rank order of the tuples in this table by performing indirect
     *  merge-sort on the given attributes.
     *  @param x  the attributes to indirectly sort on
     */
    private def rankOrder (x: String*): Array [Int] =
        var perm: Array [Int] = null                                        // permutation giving rank order

        for j <- x.indices do
            val col_j = col (on (x (j)))
            perm = if j == 0 then (new MergeSortIndirect (col_j)()).isort ()
                   else           (new MergeSortIndirect (col_j)(perm)).isort ()
        end for
        perm
    end rankOrder

end Table


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tableTest` main function tests the `Table` class with queries on the
 *  Bank database.
 *  > runMain scalation.database.table.tableTest
 */
@main def tableTest (): Unit =

    // Data Definition Language

    val customer = Table ("customer", "cname, street, ccity", "S, S, S", "cname")
    val branch   = Table ("branch", "bname, assets, bcity", "S, D, S", "bname")
    val deposit  = Table ("deposit", "accno, balance, cname, bname", "I, D, S, S", "accno")
    val loan     = Table ("loan", "loanno, amount, cname, bname", "I, D, S, S", "loanno")

    //--------------------------------------------------------------------------
    banner ("Populate Database")

    customer += ("Peter", "Oak St",   "Bogart")
             += ("Paul",  "Elm St",   "Watkinsville")
             += ("Mary",  "Maple St", "Athens")
    customer.show ()

    branch += ("Alps",     20000000.0, "Athens")
           += ("Downtown", 30000000.0, "Athens")
           += ("Lake",     10000000.0, "Bogart")
    branch.show ()

    deposit += (11, 2000.0, "Peter", "Lake")
            += (12, 1500.0, "Paul",  "Alps")
            += (13, 2500.0, "Paul",  "Downtown")
            += (14, 2500.0, "Paul",  "Lake")
            += (15, 3000.0, "Mary",  "Alps")
            += (16, 1000.0, "Mary",  "Downtown")
    deposit.show ()

    loan += (21, 2200.0, "Peter", "Alps")
         += (22, 2100.0, "Peter", "Downtown")
         += (23, 1500.0, "Paul",  "Alps")
         += (24, 2500.0, "Paul",  "Downtown")
         += (25, 3000.0, "Mary",  "Alps")
         += (26, 1000.0, "Mary",  "Lake")
    loan.show ()

    //--------------------------------------------------------------------------
    banner ("Show Table Statistics")

    customer.stats.show ()
    branch.stats.show ()
    deposit.stats.show ()
    loan.stats.show ()

    //--------------------------------------------------------------------------
    banner ("Verify Usage Queries")

    import Table._

    var a, q: Table = null

    banner (""" customer rename "client" """)
    q = customer rename "client"
    q.show ()

    banner (""" customer project (Array ("street", "ccity")) """)
    q = customer project (Array ("street", "ccity"))
    q.show ()

    banner (""" customer project (Array (1, 2)) """)
    q = customer.project (Array (1, 2))
    q.show ()

    banner (""" customer selproject ("ccity", _ > "Athens") """)
    q = customer selproject ("ccity", _ > "Athens")
    q.show ()

    banner (""" customer select ("ccity", _ == "Athens") """)
    q = customer select ("ccity", _ == "Athens")
    q.show ()

    banner (""" customer select (t => t(customer.on("ccity")) == "Athens") """)
    q = customer select (t => t(customer.on("ccity")) == "Athens")
    q.show ()

    banner (""" customer select ("ccity == 'Athens'") """)
    q = customer select ("ccity == 'Athens'")
    q.show ()

    banner (""" customer select (new KeyType ("Mary")) """)
    q = customer select (new KeyType ("Mary"))
    q.show ()

    banner (""" deposit union loan """)
    a = deposit union loan                                 // save as a for account
    a.show ()

    banner (""" a minus loan """)
    loan.create_index()
    q = a minus loan
    q.show ()

    banner (""" a intersect loan """)
    loan.create_index()
    q = a intersect loan
    q.show ()

    banner (""" customer product deposit """)
    q = customer product deposit
    q.show ()

    banner (""" customer join ((t, u) => t(customer.on("cname")) == u(deposit.on("cname")), deposit) """)
    q = customer join ((t, u) => t(customer.on("cname")) == u(deposit.on("cname")), deposit)
    q.show ()

    banner (""" customer join ("cname == cname", deposit) """)
    q = customer join ("cname == cname", deposit)
    q.show ()

    banner (""" customer join (Array ("cname"), Array ("cname"), deposit) """)
    q = customer join (Array ("cname"), Array ("cname"), deposit)
    q.show ()

    banner (""" deposit join (("cname", customer)) """)
    q = deposit join (("cname", customer))
    q.show ()

    banner (""" customer join deposit """)
    q = customer join deposit
    q.show ()

    banner (""" customer leftJoin (Array ("cname"), Array ("cname"), deposit) """)
    q = customer leftJoin (Array ("cname"), Array ("cname"), deposit)
    q.show ()

    banner (""" deposit.project ("cname, bname") divide branch.project ("bname") """)
    q = deposit.project ("cname, bname") divide branch.project ("bname")
    q.show ()

    banner (""" deposit groupBy "bname" """)
    q = deposit groupBy "bname"
    q.show ()

    banner (""" deposit aggregate ("bname", (avg, "balance")) """)
    q = deposit aggregate ("bname", (avg, "balance"))
    q.show ()

    banner (""" deposit orderBy "bname" """)
    q = deposit orderBy "bname"
    q.show ()

    banner (""" deposit orderByDesc "bname" """)
    q = deposit orderByDesc "bname"
    q.show ()

    //--------------------------------------------------------------------------
    banner ("Example Queries")

    banner ("Names of customers who live in Athens")
    val liveAthens = customer.σ ("ccity == 'Athens'").π ("cname")
    liveAthens.show ()

    banner ("Names of customers who bank (deposits) in Athens")
    val bankAthens = (deposit ⋈ branch).σ ("bcity == 'Athens'").π ("cname")
    bankAthens.show ()

    banner ("Names of customers who live or bank in Athens")
    val liveBank = customer.σ ("ccity == 'Athens'").π ("cname") ⋃
                   (deposit ⋈ branch).σ ("bcity == 'Athens'").π ("cname")
    liveBank.create_index ()
    liveBank.show ()

    banner ("Names of customers who live and bank in the same city")
    val sameCity = (customer ⋈ deposit ⋈ branch).σ ("ccity == bcity").π ("cname")
    sameCity.create_index ()
    sameCity.show ()

    banner ("Names and account numbers of customers with the largest balance")
    val largest = deposit.π ("cname, accno") - (deposit ⋈ ("balance < balance", deposit)).π ("cname, accno") 
    largest.show ()

    banner ("Names of customers who are silver club members")
    val silver = (loan.π ("cname, bname") ⋂ deposit.π ("cname, bname")).π ("cname")
    silver.create_index ()
    silver.show ()

    banner ("Names of customers who are gold club members")
    val gold = loan.π ("cname") - (loan.π ("cname, bname") - deposit.π ("cname, bname")).π ("cname")
    gold.create_index ()
    gold.show ()

    banner ("Names of branches located in Athens")
    val inAthens = branch.σ ("bcity == 'Athens'").π ("bname")
    inAthens.show ()

    banner ("Names of customers who have deposits at all branches located in Athens")
    val allAthens = deposit.π ("cname, bname") / inAthens
    allAthens.create_index ()
    allAthens.show ()

    banner ("Branch names and their average balances")
    val avgBalance = deposit.γ ("bname").aggregate ("bname", (count, "accno"), (avg, "balance"))
    avgBalance.show ()

end tableTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tableTest2` main function tests the `Table` class with queries on the
 *  Student-Course-Professor database.
 *  > runMain scalation.database.table.tableTest2
 */
@main def tableTest2 (): Unit =

    // Data Definition Language

    val student   = Table ("student",   "sid, sname, street, city, dept, level",
                                        "I, S, S, S, S, I", "sid")
    val professor = Table ("professor", "pid, pname, street, city, dept",
                                        "I, S, S, S, S", "pid")
    val course    = Table ("course",    "cid, cname, hours, dept, pid",
                                        "I, X, I, S, I", "cid")
    val takes     = Table ("takes",     "sid, cid",
                                        "I, I", "sid, cid")

    //--------------------------------------------------------------------------
    banner ("Populate Database")

    student += (101, "Peter", "Oak St",   "Bogart",       "CS", 3)
            += (102, "Paul",  "Elm St",   "Watkinsville", "CE", 4)
            += (103, "Mary",  "Maple St", "Athens",       "CS", 4)
    student.show ()

    professor += (104, "DrBill", "Plum St",  "Athens",       "CS")
              += (105, "DrJohn", "Pine St",  "Watkinsville", "CE")
    professor.show ()

    course += (4370, "Database Management", 4, "CS", 104)
           += (4720, "Comp. Architecture",  4, "CE", 104)
           += (4760, "Computer Networks",   4, "CS", 105)
    course.show ()

    takes += (101, 4370)
          += (101, 4720)
          += (102, 4370)
          += (102, 4760)
          += (103, 4760)
    takes.show ()

    // Add links for foreign key contraints and efficient joins (will make any needed primary indices)

    takes.addLinkage ("sid", student)                          // takes sid references student sid
    takes.addLinkage ("cid", course)                           // takes cid references course cid
    course.addLinkage ("pid", professor)                       // course pid references professor pid

    //--------------------------------------------------------------------------
    banner ("Example Queries")

    banner ("locations of students")
    val locs = student project ("sname, city")
    locs.show ()

    banner ("living in Athens")
    val inAthens = student select ("city == 'Athens'")
    inAthens.show ()

    banner ("not living in Athens")
    val notAthens = student minus inAthens
    notAthens.show ()

    banner ("student intersect inAthens")
    val inters = student intersect inAthens
    inters.show ()
    
    banner ("in-Athens union not-in-Athens")
    val unio = inAthens union notAthens
    unio.show ()

    banner ("course taken: course id")
    val taken_id = takes.join (("sid", student))
                        .project ("sname, cid")
    taken_id.show ()

    banner ("course taken: course name")
    val taken_nm = takes.join (("sid", student))
                        .join (("cid", course))
                        .project ("sname, cname")
    taken_nm.show ()

    banner ("student taught by")
    val taught_by = takes.join (("sid", student))
                         .join (("cid", course))
                         .join (("pid", professor))
                         .project ("sname, pname")
    taught_by.show ()

end tableTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tableTest3` main function tests the `Table` object and class by loading
 *  a dataset from a file.  It loads the ScalaTion "covid_19" dataset/CSV file.
 *  - RELATIVE PATHS are from ScalaTion's DATA-DIR (@see Util.scala)
 *  - FULL PATHS are from the OS's root directory
 *  Defaults to RELATIVE PATHS; use `setFullPath` method to change.
 *  > runMain scalation.database.table.tableTest3
 */
@main def tableTest3 (): Unit =

    //--------------------------------------------------------------------------
    // Verify access to file contents, comment out readFile once verified.
    //--------------------------------------------------------------------------

    val fileName = "covid_19.csv"
    println (s"fileName = $fileName")
//  readFile (fileName)                                      // for RELATIVE PATHS
//  readFile (fileName, fullPath = true)                     // for FULL PATHS

    //--------------------------------------------------------------------------
    // Use sample row/tuple in the middle of the file that has full information.
    //--------------------------------------------------------------------------

    val data_str = """
12/29/2020,19658043,205972,184889.714,342639,3611,2372.857,1.04,27782,122664,106708,
253765556,1887683,1484784,0.134,7.5,4387280,4282967,31140,722024,333594,325788
"""

    //--------------------------------------------------------------------------
    // Use this to guess the data-types/domains.  See last step for making corrections.
    //--------------------------------------------------------------------------

    val domain = Table.tuple2type (strim (data_str))
    println (s"domain = ${stringOf (domain)}")

    //--------------------------------------------------------------------------
    // Data stored relative to the "scalation_2.0/data" directory, if not use full path.
    // Call the Table.load method:
    //     def load (fileName: String, name: String, domain: Domain, key: String,
    //               pos_ : Array [Int] = null, sep: String = ","): Table =
    //--------------------------------------------------------------------------

    val covid = Table.load (fileName, "covid", domain, "date")
    covid.show (0 until 200)

    //--------------------------------------------------------------------------
    // If this fails due to incorrect domains, save the domain that was printed,
    // correct the domains that are incorrect, and try again.
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // for fullPath: Table.setFullPath ()
    // for limit:    Table.setLimit (200)
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // Serialize and output the data into a JSON file (covid.json) in DATA_DIR
    //--------------------------------------------------------------------------

    covid.writeJSON ()

end tableTest3

