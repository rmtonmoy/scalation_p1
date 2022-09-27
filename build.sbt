
lazy val scalation = project.in(file("."))
  .settings(
    scalaVersion  := "3.2.0",
    scalacOptions ++= Seq(
       "-deprecation",         // emit warning and location for usages of deprecated APIs
       "-explain",             // explain errors in more detail
//     "-explain-types",       // explain type errors in more detail
       "-new-syntax",          // require `then` and `do` in control expressions.
       "-Xfatal-warnings")     // fail the compilation if there are any warnings
//  scalacOptions += "-deprecation",
//  javacOptions  += "--add-modules jdk.incubator.vector"
  )

fork := true

resolvers += Opts.resolver.sonatypeSnapshots

libraryDependencies += "com.google.code.gson" % "gson" % "2.9.1"

/*
lazy val osName = System.getProperty("os.name") match {
    case n if n.startsWith("Linux") => "linux"
    case n if n.startsWith("Mac") => "mac"
    case n if n.startsWith("Windows") => "win"
    case _ => throw new Exception("Unknown platform!")
}

libraryDependencies += "org.openjfx" % "javafx-base" % "16" classifier osName
libraryDependencies += "org.openjfx" % "javafx-controls" % "16" classifier osName
libraryDependencies += "org.openjfx" % "javafx-fxml" % "16" classifier osName
libraryDependencies += "org.openjfx" % "javafx-graphics" % "16" classifier osName
libraryDependencies += "org.openjfx" % "javafx-media" % "16" classifier osName
libraryDependencies += "org.openjfx" % "javafx-swing" % "16" classifier osName
libraryDependencies += "org.openjfx" % "javafx-web" % "16" classifier osName
*/

// libraryDependencies += "org.typelevel" %% "cats-core" % "x.y.z"

