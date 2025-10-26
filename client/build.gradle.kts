// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    id("com.android.application") version "8.3.2" apply false
    id("org.jetbrains.kotlin.android") version "2.2.0" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.2.0" apply false
}

// 강제 통일은 루트에서!
allprojects {
    configurations.configureEach {
        resolutionStrategy.eachDependency {
            if (requested.group == "org.jetbrains.kotlin") {
                useVersion("2.2.0")
            }
        }
        resolutionStrategy.force(
            "org.jetbrains.kotlin:kotlin-build-tools-api:2.2.0",
            "org.jetbrains.kotlin:kotlin-gradle-plugin-api:2.2.0",
            "org.jetbrains.kotlin:kotlin-compiler-embeddable:2.2.0",
            "org.jetbrains.kotlin:kotlin-tooling-core:2.2.0",
            "org.jetbrains.kotlin:kotlin-stdlib:2.2.0",
            "org.jetbrains.kotlin:kotlin-stdlib-jdk8:2.2.0",
            "org.jetbrains.kotlin:kotlin-reflect:2.2.0"
        )
    }
}
