pluginManagement {
    repositories {
        google(); mavenCentral(); gradlePluginPortal()
        maven(url = "https://jitpack.io")
    }
    plugins {
        id("com.android.application") version "8.3.2"
        id("org.jetbrains.kotlin.android") version "2.2.0"
        id("org.jetbrains.kotlin.plugin.compose") version "2.2.0"
    }
    // (선택) 모든 kotlin 플러그인 요청을 2.2.0으로 통일
    resolutionStrategy {
        eachPlugin {
            if (requested.id.id.startsWith("org.jetbrains.kotlin")) useVersion("2.2.0")
        }
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories { google(); mavenCentral(); maven(url = "https://jitpack.io") }
}
rootProject.name = "HandsFree"
include(":app")
