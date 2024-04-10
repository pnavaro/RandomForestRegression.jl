using Documenter
using RandomForestRegression

makedocs(
    sitename = "RandomForestRegression",
    format = Documenter.HTML(),
    modules = [RandomForestRegression]
)

deploydocs(
    branch = "gh-pages",
    devbranch = "main",
    repo = "github.com/pnavaro/RandomForestRegression.jl.git",
)
