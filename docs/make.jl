using Documenter
using RandomForestRegression
using DocumenterCitations

ENV["GKSwstype"] = "100"

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"); style=:authoryear)

makedocs(
    sitename = "RandomForestRegression",
    format = Documenter.HTML(),
    modules = [RandomForestRegression],
    pages = [
        "Documentation" => "index.md",
        "Datasets" => "longitudinal_data.md",
    ],
    plugins = [bib],
)

deploydocs(
    branch = "gh-pages",
    devbranch = "main",
    repo = "github.com/pnavaro/RandomForestRegression.jl.git",
)
