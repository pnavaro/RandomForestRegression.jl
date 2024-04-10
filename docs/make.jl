using Documenter
using RandomForestRegression

makedocs(
    sitename = "RandomForestRegression",
    format = Documenter.HTML(),
    modules = [RandomForestRegression]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
