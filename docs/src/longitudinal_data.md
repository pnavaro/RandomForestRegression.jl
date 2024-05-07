# Longitudinal data generator


```@docs
DataLongGenerator
```

```@example generator
using LaTeXStrings
using Plots
using RandomForestRegression
```

```@example generator
data = DataLongGenerator(n=17, p=6, G=6)
```

```@example generator
w = findall(data.id .== 1)
p = plot(data.time[w], data.Y[w], c="grey")
for i in unique(data.id)
  w = findall(data.id .== i)
  plot!(p, data.time[w], data.Y[w], c="grey", label = :none)
end
p
```

Let's see the fixed effects predictors:

```@example generator
p = plot(layout=(2,3))
for i in axes(data.X,2)
    w = findall(data.id .== 1)
    plot!(p[i], data.time[w],data.X[w,i], label = :none)
    title!(p[i], L"$X^{%$i}$")
    for k in unique(data.id)
        w = findall(data.id .== k)
        plot!(p[i], data.time[w], data.X[w,i], label = :none)
    end
end
p
```
