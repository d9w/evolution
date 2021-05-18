using Statistics
using LinearAlgebra

function ES(layers::Array, num_generations::Int, f::Function, npop::Int=50, sigma=0.5, alpha=0.01)
    µ = Individual(layers::Array) # Center
    n = length(µ.genes)
    evaluate!(µ, f)

    expert = µ # Best individual so far

    fits = zeros(num_generations)

    for i in eachindex(fits)
        population = Array{Individual}(undef, 𝜆)
        best = 1
        for j in eachindex(population)
            new_genes = expert.genes + randn(n) .* sigma
            population[j] = Individual(layers, new_genes)
            evaluate!(population[j], f)
            if population[j].fitness > population[best].fitness
                best = j
            end
        end

        if population[best].fitness >= expert.fitness
            expert = population[best]
        end

        fits[i] = expert.fitness

        u = max.(0, log(n/2+1) .- log.(1:n))
        u .= u ./ sum(u) .- 1/n

        # Sort from highest to lowest fitness
        idx = sortperm(population, by=x -> x.fitness, rev=true)

        step = zeros(n)
        for i in eachindex(population)
            step = step.+ u[i] .* population[idx[i]].genes
        end

        µ.genes = µ.genes .+ step .* alpha

        print(i, "/",  num_generations, " > ", expert.fitness, "\t\t\r")
    end
    fits, expert
end
