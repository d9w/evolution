function one_plus_lambda(layers::Array, num_generations::Int, f::Function, 𝜆::Int)

    fits = zeros(num_generations)
    expert = Individual(layers::Array)
    evaluate!(expert, f)

    for i in eachindex(fits)
        population = Array{Individual}(undef, 𝜆)
        best = 1
        for j in eachindex(population)
            population[j] = mutate(expert)
            evaluate!(population[j], f)
            if population[j].fitness > population[best].fitness
                best = j
            end
        end

        if population[best].fitness >= expert.fitness
            expert = population[best]
        end

        fits[i] = expert.fitness
        print(i, "/",  num_generations, " > ", expert.fitness, "\t\t\r")
    end
    fits, expert
end
