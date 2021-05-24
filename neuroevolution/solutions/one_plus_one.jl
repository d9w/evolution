function one_plus_one(layers::Array, num_generations::Int, f::Function)
    fits = zeros(num_generations)
    parent = Individual(layers::Array)
    evaluate!(parent, f)

    for i in eachindex(fits)
        child = mutate(parent)
        evaluate!(child, f)

        if child.fitness >= parent.fitness
            parent = child
        end

        fits[i] = parent.fitness
        print(i, "/",  num_generations, " > ", parent.fitness, "\t\t\r")
    end
    fits, parent
end
