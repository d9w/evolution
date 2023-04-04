def tournament_selection(population, fitness, t_size=3):
    rng = np.random.default_rng()
    tournament = rng.choice(len(population), size=t_size)
    ind = tournament[np.argmin(fitness[tournament])]
    return population[ind], fitness[ind]
