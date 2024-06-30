import pde

grid = pde.UnitGrid([16, 16])
state = pde.ScalarField.random_uniform(grid, 0.49, 0.51)
eq = pde.PDE({"c": "laplace(c**3-c-laplace(c))-0.01*(c-0.5)"})

final_state = eq.solve(
    state,
    t_range=1e4,
    adaptive=True,
    solver="explicit_mpi",
    decomposition=[2, -1],
)
