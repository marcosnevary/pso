@partial(
    jit,
    static_argnames=(
        "objective_function",
        "n_dimensions",
        "n_particles",
        "max_iterations",
        "n_gradient_steps",
        "gradient_interval",
        "learning_rate",
    ),
)
def psox_jax(
    seed: random.PRNGKey,
    objective_function: callable,
    lower_bound: float,
    upper_bound: float,
    n_dimensions: int,
    n_particles: int,
    inertia_weight: float,
    social_coeff: float,
    cognitive_coeff: float,
    max_iterations: int,
    learning_rate: float,
    n_gradient_steps: int,
    gradient_interval: int,
) -> tuple:
    key = seed
    lower = jnp.array(lower_bound)
    upper = jnp.array(upper_bound)
    position_key, velocity_key, state_key = random.split(key, 3)

    search_range = upper - lower
    velocity_scale = 0.1
    velocity_limit = search_range * velocity_scale

    initial_positions = random.uniform(
        position_key,
        (n_particles, n_dimensions),
        minval=lower,
        maxval=upper,
    )
    initial_velocities = random.uniform(
        velocity_key,
        (n_particles, n_dimensions),
        minval=-velocity_limit,
        maxval=velocity_limit,
    )
    initial_fitness = vmap(objective_function)(initial_positions)

    best_particle_idx = jnp.argmin(initial_fitness)
    global_best_position = initial_positions[best_particle_idx]
    global_best_fitness = initial_fitness[best_particle_idx]

    initial_state = JaxSwarmState(
        positions=initial_positions,
        velocities=initial_velocities,
        personal_best_positions=initial_positions,
        personal_best_fitness=initial_fitness,
        global_best_position=global_best_position,
        global_best_fitness=global_best_fitness,
        rng=state_key,
    )

    gd_solver = jaxopt.GradientDescent(
        fun=objective_function,
        stepsize=learning_rate,
        implicit_diff=False,
    )

    def update_step(swarm_state: JaxSwarmState, i: int) -> tuple:
        cognitive_key, social_key, next_key = random.split(swarm_state.rng, 3)
        r1 = random.uniform(cognitive_key, (n_particles, n_dimensions))
        r2 = random.uniform(social_key, (n_particles, n_dimensions))

        inertia_term = inertia_weight * swarm_state.velocities
        cognitive_term = (
            cognitive_coeff
            * r1
            * (swarm_state.personal_best_positions - swarm_state.positions)
        )
        social_term = (
            social_coeff
            * r2
            * (swarm_state.global_best_position - swarm_state.positions)
        )

        updated_velocities = inertia_term + cognitive_term + social_term
        updated_positions = swarm_state.positions + updated_velocities
        updated_positions = jnp.clip(updated_positions, lower, upper)

        updated_fitness = vmap(objective_function)(updated_positions)

        personal_improved = updated_fitness < swarm_state.personal_best_fitness

        updated_personal_best_positions = jnp.where(
            personal_improved[:, None],
            updated_positions,
            swarm_state.personal_best_positions,
        )
        updated_personal_best_fitness = jnp.where(
            personal_improved,
            updated_fitness,
            swarm_state.personal_best_fitness,
        )

        global_best_candidate_idx = jnp.argmin(updated_personal_best_fitness)
        global_best_candidate_fitness = updated_personal_best_fitness[
            global_best_candidate_idx
        ]

        global_improved = (
            global_best_candidate_fitness < swarm_state.global_best_fitness
        )

        global_best_candidate_position = jnp.where(
            global_improved,
            updated_personal_best_positions[global_best_candidate_idx],
            swarm_state.global_best_position,
        )
        updated_global_best_fitness = jnp.where(
            global_improved,
            global_best_candidate_fitness,
            swarm_state.global_best_fitness,
        )

        def apply_gd(_: None) -> tuple:
            initial_gd_state = gd_solver.init_state(global_best_candidate_position)

            def gd_step(carry: tuple, _: None) -> tuple:
                position, gd_state = carry

                next_position, next_gd_state = gd_solver.update(position, gd_state)
                next_position = jnp.clip(next_position, lower, upper)

                return (next_position, next_gd_state), None

            (gradient_best_position, _), _ = lax.scan(
                gd_step,
                (global_best_candidate_position, initial_gd_state),
                None,
                n_gradient_steps,
            )

            gradient_best_fitness = objective_function(gradient_best_position)
            return gradient_best_position, gradient_best_fitness

        def skip_gd(_: None) -> tuple:
            return global_best_candidate_position, updated_global_best_fitness

        gradient_best_position, gradient_best_fitness = lax.cond(
            i % gradient_interval == 0,
            apply_gd,
            skip_gd,
            None,
        )

        gradient_improved = gradient_best_fitness < updated_global_best_fitness
        updated_global_best_position = jnp.where(
            gradient_improved,
            gradient_best_position,
            global_best_candidate_position,
        )
        updated_global_best_fitness = jnp.where(
            gradient_improved,
            gradient_best_fitness,
            updated_global_best_fitness,
        )

        any_improvement = updated_global_best_fitness < swarm_state.global_best_fitness

        winner_mask = (jnp.arange(n_particles) == global_best_candidate_idx)[:, None]
        should_update_mask = (gradient_improved & any_improvement) & winner_mask

        final_personal_best_positions = jnp.where(
            should_update_mask,
            updated_global_best_position,
            updated_personal_best_positions,
        )
        final_personal_best_fitness = jnp.where(
            (gradient_improved & any_improvement)
            & (jnp.arange(n_particles) == global_best_candidate_idx),
            updated_global_best_fitness,
            updated_personal_best_fitness,
        )

        next_state = JaxSwarmState(
            positions=updated_positions,
            velocities=updated_velocities,
            personal_best_positions=final_personal_best_positions,
            personal_best_fitness=final_personal_best_fitness,
            global_best_position=updated_global_best_position,
            global_best_fitness=updated_global_best_fitness,
            rng=next_key,
        )

        return next_state, updated_global_best_fitness

    final_state, fitness_history = lax.scan(
        update_step,
        initial_state,
        jnp.arange(max_iterations),
    )
    full_fitness_history = jnp.concatenate(
        [jnp.array([initial_state.global_best_fitness]), fitness_history],
    )

    return (
        final_state.global_best_position,
        final_state.global_best_fitness,
        full_fitness_history,
    )
