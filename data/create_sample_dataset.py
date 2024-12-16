if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Dataset parameters
    timesteps = 1000
    components = 3
    nx, ny, nz = 50, 50, 10  # Volume dimensions

    # Initialize the dataset
    data = np.zeros((timesteps, components, nx, ny, nz))

    # Spatial grid
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Center coordinates
    cx, cy = nx // 2, ny // 2

    # Shape parameters
    min_r_u = 5
    max_r_u = 20
    min_s_v = 5
    max_s_v = 40
    min_s_w = 5
    max_s_w = 40
    cycle_length = 14

    # Generate the dataset
    for t in range(timesteps):
        t_mod = t % cycle_length
        if t_mod < cycle_length / 2:
            frac = t_mod / (cycle_length / 2)
        else:
            frac = (cycle_length - t_mod) / (cycle_length / 2)
            
        # u component: Cylinder with varying radius
        r_u = min_r_u + (max_r_u - min_r_u) * frac
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask_u = dist <= r_u
        data[t, 0][mask_u] = 1

        # v component: Square with varying size
        s_v = min_s_v + (max_s_v - min_s_v) * frac
        half_s_v = s_v / 2
        mask_v = (np.abs(X - cx) <= half_s_v) & (np.abs(Y - cy) <= half_s_v)
        data[t, 1][mask_v] = 1

        # w component: Triangle with varying size
        s_w = min_s_w + (max_s_w - min_s_w) * frac
        base = s_w
        height = s_w
        rel_X = X - cx
        rel_Y = Y - (cy - height / 2)  # Shift Y so that the base is at y = cy - height / 2

        mask_w = (rel_Y >= 0) & \
                (rel_Y <= height) & \
                ((np.abs(rel_X) / (base / 2)) + (rel_Y / height) <= 1)

        data[t, 2][mask_w] = 1

    noise_level = 0.2  # Adjust noise level as needed
    noise = noise_level * np.random.randn(*data.shape)
    data = data + noise
    data = np.clip(data, 0, 1)  # Ensure values are in the range [0, 1]

    np.save("test_data.npy", data.astype(np.float16))
    # Plotting the components
    z_slice = nz // 2
    z_slice = 0
    timesteps_to_plot = list(range(6))

    fig, axs = plt.subplots(len(timesteps_to_plot), 3, figsize=(8, 32))

    for i, t in enumerate(timesteps_to_plot):
        u_slice = data[t, 0, :, :, z_slice]
        v_slice = data[t, 1, :, :, z_slice]
        w_slice = data[t, 2, :, :, z_slice]

        axs[i, 0].imshow(u_slice.T, origin='lower', cmap='viridis')
        axs[i, 0].set_title(f'u at timestep {t}')
        axs[i, 1].imshow(v_slice.T, origin='lower', cmap='viridis')
        axs[i, 1].set_title(f'v at timestep {t}')
        axs[i, 2].imshow(w_slice.T, origin='lower', cmap='viridis')
        axs[i, 2].set_title(f'w at timestep {t}')

        for ax in axs[i]:
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')

    plt.tight_layout()
    plt.show()
