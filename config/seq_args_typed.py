class SeqTypedArgs:
    checkpoint: str
    
    # Dataset
    dataset: str
    data_location: list[str]
    max_autoregressive_steps: int
    autoregressive_steps_valid: int
    start_n: int
    n_span: int
    stride_min: int
    stride_max: int
    stride_valid: int
    trajec_max_len: int
    trajec_max_len_valid: int

    # Model
    model_name: str
    coarse_dim: tuple[int, int, int]
    n_velocities: int
    start_n_valid: int
    n_span_valid: int
    n_layer: int
    output_hidden_states: bool
    output_attentions: bool
    num_timesteps: int
    n_embd: int
    n_head: int
    embd_pdrop: float
    layer_norm_epsilon: float
    attn_pdrop: float
    resid_pdrop: float
    activation_function: str
    initializer_range: float
    is_flatten: bool

    # Training
    previous_model_name: str
    start_autoregressive_steps: int
    d_autoregressive_steps: int
    batch_size: int
    batch_size_valid: int
    shuffle: bool
    device: str
    epoch_num: int
    epoch_valid_interval: int
    epoch_save_interval: int
    iter_per_epoch: int
    learning_rate: float
    lr_decay_factor: float
    lr_decay_patience: float
    coarse_mode: str
    march_loss_threshold: float

    # Dynamically added attributes
    coarse_product: int
    time: str
    dir_output: str
    fname: str
    experiment_path: str
    model_save_path: str
    logging_path: str
    current_model_save_path: str
    logging_epoch_path: str
