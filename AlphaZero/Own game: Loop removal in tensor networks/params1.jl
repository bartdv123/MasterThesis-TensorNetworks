#####
##### Training hyperparameters
#####

network = NetLib.ResNet

netparams = NetLib.ResNetHP(                                                    # important parameters of RESNET
  num_filters=64,
  num_blocks=3,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=16,
  num_value_head_filters=16)

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=250,
    num_workers=5,
    batch_size=5,
    use_gpu=false,
    reset_every=5,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=40,
    cpuct=1.0,
    prior_temperature=1.0,                                                      # important
    temperature=ConstSchedule(0.2),
    #FORCE EXPlORATION
    dirichlet_noise_ϵ=0.33,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  sim=SimParams(
    num_games=50,
    num_workers=5,
    batch_size=5,
    use_gpu=false,
    reset_every=5,
    flip_probability=0,
    alternate_colors=false),
  mcts=MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.2),
    dirichlet_noise_ϵ=0.3,
    dirichlet_noise_α=1),
  update_threshold=100000)

learning = LearningParams(
  use_gpu=false,
  use_position_averaging=true,
  samples_weighing_policy=LOG_WEIGHT,
  batch_size=256,
  loss_computation_batch_size=256,
  optimiser=Adam(lr=0.2),                                                       # adam = important
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=128,
  num_checkpoints=1)

  params = Params(
    arena=arena,
    self_play=self_play,
    learning=learning,
    num_iters=5,
    # Remove ternary_outcome
    use_symmetries=false,
    memory_analysis=nothing,
    mem_buffer_size=PLSchedule(
        [      0,        3],
        [4_000, 8_000]))
  
#####
##### Evaluation benchmark
#####



experiment = AlphaZero.Experiment("Toymodel", GameSpec(), params, network, netparams, benchmark)