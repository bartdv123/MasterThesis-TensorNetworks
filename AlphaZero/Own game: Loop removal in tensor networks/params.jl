#####
##### Training hyperparameters
#####

Network = NetLib.ResNet

netparams = NetLib.ResNetHP(
  num_filters=64,
  num_blocks=3,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=32,
  num_value_head_filters=32,
  batch_norm_momentum=0.1)

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=600,
    num_workers=128,
    batch_size=64,
    use_gpu=false,
    reset_every=2,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=50,
    cpuct=2.0,
    prior_temperature=1.0,
    temperature=PLSchedule([0, 20, 30], [1.0, 1.0, 0.3]),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  sim=SimParams(
    num_games=50,
    num_workers=50,
    batch_size=50,
    use_gpu=false,
    reset_every=2,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.2),
    dirichlet_noise_ϵ=0.1),
  update_threshold=50000)

learning = LearningParams(
  use_gpu=512,
  use_position_averaging=true,
  samples_weighing_policy=LOG_WEIGHT,
  batch_size=512,
  loss_computation_batch_size=512,
  optimiser=Adam(lr=1e-1),
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=512,
  num_checkpoints=1)

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=3,
  ternary_outcome=false,
  use_symmetries=false,
  memory_analysis=nothing,
  mem_buffer_size=PLSchedule(
  [      0,        15],
  [400_000, 1_000_000]))

#####
##### Evaluation benchmark
#####


alphazero_player = Benchmark.Full(arena.mcts)
network_player = Benchmark.NetworkOnly(τ=0.5)

# benchmark the network only again the full agent
benchmark_sim = SimParams(
    arena.sim;
    num_games=50,
    num_workers=1,
    batch_size=1,
    alternate_colors=false)
  
benchmark = [
    Benchmark.Single(
    Benchmark.Full(self_play.mcts),
    benchmark_sim),
    Benchmark.Single(
    Benchmark.NetworkOnly(),
    benchmark_sim)]
