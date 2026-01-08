"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

@Desc: 一个用于仿真 ptd模块逻辑的代码 （原则就是尽量保证训练过程中，保证通讯在节点之间通讯少，在节点内通讯较多NVLink）
    （区别 rank, node, local_rank, group）：
        world_size :全局进程总个数，即在一个分布式任务中rank的数量，如果一个GPU代表一个进程，那么world_size就可以理解成GPU的数量(注意， rank与GPU之间没有必然的对应关系，一个rank可以包含多个GPU；一个GPU也可以为多个rank服务)
        tensor_model_parallel_size : 张量并行度，表示在张量并行计算时，每个GPU负责处理的张量维度大小。
        pipeline_model_parallel_size :流水线并行度，表示在流水线并行计算时，每个GPU上处理的模型的子集。

    Megatron-LM中有个专家并行策略，这个和上述仿真无关，但是其具体初始化分布的时候，用了几个参数。

    要找到一个好的并行映射来帮助您实现新模型的高吞吐量，有一些一般规则可能会有所帮助。针对特定模型，最佳的并行映射会根据模型架构、训练序列长度和硬件平台而有所不同。以下是一些提高性能的通用规则：
        1.尽量保持模型并行度（MP）尽可能小： 对于大型语言模型，为了避免内存不足（OOM），通常需要使用模型并行性（MP），但这会带来通信开销并影响性能。 使用分布式优化器时，主权重和优化器状态会在所有数据并行（DP）任务中分片，并带来轻微的通信开销。因此，当训练时有大量空余的GPU内存时，尽量减少模型并行度，增加数据并行度。
        2.确保EP和TP的通信在NVLink域内： EP（专家并行）和TP（张量并行）的通信应该尽可能保持在NVLink域内，因为这两者都是通信密集型操作。
        3.如果模型过大且需要跨多个节点扩展，优先考虑PP（流水并行）而不是TP和EP： 参考第3点的细节。
        4.使用流水并行（Pipeline Parallelism）进一步扩展模型： 当PP_size（流水并行度）>= 2时，启用虚拟流水并行（Virtual Pipeline Parallelism，VPP），通过设置 .num_layers_per_virtual_pipeline_stage 来减少流水并行的空闲周期。
        5.VPP_size调整： VPP_size的合法取值为 num_layers/pp_size 的所有公约数。例如，num_layers=24，pp_size=4，那么我们可以从 {1, 2, 3, 6} 中选择vpp_size。vpp_size越大，流水空闲越少，但每个PP阶段之间的P2P通信次数越多。经验上，中间值通常能达到最佳平衡。VPP_size = num_layers / PP_size / num_layers_per_virtual_pipeline_stage。
        6.专家层中尽量选择EP而非TP： TP比EP节省更多内存，但EP可以获得更高的GEMM效率，并减少通信开销。 如果EP的大小增加到与专家数量相等，则在专家计算中可以省略本地令牌的排列/反排列操作。 简化MoE层的计算图，便于实现潜在的通信和计算重叠。
        7。在长上下文训练中启用上下文并行（Context Parallelism）： CP（上下文并行）的效率在很大程度上取决于其通信是否能与计算重叠。 经验上，当序列长度>=8K时使用CP。




@Author : tz
@Date: 2024/10/14  19:28

"""

# world_size = 16
# tensor_model_parallel_size = 2
# pipeline_model_parallel_size = 4

world_size = 192
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 8

world_size = 128
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 8

world_size = 64
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 8

world_size = 160
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 4

world_size = 224
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 4

world_size = 96
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 4


world_size = 16
tensor_model_parallel_size = 2
pipeline_model_parallel_size = 2


data_parallel_size = world_size // (tensor_model_parallel_size *
                                    pipeline_model_parallel_size)  # 2
num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size  # 8
num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size  # 4
num_data_parallel_groups = world_size // data_parallel_size  # 8

# Build the data-parallel groups.
print("Build DP(Data Parallel) Groups :")
all_data_parallel_group_ranks = []
for i in range(pipeline_model_parallel_size):
    start_rank = i * num_pipeline_model_parallel_groups
    end_rank = (i + 1) * num_pipeline_model_parallel_groups
    for j in range(tensor_model_parallel_size):
        ranks = range(start_rank + j, end_rank,
                      tensor_model_parallel_size)
        all_data_parallel_group_ranks.append(list(ranks))
print(all_data_parallel_group_ranks)

# Build the model-parallel groups.
print("Build MP(Model Parallel) Group:")
for i in range(data_parallel_size):
    ranks = [data_parallel_group_ranks[i]
             for data_parallel_group_ranks in all_data_parallel_group_ranks]
    print(list(ranks))

# Build the tensor model-parallel groups.
print("Build TP(Tensor model-Parallel) Groups:")
for i in range(num_tensor_model_parallel_groups):
    ranks = range(i * tensor_model_parallel_size,
                  (i + 1) * tensor_model_parallel_size)
    print(list(ranks))

# Build the pipeline model-parallel groups and embedding groups
print("Build PP(Pipeline model-Parallel) Groups :")
for i in range(num_pipeline_model_parallel_groups):
    ranks = range(i, world_size,
                  num_pipeline_model_parallel_groups)
    print(list(ranks))
