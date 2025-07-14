import tensorrt as trt

verbose = True
IN_NAME = 'input'
OUT_NAME = 'output'
IN_H = 224
IN_W = 224
BATCH_SIZE = 1

# 显式batch模式（TensorRT推荐）
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# 创建TensorRT日志对象
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
# 创建Builder、Config和Network对象
with trt.Builder(TRT_LOGGER) as builder, \
     builder.create_builder_config() as config, \
     builder.create_network(EXPLICIT_BATCH) as network:
    # 定义网络结构
    # 添加输入张量，shape为(BATCH_SIZE, 3, IN_H, IN_W)
    input_tensor = network.add_input(
        name=IN_NAME, dtype=trt.float32, shape=(BATCH_SIZE, 3, IN_H, IN_W))
    # 添加最大池化层，窗口为2x2
    pool = network.add_pooling_nd(
        input=input_tensor, type=trt.PoolingType.MAX, window_size=(2, 2))
    # 设置池化层步长为2x2
    pool.stride_nd = (2, 2)
    # 设置输出张量名
    pool.get_output(0).name = OUT_NAME
    # 标记网络输出
    network.mark_output(pool.get_output(0))

    # 创建动态shape profile（即使是定长也需profile）
    profile = builder.create_optimization_profile()
    # 设置输入张量的最小/最优/最大shape（都为同一个shape，表示定长）
    profile.set_shape_input('input', *[[BATCH_SIZE, 3, IN_H, IN_W]] * 3)
    # 添加profile到config
    config.add_optimization_profile(profile)
    # 设置最大工作空间为1GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    # 序列化网络为engine
    engine = builder.build_serialized_network(network, config)

    # 保存engine到文件
    with open('model.engine', mode='wb') as f:
        f.write(engine)
        print('generating file done!')
