# Execution Trace Analysis for Deep Learning Bottleneck Detection
Parallel Programming Technology Lab (Summer Term 2026), TU Darmstadt

In neural network training few operations can hinder the whole training process.  Identifying these bottlenecks is the most direct way to achieve significant performance gains, yet doing so has become increasingly difficult as training strategies grow in complexity.

The goal of this project is to develop an execution trace analysis tool designed for modern high-performance setups. While inspired by frameworks like chrome://tracing and perfetto.dev, we are extending the functionality to meet the needs of state-of-the-art distributed training:

**Support for Advanced Strategies**: Optimized for complex workloads like Fully Sharded Data Parallel (FSDP) training.

**Automatic Logic Mapping**: Automatically linking raw hardware execution data back to high-level logical operations in the model.

**Automated Detection**: Moving beyond manual visualization to actively flag and categorize performance bottlenecks.