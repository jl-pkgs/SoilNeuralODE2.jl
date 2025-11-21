= Enzyme 运行时活跃性说明

== 错误原因
错误 `EnzymeRuntimeActivityError: Detected potential need for runtime activity` 发生在 Enzyme 的静态分析无法在编译时确定内存位置的活跃状态（即变量是否携带导数）时。

== 代码中的起因
在 `time_step_forward` 函数中，数组 `h_preds` 表现出混合的活跃性：

1.  *初始化（常量）*：
    ```julia
    h_preds[:, 1, :] .= h_init
    ```
    `h_init` 来源于 `h_true`（观测数据），它作为 `Const` 传递给损失函数。因此，`h_preds` 的第一个时间步是 *常量*（不携带梯度）。

2.  *更新循环（活跃）*：
    ```julia
    h_preds[:, t+1, :] .= h_prev .+ dh
    ```
    `dh` 是神经网络的输出，依赖于参数 `ps`。由于 `ps` 正在被优化（标记为 `Duplicated` 或 `Active`），`dh` 携带梯度。因此，`h_preds` 的后续时间步存储的是 *活跃* 值。

== 冲突点
Enzyme 的默认模式依赖 *静态分析* 来决定数组是否需要梯度的影子内存（shadow memory）。它期望一个内存位置始终是常量或始终是活跃的。当 `h_preds` 同时存储常量值（在 $t=1$ 时）和活跃值（在 $t > 1$ 时），静态分析无法保证正确性。

== 解决方案
`Enzyme.set_runtime_activity(Reverse)` 指示 Enzyme 启用 *运行时活跃性* 跟踪。

-   它允许 Enzyme 在程序执行期间动态跟踪内存位置的活跃性。
-   这正确处理了 `h_preds` 的混合状态，其中部分是常量，部分是活跃的。
-   *权衡*：这确保了像 ODE 求解器或具有固定初始条件的时间步进循环等算法的正确性，代价是相比纯静态分析会有少量的性能开销。
