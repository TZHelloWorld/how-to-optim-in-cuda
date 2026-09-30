// 编译/运行见 README.md。要求 CUDA Toolkit 12+、C++17；无需 cuBLAS/CUTLASS。
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "kernels.cuh"

using namespace advanced_gemm;

static void cuda_check(cudaError_t status, const char* call, int line) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(call) + " at line " + std::to_string(line) +
                                 ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(call) cuda_check((call), #call, __LINE__)

template<class T> struct Buffer {
    T* data = nullptr;
    std::size_t count;
    explicit Buffer(std::size_t n) : count(n) {
        if (!n || n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::runtime_error("invalid allocation size");
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T)));
    }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& src) {
        if (src.size() != count) throw std::runtime_error("upload size mismatch");
        CUDA_CHECK(cudaMemcpy(data, src.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
};

struct Stream {
    cudaStream_t value{};
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Event {
    cudaEvent_t value{};
    explicit Event(unsigned flags = cudaEventDefault) { CUDA_CHECK(cudaEventCreateWithFlags(&value, flags)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

struct Graph {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    ~Graph() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
};

struct Options {
    int m = 129, n = 193, k = 65, iters = 10, split = 8, workers = 0, group_rows = 4;
    unsigned seed = 42;
    std::string demo = "all";
    bool suite = false;
};

static Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::puts("gemm_advanced [--m M --n N --k K] [--iters I] [--split S]\n"
                      "  [--workers P] [--group-rows G] [--seed S] [--check-suite]\n"
                      "  [--demo all|core|split|streamk|packed|fusion|batch|tune|graph|cache]\n"
                      "All variants check against a double-accumulating CPU reference.\n"
                      "CPU reference can be slow for large matrices.");
            std::exit(0);
        }
        if (arg == "--check-suite") { o.suite = true; continue; }
        if (i + 1 == argc) throw std::runtime_error("missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--demo") { o.demo = value; continue; }
        std::size_t used = 0;
        const long long v = std::stoll(value, &used);
        if (used != value.size() || v < 0 || v > 65536)
            throw std::runtime_error("numeric options must be in [0,65536]");
        if (arg == "--m") o.m = v;
        else if (arg == "--n") o.n = v;
        else if (arg == "--k") o.k = v;
        else if (arg == "--iters") o.iters = v;
        else if (arg == "--split") o.split = v;
        else if (arg == "--workers") o.workers = v;
        else if (arg == "--group-rows") o.group_rows = v;
        else if (arg == "--seed") o.seed = v;
        else throw std::runtime_error("unknown option " + arg);
    }
    if (!o.m || !o.n || !o.k || !o.iters || !o.split || !o.group_rows || o.group_rows > 64)
        throw std::runtime_error("M/N/K/iters/split must be positive; group-rows must be in [1,64]");
    const std::vector<std::string> demos{"all", "core", "split", "streamk", "packed", "fusion", "batch", "tune", "graph", "cache"};
    if (std::find(demos.begin(), demos.end(), o.demo) == demos.end())
        throw std::runtime_error("unknown demo " + o.demo);
    return o;
}

// 使用带 padding 的非对称随机输入，验证 lda/ldb/ldc/ldd 与边界。
// C_old 永远只读，所有计时迭代语义相同，不会发生 beta!=0 时不断叠加。
struct Fixture {
    Problem p{};
    std::vector<float> ha, hb, hc, h_bias, poison;
    std::vector<double> product, abs_sum;
    Buffer<float> a, b, c, d, bias;

    Fixture(int m, int n, int k, unsigned seed)
        : ha(static_cast<std::size_t>(m) * (k + 3), NAN),
          hb(static_cast<std::size_t>(k) * (n + 5), NAN),
          hc(static_cast<std::size_t>(m) * (n + 7), NAN), h_bias(n),
          poison(static_cast<std::size_t>(m) * (n + 9), NAN),
          product(static_cast<std::size_t>(m) * n), abs_sum(product.size()),
          a(ha.size()), b(hb.size()), c(hc.size()), d(poison.size()), bias(n) {
        p = {a.data, b.data, c.data, bias.data, d.data,
             m, n, k, k + 3, n + 5, n + 7, n + 9, 0.75f, -0.25f};
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> random(-0.5f, 0.5f);
        for (int r = 0; r < m; ++r)
            for (int x = 0; x < k; ++x) ha[static_cast<std::size_t>(r) * p.lda + x] = random(rng);
        for (int x = 0; x < k; ++x)
            for (int col = 0; col < n; ++col) hb[static_cast<std::size_t>(x) * p.ldb + col] = random(rng);
        for (int r = 0; r < m; ++r)
            for (int col = 0; col < n; ++col) hc[static_cast<std::size_t>(r) * p.ldc + col] = random(rng);
        for (auto& x : h_bias) x = random(rng);
        for (int r = 0; r < m; ++r)
            for (int col = 0; col < n; ++col) {
                const auto ix = static_cast<std::size_t>(r) * n + col;
                double sum = 0, magnitude = 0;
                for (int x = 0; x < k; ++x) {
                    const double term = static_cast<double>(ha[static_cast<std::size_t>(r) * p.lda + x]) *
                                        hb[static_cast<std::size_t>(x) * p.ldb + col];
                    sum += term; magnitude += std::abs(term);
                }
                product[ix] = sum; abs_sum[ix] = magnitude;
            }
        a.upload(ha); b.upload(hb); c.upload(hc); bias.upload(h_bias); reset();
    }

    void reset(float* target = nullptr) const {
        CUDA_CHECK(cudaMemcpy(target ? target : p.d, poison.data(), poison.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        // 测试准备阶段：也保证 pageable H2D 上传对非默认 stream 可见，不计入计时。
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void check(const std::string& name, bool fused = false, bool beta_zero = false,
               const float* target = nullptr) const {
        std::vector<float> got(poison.size());
        CUDA_CHECK(cudaMemcpy(got.data(), target ? target : p.d, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int r = 0; r < p.m; ++r)
            for (int col = 0; col < p.ldd; ++col) {
                const float value = got[static_cast<std::size_t>(r) * p.ldd + col];
                if (col >= p.n) {
                    if (!std::isnan(value)) throw std::runtime_error(name + ": output row padding overwritten");
                    continue;
                }
                const auto ix = static_cast<std::size_t>(r) * p.n + col;
                double ref = p.alpha * product[ix];
                if (!beta_zero) ref += p.beta * hc[static_cast<std::size_t>(r) * p.ldc + col];
                if (fused) ref = std::max(0.0, ref + h_bias[col]);
                // 绝对+相对误差，另给长 K / 分段求和留一个与输入量级相关的余量。
                const double tol = 1e-4 + 2e-4 * std::abs(ref) + 2e-6 * abs_sum[ix];
                if (!std::isfinite(value) || std::abs(value - ref) > tol) {
                    std::fprintf(stderr, "%s mismatch (%d,%d): got %.9g expected %.9g tol %.3g\n",
                                 name.c_str(), r, col, value, ref, tol);
                    throw std::runtime_error("numerical verification failed");
                }
            }
    }
};

static int element_grid(std::size_t count) {
    return static_cast<int>(std::min<std::size_t>((count + 255) / 256, 65535));
}

template<class T> static int tile_count(const Problem& p) {
    return ceil_div(p.m, T::BM) * ceil_div(p.n, T::BN);
}

template<class T = Base, bool Persistent = false, bool Packed = false,
         bool Fast = true, bool UseBeta = true, bool Bias = false, bool Relu = false>
static void launch(Problem p, cudaStream_t stream, int workers = 1, int group_rows = 1) {
    const int tiles = tile_count<T>(p);
    const int grid = Persistent ? std::min(tiles, workers) : tiles;
    gemm_kernel<T, Persistent, Packed, Fast, UseBeta, Bias, Relu>
        <<<grid, T::Threads, 0, stream>>>(p, group_rows);
    CUDA_CHECK(cudaGetLastError());
}

static float time_gpu(const std::function<void()>& fn, cudaStream_t stream, int iters) {
    Event start, stop;
    for (int i = 0; i < 2; ++i) fn();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(start.value, stream));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop.value, stream));
    CUDA_CHECK(cudaEventSynchronize(stop.value));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.value, stop.value));
    return ms / iters;
}

static double time_wall(const std::function<void()>& fn, cudaStream_t stream, int iters) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / iters;
}

static float report(const std::string& name, Fixture& f, const std::function<void()>& fn,
                    cudaStream_t stream, int iters, bool fused = false, bool beta_zero = false) {
    f.reset(); fn();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    f.check(name, fused, beta_zero);
    const float ms = time_gpu(fn, stream, iters);
    const double gflops = 2.0 * f.p.m * f.p.n * f.p.k / (ms * 1e6);
    std::printf("%-25s %10.4f ms %10.2f GFLOP/s  PASS\n", name.c_str(), ms, gflops);
    return ms;
}

static void unfused_chain(Problem p, cudaStream_t stream) {
    launch(p, stream);
    const int blocks = element_grid(static_cast<std::size_t>(p.m) * p.n);
    bias_kernel<<<blocks, 256, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
    relu_kernel<<<blocks, 256, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

// 只比较三个已实现、已知合法的配置；缓存限于本进程、固定 dtype/epilogue。
using TuneKey = std::tuple<int, int, int, int, int, int, int, int>;
static void launch_candidate(int id, Problem p, cudaStream_t stream) {
    if (id == 0) launch<Small>(p, stream);
    else if (id == 1) launch<Base>(p, stream);
    else launch<Wide>(p, stream);
}

static int tune(Fixture& f, cudaStream_t stream, int iters) {
    static std::map<TuneKey, int> cache;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    const auto& p = f.p;
    const TuneKey key{device, p.m, p.n, p.k, p.lda, p.ldb, p.ldc, p.ldd};
    if (const auto it = cache.find(key); it != cache.end()) {
        std::printf("autotune cache hit: candidate %d\n", it->second);
        return it->second;
    }
    float best = std::numeric_limits<float>::infinity();
    int selected = 0;
    for (int i = 0; i < 3; ++i) {
        const auto invoke = [&, i] { launch_candidate(i, p, stream); };
        std::vector<float> samples{report("tune-candidate-" + std::to_string(i), f, invoke, stream, iters),
                                   time_gpu(invoke, stream, iters), time_gpu(invoke, stream, iters)};
        std::sort(samples.begin(), samples.end());
        const float ms = samples[1];
        std::printf("candidate %d median of 3 timing batches: %.4f ms\n", i, ms);
        if (ms < best) { best = ms; selected = i; }
    }
    cache.emplace(key, selected);
    return selected;
}

static void run_one(const Options& o, cudaStream_t stream, int default_workers) {
    Fixture f(o.m, o.n, o.k, o.seed);
    const Problem p = f.p;
    const int workers = o.workers ? o.workers : default_workers;
    const auto wants = [&](const char* x) { return o.demo == "all" || o.demo == x; };
    std::printf("\nM=%d N=%d K=%d, strides=(%d,%d,%d,%d), workers=%d\n",
                p.m, p.n, p.k, p.lda, p.ldb, p.ldc, p.ldd, workers);
    if (wants("core")) {
        report("masked-64x64", f, [&] { launch<Base, false, false, false>(p, stream); }, stream, o.iters);
        report("interior-fast-64x64", f, [&] { launch(p, stream); }, stream, o.iters);
        report("shape-32x64", f, [&] { launch<Small>(p, stream); }, stream, o.iters);
        report("shape-64x128", f, [&] { launch<Wide>(p, stream); }, stream, o.iters);
        report("grouped-M-swizzle", f, [&] { launch(p, stream, 1, o.group_rows); }, stream, o.iters);
        report("persistent", f, [&] { launch<Base, true>(p, stream, workers); }, stream, o.iters);
        Problem zero = p; zero.beta = 0; zero.c = nullptr;
        report("beta-zero-specialized", f,
               [&] { launch<Base, false, false, true, false>(zero, stream); }, stream, o.iters, false, true);
    }
    if (wants("split")) {
        const int splits = std::min(o.split, ceil_div(p.k, Base::BK));
        Buffer<float> partial(static_cast<std::size_t>(splits) * p.m * p.n);
        std::printf("Split-K: %d splits, %.3f MiB workspace\n", splits, partial.count * sizeof(float) / 1048576.0);
        auto invoke = [&](bool fused) {
            split_k_kernel<Base><<<dim3(tile_count<Base>(p), 1, splits), 256, 0, stream>>>(p, splits, partial.data);
            CUDA_CHECK(cudaGetLastError());
            const int blocks = element_grid(static_cast<std::size_t>(p.m) * p.n);
            if (fused) split_reduce_kernel<true, true><<<blocks, 256, 0, stream>>>(p, splits, partial.data);
            else split_reduce_kernel<><<<blocks, 256, 0, stream>>>(p, splits, partial.data);
            CUDA_CHECK(cudaGetLastError());
        };
        report("split-k-two-pass", f, [&] { invoke(false); }, stream, o.iters);
        report("split-k-fused-reduce", f, [&] { invoke(true); }, stream, o.iters, true);
        report("sliced-k-block-reduce", f, [&] {
            sliced_k_kernel<<<ceil_div(p.m, 16) * ceil_div(p.n, 32), 256, 0, stream>>>(p);
            CUDA_CHECK(cudaGetLastError());
        }, stream, o.iters);
    }
    if (wants("streamk")) {
        const auto plan = make_streamk_plan(tile_count<Base>(p), ceil_div(p.k, Base::BK), workers);
        Buffer<Segment> segments(plan.segments.size()); segments.upload(plan.segments);
        Buffer<int> wo(plan.worker_offsets.size()); wo.upload(plan.worker_offsets);
        Buffer<int> to(plan.tile_offsets.size()); to.upload(plan.tile_offsets);
        Buffer<int> slots(plan.tile_slots.size()); slots.upload(plan.tile_slots);
        Buffer<float> partial(plan.segments.size() * Base::BM * Base::BN);
        const int actual_workers = plan.worker_offsets.size() - 1;
        std::printf("Stream-K teaching plan: %d workers, %zu segments, %.3f MiB workspace\n",
                    actual_workers, plan.segments.size(), partial.count * sizeof(float) / 1048576.0);
        report("stream-k-two-pass", f, [&] {
            streamk_partials_kernel<Base><<<actual_workers, 256, 0, stream>>>(p, segments.data, wo.data, partial.data);
            CUDA_CHECK(cudaGetLastError());
            streamk_reduce_kernel<Base><<<element_grid(static_cast<std::size_t>(p.m) * p.n), 256, 0, stream>>>
                (p, partial.data, to.data, slots.data);
            CUDA_CHECK(cudaGetLastError());
        }, stream, o.iters);
    }
    if (wants("packed")) {
        const std::size_t count = static_cast<std::size_t>(ceil_div(p.n, Base::BN)) *
                                   ceil_div(p.k, Base::BK) * Base::BK * Base::BN;
        Buffer<float> packed(count);
        auto pack = [&] {
            pack_b_kernel<Base><<<element_grid(count), 256, 0, stream>>>(p, packed.data, count);
            CUDA_CHECK(cudaGetLastError());
        };
        pack(); CUDA_CHECK(cudaStreamSynchronize(stream));
        // 直接核对打包内容，包括补零区域，而不仅是最终 GEMM 输出。
        std::vector<float> got(count);
        CUDA_CHECK(cudaMemcpy(got.data(), packed.data, count * sizeof(float), cudaMemcpyDeviceToHost));
        for (int tn = 0; tn < ceil_div(p.n, Base::BN); ++tn)
            for (int tk = 0; tk < ceil_div(p.k, Base::BK); ++tk)
                for (int r = 0; r < Base::BK; ++r)
                    for (int c = 0; c < Base::BN; ++c) {
                        const int row = tk * Base::BK + r, col = tn * Base::BN + c;
                        const float expected = row < p.k && col < p.n
                            ? f.hb[static_cast<std::size_t>(row) * p.ldb + col] : 0;
                        if (got[packed_b_index(tn, tk, r, c, ceil_div(p.k, Base::BK), Base::BK, Base::BN)] != expected)
                            throw std::runtime_error("packed B layout/padding mismatch");
                    }
        Problem pp = p; pp.b = packed.data;
        const float pack_ms = time_gpu(pack, stream, o.iters);
        const float ordinary = report("unpacked-B", f, [&] { launch(p, stream); }, stream, o.iters);
        const float steady = report("packed-B-steady", f, [&] { launch<Base, false, true>(pp, stream); }, stream, o.iters);
        report("pack-plus-gemm", f, [&] { pack(); launch<Base, false, true>(pp, stream); }, stream, o.iters);
        std::printf("pack only: %.4f ms; ", pack_ms);
        if (steady < ordinary) std::printf("measured break-even R > %.2f\n", pack_ms / (ordinary - steady));
        else std::puts("no measured steady-state gain; no finite break-even in this model");
    }
    if (wants("fusion")) {
        report("gemm+bias+relu-3launch", f, [&] { unfused_chain(p, stream); }, stream, o.iters, true);
        report("fused-epilogue", f,
               [&] { launch<Base, false, false, true, true, true, true>(p, stream); }, stream, o.iters, true);
    }
    if (wants("cache")) {
        int device;
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        if (!prop.persistingL2CacheMaxSize || !prop.accessPolicyMaxWindowSize) {
            std::puts("L2 persisting window: SKIP (device does not expose this feature)");
        } else {
            std::size_t original = 0;
            CUDA_CHECK(cudaDeviceGetLimit(&original, cudaLimitPersistingL2CacheSize));
            const std::size_t budget = std::min<std::size_t>(prop.persistingL2CacheMaxSize, prop.l2CacheSize / 2);
            const std::size_t bytes = std::min<std::size_t>(f.hb.size() * sizeof(float), prop.accessPolicyMaxWindowSize);
            report("L2-window-control", f, [&] { launch(p, stream); }, stream, o.iters);
            CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, budget));
            cudaStreamAttrValue attr{};
            attr.accessPolicyWindow.base_ptr = const_cast<float*>(p.b);
            attr.accessPolicyWindow.num_bytes = bytes;
            attr.accessPolicyWindow.hitRatio = std::min(1.0, static_cast<double>(budget) / bytes);
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
            std::printf("L2 window: %zu bytes of B; budget=%zu; hitRatio=%.3f\n", bytes, budget, attr.accessPolicyWindow.hitRatio);
            report("L2-persisting-window", f, [&] { launch(p, stream); }, stream, o.iters);
            // 先等待正在使用策略的工作结束，再清除窗口并恢复设备设置。
            attr.accessPolicyWindow.num_bytes = 0;
            CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
            CUDA_CHECK(cudaCtxResetPersistingL2Cache());
            CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, original));
        }
    }
    if (wants("tune")) {
        const int selected = tune(f, stream, o.iters);
        if (selected != tune(f, stream, o.iters)) throw std::runtime_error("tuning cache mismatch");
        report("autotune-selected", f, [&] { launch_candidate(selected, p, stream); }, stream, o.iters);
    }
    if (wants("graph")) {
        // 先预热与加载模块；capture 内不分配、不拷贝主机内存、不做同步等待。
        unfused_chain(p, stream); CUDA_CHECK(cudaStreamSynchronize(stream));
        Graph graph;
        const auto build_begin = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        unfused_chain(p, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph.graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph.exec, graph.graph, nullptr, nullptr, 0));
        const double build_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - build_begin).count();
        auto replay = [&] { CUDA_CHECK(cudaGraphLaunch(graph.exec, stream)); };
        auto ordinary = [&] { unfused_chain(p, stream); };
        report("graph-control-3launch", f, ordinary, stream, o.iters, true);
        report("graph-replay-3nodes", f, replay, stream, o.iters, true);
        std::printf("graph build/instantiate wall: %.4f ms; control/replay wall per step: %.4f / %.4f ms\n",
                    build_ms, time_wall(ordinary, stream, o.iters), time_wall(replay, stream, o.iters));
    }
}

// 固定小矩阵批次，与单矩阵 --m/--n/--k 独立，便于看清接口组织。
static void run_batches(const Options& o, cudaStream_t control, int workers) {
    constexpr int count = 8;
    std::vector<std::unique_ptr<Fixture>> fixtures;
    for (int i = 0; i < count; ++i) fixtures.emplace_back(new Fixture(33, 65, 49, o.seed + i));
    const Fixture& first = *fixtures.front();
    const BatchStrides strides{first.ha.size(), first.hb.size(), first.hc.size(), first.poison.size(), first.h_bias.size()};
    Buffer<float> a(count * strides.a), b(count * strides.b), c(count * strides.c),
                  d(count * strides.d), bias(count * strides.bias);
    for (int i = 0; i < count; ++i) {
        CUDA_CHECK(cudaMemcpy(a.data + i * strides.a, fixtures[i]->ha.data(), strides.a * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b.data + i * strides.b, fixtures[i]->hb.data(), strides.b * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(c.data + i * strides.c, fixtures[i]->hc.data(), strides.c * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bias.data + i * strides.bias, fixtures[i]->h_bias.data(), strides.bias * sizeof(float), cudaMemcpyHostToDevice));
    }
    Problem base = first.p;
    base.a = a.data; base.b = b.data; base.c = c.data; base.d = d.data; base.bias = bias.data;
    auto report_batch = [&](const std::string& name, const std::function<void()>& fn) {
        for (int i = 0; i < count; ++i) fixtures[i]->reset(d.data + i * strides.d);
        fn(); CUDA_CHECK(cudaStreamSynchronize(control));
        for (int i = 0; i < count; ++i)
            fixtures[i]->check(name, false, false, d.data + i * strides.d);
        std::printf("%-25s %10.4f ms / batch  PASS\n", name.c_str(), time_gpu(fn, control, o.iters));
    };
    std::puts("\nBatch: 8 independent (33,65,49) GEMMs, all matrices have padded row strides");
    report_batch("8-sequential-launches", [&] {
        for (int i = 0; i < count; ++i) launch(batch_problem(base, strides, i), control);
    });
    report_batch("strided-batched", [&] {
        batched_kernel<Base><<<dim3(tile_count<Base>(base), 1, count), 256, 0, control>>>(base, strides);
        CUDA_CHECK(cudaGetLastError());
    });

    std::vector<std::unique_ptr<Stream>> streams;
    std::vector<std::unique_ptr<Event>> done;
    Event gate(cudaEventDisableTiming);
    for (int i = 0; i < count; ++i) {
        streams.emplace_back(new Stream);
        done.emplace_back(new Event(cudaEventDisableTiming));
    }
    report_batch("8-streams-with-join", [&] {
        CUDA_CHECK(cudaEventRecord(gate.value, control));
        for (int i = 0; i < count; ++i) {
            CUDA_CHECK(cudaStreamWaitEvent(streams[i]->value, gate.value, 0));
            launch(batch_problem(base, strides, i), streams[i]->value);
            CUDA_CHECK(cudaEventRecord(done[i]->value, streams[i]->value));
            CUDA_CHECK(cudaStreamWaitEvent(control, done[i]->value, 0));
        }
    });

    std::vector<std::unique_ptr<Fixture>> groups;
    const int shapes[][3] = {{17, 33, 129}, {65, 31, 7}, {33, 97, 65}, {64, 64, 32}, {1, 9, 3}};
    for (int i = 0; i < 5; ++i)
        groups.emplace_back(new Fixture(shapes[i][0], shapes[i][1], shapes[i][2], o.seed + 100 + i));
    std::vector<Problem> problems;
    for (const auto& f : groups) problems.push_back(f->p);
    std::puts("Grouped: 5 GEMMs with different M/N/K and row strides");
    auto measure_group = [&](const std::string& name, const std::function<void()>& fn) {
        for (auto& f : groups) f->reset();
        fn(); CUDA_CHECK(cudaStreamSynchronize(control));
        for (auto& f : groups) f->check(name);
        std::printf("%-25s %10.4f ms / group  PASS\n", name.c_str(), time_gpu(fn, control, o.iters));
    };
    measure_group("group-sequential", [&] { for (const auto& p : problems) launch(p, control); });
    for (bool sorted : {false, true}) {
        if (sorted) std::stable_sort(problems.begin(), problems.end(), [](const Problem& x, const Problem& y) { return x.k > y.k; });
        std::vector<int> prefix{0};
        for (const auto& p : problems) prefix.push_back(prefix.back() + tile_count<Base>(p));
        Buffer<Problem> device_problems(problems.size()); device_problems.upload(problems);
        Buffer<int> device_prefix(prefix.size()); device_prefix.upload(prefix);
        measure_group(sorted ? "grouped-K-sorted" : "grouped-persistent", [&] {
            grouped_kernel<Base><<<std::min(workers, prefix.back()), 256, 0, control>>>
                (device_problems.data, device_prefix.data, problems.size());
            CUDA_CHECK(cudaGetLastError());
        });
    }
    // 共同输入的三个列投影：同一份拼接权重，三次子矩阵视图 vs 一次宽 GEMM。
    Fixture projection(17, 105, 49, o.seed + 200);
    report("QKV-three-views", projection, [&] {
        for (int i = 0; i < 3; ++i) {
            Problem view = projection.p;
            view.n = 35;
            view.b += i * 35; view.c += i * 35; view.d += i * 35; view.bias += i * 35;
            // 保留原始 105 列矩阵的 ldb/ldc/ldd，而非改成 35。
            launch(view, control);
        }
    }, control, o.iters);
    report("QKV-one-wide-GEMM", projection, [&] { launch(projection.p, control); }, control, o.iters);
}

int main(int argc, char** argv) {
    try {
        Options o = parse_options(argc, argv);
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        int resident = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, gemm_kernel<Base, true>, Base::Threads, 0));
        const int default_workers = prop.multiProcessorCount * std::max(1, resident);
        std::printf("GPU: %s; persistent Base occupancy limit: %d blocks/SM\n", prop.name, resident);
        std::puts("Teaching fp32 kernels; event timings include the complete named GPU sequence.\n"
                  "CPU reference, allocation, metadata upload and tuning search are outside steady-state timings.");
        Stream stream;
        if (o.suite) {
            o.demo = "all"; o.iters = 1;
            const int shapes[][3] = {{1, 17, 3}, {31, 65, 17}, {64, 64, 32}, {129, 193, 65}, {32, 48, 513}};
            for (const auto& shape : shapes) {
                o.m = shape[0]; o.n = shape[1]; o.k = shape[2];
                run_one(o, stream.value, default_workers);
            }
        } else if (o.demo != "batch") run_one(o, stream.value, default_workers);
        if (o.demo == "all" || o.demo == "batch")
            run_batches(o, stream.value, o.workers ? o.workers : default_workers);
        CUDA_CHECK(cudaStreamSynchronize(stream.value));
        std::puts("\nAll requested numerical and padding checks passed.");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
