// 第 13 章：CPU/GPU 共用的索引与任务划分。这里不依赖 CUDA Runtime。
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef __CUDACC__
#define GEMM_HD __host__ __device__
#else
#define GEMM_HD
#endif

namespace advanced_gemm {

GEMM_HD inline int ceil_div(int x, int y) { return x / y + (x % y != 0); }
struct TileCoord { int m, n; };
struct Range { std::int64_t begin, end; };  // 左闭右开

GEMM_HD inline Range partition(std::int64_t total, int rank, int parts) {
    // 商余数写法避免 total*rank 的中间乘积；前 rem 个任务多分一个单位。
    const auto base = total / parts;
    const auto rem = total % parts;
    const auto begin = base * rank + (rank < rem ? rank : rem);
    return {begin, begin + base + (rank < rem)};
}

GEMM_HD inline TileCoord tile_coord(int id, int mt, int nt, int group_rows) {
    // group_rows=1 即普通逐行；>1 时在一组输出块行中先沿 M 方向走。
    const int group_size = group_rows * nt;
    const int first_m = (id / group_size) * group_rows;
    const int remaining = mt - first_m;
    const int rows = remaining < group_rows ? remaining : group_rows;
    const int inner = id % group_size;
    return {first_m + inner % rows, inner / rows};
}

GEMM_HD inline std::size_t packed_b_index(int tile_n, int tile_k,
                                         int r, int c, int kt, int bk, int bn) {
    // Bp[tile_n][tile_k][r][c]；每个 BK×BN 子块连续。
    return ((static_cast<std::size_t>(tile_n) * kt + tile_k) * bk + r) * bn + c;
}

GEMM_HD inline int find_problem(int tile, const int* prefix, int count) {
    int lo = 0, hi = count;
    while (lo + 1 < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (prefix[mid] <= tile) lo = mid;
        else hi = mid;
    }
    return lo;
}

struct Segment {
    int tile, k_begin, k_end;  // k_begin/end 是 BK 迭代编号，不是元素坐标
};

struct StreamKPlan {
    std::vector<Segment> segments;   // 按 worker 排列；下标也是 workspace slot
    std::vector<int> worker_offsets;
    std::vector<int> tile_offsets;    // 每个输出 tile 的贡献 slot 列表，CSR 形式
    std::vector<int> tile_slots;
};

inline StreamKPlan make_streamk_plan(int tiles, int kt, int requested_workers) {
    if (tiles <= 0 || kt <= 0 || requested_workers <= 0)
        throw std::invalid_argument("Stream-K dimensions/workers must be positive");
    const std::int64_t total = static_cast<std::int64_t>(tiles) * kt;
    const int workers = static_cast<int>(std::min<std::int64_t>(total, requested_workers));
    StreamKPlan plan;
    plan.worker_offsets.push_back(0);
    plan.tile_offsets.assign(tiles + 1, 0);
    for (int w = 0; w < workers; ++w) {
        const Range work = partition(total, w, workers);
        for (auto pos = work.begin; pos < work.end;) {
            const int tile = static_cast<int>(pos / kt);
            const auto stop = std::min(work.end, (static_cast<std::int64_t>(tile) + 1) * kt);
            plan.segments.push_back({tile, static_cast<int>(pos % kt),
                                     static_cast<int>(stop - static_cast<std::int64_t>(tile) * kt)});
            ++plan.tile_offsets[tile + 1];
            pos = stop;
        }
        plan.worker_offsets.push_back(static_cast<int>(plan.segments.size()));
    }
    for (int t = 0; t < tiles; ++t) plan.tile_offsets[t + 1] += plan.tile_offsets[t];
    plan.tile_slots.resize(plan.segments.size());
    auto cursor = plan.tile_offsets;
    for (int s = 0; s < static_cast<int>(plan.segments.size()); ++s)
        plan.tile_slots[cursor[plan.segments[s].tile]++] = s;
    return plan;
}

}  // namespace advanced_gemm
#undef GEMM_HD
