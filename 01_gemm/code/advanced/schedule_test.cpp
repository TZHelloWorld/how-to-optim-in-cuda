// 无 GPU 时可验证的部分：调度覆盖、分段平衡、打包索引，以及由任务计划重建 GEMM。
// CUDA shared/barrier、实际设备代码和数值结果由 gemm_advanced --check-suite 验证。
#include "schedule.h"
#include <cmath>
#include <iostream>
#include <string>

using namespace advanced_gemm;

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void test_tile_mapping() {
    for (int mt = 1; mt <= 11; ++mt)
        for (int nt = 1; nt <= 13; ++nt)
            for (int group = 1; group <= 16; ++group) {
                std::vector<int> seen(mt * nt);
                for (int id = 0; id < mt * nt; ++id) {
                    const auto t = tile_coord(id, mt, nt, group);
                    require(t.m >= 0 && t.m < mt && t.n >= 0 && t.n < nt, "swizzle out of range");
                    ++seen[t.m * nt + t.n];
                }
                for (int visits : seen) require(visits == 1, "swizzle misses/duplicates output tile");
            }
}

static void test_partitions() {
    for (int total = 0; total < 100; ++total)
        for (int parts = 1; parts < 40; ++parts) {
            std::int64_t end = 0, low = total, high = 0;
            for (int i = 0; i < parts; ++i) {
                const auto r = partition(total, i, parts);
                require(r.begin == end && r.end >= r.begin && r.end <= total, "partition gap/overlap");
                low = std::min(low, r.end - r.begin);
                high = std::max(high, r.end - r.begin);
                end = r.end;
            }
            require(end == total && high - low <= 1, "partition coverage/imbalance");
        }
    const auto r = partition(INT64_C(4000000000), 2, 3);
    require(r.end == INT64_C(4000000000), "64-bit work index truncated");
}

static void test_streamk_coverage() {
    for (int tiles = 1; tiles <= 17; ++tiles)
        for (int kt = 1; kt <= 11; ++kt)
            for (int requested = 1; requested <= 31; ++requested) {
                const auto plan = make_streamk_plan(tiles, kt, requested);
                const int workers = plan.worker_offsets.size() - 1;
                std::vector<int> seen(tiles * kt), slot_visits(plan.segments.size());
                int low = tiles * kt, high = 0;
                for (int w = 0; w < workers; ++w) {
                    int work = 0;
                    for (int s = plan.worker_offsets[w]; s < plan.worker_offsets[w + 1]; ++s) {
                        const auto x = plan.segments[s];
                        require(x.tile >= 0 && x.tile < tiles && x.k_begin >= 0 && x.k_end <= kt &&
                                x.k_begin < x.k_end, "invalid Stream-K segment");
                        for (int k = x.k_begin; k < x.k_end; ++k) ++seen[x.tile * kt + k];
                        work += x.k_end - x.k_begin;
                    }
                    low = std::min(low, work); high = std::max(high, work);
                }
                for (int n : seen) require(n == 1, "Stream-K work missing/duplicated");
                require(high - low <= 1, "Stream-K unbalanced");
                for (int t = 0; t < tiles; ++t)
                    for (int i = plan.tile_offsets[t]; i < plan.tile_offsets[t + 1]; ++i) {
                        const int s = plan.tile_slots[i];
                        require(plan.segments[s].tile == t, "reduction uses another tile's contribution");
                        ++slot_visits[s];
                    }
                for (int n : slot_visits) require(n == 1, "reduction contribution missing/duplicated");
                require(plan.segments.size() <= static_cast<std::size_t>(tiles + workers - 1), "too many segments");
            }
}

static void test_grouped_mapping() {
    const std::vector<int> prefix{0, 1, 5, 7, 16, 17};
    for (int workers : {1, 3, 8, 31}) {
        std::vector<int> seen(prefix.back());
        for (int w = 0; w < workers; ++w)
            for (int id = w; id < prefix.back(); id += workers) {
                const int g = find_problem(id, prefix.data(), prefix.size() - 1);
                require(prefix[g] <= id && id < prefix[g + 1], "wrong grouped problem");
                ++seen[id];
            }
        for (int n : seen) require(n == 1, "grouped tile missing/duplicated");
    }
}

static void test_plan_reconstructs_gemm() {
    constexpr int M = 7, N = 11, K = 19, BM = 4, BN = 5, BK = 3;
    const int mt = ceil_div(M, BM), nt = ceil_div(N, BN), kt = ceil_div(K, BK);
    std::vector<double> a(M * K), b(K * N), ref(M * N);
    for (int i = 0; i < M * K; ++i) a[i] = ((i * 7 + 3) % 17 - 8) / 4.0;
    for (int i = 0; i < K * N; ++i) b[i] = ((i * 11 + 5) % 19 - 9) / 8.0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k) ref[m * N + n] += a[m * K + k] * b[k * N + n];

    std::vector<double> packed(nt * kt * BK * BN, 0);
    std::vector<int> packed_visits(packed.size());
    for (int n = 0; n < nt * BN; ++n)
        for (int k = 0; k < kt * BK; ++k) {
            const auto ix = packed_b_index(n / BN, k / BK, k % BK, n % BN, kt, BK, BN);
            require(ix < packed.size(), "pack index out of bounds");
            ++packed_visits[ix];
            packed[ix] = n < N && k < K ? b[k * N + n] : 0;
        }
    for (int visits : packed_visits) require(visits == 1, "pack layout overlaps");

    for (int workers : {1, 2, 7, 100}) {
        const auto plan = make_streamk_plan(mt * nt, kt, workers);
        std::vector<double> partial(plan.segments.size() * BM * BN, 0);
        for (std::size_t s = 0; s < plan.segments.size(); ++s) {
            const auto task = plan.segments[s];
            for (int r = 0; r < BM; ++r)
                for (int c = 0; c < BN; ++c) {
                    const int m = (task.tile / nt) * BM + r, n = (task.tile % nt) * BN + c;
                    if (m >= M || n >= N) continue;
                    for (int k = task.k_begin * BK; k < std::min(K, task.k_end * BK); ++k)
                        partial[(s * BM + r) * BN + c] += a[m * K + k] *
                            packed[packed_b_index(n / BN, k / BK, k % BK, n % BN, kt, BK, BN)];
                }
        }
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                const int t = (m / BM) * nt + n / BN;
                double sum = 0;
                for (int i = plan.tile_offsets[t]; i < plan.tile_offsets[t + 1]; ++i)
                    sum += partial[(static_cast<std::size_t>(plan.tile_slots[i]) * BM + m % BM) * BN + n % BN];
                require(sum == ref[m * N + n], "Stream-K/packed layout changes GEMM result");
            }
    }
}

int main() {
    try {
        test_tile_mapping(); test_partitions(); test_streamk_coverage();
        test_grouped_mapping(); test_plan_reconstructs_gemm();
        bool rejected = false;
        try { make_streamk_plan(0, 1, 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid plan accepted");
        std::cout << "PASS: swizzle, split partitions, Stream-K coverage/reduction, grouped mapping, packed GEMM reconstruction\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
