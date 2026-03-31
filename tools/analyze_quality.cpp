#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>
#include <unordered_map>
#include "replay_tree.hpp"

struct Metrics {
    std::size_t rank_error;
    std::size_t delay;
    std::size_t pq_size;
    std::size_t node_id;
    int ignored_node;
    int extra_work;
};

struct ReplayResult {
    std::vector<Metrics> metrics;
    std::double_t avg_rank_error;
    std::double_t avg_delay;
    std::size_t largest_rank;
    std::size_t smallest_rank;
    std::size_t largest_delay;
    std::size_t smallest_delay;
    std::double_t avg_pq_size;
    std::size_t max_pq_size;
};

void write_metrics(std::ostream& out, std::vector<Metrics> const& metrics) {
    out << "rank_error,delay,pq_size,node_id,ignored_node,extra_work\n";
    for (auto const& m : metrics) {
        out << m.rank_error << ',' << m.delay << ',' << m.pq_size << ',' << m.node_id << ',' << m.ignored_node << ',' << m.extra_work << '\n';
    }
}

struct Log {
    using key_type = long long;
    struct Pop {
        std::size_t push_index;
        std::size_t ref_index;
        std::size_t node_id;
    };
    std::vector<key_type> keys;
    std::vector<Pop> pops;
    std::unordered_map<std::size_t, key_type> min_distance;
};

Log read_log(std::istream& in) {
    std::size_t num_pushes = 0;
    std::size_t num_pops = 0;
    long long invalid_pops = 0;
    in >> num_pushes >> num_pops;
    Log log;
    log.keys.reserve(num_pushes);
    log.pops.reserve(num_pops);

    std::size_t push_index = 0;
    char op;
    while (in >> op) {
        if (op == '+') {
            std::size_t node_id;
            Log::key_type key;
            in >> key >> node_id;
            log.keys.push_back(key);

            auto& min_dist = log.min_distance[node_id];
            if (min_dist == 0) {
                min_dist = key;
            } else {
                min_dist = std::min(min_dist, key);
            }
            ++push_index;
        } else if (op == '-') {
            std::size_t node_id;
            std::size_t index;
            in >> index >> node_id;
            if (index >= push_index) {
                ++invalid_pops;
            }
            log.pops.push_back({push_index, index, node_id});
        }
    }
    std::cerr << "Invalid pops: " << invalid_pops << '\n';
    return log;
}

ReplayResult replay(Log const& log) {
    struct HeapElement {
        Log::key_type key;
        std::size_t index;
        bool operator==(HeapElement const& other) const noexcept {
            return index == other.index;
        }
        bool operator!=(HeapElement const& other) const noexcept {
            return !(*this == other);
        }
    };

    struct ExtractKey {
        static auto const& get(HeapElement const& e) noexcept {
            return e.key;
        }
    };

    ReplayTree<Log::key_type, HeapElement, ExtractKey> replay_tree{};
    std::vector<Metrics> metrics;
    std::unordered_map<std::size_t, Log::key_type> node_dist;

    std::size_t num_pops = log.pops.size();
    std::size_t total_rank = 0;
    std::size_t total_delay = 0;
    std::size_t largest_rank = 0;
    std::size_t smallest_rank = std::numeric_limits<std::size_t>::max();
    std::size_t largest_delay = 0;
    std::size_t smallest_delay = std::numeric_limits<std::size_t>::max();
    std::size_t total_pq_size = 0;
    std::size_t max_pq_size = 0;
    std::size_t pq_size = 0;
    metrics.reserve(num_pops);

    std::size_t push_index = 0;

    for (auto const& pop : log.pops) {
        for (; push_index < pop.push_index; ++push_index) {
            replay_tree.insert({log.keys[push_index], push_index});
            ++pq_size;
        }

        total_pq_size += pq_size;
        max_pq_size = std::max(max_pq_size, pq_size);

        auto [success, rank, delay] = replay_tree.erase_val({log.keys[pop.ref_index], pop.ref_index});

        if (!success) {
            std::cerr << "Failed to delete element " << pop.ref_index << " with key " << log.keys[pop.ref_index]
                      << '\n';
            std::abort();
        }

        total_rank += rank;
        total_delay += delay;
        largest_rank = std::max(largest_rank, rank);
        smallest_rank = std::min(smallest_rank, rank);
        largest_delay = std::max(largest_delay, delay);
        smallest_delay = std::min(smallest_delay, delay);

        int ignored_node = 0;
        int extra_work = 0;
        if (pop.ref_index < log.keys.size()) {
            auto pushed_distance = log.keys[pop.ref_index];
            auto it = node_dist.find(pop.node_id);
            if (it == node_dist.end()) {
                node_dist[pop.node_id] = pushed_distance;
            } else if (pushed_distance < it->second) {
                it->second = pushed_distance;
                extra_work = 1;
            } else {
                ignored_node = 1;
            }
        }

        metrics.push_back({rank, delay, pq_size, pop.node_id, ignored_node, extra_work});
        --pq_size;
    }

    return {
        metrics,
        static_cast<double_t>(total_rank)    / static_cast<double_t>(num_pops),
        static_cast<double_t>(total_delay)   / static_cast<double_t>(num_pops),
        largest_rank,
        smallest_rank,
        largest_delay,
        smallest_delay,
        static_cast<double_t>(total_pq_size) / static_cast<double_t>(num_pops),
        max_pq_size
    };
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    std::clog << "Reading log...\n";
    auto log = read_log(std::cin);
    std::clog << "Analyzing log...\n";
    auto result = replay(log);
    std::clog << "Average rank error: " << result.avg_rank_error << '\n';
    std::clog << "Largest rank error: " << result.largest_rank << '\n';
    std::clog << "Smallest rank error: " << result.smallest_rank << '\n';
    std::clog << "Average delay: " << result.avg_delay << '\n';
    std::clog << "Largest delay: " << result.largest_delay << '\n';
    std::clog << "Smallest delay: " << result.smallest_delay << '\n';
    std::clog << "Average PQ size: " << result.avg_pq_size << '\n';
    std::clog << "Max PQ size: " << result.max_pq_size << '\n';
    std::clog << "Writing metrics...\n";
    write_metrics(std::cout, result.metrics);
}