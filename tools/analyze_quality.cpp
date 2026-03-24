#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>
#include "replay_tree.hpp"

struct Metrics {
    std::size_t rank_error;
    std::size_t delay;
    std::size_t pq_size;
    std::size_t node_id;
};

struct Temp {
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
    out << "rank_error,delay,pq_size,node_id\n";
    for (auto const& m : metrics) {
        out << m.rank_error << ',' << m.delay << ',' << m.pq_size << ',' << m.node_id << '\n';
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
    char op_type;
    while (in >> op_type) {
        if (op_type == '+') {
            Log::key_type key;
            std::size_t node_id;
            in >> key >> node_id;
            log.keys.push_back(key);
            ++push_index;
        } else if (op_type == '-') {
            std::size_t ref_idx, node_id;
            in >> ref_idx >> node_id;
            if (ref_idx >= push_index) {
                ++invalid_pops;
            }
            log.pops.push_back({push_index, ref_idx, node_id});
        }
    }
    std::cerr << "Invalid pops: " << invalid_pops << '\n';
    return log;
}

// std::vector<Metrics> replay(Log const& log) {
Temp replay(Log const& log) {
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

    // init vars
    Temp temp;
    std::size_t total_rank = 0;
    std::size_t total_delay = 0;
    std::size_t largest_rank = 0;
    std::size_t smallest_rank = INFINITY;
    std::size_t largest_delay = 0;
    std::size_t smallest_delay = INFINITY;
    std::size_t sum_pq_sizes = 0;
    std::size_t max_pq_size = 0;
    std::size_t current_pq_size = 0;

    std::size_t pop_size = log.pops.size();
    metrics.reserve(pop_size);
    std::size_t push_index = 0;
    for (auto const& pop : log.pops) {
        for (; push_index < pop.push_index; ++push_index) {
            replay_tree.insert({log.keys[push_index], push_index});
            ++current_pq_size;
        }
        
        // Track PQ size before popping
        sum_pq_sizes += current_pq_size;
        max_pq_size = std::max(max_pq_size, current_pq_size);
        
        auto [success, rank, delay] = replay_tree.erase_val({log.keys[pop.ref_index], pop.ref_index});

        if (!success) {
            std::cerr << "Failed to delete element " << pop.ref_index << " with key " << log.keys[pop.ref_index]
                      << '\n';
            std::abort();
        }
        
        // Add up the rank and delay
        total_rank += rank;
        total_delay += delay;
        
        if (rank > largest_rank) {
            largest_rank = rank;
        }
        if (rank < smallest_rank) {
            smallest_rank = rank;
        }
        if (delay > largest_delay) {
            largest_delay = delay;
        }
        if (delay < smallest_delay) {
            smallest_delay = delay;
        }
        
        metrics.push_back({rank, delay, current_pq_size, pop.node_id});
        
        --current_pq_size;
    }

    temp.metrics = metrics;
    temp.avg_rank_error = static_cast<double_t>(total_rank) / static_cast<double_t>(pop_size);
    temp.avg_delay = static_cast<double_t>(total_delay) / static_cast<double_t>(pop_size);
    temp.largest_rank = largest_rank;
    temp.smallest_rank = smallest_rank;
    temp.largest_delay = largest_delay;
    temp.smallest_delay = smallest_delay;
    temp.avg_pq_size = static_cast<double_t>(sum_pq_sizes) / static_cast<double_t>(pop_size);
    temp.max_pq_size = max_pq_size;

    return temp;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    std::clog << "Reading log...\n";
    auto log = read_log(std::cin);
    std::clog << "Analyzing log...\n";
    Temp temp = replay(log);
    auto metrics = temp.metrics;
    std::clog << "Average rank error: " << temp.avg_rank_error << "\n";
    std::clog << "Largest rank error: " << temp.largest_rank << "\n";
    std::clog << "Smallest rank errror: " << temp.smallest_rank << "\n";
    std::clog << "Average delay: " << temp.avg_delay << "\n";
    std::clog << "Largest delay: " << temp.largest_delay << "\n";
    std::clog << "Smallest delay: " << temp.smallest_delay << "\n";
    std::clog << "Average PQ size: " << temp.avg_pq_size << "\n";
    std::clog << "Max PQ size: " << temp.max_pq_size << "\n";
    std::clog << "Writing metrics...\n";
    write_metrics(std::cout, metrics);
}
