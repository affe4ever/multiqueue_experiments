#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>
#include "replay_tree.hpp"

struct Metrics {
    std::size_t rank_error;
    std::size_t delay;
};

struct Temp {
    std::vector<Metrics> metrics;
    double avg_rank_error;
    double avg_delay;
    std::size_t largest_rank;
    std::size_t smallest_rank;
    std::size_t largest_delay;
    std::size_t smallest_delay;
    double avg_pq_size;
    std::size_t max_pq_size;
};

void write_metrics(std::ostream& out, std::vector<Metrics> const& metrics) {
    out << "rank_error,delay\n";
    for (auto const& m : metrics) {
        out << m.rank_error << ',' << m.delay << '\n';
    }
}

struct Log {
    using key_type = long long;
    struct Pop {
        std::size_t push_index;
        std::size_t ref_index;
    };
    std::vector<key_type> keys;
    std::vector<Pop> pops;
};

Log read_log(std::istream& in) {
    std::size_t num_pushes = 0;
    std::size_t num_pops   = 0;
    long long   invalid_pops = 0;
    in >> num_pushes >> num_pops;

    Log log;
    log.keys.reserve(num_pushes);
    log.pops.reserve(num_pops);

    for (std::size_t i = 0; i < num_pops; ++i) {
        in.ignore();
        while (in.get() == '+') {
            Log::key_type key;
            in >> key;
            log.keys.push_back(key);
            in.ignore();
        }
        std::size_t index;
        in >> index;
        std::size_t push_index = log.keys.size();
        if (index >= log.keys.size()) {
            ++invalid_pops;
            push_index = index + 1;
        }
        log.pops.push_back({push_index, index});
    }
    std::cerr << "Invalid pops: " << invalid_pops << '\n';
    return log;
}

Temp replay(Log const& log) {
    struct HeapElement {
        Log::key_type key;  // compound key: (original_key << 20) | index
        std::size_t   index;
        bool operator==(HeapElement const& other) const noexcept { return index == other.index; }
        bool operator!=(HeapElement const& other) const noexcept { return !(*this == other); }
    };

    struct ExtractKey {
        static Log::key_type const& get(HeapElement const& e) noexcept { return e.key; }
    };

    auto make_key = [](Log::key_type original_key, std::size_t index) -> Log::key_type {
        return (original_key << 20) | (static_cast<Log::key_type>(index) & 0xFFFFF);
    };

    ReplayTree<Log::key_type, HeapElement, ExtractKey> replay_tree{};

    const std::size_t pop_size = log.pops.size();

    std::size_t total_rank    = 0;
    std::size_t total_delay   = 0;
    std::size_t largest_rank  = 0;
    std::size_t smallest_rank = std::numeric_limits<std::size_t>::max();
    std::size_t largest_delay  = 0;
    std::size_t smallest_delay = std::numeric_limits<std::size_t>::max();
    std::size_t sum_pq_sizes  = 0;
    std::size_t max_pq_size   = 0;
    std::size_t current_pq_size = 0;

    std::vector<Metrics> metrics;
    metrics.reserve(pop_size);

    std::size_t push_index = 0;
    for (auto const& pop : log.pops) {
        for (; push_index < pop.push_index; ++push_index) {
            replay_tree.insert({make_key(log.keys[push_index], push_index), push_index});
            ++current_pq_size;
        }

        sum_pq_sizes += current_pq_size;
        if (current_pq_size > max_pq_size)
            max_pq_size = current_pq_size;

        auto [success, rank, delay] =
            replay_tree.erase_val({make_key(log.keys[pop.ref_index], pop.ref_index), pop.ref_index});

        if (!success) {
            std::cerr << "Failed to delete element " << pop.ref_index
                      << " with key " << log.keys[pop.ref_index] << '\n';
            std::abort();
        }
        --current_pq_size;

        total_rank  += rank;
        total_delay += delay;

        if (rank  > largest_rank)   largest_rank  = rank;
        if (rank  < smallest_rank)  smallest_rank = rank;
        if (delay > largest_delay)  largest_delay = delay;
        if (delay < smallest_delay) smallest_delay = delay;

        metrics.push_back({rank, delay});
    }

    return Temp{
        std::move(metrics),
        static_cast<double>(total_rank)   / static_cast<double>(pop_size),
        static_cast<double>(total_delay)  / static_cast<double>(pop_size),
        largest_rank,
        smallest_rank,
        largest_delay,
        smallest_delay,
        static_cast<double>(sum_pq_sizes) / static_cast<double>(pop_size),
        max_pq_size,
    };
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    std::clog << "Reading log...\n";
    auto log = read_log(std::cin);
    std::clog << "Analyzing log...\n";
    Temp temp = replay(log);

    std::clog << "Average rank error:  " << temp.avg_rank_error  << '\n'
              << "Largest rank error:  " << temp.largest_rank    << '\n'
              << "Smallest rank error: " << temp.smallest_rank   << '\n'
              << "Average delay:       " << temp.avg_delay       << '\n'
              << "Largest delay:       " << temp.largest_delay   << '\n'
              << "Smallest delay:      " << temp.smallest_delay  << '\n'
              << "Average PQ size:     " << temp.avg_pq_size     << '\n'
              << "Max PQ size:         " << temp.max_pq_size     << '\n';

    std::clog << "Writing metrics...\n";
    write_metrics(std::cout, temp.metrics);
}