#include "util/build_info.hpp"
#include "util/graph.hpp"
#include "util/selector.hpp"
#include "util/termination_detection.hpp"
#include "util/thread_coordination.hpp"
#ifdef LOG_OPERATIONS
#include <fstream>
#include <unordered_map>
#endif

#include <cxxopts.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <x86intrin.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <type_traits>
#include <vector>

using pq_type = PQ<true, unsigned long, unsigned long>;
using handle_type = pq_type::handle_type;
using node_type = pq_type::value_type;

struct Settings {
    int num_threads = 4;
    std::filesystem::path graph_file;
    unsigned int seed = 1;
    pq_type::settings_type pq_settings{};
#ifdef LOG_OPERATIONS
    std::filesystem::path log_file = "dijkstra_log.txt";
#endif
#ifdef DR_PQ_PQ
    std::optional<double> dr_pq_mean{};
    std::optional<double> dr_pq_stddev{};
    std::optional<double> dr_pq_percentile{};
#endif
};

Settings settings{};

void register_cmd_options(cxxopts::Options& cmd) {
    // clang-format off
    cmd.add_options()
        ("j,threads", "The number of threads", cxxopts::value<int>(settings.num_threads), "NUMBER")
        ("graph", "The input graph", cxxopts::value<std::filesystem::path>(settings.graph_file), "PATH")
#ifdef LOG_OPERATIONS
        ("l,log-file", "File to write the operation log to", cxxopts::value<std::filesystem::path>(settings.log_file), "PATH")
#endif
        ;
    // clang-format on
    settings.pq_settings.register_cmd_options(cmd);

#ifdef DR_PQ_PQ
    cmd.add_options()("mean", "DR_PQ mean value", cxxopts::value<double>())(
        "stddev", "DR_PQ standard deviation", cxxopts::value<double>())("percentile", "DR_PQ percentile (0 < p < 1)",
                                                                        cxxopts::value<double>());
#endif

    cmd.parse_positional({"graph"});
}

void write_settings_human_readable(std::ostream& out) {
    out << "Threads: " << settings.num_threads << '\n';
    out << "Graph: " << settings.graph_file << '\n';
    settings.pq_settings.write_human_readable(out);
}

void write_settings_json(std::ostream& out) {
    out << '{';
    out << std::quoted("num_threads") << ':' << settings.num_threads << ',';
    out << std::quoted("graph_file") << ':' << settings.graph_file << ',';
    out << std::quoted("seed") << ':' << settings.seed << ',';
    out << std::quoted("pq") << ':';
    settings.pq_settings.write_json(out);
    out << '}';
}

struct Counter {
    long long pushed_nodes{0};
    long long ignored_nodes{0};
    long long processed_nodes{0};
    long long edge_relaxations{0};
    long long discarded_edges{0};
};

#ifdef LOG_OPERATIONS
using pop_id_t = unsigned long;
inline pop_id_t compute_pop_id_hash(long long thread_id, long long pop_counter) {
    return (static_cast<unsigned long>(thread_id) << 48) | 
           (static_cast<unsigned long>(pop_counter) & 0xFFFFFFFFFFFFUL);
}

struct ThreadData {
    struct PushLog {
        std::pair<unsigned long, unsigned long> element;    // {distance, node_id}
        pop_id_t pop_id;                                    // unique hash
        std::chrono::steady_clock::time_point timestamp;
    };
    struct PopLog {
        std::pair<unsigned long, unsigned long> element;    // {distance, node_id}
        pop_id_t pop_id;                                    // unique hash
        std::chrono::steady_clock::time_point timestamp;
    };
    struct IgnoreLog {
        std::pair<unsigned long, unsigned long> element;    // {distance, node_id}
        pop_id_t pop_id;                                    // unique hash
        std::chrono::steady_clock::time_point timestamp;  
    };
    std::vector<PushLog> pushes;
    std::vector<PopLog> pops;
    std::vector<IgnoreLog> ignores;
    long long pop_counter{0};

};
#endif

struct alignas(L1_CACHE_LINE_SIZE) AtomicDistance {
    std::atomic<long long> value{std::numeric_limits<long long>::max()};
};

struct SharedData {
    Graph graph;
    std::vector<AtomicDistance> distances;
    termination_detection::TerminationDetection termination_detection;
    std::atomic_llong missing_nodes{0};
#ifdef LOG_OPERATIONS
    std::vector<ThreadData> thread_data;
#endif
};

#ifdef LOG_OPERATIONS
void push_with_logging(handle_type& handle, unsigned long distance, unsigned long node_id, Counter& counter,
                        ThreadData& thread_data, pop_id_t pop_id) {
    auto timestamp = std::chrono::steady_clock::now();
    if (handle.push({distance, node_id})) {
        ++counter.pushed_nodes;
        thread_data.pushes.push_back({{distance, node_id}, pop_id, timestamp});
    }
}
pop_id_t pop_with_logging(node_type const& node, ThreadData& thread_data,
                        std::chrono::steady_clock::time_point timestamp, bool is_ignored) {
    auto pop_counter = ++thread_data.pop_counter;
    auto thread_id = static_cast<long long>(gettid());
    pop_id_t pop_id = compute_pop_id_hash(thread_id, pop_counter);

    if (is_ignored) {
        thread_data.ignores.push_back({{node.first, node.second}, pop_id, timestamp});
    } else {
        thread_data.pops.push_back({{node.first, node.second}, pop_id, timestamp});
    }
    
    return pop_id;
}
#endif

void process_node(node_type const& node, handle_type& handle, Counter& counter, SharedData& data
#ifdef LOG_OPERATIONS
                  , 
                  ThreadData& thread_data, std::chrono::steady_clock::time_point pop_timestamp
#endif
) {
    auto current_distance = data.distances[node.second].value.load(std::memory_order_relaxed);
    auto new_distance = static_cast<long long>(node.first);
    if (new_distance > current_distance) {
        ++counter.ignored_nodes;
#ifdef LOG_OPERATIONS
        pop_with_logging(node, thread_data, pop_timestamp, true);
#endif
        return;
    }
#ifdef LOG_OPERATIONS
    auto pop_id = pop_with_logging(node, thread_data, pop_timestamp, false);
#endif
    for (auto i = data.graph.nodes[node.second]; i < data.graph.nodes[node.second + 1]; ++i) {
        auto target = data.graph.edges[i].target;
        auto d = static_cast<long long>(node.first) + data.graph.edges[i].weight;
        auto old_d = data.distances[target].value.load(std::memory_order_relaxed);
        while (d < old_d) {
            // Atomic CAS operation
            // keep trying untill successfull
            if (data.distances[target].value.compare_exchange_weak(old_d, d, std::memory_order_relaxed)) {
#ifdef LOG_OPERATIONS
                push_with_logging(handle, static_cast<unsigned long>(d), target, counter, thread_data, pop_id);
#else
                if (handle.push({d, target})) {
                    ++counter.pushed_nodes;
                }
#endif
                break;
            }
        }
        if (d >= old_d) {
            ++counter.discarded_edges;
        }
        ++counter.edge_relaxations;
    }
    ++counter.processed_nodes;
}

#ifdef LOG_OPERATIONS
void write_log(std::vector<ThreadData> const& thread_data, std::ostream& out) {
    // First pass: count total operations without buffering all at once
    std::size_t num_pushes = 0;
    std::size_t num_pops = 0;
    std::size_t num_ignores = 0;
    for (auto const& td : thread_data) {
        num_pushes += td.pushes.size();
        num_pops += td.pops.size();
        num_ignores += td.ignores.size();
    }

    out << num_pushes << ' ' << num_pops << ' ' << num_ignores << '\n';
    out.flush();

    // Stream-write without intermediate vectors: use priority queue with indices
    // to merge sorted streams from each thread
    
    enum OpType { OP_PUSH, OP_POP, OP_IGNORE };
    
    struct OpRef {
        std::chrono::steady_clock::time_point timestamp;
        std::size_t thread_id;
        OpType op_type;
        std::size_t index;  // Index in pushes, pops, or ignores array
        
        // For priority queue ordering
        bool operator>(const OpRef& other) const {
            return timestamp > other.timestamp;
        }
    };

    // Priority queue to merge operations from all threads
    std::priority_queue<OpRef, std::vector<OpRef>, std::greater<OpRef>> pq;

    // Initialize priority queue with first push/pop/ignore from each thread
    for (std::size_t t = 0; t < thread_data.size(); ++t) {
        auto const& td = thread_data[t];
        
        // Add first push if exists
        if (!td.pushes.empty()) {
            pq.push({td.pushes[0].timestamp, t, OP_PUSH, 0});
        }
        
        // Add first pop if exists
        if (!td.pops.empty()) {
            pq.push({td.pops[0].timestamp, t, OP_POP, 0});
        }
        
        // Add first ignore if exists
        if (!td.ignores.empty()) {
            pq.push({td.ignores[0].timestamp, t, OP_IGNORE, 0});
        }
    }

    // Hash function for element pairs
    struct PairHash {
        std::size_t operator()(std::pair<unsigned long, unsigned long> const& p) const noexcept {
            return std::hash<unsigned long>{}(p.first) ^ (std::hash<unsigned long>{}(p.second) << 1);
        }
    };

    // Map elements to their push indices (FIFO queue per element)
    std::unordered_map<std::pair<unsigned long, unsigned long>, std::deque<std::size_t>, PairHash> element_to_indices;
    std::size_t push_index = 0;

    // Process operations in timestamp order via priority queue
    while (!pq.empty()) {
        OpRef op = pq.top();
        pq.pop();

        std::size_t t = op.thread_id;
        auto const& td = thread_data[t];

        if (op.op_type == OP_PUSH) {
            auto const& push = td.pushes[op.index];
            out << '+' << push.element.first << ' ' << push.element.second << ' ' << push.pop_id << '\n';
            element_to_indices[push.element].push_back(push_index);
            ++push_index;
            
            // Add next push from this thread
            if (op.index + 1 < td.pushes.size()) {
                pq.push({td.pushes[op.index + 1].timestamp, t, OP_PUSH, op.index + 1});
            }
        } else if (op.op_type == OP_POP) {
            auto const& pop = td.pops[op.index];
            auto it = element_to_indices.find(pop.element);
            if (it != element_to_indices.end() && !it->second.empty()) {
                out << '-' << it->second.front() << ' ' << pop.element.second << ' ' << pop.pop_id << '\n';
                it->second.pop_front();
                
                // Erase entry when deque becomes empty to avoid memory bloat
                // This limits map size to max pending pushes, not total pushes
                if (it->second.empty()) {
                    element_to_indices.erase(it);
                }
            }
            
            // Add next pop from this thread
            if (op.index + 1 < td.pops.size()) {
                pq.push({td.pops[op.index + 1].timestamp, t, OP_POP, op.index + 1});
            }
        } else if (op.op_type == OP_IGNORE) {
            auto const& ignore = td.ignores[op.index];
            auto it = element_to_indices.find(ignore.element);
            if (it != element_to_indices.end() && !it->second.empty()) {
                out << '=' << it->second.front() << ' ' << ignore.element.second << ' ' << ignore.pop_id << '\n';
                it->second.pop_front();
                
                // Erase entry when deque becomes empty to avoid memory bloat
                if (it->second.empty()) {
                    element_to_indices.erase(it);
                }
            }
            
            // Add next ignore from this thread
            if (op.index + 1 < td.ignores.size()) {
                pq.push({td.ignores[op.index + 1].timestamp, t, OP_IGNORE, op.index + 1});
            }
        }
        
        out.flush();  // Flush frequently to avoid buffer buildup
    }
}
#endif

[[gnu::noinline]] Counter benchmark_thread(thread_coordination::Context& thread_context, pq_type& pq,
                                           SharedData& data) {
    Counter counter;
    auto handle = pq.get_handle();
#ifdef LOG_OPERATIONS
    ThreadData thread_data;
    thread_data.pushes.reserve(data.graph.num_edges());
    thread_data.pops.reserve(data.graph.num_nodes());
#endif

    if (thread_context.id() == 0) {
        data.distances[0].value = 0;
#ifdef LOG_OPERATIONS
        push_with_logging(handle, 0, 0, counter, thread_data, 0);
#else
        handle.push({0, 0});
        ++counter.pushed_nodes;
#endif
    }
    thread_context.synchronize();
    while (true) {
        std::optional<node_type> node;
#ifdef LOG_OPERATIONS
        std::chrono::steady_clock::time_point pop_timestamp;
#endif
        while (data.termination_detection.repeat([&]() {
            node = handle.try_pop();
            bool has_value = node.has_value();
#ifdef LOG_OPERATIONS
            if (has_value) {
                pop_timestamp = std::chrono::steady_clock::now();
            }
#endif
            return has_value;
        })) {
#ifdef LOG_OPERATIONS
            process_node(*node, handle, counter, data, thread_data, pop_timestamp);
#else
            process_node(*node, handle, counter, data);
#endif
        }
        data.missing_nodes.fetch_add(counter.pushed_nodes - counter.processed_nodes - counter.ignored_nodes,
                                     std::memory_order_relaxed);
        thread_context.synchronize();
        if (data.missing_nodes.load(std::memory_order_relaxed) == 0) {
            break;
        }
        thread_context.synchronize();
        if (thread_context.id() == 0) {
            data.missing_nodes.store(0, std::memory_order_relaxed);
            data.termination_detection.reset();
        }
        thread_context.synchronize();
    }
#ifdef LOG_OPERATIONS
    data.thread_data[static_cast<std::size_t>(thread_context.id())] = std::move(thread_data);
#endif
    return counter;
}

void run_benchmark() {
    std::clog << "Reading graph...\n";
    SharedData shared_data{{}, {}, termination_detection::TerminationDetection(settings.num_threads), {}
#ifdef LOG_OPERATIONS
        , {}
#endif
    };
    try {
        shared_data.graph = Graph(settings.graph_file);
    } catch (std::runtime_error const& e) {
        std::clog << "Error: " << e.what() << '\n';
        std::exit(EXIT_FAILURE);
    }
    std::clog << "Graph has " << shared_data.graph.num_nodes() << " nodes and " << shared_data.graph.num_edges()
              << " edges\n";
    shared_data.distances = std::vector<AtomicDistance>(shared_data.graph.num_nodes());
#ifdef LOG_OPERATIONS
    shared_data.thread_data.resize(static_cast<std::size_t>(settings.num_threads));
#endif

    std::vector<Counter> thread_counter(static_cast<std::size_t>(settings.num_threads));
    auto pq = pq_type(settings.num_threads, shared_data.graph.num_nodes(), settings.pq_settings
#ifdef DR_PQ_PQ
                      ,
                      settings.dr_pq_mean, settings.dr_pq_stddev, settings.dr_pq_percentile
#endif
    );
    std::clog << "Working...\n";
    auto start_time = std::chrono::steady_clock::now();
    thread_coordination::Dispatcher dispatcher{settings.num_threads, [&](auto ctx) {
                                                   auto t_id = static_cast<std::size_t>(ctx.id());
                                                   thread_counter[t_id] = benchmark_thread(ctx, pq, shared_data);
                                               }};
    dispatcher.wait();
    auto end_time = std::chrono::steady_clock::now();

    std::clog << "Done\n";
#ifdef LOG_OPERATIONS
    std::clog << "Writing logs...\n";
    std::ofstream log_out(settings.log_file);
    if (log_out) {
        write_log(shared_data.thread_data, log_out);
        log_out.close();
        std::clog << "Log written to " << settings.log_file << "\n";
    } else {
        std::cerr << "Warning: Could not write log file " << settings.log_file << "\n";
    }
#endif
    auto total_counts =
        std::accumulate(thread_counter.begin(), thread_counter.end(), Counter{}, [](auto sum, auto const& counter) {
            sum.pushed_nodes += counter.pushed_nodes;
            sum.processed_nodes += counter.processed_nodes;
            sum.ignored_nodes += counter.ignored_nodes;
            sum.edge_relaxations += counter.edge_relaxations;
            sum.discarded_edges += counter.discarded_edges;
            return sum;
        });
    std::clog << '\n';
    auto furthest_node =
        std::max_element(shared_data.distances.begin(), shared_data.distances.end(), [](auto const& a, auto const& b) {
            auto a_val = a.value.load(std::memory_order_relaxed);
            auto b_val = b.value.load(std::memory_order_relaxed);
            if (b_val == std::numeric_limits<long long>::max()) {
                return false;
            }
            if (a_val == std::numeric_limits<long long>::max()) {
                return true;
            }
            return a_val < b_val;
        });
    std::clog << "= Results =\n";
    std::clog << "Time (s): " << std::fixed << std::setprecision(3)
              << std::chrono::duration<double>(end_time - start_time).count() << '\n';
    std::clog << "Furthest node: " << furthest_node - shared_data.distances.begin() << '\n';
    std::clog << "Longest distance: " << furthest_node->value.load(std::memory_order_relaxed) << '\n';
    std::clog << "Processed nodes: " << total_counts.processed_nodes << '\n';
    std::clog << "Ignored nodes: " << total_counts.ignored_nodes << '\n';
    std::clog << "Edge relaxations: " << total_counts.edge_relaxations << '\n';
    std::clog << "Discarded edges: " << total_counts.discarded_edges << '\n';
    std::clog << "Min degree: " << shared_data.graph.min_degree << '\n';
    std::clog << "Avg degree: " << std::fixed << std::setprecision(2) << shared_data.graph.avg_degree << '\n';
    std::clog << "Max degree: " << shared_data.graph.max_degree << '\n';
    if (total_counts.processed_nodes + total_counts.ignored_nodes != total_counts.pushed_nodes) {
        std::cerr << "Warning: Not all nodes were popped\n";
        std::cerr << "Probably the priority queue discards duplicate keys\n";
    }
    std::cout << '{';
    std::cout << std::quoted("settings") << ':';
    write_settings_json(std::cout);
    std::cout << ',';
    std::cout << std::quoted("graph") << ':';
    std::cout << '{';
    std::cout << std::quoted("num_nodes") << ':' << shared_data.graph.num_nodes() << ',';
    std::cout << std::quoted("num_edges") << ':' << shared_data.graph.num_edges() << ',';
    std::cout << std::quoted("min_degree") << ':' << shared_data.graph.min_degree << ',';
    std::cout << std::quoted("avg_degree") << ':' << shared_data.graph.avg_degree << ',';
    std::cout << std::quoted("max_degree") << ':' << shared_data.graph.max_degree;
    std::cout << '}' << ',';
    std::cout << std::quoted("results") << ':';
    std::cout << '{';
    std::cout << std::quoted("time_ns") << ':' << std::chrono::nanoseconds{end_time - start_time}.count() << ',';
    std::cout << std::quoted("furthest_node") << ':' << furthest_node - shared_data.distances.begin() << ',';
    std::cout << std::quoted("longest_distance") << ':' << furthest_node->value.load(std::memory_order_relaxed) << ',';
    std::cout << std::quoted("processed_nodes") << ':' << total_counts.processed_nodes << ',';
    std::cout << std::quoted("ignored_nodes") << ':' << total_counts.ignored_nodes;
    std::cout << std::quoted("edge_relaxations") << ':' << total_counts.edge_relaxations << ',';
    std::cout << std::quoted("discarded_edges") << ':' << total_counts.discarded_edges << ',';
    std::cout << '}';
    std::cout << '}' << '\n';
}

int main(int argc, char* argv[]) {
    write_build_info(std::clog);
    std::clog << '\n';

    std::clog << "= Priority queue =\n";
    pq_type::write_human_readable(std::clog);
    std::clog << '\n';

    std::clog << "= Command line =\n";
    for (int i = 0; i < argc; ++i) {
        std::clog << argv[i];
        if (i != argc - 1) {
            std::clog << ' ';
        }
    }
    std::clog << '\n' << '\n';

    cxxopts::Options cmd(argv[0]);
    cmd.add_options()("h,help", "Print this help");
    register_cmd_options(cmd);

    try {
        auto args = cmd.parse(argc, argv);
        if (args.count("help") > 0) {
            std::cerr << cmd.help() << '\n';
            return EXIT_SUCCESS;
        }

#ifdef DR_PQ_PQ
        if (args.count("mean") > 0) {
            settings.dr_pq_mean = args["mean"].as<double>();
        }
        if (args.count("stddev") > 0) {
            settings.dr_pq_stddev = args["stddev"].as<double>();
        }
        if (args.count("percentile") > 0) {
            settings.dr_pq_percentile = args["percentile"].as<double>();
        }
#endif

    } catch (cxxopts::OptionParseException const& e) {
        std::cerr << "Error parsing command line: " << e.what() << '\n';
        std::cerr << "Use --help for usage information" << '\n';
        return EXIT_FAILURE;
    }

    std::clog << "= Settings =\n";
    write_settings_human_readable(std::clog);
    std::clog << '\n';

    std::clog << "= Running benchmark =\n";
    run_benchmark();
    return EXIT_SUCCESS;
}
