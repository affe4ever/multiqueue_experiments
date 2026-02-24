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
#include <atomic>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

using pq_type = PQ<true, unsigned long, unsigned long>;
using handle_type = pq_type::handle_type;
using node_type = pq_type::value_type;

struct Settings {
    int num_threads = 4;
    std::filesystem::path graph_file;
    unsigned int seed = 1;
#ifdef LOG_OPERATIONS
    std::filesystem::path log_file = "dijkstra_log.txt";
#endif
    pq_type::settings_type pq_settings{};
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
};

#ifdef LOG_OPERATIONS
struct ThreadData {
    struct PushLog {
        std::chrono::high_resolution_clock::time_point tick;
        std::pair<unsigned long, unsigned long> element;  // {distance, node_id}
        std::size_t sequence_id;
    };
    struct PopLog {
        std::chrono::high_resolution_clock::time_point tick;
        std::pair<unsigned long, unsigned long> element;  // {distance, node_id}
    };
    std::vector<PushLog> pushes;
    std::vector<PopLog> pops;
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
    std::atomic<std::size_t> push_counter{0};
    std::vector<ThreadData> thread_data;
#endif
};

#ifdef LOG_OPERATIONS
void push_with_logging(handle_type& handle, unsigned long distance, unsigned long node_id, 
                       Counter& counter, SharedData& data, ThreadData& thread_data) {
    // Get unique sequence ID
    auto seq_id = data.push_counter.fetch_add(1, std::memory_order_relaxed);
    
    // Actually push to queue
    if (handle.push({distance, node_id})) {
        ++counter.pushed_nodes;
        
        // Record the push with timestamp
        auto tick = std::chrono::high_resolution_clock::now();
        thread_data.pushes.push_back({tick, {distance, node_id}, seq_id});
    }
}
#endif

void process_node(node_type const& node, handle_type& handle, Counter& counter, SharedData& data
#ifdef LOG_OPERATIONS
                  , ThreadData& thread_data
#endif
                  ) {
    auto current_distance = data.distances[node.second].value.load(std::memory_order_relaxed);
    if (static_cast<long long>(node.first) > current_distance) {
        ++counter.ignored_nodes;
        return;
    }
    for (auto i = data.graph.nodes[node.second]; i < data.graph.nodes[node.second + 1]; ++i) {
        auto target = data.graph.edges[i].target;
        auto d = static_cast<long long>(node.first) + data.graph.edges[i].weight;
        auto old_d = data.distances[target].value.load(std::memory_order_relaxed);
        while (d < old_d) {
            if (data.distances[target].value.compare_exchange_weak(old_d, d, std::memory_order_relaxed)) {
#ifdef LOG_OPERATIONS
                push_with_logging(handle, static_cast<unsigned long>(d), target, counter, data, thread_data);
#else
                if (handle.push({d, target})) {
                    ++counter.pushed_nodes;
                }
#endif
                break;
            }
        }
    }
    ++counter.processed_nodes;
}

#ifdef LOG_OPERATIONS
void write_log(std::vector<ThreadData> const& thread_data, std::ostream& out) {
    // Merge all pushes and pops from all threads
    std::vector<ThreadData::PushLog> pushes;
    std::vector<ThreadData::PopLog> pops;
    for (auto const& td : thread_data) {
        pushes.insert(pushes.end(), td.pushes.begin(), td.pushes.end());
        pops.insert(pops.end(), td.pops.begin(), td.pops.end());
    }
    
    // Sort by timestamp
    std::sort(pushes.begin(), pushes.end(), 
              [](auto const& lhs, auto const& rhs) { return lhs.tick < rhs.tick; });
    std::sort(pops.begin(), pops.end(), 
              [](auto const& lhs, auto const& rhs) { return lhs.tick < rhs.tick; });
    
    // Map sequence_id to push index for output
    std::unordered_map<std::size_t, std::size_t> sequence_to_index;
    for (std::size_t i = 0; i < pushes.size(); ++i) {
        sequence_to_index[pushes[i].sequence_id] = i;
    }
    
    // Build element-to-sequence_id map (using last push for each element)
    std::map<std::pair<unsigned long, unsigned long>, std::size_t> element_to_seq;
    for (auto const& push : pushes) {
        element_to_seq[push.element] = push.sequence_id;
    }
    
    // Write in analyze_quality format
    out << pushes.size() << ' ' << pops.size() << '\n';
    std::size_t i = 0;
    for (auto const& pop : pops) {
        // Write pushes that happened before this pop
        while (i < pushes.size() && pushes[i].tick < pop.tick) {
            out << '+' << pushes[i].element.first << '\n';
            ++i;
        }
        // Find which push this pop corresponds to by matching element
        auto elem_it = element_to_seq.find(pop.element);
        if (elem_it != element_to_seq.end()) {
            auto it = sequence_to_index.find(elem_it->second);
            if (it != sequence_to_index.end()) {
                out << 'd' << it->second << '\n';
            }
        }
    }
    // Write remaining pushes
    for (; i < pushes.size(); ++i) {
        out << '+' << pushes[i].element.first << '\n';
    }
}
#endif

[[gnu::noinline]] Counter benchmark_thread(thread_coordination::Context& thread_context, pq_type& pq,
                                           SharedData& data) {
    Counter counter;
    auto handle = pq.get_handle();
#ifdef LOG_OPERATIONS
    ThreadData thread_data;
    // Reserve space for logs
    thread_data.pushes.reserve(data.graph.num_edges());
    thread_data.pops.reserve(data.graph.num_nodes());
#endif
    
    if (thread_context.id() == 0) {
        data.distances[0].value = 0;
#ifdef LOG_OPERATIONS
        push_with_logging(handle, 0, 0, counter, data, thread_data);
#else
        handle.push({0, 0});
        ++counter.pushed_nodes;
#endif
    }
    thread_context.synchronize();
    while (true) {
        std::optional<node_type> node;
        while (data.termination_detection.repeat([&]() {
            auto tick = std::chrono::high_resolution_clock::now();
            node = handle.try_pop();
#ifdef LOG_OPERATIONS
            if (node.has_value()) {
                thread_data.pops.push_back({tick, {node->first, node->second}});
            }
#endif
            return node.has_value();
        })) {
#ifdef LOG_OPERATIONS
            process_node(*node, handle, counter, data, thread_data);
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
    SharedData shared_data{{}, {}, termination_detection::TerminationDetection(settings.num_threads)};
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
    auto pq = pq_type(settings.num_threads, shared_data.graph.num_nodes(), settings.pq_settings);
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
    std::cout << std::quoted("num_edges") << ':' << shared_data.graph.num_edges();
    std::cout << '}' << ',';
    std::cout << std::quoted("results") << ':';
    std::cout << '{';
    std::cout << std::quoted("time_ns") << ':' << std::chrono::nanoseconds{end_time - start_time}.count() << ',';
    std::cout << std::quoted("furthest_node") << ':' << furthest_node - shared_data.distances.begin() << ',';
    std::cout << std::quoted("longest_distance") << ':' << furthest_node->value.load(std::memory_order_relaxed) << ',';
    std::cout << std::quoted("processed_nodes") << ':' << total_counts.processed_nodes << ',';
    std::cout << std::quoted("ignored_nodes") << ':' << total_counts.ignored_nodes;
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
