#include "util/build_info.hpp"
#include "util/graph.hpp"

#include <cxxopts.hpp>

#include <chrono>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <random>

using clock_type = std::chrono::steady_clock;

struct Node {
    long long distance;
    std::size_t id;

    friend bool operator>(Node const& lhs, Node const& rhs) noexcept {
        return lhs.distance > rhs.distance;
    }
};

template<typename T, typename Compare = std::greater<T>>
class RelaxedPQ {
public:
    RelaxedPQ() = default;

    bool empty() const noexcept {
        return data_.empty();
    }

    std::size_t size() const noexcept {
        return data_.size();
    }

    void push(const T& value) {
        data_.push_back(value);
        std::push_heap(data_.begin(), data_.end(), comp_);
    }

    const T& top() const {
        return data_.front();
    }

    void pop() {
        std::pop_heap(data_.begin(), data_.end(), comp_);
        data_.pop_back();
    }

    const T& at(std::size_t i) const {
        return data_[i];
    }

    T pop_at(std::size_t i) {
        T result = data_[i];

        if (i == data_.size() - 1) {
            data_.pop_back();
            return result;
        }

        std::swap(data_[i], data_.back());
        data_.pop_back();

        // Restore heap property from position i
        std::make_heap(data_.begin(), data_.end(), comp_);
        return result;
    }


private:
    std::vector<T> data_;
    Compare comp_;
};

Node pop_normal(RelaxedPQ<Node, std::greater<Node>>& pq,
                double mean,
                double stddev) {

    static thread_local std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(mean, stddev);

    int index = static_cast<int>(std::round(dist(gen)));
    index = std::clamp(index, 0, static_cast<int>(pq.size() - 1));

    return pq.pop_at(index);
}


void dijkstra(std::filesystem::path const& graph_file,
            double normal_mean,
            double normal_stddev) noexcept
 {
    std::clog << "Reading graph...\n";
    Graph graph;
    try {
        graph = Graph(graph_file);
    } catch (std::runtime_error const& e) {
        std::clog << "Error: " << e.what() << '\n';
        std::exit(EXIT_FAILURE);
    }
    std::clog << "Graph has " << graph.num_nodes() << " nodes and " << graph.num_edges() << " edges\n";
    std::vector<long long> distances(graph.num_nodes(), std::numeric_limits<long long>::max());
    long long processed_nodes{0};
    long long ignored_nodes{0};
    std::size_t sum_sizes{0};
    std::size_t max_size{0};
    std::vector<Node> container;
    container.reserve(graph.num_nodes());
    RelaxedPQ<Node, std::greater<Node>> pq;
    std::clog << "Working...\n";
    auto t_start = std::chrono::steady_clock::now();
    distances[0] = 0;
    pq.push({0, 0});
    while (!pq.empty()) {
        sum_sizes += pq.size();
        max_size = std::max(max_size, pq.size());
        auto node = pop_normal(pq, normal_mean, normal_stddev);
        // Ignore stale nodes
        if (node.distance > distances[node.id]) {
            ++ignored_nodes;
            continue;
        }
        for (std::size_t i = graph.nodes[node.id]; i < graph.nodes[node.id + 1]; ++i) {
            auto d = node.distance + graph.edges[i].weight;
            if (d < distances[graph.edges[i].target]) {
                distances[graph.edges[i].target] = d;
                pq.push({d, graph.edges[i].target});
            }
        }
        ++processed_nodes;
    }
    auto t_end = std::chrono::steady_clock::now();
    std::clog << "Done\n\n";
    auto furthest_node = std::max_element(distances.begin(), distances.end(), [](auto const& a, auto const& b) {
        if (b == std::numeric_limits<long long>::max()) {
            return false;
        }
        if (a == std::numeric_limits<long long>::max()) {
            return true;
        }
        return a < b;
    });
    std::clog << "= Results =\n";
    std::clog << "Time (s): " << std::fixed << std::setprecision(3)
              << std::chrono::duration<double>(t_end - t_start).count() << '\n';
    std::clog << "Furthest node: " << furthest_node - distances.begin() << '\n';
    std::clog << "Longest distance: " << *furthest_node << '\n';
    std::clog << "Processed nodes: " << processed_nodes << '\n';
    std::clog << "Ignored nodes: " << ignored_nodes << '\n';
    std::clog << "Average PQ size: " << static_cast<double>(sum_sizes) / static_cast<double>(processed_nodes + ignored_nodes) << '\n';
    std::clog << "Max PQ size: " << max_size << '\n';

    std::cout << '{';
    std::cout << std::quoted("settings") << ':';
    std::cout << '{';
    std::cout << std::quoted("graph_file") << ':' << graph_file;
    std::cout << '}' << ',';
    std::cout << std::quoted("graph") << ':';
    std::cout << '{';
    std::cout << std::quoted("num_nodes") << ':' << graph.num_nodes() << ',';
    std::cout << std::quoted("num_edges") << ':' << graph.num_edges();
    std::cout << '}' << ',';
    std::cout << std::quoted("results") << ':';
    std::cout << '{';
    std::cout << std::quoted("time_ns") << ':' << std::chrono::nanoseconds{t_end - t_start}.count() << ',';
    std::cout << std::quoted("furthest_node") << ':' << furthest_node - distances.begin() << ',';
    std::cout << std::quoted("longest_distance") << ':' << *furthest_node << ',';
    std::cout << std::quoted("processed_nodes") << ':' << processed_nodes << ',';
    std::cout << std::quoted("ignored_nodes") << ':' << ignored_nodes << ',';
    std::cout << std::quoted("average_pq_size") << ':'
              << static_cast<double>(sum_sizes) / static_cast<double>(processed_nodes + ignored_nodes) << ',';
    std::cout << std::quoted("max_pq_size") << ':' << max_size;
    std::cout << '}';
    std::cout << '}' << '\n';
}

int main(int argc, char* argv[]) {
    write_build_info(std::clog);
    std::clog << '\n';

    std::clog << "= Command line =\n";
    for (int i = 0; i < argc; ++i) {
        std::clog << argv[i];
        if (i != argc - 1) {
            std::clog << ' ';
        }
    }
    std::clog << "\n\n";

    std::filesystem::path graph_file;
    double normal_mean;
    double normal_stddev;

    cxxopts::Options cmd(argv[0], "Dijkstra with relaxed Gaussian extraction");

    cmd.add_options()
        ("h,help", "Print help")
        ("graph", "Input graph file",
            cxxopts::value<std::filesystem::path>(graph_file))
        ("mean", "Normal distribution mean (index space)",
            cxxopts::value<double>(normal_mean)->default_value("0.0"))
        ("stddev", "Normal distribution standard deviation (index space)",
            cxxopts::value<double>(normal_stddev)->default_value("1.0"));

    cmd.parse_positional({"graph"});

    cxxopts::ParseResult args;

    try {
        args = cmd.parse(argc, argv);
    } catch (const cxxopts::OptionParseException& e) {
        std::cerr << "Error parsing command line: "
                  << e.what() << '\n';
        std::cerr << "Use --help for usage information\n";
        return EXIT_FAILURE;
    }

    if (args.count("help") || !args.count("graph")) {
        std::cout << cmd.help() << '\n';
        return EXIT_SUCCESS;
    }

    if (normal_stddev < 0.0) {
        std::cerr << "Error: stddev must be non-negative\n";
        return EXIT_FAILURE;
    }

    std::clog << "= Settings =\n";
    std::clog << "Graph: " << graph_file << '\n';
    std::clog << "Mean: " << normal_mean << '\n';
    std::clog << "Stddev: " << normal_stddev << "\n\n";

    std::clog << "= Running benchmark =\n";
    dijkstra(graph_file, normal_mean, normal_stddev);

    return EXIT_SUCCESS;
}
