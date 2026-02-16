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

    void sift_up(std::size_t i) {
    while (i > 0) {
        std::size_t parent = (i - 1) / 2;
        if (!comp_(data_[parent], data_[i]))
            break;
        std::swap(data_[parent], data_[i]);
        i = parent;
    }
}

void sift_down(std::size_t i) {
    std::size_t n = data_.size();
        while (true) {
            std::size_t left = 2 * i + 1;
            std::size_t right = 2 * i + 2;
            std::size_t largest = i;
        
            if (left < n && comp_(data_[largest], data_[left]))
                largest = left;
        
            if (right < n && comp_(data_[largest], data_[right]))
                largest = right;
        
            if (largest == i)
                break;
        
            std::swap(data_[i], data_[largest]);
            i = largest;
        }
    }


    T pop_at(std::size_t i) {
        T result = data_[i];

    if (i == data_.size() - 1) {
        data_.pop_back();
        return result;
    }

    std::swap(data_[i], data_.back());
    data_.pop_back();

    sift_down(i);
    sift_up(i);

    return result;
}


private:
    std::vector<T> data_;
    Compare comp_;
};

Node pop_normal(RelaxedPQ<Node, std::greater<Node>>& pq,
                std::normal_distribution<double>& dist,
                std::mt19937& gen) {

    int index = static_cast<int>(std::round(dist(gen)));
    index = std::clamp(index, 0, static_cast<int>(pq.size() - 1));

    return pq.pop_at(index);
}

double inverse_normal_cdf(double p) {
    if (p <= 0.0 || p >= 1.0) {
        throw std::invalid_argument("Percentile must be in (0,1)");
    }

    // Coefficients for Acklam approximation
    static const double a1 = -3.969683028665376e+01;
    static const double a2 =  2.209460984245205e+02;
    static const double a3 = -2.759285104469687e+02;
    static const double a4 =  1.383577518672690e+02;
    static const double a5 = -3.066479806614716e+01;
    static const double a6 =  2.506628277459239e+00;

    static const double b1 = -5.447609879822406e+01;
    static const double b2 =  1.615858368580409e+02;
    static const double b3 = -1.556989798598866e+02;
    static const double b4 =  6.680131188771972e+01;
    static const double b5 = -1.328068155288572e+01;

    static const double c1 = -7.784894002430293e-03;
    static const double c2 = -3.223964580411365e-01;
    static const double c3 = -2.400758277161838e+00;
    static const double c4 = -2.549732539343734e+00;
    static const double c5 =  4.374664141464968e+00;
    static const double c6 =  2.938163982698783e+00;

    static const double d1 =  7.784695709041462e-03;
    static const double d2 =  3.224671290700398e-01;
    static const double d3 =  2.445134137142996e+00;
    static const double d4 =  3.754408661907416e+00;

    const double plow = 0.02425;
    const double phigh = 1 - plow;

    double q, r;

    if (p < plow) {
        q = std::sqrt(-2 * std::log(p));
        return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
               ((((d1*q + d2)*q + d3)*q + d4)*q + 1);
    }

    if (phigh < p) {
        q = std::sqrt(-2 * std::log(1 - p));
        return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
                ((((d1*q + d2)*q + d3)*q + d4)*q + 1);
    }

    q = p - 0.5;
    r = q * q;
    return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q /
           (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1);
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
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(normal_mean, normal_stddev);

    while (!pq.empty()) {
        sum_sizes += pq.size();
        max_size = std::max(max_size, pq.size());
        auto node = pop_normal(pq, dist, gen);
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
    double percentile;

    cxxopts::Options cmd(argv[0], "Dijkstra with relaxed Gaussian extraction");

    cmd.add_options()
    ("h,help", "Print help")
    ("graph", "Input graph file",
        cxxopts::value<std::filesystem::path>(graph_file))
    ("mean", "Normal distribution mean (index space)",
        cxxopts::value<double>(normal_mean))
    ("stddev", "Normal distribution standard deviation (index space)",
        cxxopts::value<double>(normal_stddev))
    ("percentile", "Percentile for index 0 (value in (0,1))",
        cxxopts::value<double>(percentile));

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

    if (!args.count("mean")) {
    std::cerr << "Error: --mean must be provided\n";
    return EXIT_FAILURE;
    }

    bool stddev_given = args.count("stddev");
    bool percentile_given = args.count("percentile");

    if (!stddev_given && !percentile_given) {
        std::cerr << "Error: either --stddev or --percentile must be provided\n";
        return EXIT_FAILURE;
    }

    if (stddev_given && percentile_given) {
        std::cerr << "Error: provide either --stddev or --percentile, not both\n";
        return EXIT_FAILURE;
    }

    if (percentile_given) {
        double z = inverse_normal_cdf(percentile);

        if (z == 0.0) {
            std::cerr << "Error: percentile cannot be 0.5\n";
            return EXIT_FAILURE;
        }

        normal_stddev = (0.0 - normal_mean) / z;

        if (normal_stddev < 0.0) {
            normal_stddev = -normal_stddev;
        }
    }

    std::clog << "= Settings =\n";
    std::clog << "Graph: " << graph_file << '\n';
    std::clog << "Mean: " << normal_mean << '\n';
    std::clog << "Stddev: " << normal_stddev << "\n\n";

    std::clog << "= Running benchmark =\n";
    dijkstra(graph_file, normal_mean, normal_stddev);

    return EXIT_SUCCESS;
}
