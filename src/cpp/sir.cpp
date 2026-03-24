/**
 * Fast SIR simulation using Gillespie's event-driven algorithm.
 *
 * Continuous-time SIR on a weighted graph. Each edge e=(i,j) with weight w_e
 * transmits infection at rate beta * w_e. Infected nodes recover at rate gamma.
 * Uses a min-heap priority queue for O(log n) event processing.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <limits>
#include <tuple>

namespace py = pybind11;

enum EventType : uint8_t { INFECTION = 0, RECOVERY = 1 };
enum State : uint8_t { SUSCEPTIBLE = 0, INFECTED = 1, RECOVERED = 2 };

struct Event {
    double time;
    EventType type;
    int source;
    int target;

    bool operator>(const Event& other) const { return time > other.time; }
};

using MinHeap = std::priority_queue<Event, std::vector<Event>, std::greater<Event>>;

/**
 * Run a single SIR simulation.
 *
 * @param indptr  CSR row pointers (n+1 elements)
 * @param indices CSR column indices
 * @param data    CSR edge weights
 * @param n       Number of nodes
 * @param beta    Infection rate per unit weight
 * @param gamma   Recovery rate
 * @param initial_infected  Initially infected node indices
 * @param t_max   Maximum simulation time
 * @param seed    RNG seed
 *
 * @return (arrival_times, recovery_times) as numpy arrays
 */
std::pair<py::array_t<double>, py::array_t<double>>
sir_simulation_cpp(
    py::array_t<int32_t> indptr,
    py::array_t<int32_t> indices,
    py::array_t<double> data,
    int n,
    double beta,
    double gamma,
    py::array_t<int32_t> initial_infected,
    double t_max,
    int seed)
{
    auto ip = indptr.unchecked<1>();
    auto idx = indices.unchecked<1>();
    auto dat = data.unchecked<1>();
    auto init = initial_infected.unchecked<1>();

    std::mt19937_64 gen(seed);
    std::exponential_distribution<double> exp_gamma(gamma);

    // State arrays
    std::vector<State> state(n, SUSCEPTIBLE);
    std::vector<double> arrival_times(n, std::numeric_limits<double>::infinity());
    std::vector<double> recovery_times(n, std::numeric_limits<double>::infinity());

    MinHeap heap;

    auto schedule_events = [&](int node, double t_infect) {
        // Schedule recovery
        double dt_recover = exp_gamma(gen);
        heap.push({t_infect + dt_recover, RECOVERY, node, -1});

        // Schedule infection attempts to all neighbors
        for (int k = ip(node); k < ip(node + 1); ++k) {
            int nbr = idx(k);
            double w = dat(k);
            if (w > 0) {
                std::exponential_distribution<double> exp_inf(beta * w);
                double dt_infect = exp_inf(gen);
                heap.push({t_infect + dt_infect, INFECTION, node, nbr});
            }
        }
    };

    // Initialize infected nodes
    auto n_init = init.shape(0);
    for (decltype(n_init) i = 0; i < n_init; ++i) {
        int node = init(i);
        state[node] = INFECTED;
        arrival_times[node] = 0.0;
        schedule_events(node, 0.0);
    }

    // Process events
    while (!heap.empty()) {
        Event ev = heap.top();
        heap.pop();

        if (ev.time > t_max) break;

        if (ev.type == RECOVERY) {
            if (state[ev.source] == INFECTED) {
                state[ev.source] = RECOVERED;
                recovery_times[ev.source] = ev.time;
            }
        } else {
            // Infection: source tries to infect target
            if (state[ev.source] == INFECTED && state[ev.target] == SUSCEPTIBLE) {
                state[ev.target] = INFECTED;
                arrival_times[ev.target] = ev.time;
                schedule_events(ev.target, ev.time);
            }
        }
    }

    // Convert to numpy arrays
    auto arr_times = py::array_t<double>(n);
    auto rec_times = py::array_t<double>(n);
    auto at = arr_times.mutable_unchecked<1>();
    auto rt = rec_times.mutable_unchecked<1>();

    for (int i = 0; i < n; ++i) {
        at(i) = arrival_times[i];
        rt(i) = recovery_times[i];
    }

    return {arr_times, rec_times};
}

PYBIND11_MODULE(_sir_cpp, m) {
    m.doc() = "Fast C++ SIR simulation with Gillespie algorithm";
    m.def("sir_simulation_cpp", &sir_simulation_cpp,
          "Run a single SIR simulation on a CSR graph",
          py::arg("indptr"),
          py::arg("indices"),
          py::arg("data"),
          py::arg("n"),
          py::arg("beta"),
          py::arg("gamma"),
          py::arg("initial_infected"),
          py::arg("t_max"),
          py::arg("seed"));
}
