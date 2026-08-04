#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#if !defined(NND_BENCH_CANDIDATE) && !defined(NND_BENCH_MAIN)
#error "Compile with NND_BENCH_CANDIDATE or NND_BENCH_MAIN"
#endif

namespace nnd = cuvs::neighbors::nn_descent;

template <typename T>
void fill_data(raft::resources const& res, T* data, std::size_t size)
{
  raft::random::RngState rng(1234ULL);
  raft::random::normal(res, rng, data, size, 0.1f, 2.0f);
}

void run_case(char const* variant,
              char const* mode_name,
              std::int64_t rows,
              std::int64_t dim,
              int graph_degree,
              int iterations,
              int repeats,
              nnd::DIST_COMP_DTYPE mode)
{
  raft::resources res;
  auto data = raft::make_device_matrix<float, std::int64_t>(res, rows, dim);
  fill_data(res, data.data_handle(), static_cast<std::size_t>(rows * dim));
  raft::resource::sync_stream(res);

  nnd::index_params params;
  params.metric                    = cuvs::distance::DistanceType::InnerProduct;
  params.graph_degree              = graph_degree;
  params.intermediate_graph_degree = 2 * graph_degree;
  params.max_iterations            = iterations;
  params.termination_threshold     = 0.0f;
  params.return_distances          = false;
  params.dist_comp_dtype           = mode;

  auto view = raft::make_const_mdspan(data.view());
  {
    auto warmup = nnd::build(res, params, view);
    raft::resource::sync_stream(res);
  }

  std::vector<double> samples;
  samples.reserve(repeats);
  for (int repeat = 0; repeat < repeats; ++repeat) {
    auto begin = std::chrono::steady_clock::now();
    auto index = nnd::build(res, params, view);
    raft::resource::sync_stream(res);
    auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }

  std::sort(samples.begin(), samples.end());
  double mean   = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  double median = samples[samples.size() / 2];
  std::cout << variant << ',' << mode_name << ",rows=" << rows << ",dim=" << dim
            << ",degree=" << graph_degree << ",iterations=" << iterations
            << ",median_ms=" << std::fixed << std::setprecision(3) << median << ",mean_ms=" << mean
            << ",samples_ms=";
  for (double sample : samples) {
    std::cout << sample << ';';
  }
  std::cout << '\n';
}

std::vector<std::int64_t> parse_dims(std::string const& dims_csv)
{
  std::vector<std::int64_t> dims;
  std::stringstream stream(dims_csv);
  std::string value;
  while (std::getline(stream, value, ',')) {
    if (!value.empty()) { dims.push_back(std::stoll(value)); }
  }
  if (dims.empty()) { throw std::runtime_error("At least one dimension is required"); }
  return dims;
}

int main(int argc, char** argv)
{
  std::int64_t rows = argc > 1 ? std::atoll(argv[1]) : 20000;
  int repeats       = argc > 2 ? std::atoi(argv[2]) : 7;
  int iterations    = argc > 3 ? std::atoi(argv[3]) : 20;
  int graph_degree  = argc > 4 ? std::atoi(argv[4]) : 64;
  std::string dims  = argc > 5 ? argv[5] : "64,256,1024";

  if (rows <= 0 || repeats <= 0 || iterations <= 0 || graph_degree <= 0) {
    std::cerr << "rows, repeats, iterations, and graph degree must be positive\n";
    return 2;
  }

  for (std::int64_t dim : parse_dims(dims)) {
#if defined(NND_BENCH_CANDIDATE)
    run_case("candidate",
             "tf32",
             rows,
             dim,
             graph_degree,
             iterations,
             repeats,
             nnd::DIST_COMP_DTYPE::TF32);
    run_case("candidate",
             "fp16",
             rows,
             dim,
             graph_degree,
             iterations,
             repeats,
             nnd::DIST_COMP_DTYPE::FP16);
#else
    run_case(
      "main", "fp32", rows, dim, graph_degree, iterations, repeats, nnd::DIST_COMP_DTYPE::FP32);
    run_case(
      "main", "fp16", rows, dim, graph_degree, iterations, repeats, nnd::DIST_COMP_DTYPE::FP16);
#endif
  }
  return 0;
}
