#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <dpct/rng_utils.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>

#define N 256

typedef unsigned long long ull;

// data API for python
struct Ising_info
{
  double E;
  double M;
  double C;
};

dpct::constant_memory<int, 1> dx(sycl::range<1>(4), {1, 0, -1, 0});
dpct::constant_memory<int, 1> dy(sycl::range<1>(4), {0, 1, 0, -1});

// calculate the probability to Filp
/*
DPCT1032:0: A different random number generator is used. You may need to adjust
the code.
*/
bool toFlip(double p, dpct::rng::device::rng_generator<
                          oneapi::mkl::rng::device::philox4x32x10<4>> &s)
{
  if (p > 1.)
    return 1;
  return s.generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 1>() <
         (int)(p * RAND_MAX);
}

// the process of Monte Carlo
void Ising(bool *oldG, bool *NewG, const double T, ull ste,
           sycl::nd_item<3> item_ct1, int *dx, int *dy)
{
  uint seed = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
  /*
  DPCT1032:1: A different random number generator is used. You may need to
  adjust the code.
  */
  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>
      s;
  s = dpct::rng::device::rng_generator<
      oneapi::mkl::rng::device::philox4x32x10<4>>(seed, {0, 0 * 8});
  while (ste--)
  {
    int x = s.generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 1>() % N;
    int y = s.generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 1>() % N;
    int Sum1 = 0, Sum2 = 0;
    for (int i = 0; i < 4; i++)
    {
      int Nx = x + dx[i], Ny = y + dy[i];
      if (Nx < 0 || Nx > N || Ny < 0 || Ny > N)
        continue;
      oldG[Nx * N + Ny] ^ oldG[x * N + y] ? Sum1++ : Sum1--;
      Sum2++;
    }
    Sum2 -= Sum1;
    if (Sum2 < Sum1 || toFlip(sycl::exp(-1.0 * (Sum2 - Sum1) / T), s))
      NewG[x * N + y] ^= 1;
  }
}

// a iteration of Ising,which lasts for "time"
inline void iter(ull time, bool *G, bool *G_cpy, double T)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  ull threadsPerBlock = 1024;
  ull blockPerGrid = 1024;
  dpct::get_default_queue().memcpy(&G_cpy, &G, N * N * sizeof(bool)).wait();
  /*
  DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    dx.init();
    dy.init();

    auto dx_ptr_ct1 = dx.get_ptr();
    auto dy_ptr_ct1 = dy.get_ptr();

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blockPerGrid) *
                              sycl::range<3>(1, 1, threadsPerBlock),
                          sycl::range<3>(1, 1, threadsPerBlock)),
        [=](sycl::nd_item<3> item_ct1) {
          Ising(G_cpy, G, T, (time / threadsPerBlock / blockPerGrid), item_ct1,
                dx_ptr_ct1, dy_ptr_ct1);
        });
  });
}

// thrust sturct for calculate energy
/*
DPCT1044:3: thrust::unary_function was removed because std::unary_function has
been deprecated in C++11. You may need to remove references to typedefs from
thrust::unary_function in the class definition.
*/
struct estimate_energy {
  bool *G;
  estimate_energy(bool *_G) : G(_G) {}
  int operator()(uint id)
  {
    int H = 0;
    if (id + 1 < N * N)
      G[id + 1] ^ G[id] ? H++ : H--;
    if (id + N < N * N)
      G[id + N] ^ G[id] ? H++ : H--;
    return H;
  }
};

// thrust sturct for calculate magnetic
/*
DPCT1044:4: thrust::unary_function was removed because std::unary_function has
been deprecated in C++11. You may need to remove references to typedefs from
thrust::unary_function in the class definition.
*/
struct estimate_mag {
  bool *G;
  estimate_mag(bool *_G) : G(_G) {}
  int operator()(uint id)
  {
    return G[id] ? 1 : -1;
  }
};

// to calculate energy by a reduce way
int Energy(bool *G)
{
  return std::transform_reduce(oneapi::dpl::execution::seq,
                               oneapi::dpl::counting_iterator<int>(0),
                               oneapi::dpl::counting_iterator<int>(N * N), 0,
                               std::plus<int>(), estimate_energy(G));
}

// to calculate magnetic by a reduce way
int Magnetic(bool *G)
{
  return abs(std::transform_reduce(oneapi::dpl::execution::seq,
                                   oneapi::dpl::counting_iterator<int>(0),
                                   oneapi::dpl::counting_iterator<int>(N * N),
                                   0, std::plus<int>(), estimate_mag(G)));
}

// the python API function, also the main function of Ising program
// T is the temperature of ising model, GPUid is the GPU we used to run our program
extern "C" Ising_info *Ising_E(double T, int GPUid)
{
  /*
  DPCT1093:5: The "GPUid" may not be the best XPU device. Adjust the selected
  device if needed.
  */
  dpct::select_device(GPUid);
  ull time = 1LL << 20;

  bool *dev_map, *dev_map_cpy;

  dev_map = sycl::malloc_device<bool>(N * N, dpct::get_default_queue());
  dev_map_cpy = sycl::malloc_device<bool>(N * N, dpct::get_default_queue());

  // warm up the ising model
  iter(time << 4, dev_map, dev_map_cpy, T);

  int eNum = 2048;
  double eAver = 0., eAverSq = 0.;
  // the sample process,to calculate the C
  for (int i = 0; i < eNum; i++)
  {
    double e = 1. * Energy(dev_map) / (N * N * 1.);
    eAver += e;
    eAverSq += e * e;
    iter(time, dev_map, dev_map_cpy, T);
  }
  eAver /= eNum;
  eAverSq /= eNum;
  double C = eAverSq - eAver * eAver;
  C /= T * T;
  // load data to python data api
  Ising_info *pans = (Ising_info *)malloc(sizeof(Ising_info));
  pans->E = Energy(dev_map) * 1. / N / N;
  pans->M = Magnetic(dev_map) * 1. / N / N;
  pans->C = C;

  sycl::free(dev_map, dpct::get_default_queue());
  sycl::free(dev_map_cpy, dpct::get_default_queue());
  return pans;
}
