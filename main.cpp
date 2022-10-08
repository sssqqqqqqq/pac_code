#include "Defines.h"
#include <CL/sycl.hpp>
#include <omp.h>
#include <mpi.h>
using namespace sycl;
inline void correntess(ComplexType result1, ComplexType result2, ComplexType result3) {
  double re_diff, im_diff;
  int numThreads = 64;
  // #pragma omp parallel
  //     {
  //       int ttid = omp_get_thread_num();
  //       if (ttid == 0)
  //         numThreads = omp_get_num_threads();
  //     }
  //  printf("here are %d threads \n",numThreads);
   numThreads = 65;
if(numThreads<= 64){
  re_diff = fabs(result1.real() - -264241151.454552);
  im_diff = fabs(result1.imag() - 1321205770.975190);
  re_diff += fabs(result2.real() - -137405397.758745);
  im_diff += fabs(result2.imag() - 961837795.884157);
  re_diff += fabs(result3.real() - -83783779.241634);
  im_diff += fabs(result3.imag() - 754054017.424472);
  printf("%f,%f\n",re_diff,im_diff);
}else{
  re_diff = fabs(result1.real() - -264241151.200123);
  im_diff = fabs(result1.imag() - 1321205763.246570);
  re_diff += fabs(result2.real() - -137405398.773852);
  im_diff += fabs(result2.imag() - 961837794.726070);
  re_diff += fabs(result3.real() - -83783779.939936);
  im_diff += fabs(result3.imag() - 754054018.099450);
}
  if (re_diff < 10 && im_diff < 10)
    printf("\n!!!! SUCCESS - !!!! Correctness test passed :-D :-D\n\n");
  else
    printf("\n!!!! FAILURE - Correctness test failed :-( :-(  \n");
}



int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);


  int number_bands = 0, nvband = 0, ncouls = 0, nodes_per_group = 0;
  int npes = 1;
  if (argc == 1) {
    number_bands = 512;
    nvband = 2;
    ncouls = 32768;
    nodes_per_group = 20;
  } else if (argc == 5) {
    number_bands = atoi(argv[1]);
    nvband = atoi(argv[2]);
    ncouls = atoi(argv[3]);
    nodes_per_group = atoi(argv[4]);
  } else {
    if(world_rank == 0){
          std::cout << "The correct form of input is : " << std::endl;
          std::cout << " ./main.exe <number_bands> <number_valence_bands> "
                 "<number_plane_waves> <nodes_per_mpi_group> "
              << std::endl;
    }

    exit(0);
  }
  int ngpown = ncouls / (nodes_per_group * npes);
  // Constants that will be used later
  const DataType e_lk = 10;
  const DataType dw = 1;
  const DataType to1 = 1e-6;
  const DataType limittwo = pow(0.5, 2);
  const DataType e_n1kq = 6.0;

  // Using time point and system_clock
  time_point<system_clock> start, end, k_start, k_end;

  start = system_clock::now();
  double elapsedKernelTimer;

  // Printing out the params passed.
  if(world_rank == 0){
    std::cout << "Sizeof(ComplexType = "
              << sizeof(ComplexType) << " bytes" << std::endl;
    std::cout << "number_bands = " << number_bands << "\t nvband = " << nvband
              << "\t ncouls = " << ncouls
              << "\t nodes_per_group  = " << nodes_per_group
              << "\t ngpown = " << ngpown << "\t nend = " << nend
              << "\t nstart = " << nstart << std::endl;
  }

  size_t memFootPrint = 0.00;



  std::vector<sycl::queue> qs;
  auto platforms = sycl::platform::get_platforms();
  for (auto & p : platforms) {
    p.get_info<sycl::info::platform::name>();
    auto devices = p.get_devices();
    for (auto & d : devices ) {
        d.get_info<sycl::info::device::name>();
        if (d.is_gpu()) {
          qs.push_back(sycl::queue(d));
        }
    }
  }




  memFootPrint += 3 * (nend - nstart) * sizeof(DataType);
  memFootPrint += ngpown * sizeof(int);
  memFootPrint += (ncouls + 1) * sizeof(int);
  memFootPrint += ncouls * sizeof(DataType);
  memFootPrint += 2 * (ngpown * ncouls) * sizeof(ComplexType);
  memFootPrint += 2 * (number_bands * ncouls) * sizeof(ComplexType);
  memFootPrint += (nend - nstart) * sizeof(ComplexType);
  // Print Memory Foot print
  if(world_rank == 0){
    cout << "MPI All Memory Foot Print = " << memFootPrint*2*world_size / pow(1024, 3) << " GBs"
        << std::endl;
  }
  size_t r1 = static_cast<size_t>(number_bands);
  size_t r2 = static_cast<size_t>(ncouls);
  size_t r3 = static_cast<size_t>(ngpown);

  sycl::queue q = qs[world_rank];
  // std::cout << "Rank:"<<world_rank<<" Size:"<<world_size<<" Device-Size:"<<qs.size()<< 
  // " Device: " << q.get_device().get_info<info::device::name>() << "\n";


  
  DataType *result = new DataType((nend-nstart)*2);
  ComplexType *achtemp = new ComplexType(nend-nstart);

  ComplexType *aqsmtemp_host = malloc_host<ComplexType>(number_bands*ncouls,q);
  ComplexType *aqsntemp_host = malloc_host<ComplexType>(number_bands*ncouls,q);
  ComplexType *I_eps_array_host = malloc_host<ComplexType>(ngpown*ncouls,q);
  ComplexType *wtilde_array_host = malloc_host<ComplexType>(ngpown*ncouls,q);
  DataType *vcoul_host = malloc_host<DataType>(ncouls,q);
  int *inv_igp_index_host = malloc_host<int>(ngpown,q);
  int *indinv_host = malloc_host<int>(ncouls + 1,q);
  DataType *wx_array_host = malloc_host<DataType>(nend - nstart,q);

  DataType *sum_vec = malloc_shared<DataType>((nend-nstart)*2, q);
  ComplexType *aqsmtemp = malloc_device<ComplexType>(number_bands*ncouls, q);
  ComplexType *aqsntemp = malloc_device<ComplexType>(number_bands*ncouls, q);
  ComplexType *I_eps_array = malloc_device<ComplexType>(ngpown*ncouls, q);
  ComplexType *wtilde_array = malloc_device<ComplexType>(ngpown*ncouls, q);
  DataType *vcoul = malloc_device<DataType>(ncouls, q);
  int *inv_igp_index = malloc_device<int>(ngpown, q);
  int *indinv = malloc_device<int>(ncouls + 1, q);
  DataType *wx_array = malloc_device<DataType>(nend - nstart, q);


  ComplexType  expr(.5, .5);

  for (int i = 0; i < number_bands; i++){
    for (int j = 0; j < ncouls; j++) {
      aqsmtemp_host[i*ncouls+j] = expr;
      aqsntemp_host[i*ncouls+j] = expr;
    }
  }

  for (int i = 0; i < ngpown; i++)
    for (int j = 0; j < ncouls; j++) {
      I_eps_array_host[i*ncouls+j] = expr;
      wtilde_array_host[i*ncouls+j] = expr;
    }

  for (int i = 0; i < ncouls; i++)
    vcoul_host[i] = 1.0;

  for (int ig = 0; ig < ngpown; ++ig)
    inv_igp_index_host[ig] = (ig + 1) * ncouls / ngpown;

  for (int ig = 0; ig < ncouls; ++ig)
    indinv_host[ig] = ig;
  indinv_host[ncouls] = ncouls - 1;
  for (int iw = nstart; iw < nend; ++iw) {
    wx_array_host[iw] = e_lk - e_n1kq + dw * ((iw + 1) - 2);
    if (wx_array_host[iw] < to1)
      wx_array_host[iw] = to1;
  }
  for(int i = 0;i<(nend-nstart)*2;i++){
    sum_vec[i] = 0.0;
  }

    
  k_start = system_clock::now();
  q.memcpy(aqsmtemp,aqsmtemp_host,sizeof(ComplexType)*number_bands*ncouls).wait();
  q.memcpy(aqsntemp,aqsntemp_host,sizeof(ComplexType)*number_bands*ncouls).wait();
  q.memcpy(I_eps_array,I_eps_array_host,sizeof(ComplexType)*ngpown*ncouls).wait();
  q.memcpy(wtilde_array,wtilde_array_host,sizeof(ComplexType)*ngpown*ncouls).wait();
  q.memcpy(vcoul,vcoul_host,sizeof(DataType)*ncouls).wait();
  q.memcpy(inv_igp_index,inv_igp_index_host,sizeof(int)*ngpown).wait();
  q.memcpy(indinv,indinv_host,sizeof(int)*(ncouls+1)).wait();
  q.memcpy(wx_array,wx_array_host,sizeof(DataType)*(nend-nstart)).wait();

  // noflagOCC_solver(number_bands, ngpown, ncouls, inv_igp_index, indinv,
  //                  wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array,
  //                  vcoul, achtemp);



  double time;

  time = MPI_Wtime();    
  q.submit([&](handler& h) {
          h.parallel_for(nd_range<3>{range<3>{r1/world_size,r2,1},range<3>{32,16,1}},
          reduction(&sum_vec[0], sycl::plus<>()),reduction(&sum_vec[1], sycl::plus<>()),
              reduction(&sum_vec[2], sycl::plus<>()),reduction(&sum_vec[3], sycl::plus<>()),
              reduction(&sum_vec[4], sycl::plus<>()),reduction(&sum_vec[5], sycl::plus<>()),
              [=](nd_item<3> it,auto &tmp,auto & tmp1,auto &tmp2,auto &tmp3,auto &tmp4,auto &tmp5)
              [[intel::reqd_sub_group_size(32)]] { 
            int n1 = it.get_global_id(0)+(world_rank*(r1/world_size));
            int ig = it.get_global_id(1);
            double re_tmp[nend-nstart];
            double im_tmp[nend-nstart];
            for (int iw = nstart; iw < nend; ++iw) {
              re_tmp[iw] = 0.0;
              im_tmp[iw] = 0.0;
            }
            for(int my_igp = 0;my_igp<ngpown;my_igp++){
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];
                ComplexType sch_store =
                ComplexType_conj(aqsmtemp[n1*ncouls+igp]) * aqsntemp[n1*ncouls+igp] * 0.5 *
                vcoul[igp] * wtilde_array[my_igp*ncouls+igp];
                for (int iw = nstart; iw < nend; ++iw) {
                  ComplexType wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls+ig];
                  ComplexType delw = ComplexType_conj(wdiff) * (1 / (wdiff * ComplexType_conj(wdiff)).real());
                  ComplexType sch_array = delw * I_eps_array[my_igp*ncouls+ig] * sch_store;
                  re_tmp[iw] += (sch_array).real();
                  im_tmp[iw] += (sch_array).imag();
                }
            }
            tmp.combine(re_tmp[0]);
            tmp1.combine(re_tmp[1]);
            tmp2.combine(re_tmp[2]);
            tmp3.combine(im_tmp[0]);
            tmp4.combine(im_tmp[1]);
            tmp5.combine(im_tmp[2]);
      });
  }).wait();

  MPI_Reduce(&sum_vec[0], &result[0], (nend-nstart)*2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  sycl::free(aqsntemp , q);
  sycl::free(aqsmtemp , q);
  sycl::free(I_eps_array , q);
  sycl::free(wtilde_array, q);
  sycl::free(vcoul , q);
  sycl::free(inv_igp_index, q);
  sycl::free(indinv , q);
  sycl::free(wx_array, q);
  sycl::free(sum_vec, q);

  sycl::free(aqsntemp_host, q);
  sycl::free(aqsmtemp_host , q);
  sycl::free(I_eps_array_host , q);
  sycl::free(wtilde_array_host, q);
  sycl::free(vcoul_host , q);
  sycl::free(inv_igp_index_host, q);
  sycl::free(indinv_host , q);
  sycl::free(wx_array_host, q);
  // time = MPI_Wtime() - time;
  // printf("world_rank = %d  time = %f ms\n", world_rank,time * 1000); 
  // Check for correctness
  // correntess0(achtemp(0));
  // correntess1(achtemp(1));
  // correntess2(achtemp(2));
  k_end = system_clock::now();
  duration<double> elapsed = k_end - k_start;
  elapsedKernelTimer = elapsed.count();
  if(world_rank == 0){ 
    achtemp[0] = ComplexType(result[0],result[3]);
    achtemp[1] = ComplexType(result[1],result[4]);
    achtemp[2] = ComplexType(result[2],result[5]);
    correntess(achtemp[0],achtemp[1],achtemp[2]);
    printf("\n Final achtemp\n");
    ComplexType_print(achtemp[0]);
    ComplexType_print(achtemp[1]);
    ComplexType_print(achtemp[2]);
  }


  end = system_clock::now();
  elapsed = end - start;

  if(world_rank == 0){   
    cout << "********** Kernel Time Taken **********= " << elapsedKernelTimer
          << " secs" << std::endl;
    cout << "********** Total Time Taken **********= " << elapsed.count()
          << " secs" << std::endl;
  }
  MPI_Finalize();
  return 0;

}


