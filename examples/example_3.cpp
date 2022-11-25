#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
 
void matSum(uchar *dataIn, int *dataOutSum, int *dataOutMax, int *dataOutMin, int imgHeight, int imgWidth,
            sycl::nd_item<3> item_ct1, uint8_t *dpct_local, int *_max, int *_min)
{
    //__shared__ int _data[1600];
    const int number = 2048;

    auto _sum = (int *)dpct_local; // 小图像块中求和共享数组
                                   // 小图像块中求最大值共享数组
                                   // 小图像块中求最小值共享数组

    int thread =
        item_ct1.get_local_id(2) +
        item_ct1.get_local_id(1) *
            item_ct1.get_local_range(2); // 一个block中所有thread的索引值
    int threadIndex =
        item_ct1.get_local_id(2) +
        item_ct1.get_local_id(1) * imgWidth; // 每个小块中存放数据的thread索引值
    //每个小块中存放数据的block索引值
    int blockIndex1 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                      2 * item_ct1.get_group(1) * item_ct1.get_local_range(1) *
                          imgWidth; // 40*20的上半block索引值
    int blockIndex2 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                      (2 * item_ct1.get_group(1) + 1) *
                          item_ct1.get_local_range(1) *
                          imgWidth; // 40*20的下半block索引值

    int index1 = threadIndex + blockIndex1; //每个block中上半部分索引值
    int index2 = threadIndex + blockIndex2; //每个block中下半部分索引值
 
    //将待计算的40*40小图像块中的所有像素值分两次传送到共享数组中
    _sum[thread] = dataIn[index1]; //将上半部分的40*20中所有数据赋值到共享数组中
    _sum[thread + item_ct1.get_local_range(2) * item_ct1.get_local_range(1)] =
        dataIn[index2]; // 将下半部分的40*20中所有数据赋值到共享数组中

    _max[thread] = dataIn[index1];
    _max[thread + item_ct1.get_local_range(2) * item_ct1.get_local_range(1)] =
        dataIn[index2];

    _min[thread] = dataIn[index1];
    _min[thread + item_ct1.get_local_range(2) * item_ct1.get_local_range(1)] =
        dataIn[index2];

    //memcpy(_sum, _data, 1600 * sizeof(int));
    //memcpy(_max, _data, 1600 * sizeof(int));
    //memcpy(_min, _data, 1600 * sizeof(int));  在GPU（Device）中用memcpy函数进行拷贝会导致显卡混乱，故不选择此种方式
 
    //利用归约算法求出40*40小图像块中1600个像素值中的和、最大值以及最小值
    for (unsigned int s = number / 2; s > 0; s >>= 1)
    {
        if (thread < s)
        { 
            _sum[thread] += _sum[thread + s]; 
            if (_max[thread] < _max[thread + s]) { _max[thread] = _max[thread + s]; }
            if (_min[thread] > _min[thread + s]) { _min[thread] = _min[thread + s]; }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space); // 所有线程同步
    }
    if (threadIndex == 0) 
    { 
        //将每个小块中的结果储存到输出中
        dataOutSum[item_ct1.get_group(2) +
                   item_ct1.get_group(1) * item_ct1.get_group_range(2)] =
            _sum[0];
        dataOutMax[item_ct1.get_group(2) +
                   item_ct1.get_group(1) * item_ct1.get_group_range(2)] =
            _max[0];
        dataOutMin[item_ct1.get_group(2) +
                   item_ct1.get_group(1) * item_ct1.get_group_range(2)] =
            _min[0];
    }
 
}
 
int main()
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    Mat image = imread("test.jpg", 0); //读取待检测图片
    int sum[5000]; //求和结果数组
    int max[5000]; //最大值结果数组
    int min[5000]; //最小值结果数组
    imshow("src", image);
 
    size_t memSize = image.cols*image.rows*sizeof(uchar);
    int size = 5000 * sizeof(int);
 
    uchar *d_src = NULL;
    int *d_sum = NULL;
    int *d_max = NULL;
    int *d_min = NULL;

    d_src = (uchar *)sycl::malloc_device(memSize, q_ct1);
    d_sum = (int *)sycl::malloc_device(size, q_ct1);
    d_max = (int *)sycl::malloc_device(size, q_ct1);
    d_min = (int *)sycl::malloc_device(size, q_ct1);

    q_ct1.memcpy(d_src, image.data, memSize).wait();

    int imgWidth = image.cols;
    int imgHeight = image.rows;

    sycl::range<3> threadsPerBlock(1, 20, 40); // 每个block大小为40*20
    sycl::range<3> blockPerGrid(1, 200,
                                25); // 将8000*1000的图片分为25*200个小图像块

    double time0 = static_cast<double>(getTickCount()); //计时器开始

    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh) {
        /*
        DPCT1083:1: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(4096 * sizeof(int)), cgh);
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            _max_acc_ct1(sycl::range<1>(2048 /*number*/), cgh);
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            _min_acc_ct1(sycl::range<1>(2048 /*number*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(blockPerGrid * threadsPerBlock, threadsPerBlock),
            [=](sycl::nd_item<3> item_ct1) {
                matSum(d_src, d_sum, d_max, d_min, imgHeight, imgWidth,
                       item_ct1, dpct_local_acc_ct1.get_pointer(),
                       _max_acc_ct1.get_pointer(), _min_acc_ct1.get_pointer());
            });
    });

    time0 = ((double)getTickCount() - time0) / getTickFrequency(); //计时器结束
    cout << "The Run Time is :" << time0 << "s" << endl; //输出运行时间

    q_ct1.memcpy(sum, d_sum, size);
    q_ct1.memcpy(max, d_max, size);
    q_ct1.memcpy(min, d_min, size).wait();

    cout << "The sum is :" << sum[0] << endl;
    cout << "The max is :" << max[0] << endl;
    cout << "The min is :" << min[0] << endl;
 
    waitKey(0);

    sycl::free(d_src, q_ct1);
    sycl::free(d_sum, q_ct1);
    sycl::free(d_max, q_ct1);
    sycl::free(d_min, q_ct1);

    return 0;
}
