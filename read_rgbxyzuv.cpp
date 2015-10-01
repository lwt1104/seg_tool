#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
/*
 * Sample code for reading point clouds in the RGB-D Object Dataset using the Point Cloud Library
 *
 * Author: Kevin Lai
 */
struct PointXYZRGBIM
{
  union
  {
    struct
    {
      float x;
      float y;
      float z;
      float rgb;
      float imX;
      float imY;
    };
    float data[6];
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZRGBIM,
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (float, rgb, rgb)
                                    (float, imX, imX)
                                    (float, imY, imY)
)

int
execute (int argc, char** argv)
{
  pcl::PointCloud<PointXYZRGBIM>::Ptr cloud (new pcl::PointCloud<PointXYZRGBIM>);

  if (pcl::io::loadPCDFile<PointXYZRGBIM> ("test_pcd.pcd", *cloud) == -1) //* load the file
  {
    printf ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }
  std::cout << "Loaded "
            << cloud->width * cloud->height
            << " data points from test_pcd.pcd with the following fields: "
            << std::endl;

  for (size_t i = 0; i < cloud->points.size (); ++i){
    uint32_t rgb = *reinterpret_cast<int*>(&cloud->points[i].rgb);
    uint8_t r = (rgb >> 16) & 0x0000ff;
    uint8_t g = (rgb >> 8)  & 0x0000ff;
    uint8_t b = (rgb)       & 0x0000ff;
    std::cout << "    " << cloud->points[i].x
              << " "    << cloud->points[i].y
              << " "    << cloud->points[i].z
              << " "    << (int)r
              << " "    << (int)g
              << " "    << (int)b
              << " "    << cloud->points[i].imX
              << " "    << cloud->points[i].imY << std::endl;
  }

  return (0);
}