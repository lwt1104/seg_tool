#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/vfh.h>
#include <pcl/filters/filter.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;

double thresh = 0.3;
int save = 0;
float radius = 0.04;
double max_metric = -1;
double min_metric = 100000;
double min_z = -1;
double max_z = 100;

float get_depth(const pcl::PointCloud<PointType>::Ptr& cloud) {
  float zsum = 0;
  int num = 0;
  for (size_t i = 0; i < cloud->points.size(); i++) {
    PointType &p = cloud->points[i];
    if (!pcl::isFinite(p)) {
      continue;
    }
    zsum += p.z;
    num++;
  }
  float distance = zsum / num;
  return distance;
}

int matchPLane(const boost::filesystem::path &path) {
  pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
  pcl::PointCloud<NormalType>::Ptr cloud_normals (new pcl::PointCloud<NormalType> ());
  pcl::NormalEstimation<PointType, pcl::Normal> ne;  
  pcl::SACSegmentationFromNormals<PointType, pcl::Normal> seg; 
  pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices);
  pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());

  pcl::PCDWriter writer;
  pcl::ExtractIndices<PointType> extract;

  if (pcl::io::loadPCDFile<PointType> (path.string(), *cloud) == -1) {
    PCL_ERROR ("Couldn't read file %s.pcd \n", path.string().c_str());
    return -1;
  }
  float depth = get_depth(cloud);
  if (depth < min_z || depth > max_z) {
    return -1;
  }

  
  // Estimate point normals
  ne.setSearchMethod (tree);
  ne.setInputCloud (cloud);
  ne.setKSearch (30);
  ne.compute (*cloud_normals);

  // Create the segmentation object for the planar model and set all the parameters
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
  seg.setNormalDistanceWeight (0.1);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.03);
  seg.setInputCloud (cloud);
  seg.setInputNormals (cloud_normals);
  // Obtain the plane inliers and coefficients
  seg.segment (*inliers_plane, *coefficients_plane);
  // std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

  // Extract the planar inliers from the input cloud
  extract.setInputCloud (cloud);
  extract.setIndices (inliers_plane);
  extract.setNegative (false);

  // Write the planar inliers to disk
  pcl::PointCloud<PointType>::Ptr cloud_plane (new pcl::PointCloud<PointType> ());
  extract.filter (*cloud_plane);  
  if (cloud_plane->points.empty ()) {
    std::cerr << "Can't find the cylindrical component.\t" << path.filename().string()<< std::endl;
    return 0;
  } else {
    float ratio = (float)cloud_plane->points.size () / cloud->points.size(); 
    std::cerr <<  path.filename().string() << " " << cloud_plane->points.size () 
      << "\t total size: " << cloud->points.size()
      << "\tplane points ratio " << ratio << std::endl;
	  
    if (save) {
      std::string current_dir = path.parent_path().string();
      std::string plane_name = "plane_" + path.filename().string();
  	  writer.write (current_dir + "/" + plane_name, *cloud_plane, false);
    }

    if (ratio > max_metric) {
      max_metric = ratio;
    } 
    if (ratio < min_metric) {
      min_metric = ratio;
    }

    if (ratio > thresh) {
    	return 1;
    } else {
      return 0;
    }
  }
}

void testModel(const boost::filesystem::path &base_dir, const std::string &extension) {
  if (!boost::filesystem::exists (base_dir) && !boost::filesystem::is_directory (base_dir))
    return;
  int num = 0;
  int detect = 0;
  for (boost::filesystem::directory_iterator it(base_dir); it != boost::filesystem::directory_iterator (); ++it)
  {
    if (!boost::filesystem::is_regular_file (it->status ()) || boost::filesystem::extension (it->path ()) != extension) {
      continue;
    }
    int result = matchPLane(base_dir / it->path ().filename ());
    if ( result != -1) {
      num++;
    }
    if (result == 1) {
    	detect++;
    }
  }
  pcl::console::print_info ("%d postive of total %d, rate: %f\n", detect, num, (float)(detect) / num);
  pcl::console::print_info ("min: %f \t max: %f\n", min_metric, max_metric);
}



int
main (int argc, char** argv)
{
  if (argc < 2)
  {
    PCL_ERROR ("Need at two parameters! Syntax is: %s [test_dir]\n", argv[0]);
    return (-1);
  }
  int k = 2;

  pcl::console::parse_argument (argc, argv, "-thresh", thresh);
  // Search for the k closest matches
  pcl::console::parse_argument (argc, argv, "-k", k);

  pcl::console::parse_argument (argc, argv, "-s", save);

  pcl::console::parse_argument (argc, argv, "-r", radius);

  pcl::console::parse_argument (argc, argv, "-min_z", min_z);
  pcl::console::parse_argument (argc, argv, "-max_z", max_z);

  std::string extension (".pcd");
  transform (extension.begin (), extension.end (), extension.begin (), (int(*)(int))tolower);
  testModel(argv[1], extension);

  return 0;
}