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

typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;

double thresh = 0.2;
int save_cylinder = 0;
float radius = 0.05;

int matchCylinder(const boost::filesystem::path &path) {
  pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
  pcl::PointCloud<NormalType>::Ptr cloud_normals (new pcl::PointCloud<NormalType> ());
  pcl::NormalEstimation<PointType, pcl::Normal> ne;  
  pcl::SACSegmentationFromNormals<PointType, pcl::Normal> seg; 
  pcl::ModelCoefficients::Ptr coefficients_cylinder (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_cylinder (new pcl::PointIndices);
  pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());

  pcl::PCDWriter writer;
  pcl::ExtractIndices<PointType> extract;

  if (pcl::io::loadPCDFile<PointType> (path.string(), *cloud) == -1) {
    PCL_ERROR ("Couldn't read file %s.pcd \n", path.string().c_str());
    return -1;
  }
  // Estimate point normals
  ne.setSearchMethod (tree);
  ne.setInputCloud (cloud);
  ne.setKSearch (30);
  ne.compute (*cloud_normals);

  // Create the segmentation object for cylinder segmentation and set all the parameters
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_CYLINDER);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight (0.1);
  seg.setMaxIterations (10000);
  seg.setDistanceThreshold (0.05);
  seg.setRadiusLimits (0, radius);
  seg.setInputCloud (cloud);
  seg.setInputNormals (cloud_normals);

  // Obtain the cylinder inliers and coefficients
  seg.segment (*inliers_cylinder, *coefficients_cylinder);
  // std::cerr << "Cylinder coef7cients: " << *coefficients_cylinder << std::endl;

  // Write the cylinder inliers to disk
  extract.setInputCloud (cloud);
  extract.setIndices (inliers_cylinder);
  extract.setNegative (false);
  pcl::PointCloud<PointType>::Ptr cloud_cylinder (new pcl::PointCloud<PointType> ());
  extract.filter (*cloud_cylinder);
  if (cloud_cylinder->points.empty ()) {
    std::cerr << "Can't find the cylindrical component.\t" << path.filename().string()<< std::endl;
    return 0;
  } else {
    float ratio = (float)cloud_cylinder->points.size () / cloud->points.size();
	  std::cerr << "Cylindrical: " << cloud_cylinder->points.size () 
	    << "\t total size: " << cloud->points.size()
	    << "\tCylinder points ratio " << ratio << std::endl;
	  
    if (save_cylinder) {
      std::string current_dir = path.parent_path().string();
      std::string cylinder_name = "cylinder_" + path.filename().string();
  	  writer.write (current_dir + "/" + cylinder_name, *cloud_cylinder, false);
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
    int result = matchCylinder(base_dir / it->path ().filename ());
    if ( result != -1) {
      num++;
    }
    if (result == 1) {
    	detect++;
    }
  }
  pcl::console::print_info ("%d postive of total %d, rate: %f\n", detect, num, (float)(detect) / num);

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

  pcl::console::parse_argument (argc, argv, "-cylinder", save_cylinder);

  pcl::console::parse_argument (argc, argv, "-r", radius);
  std::string extension (".pcd");
  transform (extension.begin (), extension.end (), extension.begin (), (int(*)(int))tolower);
  testModel(argv[1], extension);

  return 0;
}