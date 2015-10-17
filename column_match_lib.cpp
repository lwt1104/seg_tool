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

float radius = 0.05;

double matchCylinder(pcl::PointCloud<PointType>::Ptr cloud, int silent = 0) {
  pcl::PointCloud<NormalType>::Ptr cloud_normals (new pcl::PointCloud<NormalType> ());
  pcl::NormalEstimation<PointType, pcl::Normal> ne;  
  pcl::SACSegmentationFromNormals<PointType, pcl::Normal> seg; 
  pcl::ModelCoefficients::Ptr coefficients_cylinder (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_cylinder (new pcl::PointIndices);
  pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());

  pcl::ExtractIndices<PointType> extract;

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
    if (!silent) {
      // std::cerr << "Can't find the cylindrical component.\t" << std::endl;
      std::cerr <<  " 0"   
        << "\t total size: " << cloud->points.size()
        << "\tCylinder points ratio " << 0 << std::endl;
    }
    return 0;
  } else {
    double ratio = (double)cloud_cylinder->points.size () / cloud->points.size();
	  if (!silent) {
      std::cerr << cloud_cylinder->points.size () 
  	    << "\t total size: " << cloud->points.size()
  	    << "\tCylinder points ratio " << ratio << std::endl;
    }
	  return ratio;
  }

}
