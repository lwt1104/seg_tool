#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include "png++/png.hpp"
#include <pcl/console/parse.h>
#include <pcl/filters/passthrough.h>


struct _PointXYZRGBAUV
{
  PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
  PCL_ADD_RGB;
  uint32_t u;   // col number
  uint32_t v;   // row number
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

 struct EIGEN_ALIGN16 PointXYZRGBAUV : public _PointXYZRGBAUV
  {
    inline PointXYZRGBAUV(const _PointXYZRGBAUV &p)
    {
      x = p.x; y = p.y; z = p.z; data[3] = 1.0f;
      rgba = p.rgba;
      u = p.u; v = p.v;
    }

    inline PointXYZRGBAUV ()
    {
      x = y = z = 0.0f;
      data[3] = 1.0f;
      r = g = b = a = 0;
      u = v = 0;
    }
    inline PointXYZRGBAUV (const pcl::PointXYZRGBA &p, uint32_t uu, uint32_t vv)
    {
      x = p.x; y = p.y; z = p.z; data[3] = 1.0f;
      rgba = p.rgba;
      u = uu; v = vv;
    }

    friend std::ostream& operator << (std::ostream& os, const PointXYZRGBAUV& p);
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZRGBAUV,           // here we assume a XYZ + "test" (as fields)
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, rgb, rgb)
                                   (uint32_t, u, u)
                                   (uint32_t, v, v)
);

class ObjectFilter{

  typedef pcl::PointXYZRGBA PointType;
  typedef PointXYZRGBAUV UVPointType;

  bool pass_through_filter(pcl::PointCloud<UVPointType>::Ptr cloud, float min_depth , float max_depth, float min_x, float max_x, float min_y, float max_y) {
    // Create the filtering object
    pcl::PointCloud<UVPointType>::Ptr cloud_filtered(new pcl::PointCloud<UVPointType>);
    pcl::PassThrough<UVPointType> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (min_depth, max_depth);
    pass.filter (*cloud);

    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (min_x, max_x);
    pass.filter (*cloud);

    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (min_y, max_y);
    pass.filter (*cloud);
    // *cloud = *(cloud_filtered);
    return true;
  }
  bool
  remove_plane(pcl::PointCloud<UVPointType>::Ptr cloud, float min_depth , float max_depth, float min_x, float max_x, float min_y, float max_y) {
      
      pass_through_filter(cloud, min_depth, max_depth, min_x, max_x, min_y, max_y);

      pcl::PointCloud<UVPointType>::Ptr cloud_filtered(new pcl::PointCloud<UVPointType>);
      pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
      // Create the segmentation object
      pcl::SACSegmentation<UVPointType> seg;
      // Optional
      seg.setOptimizeCoefficients (true);
      // Mandatory
      seg.setModelType (pcl::SACMODEL_PLANE);
      seg.setMethodType (pcl::SAC_RANSAC);
      seg.setDistanceThreshold (0.01);
      int nr_points = cloud->points.size();
      while (cloud->points.size () > 0.3 * nr_points) {
        seg.setInputCloud (cloud);
        seg.segment (*inliers, *coefficients);

        if (inliers->indices.size () == 0){
          PCL_ERROR ("Could not estimate a planar model for the given dataset.");
          return (-1);
        }
        // std::cerr << "Total number of points before segment plane: " << cloud->size() << std::endl;

        // std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
        //                                     << coefficients->values[1] << " "
        //                                     << coefficients->values[2] << " " 
        //                                     << coefficients->values[3] << std::endl;

        // std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;

        if (fabs(coefficients->values[2]) > 3 * (fabs(coefficients->values[1]))) {
          std::cerr << "Not allowed" << std::endl;
          break;
        }

        pcl::ExtractIndices<UVPointType> extract;
        // Extract the outliers
        extract.setInputCloud (cloud);
        extract.setIndices (inliers);
        extract.setNegative (true);
        extract.filter (*cloud_filtered);
        *cloud = *(cloud_filtered);
      }

    return true;
  }

  public:

  int 
  execute (pcl::PointCloud<PointType>::Ptr cloud, std::string filename, bool should_write)
  {
    // Read in the cloud data
    pcl::PCDReader reader;
    pcl::PCDWriter writer;
    
    std::string pcd_filename = filename + ".pcd";
    std::string png_filename = filename + ".png";

    // Take the origional png image out
    png::image<png::rgb_pixel> origin_image(cloud->width, cloud->height);
    int origin_index = 0;
    for (size_t y = 0; y < origin_image.get_height (); ++y) {
      for (size_t x = 0; x < origin_image.get_width (); ++x) {
        const PointType & p = cloud->points[origin_index++];
        origin_image[y][x] = png::rgb_pixel(p.r, p.g, p.b);
      }
    }

    // origin_image.write(png_filename);

    // std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*  


    pcl::PointCloud<UVPointType>::Ptr cloud_uv (new pcl::PointCloud<UVPointType>);
    for (size_t index = 0; index < cloud->points.size(); index++) {
      const PointType & p = cloud->points[index];
      if (p.x != p.x || p.y != p.y || p.z != p.z) { // if current point is invalid
        continue;
      }
      UVPointType cp = UVPointType(p, index % cloud-> width, index / cloud->width);
      cloud_uv->points.push_back (cp); 
    }
    cloud_uv->width = cloud_uv->points.size ();
    cloud_uv->height = 1;

    remove_plane(cloud_uv, min_z, max_z, min_x, max_x, min_y, max_y);

    std::string remove_plane_pcd = filename + "plane.pcd";
    // pcl::io::savePCDFile(remove_plane_pcd, *cloud_uv);

    // std::cout << "PointCloud after removing the plane has: " << cloud->points.size () << " data points." << std::endl; //*
    uint32_t xmin = 1000, xmax = 0, ymin = 1000, ymax = 0;
    
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<UVPointType>::Ptr tree (new pcl::search::KdTree<UVPointType>);
    tree->setInputCloud (cloud_uv);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<UVPointType> ec;
    ec.setClusterTolerance (0.035); // 2cm
    ec.setMinClusterSize (600);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_uv);
    ec.extract (cluster_indices);
    
    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
      if (j >= k) {
        break;
      }
      pcl::PointCloud<PointType>::Ptr cloud_cluster (new pcl::PointCloud<PointType>);
      xmin = 1000;
      xmax = 0; 
      ymin = 1000;
      ymax = 0;

      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
          UVPointType& p = cloud_uv->points[*pit];
          PointType cp_rgb;
          cp_rgb.x = p.x; cp_rgb.y = p.y; cp_rgb.z = p.z;
          cp_rgb.rgba = p.rgba; 
          cloud_cluster->points.push_back(cp_rgb);

          xmin = std::min(xmin, p.u);
          xmax = std::max(xmax, p.u);
          ymin = std::min(ymin, p.v);
          ymax = std::max(ymax, p.v);
      }
      cloud_cluster->is_dense = true;
      cloud_cluster->width = cloud_cluster->points.size();
      cloud_cluster->height = 1;

      std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    
      std::stringstream ss_pcd;
      ss_pcd << filename << "cluster_" << j << ".pcd";
      *cloud = *cloud_cluster;
      if (should_write) {
        pcl::io::savePCDFile(ss_pcd.str(), *cloud_cluster);
      }

      // output image in a line
      png::image<png::rgb_pixel> image(cloud_cluster->width, cloud_cluster->height);
      int i = 0;
      for (size_t y = 0; y < image.get_height (); ++y) {
        for (size_t x = 0; x < image.get_width (); ++x) {
          const PointType & p = cloud_cluster->points[i++];
          image[y][x] = png::rgb_pixel(p.r, p.g, p.b);
        }
      }
      std::stringstream ss_png;
      ss_png << filename << "cluster_" << j << ".png";
      if (should_write) {
        image.write(ss_png.str());
      }
      //crop out image patch

      png::image<png::rgb_pixel> image_patch(xmax - xmin + 1, ymax - ymin + 1);
      for (size_t y = 0; y < image_patch.get_height (); ++y) {
        for (size_t x = 0; x < image_patch.get_width (); ++x) {
          image_patch[y][x] = origin_image[y+ymin][x+xmin];
        }
      }
      std::stringstream ss_box_png;
      ss_box_png << filename << "cluster_" << j << "box.png";
      if (should_write) {
        image_patch.write(ss_box_png.str());
      }
      j++;
      return 1;

    }

    return (0);
  }


  ObjectFilter() { 
    min_z = 0.1;
    max_z = 3.0;
    max_x = 3.0;
    min_x = -3.0;
    max_y = 3.0;
    min_y = -3.0;
    k = 1;
  }

  ObjectFilter(int argc, char** argv) { 
    this->initialize(argc, argv);
  }
  void initialize(int argc, char** argv) {
    pcl::console::parse_argument (argc, argv, "-min_z", min_z);

    pcl::console::parse_argument (argc, argv, "-max_z", max_z);

    pcl::console::parse_argument (argc, argv, "-max_x", max_x);

    pcl::console::parse_argument (argc, argv, "-min_x", min_x);

    pcl::console::parse_argument (argc, argv, "-max_y", max_y);

    pcl::console::parse_argument (argc, argv, "-min_y", min_y);

    pcl::console::parse_argument (argc, argv, "-k", k);
  }

  float min_z;
  float max_z;
  float max_x;
  float min_x;
  float max_y;
  float min_y;
  int k;
  
};