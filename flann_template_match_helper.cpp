  /*
 * =====================================================================================
 *
 *       Filename:  build_tree.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  09/12/2015 12:20:17 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Wentao Luan (wluan), wluan@umd.edu
 *   Organization:  
 *
 * =====================================================================================
 */

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <fstream>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/vfh.h>
#include <pcl/filters/filter.h>

typedef std::pair<std::string, std::vector<float> > vfh_model;
typedef std::pair<std::string, std::vector<float> > xyz_model;
typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;

float radius;
flann::Index<flann::L2_Simple<float> >* index_models;
flann::Matrix<float> data_models;
std::vector<xyz_model> templates_models;
    
// class FlannVFH{

  
  // public:

  bool
  loadHist (const boost::filesystem::path &path, vfh_model &vfh)
  {

    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
    pcl::PointCloud<NormalType>::Ptr cloud_normals (new pcl::PointCloud<NormalType> ());

    if (pcl::io::loadPCDFile<PointType> (path.string(), *cloud) == -1) {
      PCL_ERROR ("Couldn't read file %s.pcd \n", path.string().c_str());
      return false;
    }
    pcl::PointCloud<PointType>::Ptr cloud_filtered (new pcl::PointCloud<PointType>);
    std::vector<int> filter_index;
    pcl::removeNaNFromPointCloud (*cloud, *cloud_filtered, filter_index);
    cloud = cloud_filtered;

    pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
    norm_est.setKSearch (10);
    norm_est.setInputCloud (cloud);
    norm_est.compute (*cloud_normals);

    pcl::VFHEstimation<PointType, NormalType, pcl::VFHSignature308> vfh_est;
    vfh_est.setInputCloud (cloud);
    vfh_est.setInputNormals (cloud_normals);

    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
    vfh_est.setSearchMethod (tree);
    pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());
    vfh_est.compute (*vfhs);

    vfh.second.resize (308);

    for (size_t i = 0; i < pcl::VFHSignature308::descriptorSize(); ++i)
    {
      vfh.second[i] = vfhs->points[0].histogram[i];
    }
    vfh.first = path.string ();
    return (true);
  }


bool
  loadPCDHist (const pcl::PointCloud<PointType>::Ptr cloud, vfh_model &vfh)
  {

    pcl::PointCloud<NormalType>::Ptr cloud_normals (new pcl::PointCloud<NormalType> ());

    pcl::PointCloud<PointType>::Ptr cloud_filtered (new pcl::PointCloud<PointType>);
    std::vector<int> filter_index;
    pcl::removeNaNFromPointCloud (*cloud, *cloud_filtered, filter_index);

    pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
    norm_est.setKSearch (10);
    norm_est.setInputCloud (cloud_filtered);
    norm_est.compute (*cloud_normals);

    pcl::VFHEstimation<PointType, NormalType, pcl::VFHSignature308> vfh_est;
    vfh_est.setInputCloud (cloud_filtered);
    vfh_est.setInputNormals (cloud_normals);

    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
    vfh_est.setSearchMethod (tree);
    pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());
    vfh_est.compute (*vfhs);

    vfh.second.resize (308);

    for (size_t i = 0; i < pcl::VFHSignature308::descriptorSize(); ++i)
    {
      vfh.second[i] = vfhs->points[0].histogram[i];
    }
    vfh.first = "test data";
    return (true);
  }




  /** \brief Load a set of VFH features that will act as the model (training data)
    * \param argc the number of arguments (pass from main ())
    * \param argv the actual command line arguments (pass from main ())
    * \param extension the file extension containing the VFH features
    * \param models the resultant vector of histogram models
    */
  void
  loadFeatureModels(const std::vector<xyz_model> templates, flann::Matrix<int>& indices, int nums, 
                    std::vector<vfh_model> &models)
  {

    for (size_t i = 0; i < nums; i++) {
      vfh_model m;
      if (loadHist (templates.at(indices[0][i]).first, m))
        models.push_back (m);
      
    }
  }


  /** \brief Search for the closest k neighbors
    * \param index the tree
    * \param model the query model
    * \param k the number of neighbors to search for
    * \param indices the resultant neighbor indices
    * \param distances the resultant neighbor distances
    */
  inline void
  nearestKSearch (const flann::Index<flann::ChiSquareDistance<float> > &index, const vfh_model &model, 
                  int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
  {
    // Query point
    flann::Matrix<float> p = flann::Matrix<float>(new float[model.second.size ()], 1, model.second.size ());
    memcpy (&p.ptr()[0], &model.second[0], p.cols * p.rows * sizeof (float));

    indices = flann::Matrix<int>(new int[k], 1, k);
    distances = flann::Matrix<float>(new float[k], 1, k);
    index.knnSearch (p, indices, distances, k, flann::SearchParams (512));
    delete[] p.ptr ();
  }

  float testModel(const pcl::PointCloud<PointType>::Ptr cloud, std::vector<vfh_model> &models,
                 const flann::Index<flann::ChiSquareDistance<float> >& index, int k) {
    vfh_model m;
    if (!loadPCDHist (cloud, m)) {
      PCL_ERROR("test data hist loading error \n");
      return 100000;
    }

    flann::Matrix<int> k_indices;
    flann::Matrix<float> k_distances;
    nearestKSearch (index, m, k, k_indices, k_distances);

    // pcl::console::print_info("Test data  match %s a distance of: %f\n", 
    //    models.at(k_indices[0][0]).first.c_str (), k_distances[0][0]);
    return k_distances[0][0];
  }



  bool
  loadLocation (const boost::filesystem::path &path, xyz_model &xyz) {
    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType> (path.string(), *cloud) == -1) {
      PCL_ERROR ("Couldn't read file %s.pcd \n", path.string().c_str());
      return false;
    }

    float zsum = 0, xsum = 0, ysum = 0;
    int num = 0;
    for (size_t i = 0; i < cloud->points.size(); i++) {
      PointType &p = cloud->points[i];
      if (!pcl::isFinite(p)) {
        continue;
      }
      zsum += p.z;
      xsum += p.x;
      ysum += p.y;
      num++;
    }
      
    float zavg = zsum / num, yavg = ysum / num, xavg = xsum /num;
    float distance = sqrt(zavg * zavg + xavg * xavg + yavg * yavg);
    xyz.second.resize(3);
    xyz.second[0] = xavg * 1;
    xyz.second[1] = yavg * 1;
    xyz.second[2] = zavg * 100;
    
    xyz.first = path.string ();

    return true;
  }


  bool
  loadPCDLocation (const pcl::PointCloud<PointType>::Ptr cloud, xyz_model &xyz) {
    float zsum = 0, xsum = 0, ysum = 0;
    int num = 0;
    for (size_t i = 0; i < cloud->points.size(); i++) {
      PointType &p = cloud->points[i];
      if (!pcl::isFinite(p)) {
        continue;
      }
      zsum += p.z;
      xsum += p.x;
      ysum += p.y;
      num++;
    }
      
    float zavg = zsum / num, yavg = ysum / num, xavg = xsum /num;
    float distance = sqrt(zavg * zavg + xavg * xavg + yavg * yavg);
    xyz.second.resize(3);
    xyz.second[0] = xavg * 1;
    xyz.second[1] = yavg * 1;
    xyz.second[2] = zavg * 100;
    
    xyz.first = "test_data";

    return true;
  }

  double match(pcl::PointCloud<PointType>::Ptr cloud) {
    
    xyz_model m;
    if (!loadPCDLocation(cloud, m)) {
      return 100000;
    }

    int max_num = 10;

    flann::Matrix<int> indices = flann::Matrix<int>(new int[max_num], 1, max_num);
    flann::Matrix<float> distances = flann::Matrix<float>(new float[max_num], 1, max_num);;
    flann::Matrix<float> p = flann::Matrix<float>(new float[m.second.size ()], 1, m.second.size ());
    memcpy(&p.ptr()[0], &m.second[0], p.cols * p.rows * sizeof (float));
    

    int nums = (*index_models).radiusSearch(p, indices, distances, radius, flann::SearchParams (512));
    delete []p.ptr();
    if (nums == 0) {
      std::cout << "No nearby templates" << std::endl;
      return 100000;
    }
    std::vector<vfh_model> models;
    // Load the model histograms
    loadFeatureModels(templates_models, indices, nums, models);
    // pcl::console::print_highlight ("Loaded %d VFH models.\n", (int)models.size ());

    // Convert data into FLANN format
    flann::Matrix<float> data (new float[models.size () * models[0].second.size ()], models.size (), models[0].second.size ());

    for (size_t i = 0; i < data.rows; ++i)
      for (size_t j = 0; j < data.cols; ++j)
        data[i][j] = models[i].second[j];
  
    // Build the tree index and save it to disk
    // pcl::console::print_error ("Building the kdtree index for %d elements...\n", (int)data.rows);
    flann::Index<flann::ChiSquareDistance<float> > model_index (data, flann::LinearIndexParams ());
    model_index.buildIndex ();
    int k = nums;
    double thresh = 30;
    double result = testModel(cloud, models, model_index, 1);
    delete[] data.ptr ();
    return result;
  }

  void
  loadLocationModels(const boost::filesystem::path &base_dir, const std::string &extension, 
                    std::vector<xyz_model> &models) {
    if (!boost::filesystem::exists (base_dir) && !boost::filesystem::is_directory (base_dir))
      return;

    for (boost::filesystem::directory_iterator it(base_dir); it != boost::filesystem::directory_iterator (); ++it)
    {
      if (!boost::filesystem::is_regular_file (it->status ()) || boost::filesystem::extension (it->path ()) != extension) {
        continue;
      }
      xyz_model m;
     
      if (loadLocation(base_dir / it->path ().filename (), m))
        models.push_back (m);
    }
  }

  void loadFlannVFH(std::string dir,  float r = 10) {
    radius = r;
    radius = radius * radius;
    std::string extension (".pcd");
    transform (extension.begin (), extension.end (), extension.begin (), (int(*)(int))tolower);
    loadLocationModels (dir, extension, templates_models);
    pcl::console::print_highlight ("%d Templates.\n", (int)templates_models.size ());
    // Convert data into FLANN format
    flann::Matrix<float> data (new float[templates_models.size () * templates_models[0].second.size ()], 
      templates_models.size (), templates_models[0].second.size ());
    data_models = data;
    for (size_t i = 0; i < data_models.rows; ++i) {
      for (size_t j = 0; j < data_models.cols; ++j) {
        data_models[i][j] = templates_models[i].second[j];
      }
    }

    // Build the tree index and save it to disk
    index_models = new flann::Index<flann::L2_Simple<float> >(data_models, flann::LinearIndexParams ());
    (*index_models).buildIndex();
    
  }

  // int
  // execute (int argc, char** argv)
  // {
  //   if (argc < 3)
  //   {
  //     PCL_ERROR ("Need at least three parameters! Syntax is: %s [train_dir] [test_dir]\n", argv[0]);
  //     return (-1);
  //   }

  //   float thresh = 50;
  //   float radius = 10;

  //   pcl::console::parse_argument (argc, argv, "-thresh", thresh);
  //   // Search for the k closest matches
  //   pcl::console::parse_argument (argc, argv, "-r", radius);

  //   radius = radius * radius;
  //   std::string extension (".pcd");
  //   transform (extension.begin (), extension.end (), extension.begin (), (int(*)(int))tolower);

  //   std::vector<xyz_model> templates;

  //   // Load the model histograms
  //   loadLocationModels (argv[1], extension, templates);
  //   pcl::console::print_highlight ("%d Templates.\n", (int)templates.size ());

  //   // Convert data into FLANN format
  //   flann::Matrix<float> data (new float[templates.size () * templates[0].second.size ()], templates.size (), templates[0].second.size ());

  //   for (size_t i = 0; i < data.rows; ++i) {
  //     for (size_t j = 0; j < data.cols; ++j) {
  //       data[i][j] = templates[i].second[j];
  //     }
  //   }

  //   // Build the tree index and save it to disk
  //   flann::Index<flann::L2_Simple<float> > index (data, flann::LinearIndexParams ());
  //   index.buildIndex ();
  //   flann::Matrix<int> indices;
  //   int nums;
  //   match(argv[2], extension, templates, index, radius, indices, nums);
    
  //   // testModel(argv[2], extension, models, index, k, thresh);


  //   delete[] data.ptr ();
  //   return (0);
  // }

