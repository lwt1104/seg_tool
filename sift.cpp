#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <math.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include "opencv2/imgproc/imgproc.hpp"


using namespace cv;

float thresh = 0.2;
double max_metric = -1;
double min_metric = 100000;

Mat img_object;
/** @function main */
int execute(Mat img_scene)
{

  if( !img_object.data || !img_scene.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  detector.detect( img_object, keypoints_object );
  detector.detect( img_scene, keypoints_scene );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_object, descriptors_scene;

  extractor.compute( img_object, keypoints_object, descriptors_object );
  extractor.compute( img_scene, keypoints_scene, descriptors_scene );
 
  if (descriptors_scene.rows == 0 || descriptors_scene.cols == 0) {
    return 0;
  }
  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 3*min_dist )
     { good_matches.push_back( matches[i]); }
  }

  Mat img_matches;

  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  Mat H = findHomography( obj, scene, CV_RANSAC );

  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);
  double area0 = contourArea(scene_corners) / img_scene.rows / img_scene.cols;
  std::cout << area0 << std::endl;


  // double ratio =  fabs(H.at<double>(0, 1)) + fabs(H.at<double>(1, 0));
  // std::cout << H.at<double>(0, 1) << "\t" << H.at<double>(1, 0) << "\t" << ratio << std::endl;

  double ratio = area0;
  if (ratio > max_metric) {
    max_metric = ratio;
  } 
  if (ratio != 0.0 && ratio < min_metric) {
    min_metric = ratio;
  }
  return 0;
  }

void testModel(const boost::filesystem::path &base_dir, const std::string &extension) {
  if (!boost::filesystem::exists (base_dir) && !boost::filesystem::is_directory (base_dir))
    return;
  int num = 0;
  int detect = 0;
  for (boost::filesystem::directory_iterator it(base_dir); it != boost::filesystem::directory_iterator (); ++it)
  {
    if (!boost::filesystem::is_regular_file (it->status ()) || boost::filesystem::extension (it->path ()) != extension
        || it->path ().filename().string().find("box") == std::string::npos) {
      continue;
    }
    Mat img;
    img = imread(it->path().string(), CV_LOAD_IMAGE_GRAYSCALE);
    std::cout<<it->path ().filename().string() << "  ";
    int result = execute(img);

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
  if (argc < 3)
  {
    PCL_ERROR ("Need at two parameters! Syntax is: %s [src_dir] [test_dir] \n", argv[0]);
    return (-1);
  }
  img_object = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE);

  pcl::console::parse_argument (argc, argv, "-thresh", thresh);

  std::string extension (".png");
  transform (extension.begin (), extension.end (), extension.begin (), (int(*)(int))tolower);

  // std::cout << std::endl << hsv_base.size() 
  //   << std::endl << hsv_base.type()
  //   << std::endl << hsv_base.channels()
  //   <<std::endl;

  testModel(argv[2], extension);

  return 0;
}