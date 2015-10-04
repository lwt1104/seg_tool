/**
 * @file compareHist_Demo.cpp
 * @brief Sample code to use the function compareHist
 * @author OpenCV team
 */
// #include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>



using namespace std;
using namespace cv;


std::vector<Mat> hist_bases;
float thresh = 0.2;
double max_metric = -1;
double min_metric = 100000;


int getHist(Mat src_rgb, Mat& hist) {
  Mat src_hsv;
  cvtColor(src_rgb, src_hsv, COLOR_BGR2HSV);
  /// Using x bins for hue and x for saturation
  int h_bins = 5; int s_bins = 7;
  int histSize[] = { h_bins, s_bins };
// hue varies from 0 to 179, saturation from 0 to 255
  float h_ranges[] = { 0, 180 };
  float s_ranges[] = { 0, 256 };
  const float* ranges[] = { h_ranges, s_ranges };
// Use the o-th and 1-st channels
  int channels[] = { 0, 1 };

/// Calculate the histograms for the HSV images
  calcHist( &src_hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false );
  normalize( hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );

  // std::cout << hist;
  return 1;
}

int getHist2(Mat src_rgb, Mat& hist) {
  Mat src = src_rgb;
  /// Using x bins for hue and x for saturation
  vector<Mat> bgr_planes;
  split( src, bgr_planes );

  /// Establish the number of bins
  int histSize = 5;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat b_hist, g_hist, r_hist;

  /// Compute the histograms:
  calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
  
  hist = b_hist;
  vconcat(hist, g_hist, hist);
  vconcat(hist, r_hist, hist);

  // std::cout << hist << std::endl;
  normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );
  // std::cout << hist << std::endl;
  
  return 1;
}

int histMatch(Mat img) {
  Mat img_hist;
  getHist(img, img_hist);
  double min_d = 10000;
  for (int i = 0; i < hist_bases.size(); i++) {
    double d1 = compareHist( hist_bases[i], img_hist, 1 );
    double d3 = compareHist( hist_bases[i], img_hist, 3 );
    if (d3 < min_d) {
      min_d = d3;
    }
  }

  std::cout << min_d << std::endl;
  if (min_d != 10000 && min_d > max_metric) {
    max_metric = min_d;
  } 
  if (min_d != 0 && min_d < min_metric) {
    min_metric = min_d;
  }

  if (min_d < thresh) {
    return 1;
  } else {
    return 0;
  }
}

void testModel(const boost::filesystem::path &base_dir, const std::string &extension) {
  if (!boost::filesystem::exists (base_dir) && !boost::filesystem::is_directory (base_dir))
    return;
  int num = 0;
  int detect = 0;
  for (boost::filesystem::directory_iterator it(base_dir); it != boost::filesystem::directory_iterator (); ++it)
  {
    if (!boost::filesystem::is_regular_file (it->status ()) || boost::filesystem::extension (it->path ()) != extension
        || it->path ().filename().string().find("box") != std::string::npos) {
      continue;
    }
    Mat img;
    img = imread(it->path().string(), 1);
    std::cout<<it->path ().filename().string() << "  ";
    int result = histMatch(img);

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

int loadBases(const boost::filesystem::path &base_dir, const std::string &extension) {
  if (!boost::filesystem::exists (base_dir) && !boost::filesystem::is_directory (base_dir))
    return -1;
  for (boost::filesystem::directory_iterator it(base_dir); it != boost::filesystem::directory_iterator (); ++it)
  {
    if (!boost::filesystem::is_regular_file (it->status ()) || boost::filesystem::extension (it->path ()) != extension
        || it->path ().filename().string().find("box") != std::string::npos) {
      continue;
    }

    Mat img, img_hist;

    img = imread(it->path().string(), 1);
    getHist(img, img_hist);
    hist_bases.push_back(img_hist);
  }
  return 0;
}

int
main (int argc, char** argv)
{
  if (argc < 3)
  {
    PCL_ERROR ("Need at two parameters! Syntax is: %s [src_dir] [test_dir] \n", argv[0]);
    return (-1);
  }

  pcl::console::parse_argument (argc, argv, "-thresh", thresh);

  std::string extension (".png");
  transform (extension.begin (), extension.end (), extension.begin (), (int(*)(int))tolower);

  loadBases(argv[1], extension);
  // std::cout << std::endl << hsv_base.size() 
  //   << std::endl << hsv_base.type()
  //   << std::endl << hsv_base.channels()
  //   <<std::endl;

  testModel(argv[2], extension);

  return 0;
}