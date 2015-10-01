/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2012, Sudarshan Srinivasan <sudarshan85@gmail.com>
 *  Copyright (c) 2012-, Open Perception, Inc.
 * 
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 * 
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */
#define PCL_NO_PRECOMPILE
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <boost/thread/condition.hpp>
#include <boost/circular_buffer.hpp>
#include <csignal>
#include <limits>
#include <pcl/io/pcd_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/common/time.h> //fps calculations
#include <termios.h>
#include <stdio.h>
#include <pcl/visualization/cloud_viewer.h>
#include <math.h>  


#include "remove_plane_cluster2.cpp"

using namespace std;
using namespace pcl;
using namespace pcl::console;

ObjectFilter of;

bool is_done = false;
boost::mutex io_mutex;

#if defined(__linux__) || defined (TARGET_OS_MAC)
#include <unistd.h>
char getch(){
    /*#include <unistd.h>   //_getch*/
    /*#include <termios.h>  //_getch*/
    char buf=0;
    struct termios old={0};
    fflush(stdout);
    if(tcgetattr(0, &old)<0)
        perror("tcsetattr()");
    old.c_lflag&=~ICANON;
    old.c_lflag&=~ECHO;
    old.c_cc[VMIN]=1;
    old.c_cc[VTIME]=0;

    if(tcsetattr(0, TCSANOW, &old)<0)
        perror("tcsetattr ICANON");

    if(read(0,&buf,1) < 0)
        perror("read()");

    old.c_lflag|=ICANON;
    old.c_lflag|=ECHO;

    if(tcsetattr(0, TCSADRAIN, &old)<0)
        perror ("tcsetattr ~ICANON");
    printf("%c\n",buf);
    return buf;
 }

size_t 
getTotalSystemMemory ()
  {
  uint64_t memory = std::numeric_limits<size_t>::max ();

#ifdef _SC_AVPHYS_PAGES
  uint64_t pages = sysconf (_SC_AVPHYS_PAGES);
  uint64_t page_size = sysconf (_SC_PAGE_SIZE);
  
  memory = pages * page_size;
  
#elif defined(HAVE_SYSCTL) && defined(HW_PHYSMEM)
  // This works on *bsd and darwin.
  unsigned int physmem;
  size_t len = sizeof physmem;
  static int mib[2] = { CTL_HW, HW_PHYSMEM };

  if (sysctl (mib, ARRAY_SIZE (mib), &physmem, &len, NULL, 0) == 0 && len == sizeof (physmem))
  {
    memory = physmem;
  }
#endif

  if (memory > uint64_t (std::numeric_limits<size_t>::max ()))
  {
    memory = std::numeric_limits<size_t>::max ();
  }
  
  print_info ("Total available memory size: %lluMB.\n", memory / 1048576ull);
  return size_t (memory);
}

const size_t BUFFER_SIZE = size_t (getTotalSystemMemory () / (640 * 480 * sizeof (pcl::PointXYZRGBA)));
#else

const size_t BUFFER_SIZE = 200;
#endif

//////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
class PCDBuffer
{
  public:
    PCDBuffer () {}

    bool 
    pushBack (typename PointCloud<PointT>::ConstPtr); // thread-save wrapper for push_back() method of ciruclar_buffer

    typename PointCloud<PointT>::ConstPtr 
    getFront (); // thread-save wrapper for front() method of ciruclar_buffer

    inline bool 
    isFull ()
    {
      boost::mutex::scoped_lock buff_lock (bmutex_);
      return (buffer_.full ());
    }

    inline bool
    isEmpty ()
    {
      boost::mutex::scoped_lock buff_lock (bmutex_);
      return (buffer_.empty ());
    }

    inline int 
    getSize ()
    {
      boost::mutex::scoped_lock buff_lock (bmutex_);
      return (int (buffer_.size ()));
    }

    inline int 
    getCapacity ()
    {
      return (int (buffer_.capacity ()));
    }

    inline void 
    setCapacity (int buff_size)
    {
      boost::mutex::scoped_lock buff_lock (bmutex_);
      buffer_.set_capacity (buff_size);
    }
    inline void releaseBuff_Empty() {
      buff_empty_.notify_one ();
    }
  private:
    PCDBuffer (const PCDBuffer&); // Disabled copy constructor
    PCDBuffer& operator = (const PCDBuffer&); // Disabled assignment operator

    boost::mutex bmutex_;
    boost::condition_variable buff_empty_;
    boost::circular_buffer<typename PointCloud<PointT>::ConstPtr> buffer_;
};

//////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> bool 
PCDBuffer<PointT>::pushBack (typename PointCloud<PointT>::ConstPtr cloud)
{
  bool retVal = false;
  {
    boost::mutex::scoped_lock buff_lock (bmutex_);
    if (!buffer_.full ())
      retVal = true;
    buffer_.push_back (cloud);
  }
  buff_empty_.notify_one ();
  return (retVal);
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> typename PointCloud<PointT>::ConstPtr 
PCDBuffer<PointT>::getFront ()
{
  typename PointCloud<PointT>::ConstPtr cloud;
  {
    boost::mutex::scoped_lock buff_lock (bmutex_);
    while (buffer_.empty ())
    {
      if (is_done)
        break;
      {
        boost::mutex::scoped_lock io_lock (io_mutex);
        //cerr << "No data in buffer_ yet or buffer is empty." << endl;
        print_warn ("Before wait buff_lock\n");
      }

      buff_empty_.wait (buff_lock);
    }

    cloud = buffer_.front ();
    buffer_.pop_front ();
  }
  return (cloud);
}


//////////////////////////////////////////////////////////////////////////////////////////
// Producer thread class

// int write_counter = 25;
boost::mutex wflag_mutex;
boost::mutex noise_flag_mutex;

bool write_once = false;
bool calculate_noise = false;

template <typename PointT>
class Producer
{
  private:
    ///////////////////////////////////////////////////////////////////////////////////////
    void printLocation(const typename PointCloud<PointT>::Ptr cloud) {
      
      float zsum = 0;
      float xsum = 0;
      float ysum = 0;
      int num = 0;
      for (size_t i = 0; i < cloud->points.size(); i++) {
        PointT &p = cloud->points[i];
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
     std::cout << "   x: " << xsum / num << "   y: " << ysum / num 
        << "   z: " << zsum / num << "   distance: " << distance << std::endl;

    }

    float getMeanDepth(const typename PointCloud<PointT>::ConstPtr cloud, int row_up, int row_down, int col_left, int col_right) {
      float sum = 0;
      int count = 0;
      for (int row = row_up; row < row_down; row++) {
        for (int col = col_left; col < col_right; col++) {
          const PointT &p = cloud->at(col, row);
          if (!isFinite(p)) {
            continue;
          }
          sum += p.z;
          count++;
        }
      }
      return sum / count;
    }

    float getSTD(const std::vector<float>& nums) {
      float mean = getMean(nums);
      float error = 0;
      for (std::vector<float>::const_iterator it = nums.begin(); it != nums.end(); ++ it) {
        error += ((*it) - mean) * ((*it) - mean);
      }
      return sqrt(error / nums.size());
    }

    float getMean(const std::vector<float>& nums) {
      float sum = 0;
      for (std::vector<float>::const_iterator it = nums.begin(); it != nums.end(); ++it) {
        sum += *it;
      }
      return sum / nums.size();
    }

    void 
    grabberCallBack (const typename PointCloud<PointT>::ConstPtr& cloud_in)
    {
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
      pcl::copyPointCloud(*cloud_in, *cloud);
      stringstream ss;
      ss << prefix << pcd_nb;

      int status = 0;
      {
        boost::mutex::scoped_lock wflag_lock (wflag_mutex);
        status = of.execute(cloud, ss.str(), write_once);
        if (write_once) {
          pcd_nb++;
          PCL_ERROR("\nWrite once\n");
        }
        write_once = false;
      }
      if (status == 1) {
        viewer.showCloud (cloud);
        std::cout << "Found " << cloud->points.size() << std::endl;
        printLocation(cloud);
      } else {
        std::cout << cloud->points.size() << std::endl;
      }

      // {
      //   boost::mutex::scoped_lock wflag_lock (wflag_mutex);
      //   if (!write_once) {
      //     return;
      //   }
      // }
      // {
      //   boost::mutex::scoped_lock wflag_lock (wflag_mutex);
      //   write_once = false;
      // }

      // if (!buf_.pushBack (cloud))
      // {
      //   {
      //     boost::mutex::scoped_lock io_lock (io_mutex);
      //     print_warn ("Warning! Buffer was full, overwriting data!\n");
      //   }
      // }
      // // FPS_CALC ("Write cloud callback.", buf_);
      // {
      //  boost::mutex::scoped_lock io_lock (io_mutex);
      //  print_warn ("Write once\n");
      // }
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    void 
    grabAndSend ()
    {
      OpenNIGrabber* grabber = new OpenNIGrabber ();
      grabber->getDevice ()->setDepthOutputFormat (depth_mode_);

      Grabber* interface = grabber;
      boost::function<void (const typename PointCloud<PointT>::ConstPtr&)> f = boost::bind (&Producer::grabberCallBack, this, _1);
      interface->registerCallback (f);
      interface->start ();

     while (true)
      {
        if (is_done) {
          break;
        }
        char c;
        {
       //  boost::mutex::scoped_lock io_lock (io_mutex);
          c = getch();
        }
        if (c == 'q') {
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_info ("\n'q' detected, exit condition set to true.\n");
          is_done = true;
        } else if (!is_done && c == 'w') {
          {
            boost::mutex::scoped_lock wflag_lock (wflag_mutex);
            write_once = true;
          }
        } else if (!is_done && c == 'a'){
          of.min_x += 0.1;
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_info ("\n min_x %f.\n", of.min_x);
        } else if (!is_done && c == 'z'){
          of.min_x -= 0.1;
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_info ("\n min_x %f.\n", of.min_x);
        } else if (!is_done && c == 's'){
          of.max_x += 0.1;
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_info ("\n max_x %f.\n", of.max_x);
        } else if (!is_done && c == 'x'){
          of.max_x -= 0.1;
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_info ("\n max_x %f.\n", of.max_x);
        } else if (!is_done && c == 'd'){
          of.min_y += 0.1;
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_info ("\n min_y %f.\n", of.min_y);
        } else if (!is_done && c == 'c'){
          of.min_y -= 0.1;
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_info ("\n min_y %f.\n", of.min_y);
        } else if (!is_done && c == 'f'){
          of.max_y += 0.1;
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_info ("\n max_y %f.\n", of.max_y);
        } else if (!is_done && c == 'v'){
          of.max_y -= 0.1;
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_info ("\n max_y %f.\n", of.max_y);
        } else if (!is_done && c == 'g'){
          of.min_z += 0.1;
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_info ("\n min_z %f.\n", of.min_z);
        } else if (!is_done && c == 'b'){
          of.min_z -= 0.1;
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_info ("\n min_z %f.\n", of.min_z);
        } else if (!is_done && c == 'h'){
          of.max_z += 0.1;
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_info ("\n max_z %f.\n", of.max_z);
        } else if (!is_done && c == 'n'){
          of.max_z -= 0.1;
          boost::mutex::scoped_lock io_lock (io_mutex);
          print_info ("\n max_z %f.\n", of.max_z);
        }

        boost::this_thread::sleep (boost::posix_time::seconds (1));
      }

      interface->stop ();
    }

  public:
    Producer (PCDBuffer<PointT> &buf, openni_wrapper::OpenNIDevice::DepthMode depth_mode, string prefix_str)
      : buf_ (buf),
        depth_mode_ (depth_mode),
        prefix(prefix_str),
        viewer ("PCL OpenNI Viewer"),
        num_pcd_sample(0)

    {
      thread_.reset (new boost::thread (boost::bind (&Producer::grabAndSend, this)));
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    void
    stop ()
    {
      thread_->join ();
      boost::mutex::scoped_lock io_lock (io_mutex);
      print_highlight ("Producer done.\n");
      buf_.releaseBuff_Empty();

    }

  private:
    PCDBuffer<PointT> &buf_;
    openni_wrapper::OpenNIDevice::DepthMode depth_mode_;
    boost::shared_ptr<boost::thread> thread_;
    pcl::visualization::CloudViewer viewer;
    int num_pcd_sample;
    std::vector<float> depths_sample;
    static int pcd_nb;
    string prefix;
};

template <typename PointT>
int Producer<PointT>::pcd_nb = 0;
//////////////////////////////////////////////////////////////////////////////////////////
// Consumer thread class
template <typename PointT>
class Consumer
{
  private:
    ///////////////////////////////////////////////////////////////////////////////////////
    void 
    writeToDisk (const typename PointCloud<PointT>::ConstPtr& cloud)
    {
      stringstream ss;
      ss << prefix << pcd_nb++ << ".pcd";
      writer_.writeBinaryCompressed (ss.str (), *cloud);
      // FPS_CALC ("cloud write.", buf_);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // Consumer thread function
    void 
    receiveAndProcess ()
    {
      while (true) {
        if (is_done)
          break;
        if (!buf_.isEmpty()) {
          writeToDisk (buf_.getFront ());
        }
        boost::this_thread::sleep (boost::posix_time::seconds (1));
      }

      {
        boost::mutex::scoped_lock io_lock (io_mutex);
        print_info ("%ld  remains. \n", buf_.getSize ());
      }
    }

  public:
    Consumer (PCDBuffer<PointT> &buf, string prefix_str)
      : buf_ (buf),
        prefix(prefix_str)
    {
      thread_.reset (new boost::thread (boost::bind (&Consumer::receiveAndProcess, this)));
    }

    /////////////////////////////////////////////////////////////boost::this_thread::sleep (boost::posix_time::seconds (1));//////////////////////////
    void
    stop ()
    {
      thread_->join ();
      boost::mutex::scoped_lock io_lock (io_mutex);
      print_highlight ("Consumer done.\n");
    }
   
    // int& pcd_nb() {
    //   return this->pcd_nb;
    // }
  private:
    PCDBuffer<PointT> &buf_;
    boost::shared_ptr<boost::thread> thread_;
    PCDWriter writer_;
    static int pcd_nb;
    string prefix;
  };

template <typename PointT>
int Consumer<PointT>::pcd_nb = 0;

//////////////////////////////////////////////////////////////////////////////////////////



// void 
// keyS (int)
// {
//   boost::mutex::scoped_lock io_lock (io_mutex);
//   print_info ("\nkey S detected, save point cloud.\n");
//   saveOne = true;
// }

//////////////////////////////////////////////////////////////////////////////////////////
void
printHelp (int default_buff_size, int, char **argv)
{
  using pcl::console::print_error;
  using pcl::console::print_info;

  print_error ("Syntax is: %s ((<device_id> | <path-to-oni-file>) [-xyz] [-shift] [-buf X]  | -l [<device_id>] | -h | --help)]\n", argv [0]);
  print_info ("%s -h | --help : shows this help\n", argv [0]);
  print_info ("%s -xyz : save only XYZ data, even if the device is RGB capable\n", argv [0]);
  print_info ("%s -shift : use OpenNI shift values rather than 12-bit depth\n", argv [0]);
  print_info ("%s -buf X ; use a buffer size of X frames (default: ", argv [0]);
  print_value ("%d", default_buff_size); print_info (")\n");
  print_info ("%s -l : list all available devices\n", argv [0]);
  print_info ("%s -l <device-id> :list all available modes for specified device\n", argv [0]);
  print_info ("\t\t<device_id> may be \"#1\", \"#2\", ... for the first, second etc device in the list\n");
#ifndef _WIN32
  print_info ("\t\t                   bus@address for the device connected to a specific usb-bus / address combination\n");
  print_info ("\t\t                   <serial-number>\n");
#endif
  print_info ("\n\nexamples:\n");
  print_info ("%s \"#1\"\n", argv [0]);
  print_info ("\t\t uses the first device.\n");
  print_info ("%s  \"./temp/test.oni\"\n", argv [0]);
  print_info ("\t\t uses the oni-player device to play back oni file given by path.\n");
  print_info ("%s -l\n", argv [0]);
  print_info ("\t\t list all available devices.\n");
  print_info ("%s -l \"#2\"\n", argv [0]);
  print_info ("\t\t list all available modes for the second device.\n");
  #ifndef _WIN32
  print_info ("%s A00361800903049A\n", argv [0]);
  print_info ("\t\t uses the device with the serial number \'A00361800903049A\'.\n");
  print_info ("%s 1@16\n", argv [0]);
  print_info ("\t\t uses the device on address 16 at USB bus 1.\n");
  #endif
}

//////////////////////////////////////////////////////////////////////////////////////////
int
main (int argc, char** argv)
{
  // execute(argc, argv);
  of.initialize(argc, argv);
  print_highlight ("PCL OpenNI Recorder for saving buffered PCD (binary compressed to disk). See %s -h for options.\n", argv[0]);

  std::string device_id ("");
  int buff_size = BUFFER_SIZE;

  openni_wrapper::OpenNIDriver& driver = openni_wrapper::OpenNIDriver::getInstance ();
  if (driver.getNumberDevices () > 0) {
    cout << "Device Id not set, using first device." << endl;
  }
  

  openni_wrapper::OpenNIDevice::DepthMode depth_mode = openni_wrapper::OpenNIDevice::OpenNI_12_bit_depth;
  if (find_switch (argc, argv, "-shift"))
    depth_mode = openni_wrapper::OpenNIDevice::OpenNI_shift_values;

  if (parse_argument (argc, argv, "-buf", buff_size) != -1)
    print_highlight ("Setting buffer size to %d frames.\n", buff_size);
  else
    print_highlight ("Using default buffer size of %d frames.\n", buff_size);

  print_highlight ("Starting the producer and consumer threads... Press 'w' to  capture and 'q' to quit\n");
 
  OpenNIGrabber grabber (device_id);
  if (grabber.providesCallback<OpenNIGrabber::sig_cb_openni_point_cloud_rgba> ()) {
    print_highlight ("PointXYZRGBA enabled.\n");
    PCDBuffer<PointXYZRGBA> buf;
    buf.setCapacity (buff_size);
    string prefix = "frame";
    pcl::console::parse_argument (argc, argv, "-name", prefix);

    Producer<PointXYZRGBA> producer (buf, depth_mode, prefix);
    // boost::this_thread::sleep (boost::posix_time::seconds (2));
  
    Consumer<PointXYZRGBA> consumer (buf, prefix);
    // consumer.pcd_nb() = 0;

    producer.stop ();
    consumer.stop ();
  }
  return (0);
}

